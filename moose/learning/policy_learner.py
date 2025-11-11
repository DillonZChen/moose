import concurrent.futures
import logging
import math
import os
import random
from typing import Any, Optional, Union

import pddl
import termcolor as tc
from pddl.action import Action
from pddl.core import Formula, Problem
from pddl.logic import Constant, Predicate
from pddl.logic.base import And, Not
from pddl.logic.functions import EqualTo, GreaterEqualThan, NumericFunction, NumericValue
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from moose.options import Options
from moose.planning import Plan, State, get_literals_list, get_plan, is_numeric_domain, new_problem
from moose.planning.progression import apply_action, check_formula
from moose.planning.regression import check_regression_support, regress_action
from moose.planning.strings import state_to_string, strings_to_actions, token_to_numeric_value
from moose.policy import Policy
from moose.precedence import PrecedenceValue
from moose.rule import OBJECT_TYPE, Rule
from moose.util import natural_sort
from moose.util.managers import TimerContextManager


def get_problem_pddls(options: Options, validation: bool = False) -> list[str]:
    problem_dir = os.path.normpath(options.training_dir)
    assert os.path.exists(problem_dir), problem_dir
    assert os.path.isdir(problem_dir), problem_dir

    problem_pddls = []
    n_problems = options.num_training if not validation else options.num_validation
    n_problems = n_problems if n_problems >= 0 else len(os.listdir(problem_dir))
    for f in natural_sort(os.listdir(problem_dir))[:n_problems]:
        assert f.endswith(".pddl"), f"Unexpected file: {f}"
        problem_pddls.append(f"{problem_dir}/{f}")

    return problem_pddls


class PolicyLearner:
    def __init__(self, options: Options) -> None:
        if options.training_dir is None:
            options.training_dir = os.path.dirname(options.domain_file) + "/training"
        self._domain_pddl = os.path.normpath(options.domain_file)
        self._problem_dir = os.path.normpath(options.training_dir)
        self._options = options

        assert os.path.exists(self._domain_pddl), self._domain_pddl
        assert os.path.exists(self._problem_dir), self._problem_dir
        assert os.path.isdir(self._problem_dir), self._problem_dir

        # Load domain and useful information
        self._domain = pddl.parse_domain(self._domain_pddl)
        self._constants = {c.name: c for c in self._domain.constants}
        self._name_to_action = {action.name: action for action in self._domain.actions}
        self._is_numeric = is_numeric_domain(self._domain)
        logging.info(f"Detected domain is_numeric={self._is_numeric}")

        # Checks for whether learner would work
        check_regression_support(self._domain)

        # Initialise
        self._policy = Policy(domain=self._domain)
        self._policy.set_train_options(self._options)

    @property
    def policy(self) -> Policy:
        return self._policy

    def set_policy(self, policy: Policy) -> None:
        self._policy = policy

    def learn(self) -> None:
        """Main learning routine."""

        problems = get_problem_pddls(self._options)
        rules = [None for _ in range(len(problems))]

        with logging_redirect_tqdm():
            pbar = tqdm(list(enumerate(problems)))

            max_workers = self._options.num_workers

            # Define a worker function that will process each problem
            def process_problem(args):
                i, problem_pddl = args
                problem = pddl.parse_problem(problem_pddl)
                with TimerContextManager(f"learning from {os.path.basename(problem_pddl)}"):
                    return i, self.learn_from_problem(problem, inplace=False, prob_id=i)

            # Use ThreadPoolExecutor to process problems in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_problem, item): item for item in list(enumerate(problems))}

                # Process results as they complete
                for future in concurrent.futures.as_completed(futures):
                    i, problem_rules = future.result()
                    rules[i] = problem_rules
                    pbar.update(1)

        for i, rule_set in enumerate(rules):
            rules_added = 0
            for rule in rule_set:
                rules_added += self._policy.insert_rule(rule)
            logging.info(f"Extracted {rules_added} new rules from problem {i}.")

    def learn_from_problem(self, problem: Problem, inplace: bool, prob_id: int = 0) -> Optional[list[Rule]]:
        """Plan for some permutations of goals sequentially to derive lifted rules via regression.

        One optimisation one could come up with is to skip learning for a problem if a policy can
        solve it. We don't do this however, as even if a policy can solve a problem, it could
        still learn rules that generalise to unseen problems.

        Args:
            problem (Problem): problem to learn rules from
            inplace (bool): to add rules inplace or the return them
            prob_id (int, optional): for multithreading. Defaults to 0.

        Returns:
            Optional[list[Rule]]: list of rules if inplace is False, otherwise None
        """

        rules = []

        goal_atoms = get_literals_list(problem.goal)
        n_goals = len(goal_atoms)
        n_perms = min(self._options.num_permutations, math.factorial(n_goals))

        # Abstract numeric goals of the form v = 0 by making states having v = 1
        state = problem.init
        new_state = set()
        numeric_goals = {}
        for goal in goal_atoms:
            if isinstance(goal, EqualTo):
                numeric_goals[goal.operands[0]] = token_to_numeric_value(goal.operands[1])
        for atom in state:
            if isinstance(atom, EqualTo):
                function = atom.operands[0]
                value = token_to_numeric_value(atom.operands[1])
                if function in numeric_goals and numeric_goals[function] == 0 and value > 0:
                    new_state.add(EqualTo(function, NumericValue(1)))
                else:
                    new_state.add(atom)
            else:
                new_state.add(atom)
        problem = self.new_problem(problem, frozenset(new_state))

        # Learn from different permutation of goals
        seen_permutations = set()
        for perm_id in range(n_perms):
            while True:
                random.shuffle(goal_atoms)
                goal_atoms_str = state_to_string(goal_atoms, sort=False)
                if goal_atoms_str not in seen_permutations:
                    seen_permutations.add(goal_atoms_str)
                    break

            for goal_size in range(1, self._options.goal_max_size + 1):
                rules += self.learn_from_goal_order(problem, goal_atoms, prob_id, perm_id, goal_size)

        if inplace:
            for rule in rules:
                self._policy.insert_rule(rule)
            return None
        else:
            return rules

    def learn_from_goal_order(
        self,
        problem: Problem,
        goal_atoms: list[Predicate],
        prob_id: int,
        perm_id: int,
        goal_size: int,
    ) -> list[Rule]:
        """Learns a set of rules from a problem and goal ordering."""

        rules = []

        n_goals = len(goal_atoms)
        state = problem.init
        planner = "lmcut-numeric" if self._is_numeric else "lmcut"

        def get_new_rules_and_state_from_goals(s, g):
            new_problem = self.new_problem(problem, s, And(*g))
            plans = get_plan(self._domain, new_problem, planner=planner, uid=f"{prob_id}_{perm_id}")
            if plans is None:
                new_rules = []
            else:
                assert len(plans) == 1  # had a symk option before which returned multiple plans
                new_rules, s = self.regress_and_progress(plans[0], s, set(g), problem)
            return new_rules, s

        for g_i in range(0, n_goals, goal_size):
            goal = goal_atoms[g_i : min(g_i + goal_size, n_goals)]
            rules_i, state = get_new_rules_and_state_from_goals(state, goal)
            rules += rules_i

        return rules

    def regress_and_progress(
        self, plan: Plan, state: State, goals: State, problem: Problem
    ) -> tuple[list[Rule], State]:
        """Returns a set of rules from regression and the progressed state from a plan."""

        rules = set()

        plan = strings_to_actions(self._domain, problem, plan)
        reversed_plan = list(reversed(plan))

        """Regress"""
        reg_state = goals
        for a_i, action in enumerate(reversed_plan):
            # Get ground regressed state
            h_opt = a_i + 1
            reg_state = regress_action(state=reg_state, action=action)
            actions = list(reversed(reversed_plan[:h_opt]))

            s_cond = set()
            g_cond = set()
            unified_actions: list[tuple[str, Any]] = []

            variables = {}
            type_to_i = {}

            def get_unified_terms(atom: Union[Predicate, Action]) -> list[Constant]:
                nonlocal type_to_i
                new_args = []
                for term in atom.terms:
                    # https://github.com/AI-Planning/pddl/issues/134
                    if not isinstance(term, Constant):
                        term = Constant(term.name)
                    # assert isinstance(term, Constant)
                    obj = term.name
                    if obj not in variables:
                        if term in self._domain.constants:
                            new_variable = term
                            # do not increment added_variables_i for constants
                        else:
                            obj_type = term.type_tag
                            if obj_type is None:
                                obj_type = OBJECT_TYPE
                            if obj_type not in type_to_i:
                                type_to_i[obj_type] = 0
                            added_variables_i = type_to_i[obj_type]
                            type_to_i[obj_type] += 1
                            new_variable = Constant(f"{obj_type}{added_variables_i}", obj_type)
                        variables[obj] = new_variable
                    new_args.append(variables[obj])
                return new_args

            def get_condition(atom: Predicate) -> Formula:
                if isinstance(atom, Predicate):
                    condition = Predicate(atom.name, *get_unified_terms(atom))
                elif isinstance(atom, EqualTo | GreaterEqualThan):
                    function = atom.operands[0]
                    function = NumericFunction(function.name, *get_unified_terms(function))
                    condition = type(atom)(function, atom.operands[1])
                else:
                    raise ValueError(f"Condition of type {type(atom)} not supported yet: {atom}")
                return condition

            for action in actions:
                parameters = tuple(get_unified_terms(action))
                unified_actions.append((action.name, parameters))

            for fact in sorted(reg_state, key=lambda x: str(x)):
                s_cond.add(get_condition(fact))

            for fact in sorted(goals, key=lambda x: str(x)):
                g_cond.add(get_condition(fact))

            rule = Rule(
                s_cond=s_cond,
                g_cond=g_cond,
                actions=unified_actions,
                precedence=PrecedenceValue(goal_size=len(g_cond), p0=h_opt),
            )
            rules.add(rule)

        """Progress"""
        for action in plan:
            assert check_formula(state=state, formula=action.precondition), f"{state_to_string(state)}\n{action}"
            state = apply_action(state=state, action=action)

        rules = sorted(rules)
        return rules, state

    def get_transitive_closures(self):
        """[wip]"""
        # compute static predicates
        static_predicates = set(p.name for p in self._domain.predicates)
        for action in self._domain.actions:
            for effect in get_literals_list(action.effect):
                if isinstance(effect, Not):
                    predicate = effect.argument.name
                elif isinstance(effect, Predicate):
                    predicate = effect.name
                else:
                    raise NotImplementedError
                static_predicates.discard(predicate)

        # find repeated actions
        rules = self.policy.rules
        schema_name_to_different_terms = {}
        for rule in rules:
            actions = rule.actions
            # print([a[0] for a in actions])
            for i in range(1, len(actions)):
                a0 = actions[i - 1]
                a1 = actions[i]
                if a0[0] != a1[0]:
                    continue
                schema_name = a0[0]
                a0_terms = a0[1]
                a1_terms = a1[1]
                assert len(a0_terms) == len(a1_terms)
                different_terms = set()
                for j in range(len(a0_terms)):
                    if a0_terms[j] != a1_terms[j]:
                        different_terms.add(j)
                if schema_name not in schema_name_to_different_terms:
                    schema_name_to_different_terms[schema_name] = set()
                schema_name_to_different_terms[schema_name] |= different_terms

        ret = {}
        for schema_name, different_terms in schema_name_to_different_terms.items():
            schema = self._name_to_action[schema_name]
            differ_terms = set(schema.parameters[j] for j in different_terms)
            if len(differ_terms) != 2:
                continue

            tc_predicates = set()

            for pre in get_literals_list(schema.precondition):
                if set(pre.terms).intersection(differ_terms) != differ_terms:
                    continue
                if pre.name not in static_predicates:
                    continue
                tc_predicates.add(pre)

            if len(tc_predicates) == 0:
                continue

            ret[schema_name] = {
                "terms": differ_terms,
                "preconditions": tc_predicates,
            }

        return ret

    def new_problem(self, problem: Problem, state: State, goal: Optional[Formula] = None) -> Problem:
        """Create a new problem with the given state and goal."""
        return new_problem(self._domain, problem, state, goal)

    def save(self, save_file: str | None) -> str:
        """Saves the stored policy to a file."""
        if save_file is None:
            logging.info("No save file specified, model not saved.")
            return
        self._policy.save(save_file)
        self._policy.save_readable(f"{save_file}.readable")
        logging.info(f"Policy saved to {tc.colored(save_file, 'cyan')}")
        return save_file

    def dump(self) -> None:
        """Print the rules in the policy."""
        self._policy.dump()
