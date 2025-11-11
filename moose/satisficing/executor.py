import logging
from abc import ABC, abstractmethod
from typing import Optional

import termcolor as tc
from pddl.core import Domain, Problem
from pddl.logic import Predicate
from pddl.logic.functions import EqualTo, NumericFunction
from tqdm.contrib.logging import logging_redirect_tqdm

from moose.planning import Plan, dump_statistics, get_literals_list
from moose.planning.strings import ClassicCondition, NumericCondition, state_to_string
from moose.policy import Policy
from moose.rule import Rule
from moose.util.logging import mat_to_str
from moose.util.managers import TimerContextManager


class PolicyExecutor(ABC):
    def __init__(
        self,
        domain: Domain,
        problem: Problem,
        policy: Policy,
        precedence_strategy: Optional[str] = None,
        bound: int = 0,
        disallow_cycles: bool = False,
        verbosity: int = 0,
    ) -> None:

        dump_statistics(domain, problem, description_prefix="Instance statistics:\n")

        self._domain = domain
        self._problem = problem
        self._policy = policy

        # Options
        if precedence_strategy is None:
            precedence_strategy = policy.precedence_strategy
        else:
            logging.info(
                f"Overriding learned precedence strategy `{policy.precedence_strategy}` ->`{precedence_strategy}`"
            )

        self._precedence_strategy = precedence_strategy

        self._bound = bound
        self._disallow_cycles = disallow_cycles

        self._verbosity = verbosity

        # Instance information
        self._predicates = domain.predicates
        self._functions = domain.functions
        self._name_to_schema = {action.name: action for action in domain.actions}
        self._name_to_predicate = {predicate.name: predicate for predicate in self._predicates}
        self._name_to_function = {function.name: function for function in self._functions}

        self._goals = get_literals_list(problem.goal)
        if any(not isinstance(goal, Predicate | EqualTo) for goal in self._goals):
            raise NotImplementedError("Executor only supports conjunctive positive goals")
        self._numeric_goals: dict[NumericFunction, float] = {}
        self._classic_goals: dict[Predicate] = set()
        for goal in sorted(self._goals, key=lambda x: str(x)):
            if isinstance(goal, ClassicCondition):
                self._classic_goals.add(goal)
            elif isinstance(goal, NumericCondition):
                assert isinstance(goal, EqualTo), "Inequality goals not supported yet"
                self._numeric_goals[goal.operands[0]] = goal.operands[1]

        # Planning information
        self._plan: Plan = []
        self._fired_rules: list[Rule] = []
        self._plan_to_fired_rules: list[Rule] = []
        self._seen: set[str] = set()
        self._detected_cycle: bool = False
        self._planning_time: float = 0

        # Initialise
        self._init_impl()

    @abstractmethod
    def _init_impl(self) -> None:
        raise NotImplementedError

    @property
    def precedence(self) -> str:
        return self._precedence_strategy

    @property
    def verbosity(self) -> int:
        return self._verbosity

    def reached_bound(self) -> bool:
        """If input bound is non-positive, then we don't have a bound."""
        return (self._bound == len(self.get_partial_plan())) and (self._bound > 0)

    def detected_cycle(self) -> bool:
        return self._detected_cycle

    def get_fired_rules(self) -> list[Rule]:
        return self._fired_rules

    def get_plan_to_fired_rules(self) -> list[Rule]:
        return self._plan_to_fired_rules

    def dump_planning_stats(self) -> None:
        logging.warning(f"Planning stats not implemented for {self.__class__.__name__}")

    def dump_profiling_stats(self) -> None:
        logging.warning(f"Profile stats not implemented for {self.__class__.__name__}")

    def dump_fired_rules(self) -> None:
        stats = {}
        for rule in self._fired_rules:
            rule = str(rule)
            if rule not in stats:
                stats[rule] = 0
            stats[rule] += 1
        stats = [(v, k) for k, v in stats.items()]
        stats = sorted(stats, reverse=True)
        stats = "\n".join([f"{v}\n{k}" for v, k in stats])
        logging.info(f"Fired rules:\n{stats}")

    def solve(self) -> Optional[Plan]:
        solved = False
        with logging_redirect_tqdm():
            with TimerContextManager("planning with policy") as timer:
                solved = self._solve_impl()
            self._planning_time = timer.get_time()
            if solved:
                return self._plan
            else:
                return None

    @abstractmethod
    def _solve_impl(self) -> None:
        raise NotImplementedError

    @property
    def n_cycles(self) -> int:
        return self._cycles

    def _log_goal_set(self, i: int, n_goals: int, goal: Predicate) -> None:
        if self._verbosity >= 1:
            logging.info(f"Setting goal {i+1}/{n_goals} to {tc.colored(state_to_string({goal}), 'blue')}")

    def _log_goal_achieved(self, i: int, n_goals: int) -> None:
        if self._verbosity >= 1:
            logging.info(f"Achieved goal {i+1}/{n_goals}")

    def get_partial_plan(self) -> Plan:
        return self._plan

    def get_fired_rules(self) -> list[Rule]:
        return self._fired_rules

    def log_stats(self) -> None:
        self.dump_fired_rules()
        self.dump_profiling_stats()
        self.dump_planning_stats()
