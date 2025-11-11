from abc import ABC
from typing import Sequence

from pddl.action import Action
from pddl.core import Domain, Problem
from pddl.logic import Constant, Variable
from pddl.logic.base import And, ExistsCondition, Formula, Not
from pddl.logic.predicates import DerivedPredicate, Predicate, Term
from pddl.requirements import Requirements

from moose.options import Options
from moose.planning import get_literals_list
from moose.policy import Policy
from moose.rule import OBJECT_TYPE, UG_SUFFIX, Rule


class Transformer(ABC):
    def __init__(self, domain: Domain, problem: Problem, policy: Policy, options: Options):
        self._domain = domain
        self._problem = problem
        self._policy = policy

        self._options = options

        self._goals = get_literals_list(problem.goal)
        self._original_schemata = {a.name: a for a in domain.actions}
        self._original_predicates = domain.predicates

        self._distances = policy.precedences
        if len(self._distances) == 0:
            raise ValueError("Loaded policy is empty")

        # new domain info
        self._new_domain_name = f"{domain.name}_ext"
        self._new_requirements = domain.requirements | {
            Requirements.DERIVED_PREDICATES,
            Requirements.EXISTENTIAL_PRECONDITION,
        }
        self._new_types = domain.types
        self._new_constants = domain.constants
        self._new_predicates = {p.name: p for p in self._original_predicates}
        self._new_axioms = []
        self._new_functions = domain.functions
        self._new_schemata = []

        # new problem info
        self._new_problem_name = f"{problem.name}_ext"
        self._new_objects = sorted(problem.objects)
        self._new_init = list(problem.init)
        self._new_goal: Formula = problem.goal

    def transform(self) -> None:
        self.add_ug_predicates_and_axioms()
        self.add_schemata()

        for rule in self.get_rules():
            schema = rule.actions[0]
            original_schema = self._original_schemata[schema[0]]
            original_schema_params = original_schema.parameters

            body, variables = self.get_body_from_rule(rule)
            ## unintuitively, having duplicate literals is faster
            # body = sorted(set(body) - set(get_literals_list(schema.precondition)), key=lambda x: str(x))
            # body += [self.get_schema_predicate_applicable(schema, make=False)(*original_schema_params)]
            body = And(*body)

            free_variables = sorted([var for var, value in variables.items() if value], key=lambda v: v.name)

            head = self.get_schema_predicate_prune(original_schema)(*original_schema_params)
            body = ExistsCondition(body, free_variables) if len(free_variables) > 0 else body

            self._new_axioms.append(DerivedPredicate(head, body))
        return

    def get_transformed_instance(self) -> tuple[Domain, Problem]:
        new_domain = Domain(
            name=self._new_domain_name,
            requirements=self._new_requirements,
            types=self._new_types,
            constants=self._new_constants,
            predicates=list(self._new_predicates.values()),
            derived_predicates=self._new_axioms,
            functions=self._new_functions,
            actions=self._new_schemata,
        )
        new_problem = Problem(
            name=self._new_problem_name,
            domain=new_domain,
            domain_name=self._new_domain_name,
            objects=self._new_objects,
            init=self._new_init,
            goal=self._new_goal,
        )

        # with open("domain.pddl", "w") as f:
        #     f.write(str(new_domain))
        # with open("problem.pddl", "w") as f:
        #     f.write(str(new_problem))
        # breakpoint()

        return new_domain, new_problem

    def get_predicate(self, name: str, terms: Sequence[Term], make: bool) -> Predicate:
        if make:
            assert name not in self._new_predicates, name
            predicate = Predicate(name, *terms)
            self._new_predicates[name] = predicate
        else:
            predicate = self._new_predicates.get(name)(*terms)
        return predicate

    def add_ug_predicates_and_axioms(self) -> None:
        g_predicates = {p.name for p in self._goals}
        for predicate in self._original_predicates:
            predicate_name = predicate.name
            if predicate_name not in g_predicates:
                # skip derivation of unnecessary ug predicates
                continue
            g_predicate = self.get_predicate(f"{predicate_name}_g", predicate.terms, make=True)
            ug_predicate = self.get_predicate(f"{predicate_name}{UG_SUFFIX}", predicate.terms, make=True)
            axiom = DerivedPredicate(ug_predicate, And(Not(predicate), g_predicate))
            self._new_axioms.append(axiom)
        for goal in self._goals:
            self._new_init.append(self.get_predicate(f"{goal.name}_g", goal.terms, make=False))

    def get_schema_predicate_prune(self, action: Action, make: bool = False) -> Predicate:
        return self.get_predicate(f"{action.name}_pp", action.parameters, make=make)

    def get_schema_predicate_applicable(self, action: Action, make: bool = False) -> Predicate:
        return self.get_predicate(f"{action.name}_ap", action.parameters, make=make)

    def add_schemata(self) -> None:
        """New *singleton* precondition derived from partial state rules as axioms"""
        for action in self._original_schemata.values():
            new_precondition = self.get_schema_predicate_prune(action, make=True)
            new_precondition = And(action.precondition, new_precondition)
            new_action = Action(
                name=action.name,
                parameters=action.parameters,
                precondition=new_precondition,
                effect=action.effect,
            )
            self._new_schemata.append(new_action)

    def get_rules(self) -> Sequence[Rule]:
        return [r for r in self._policy.rules if any(f"{g.name}" in self._new_predicates for g in r.g_cond)]

    def get_body_from_rule(self, rule: Rule) -> tuple[list[Predicate], dict[Variable, bool]]:
        partial_state = rule.s_cond
        goals = rule.g_cond
        action = rule.actions[0]

        for g in goals:
            assert g.name in self._new_predicates

        schema = self._original_schemata[action[0]]
        schema_params = schema.parameters
        obj_to_var = {obj: var for obj, var in zip(action[1], schema_params)}
        variables = {var: False for var in schema_params}

        def get_params(atom: Predicate) -> Sequence[Variable]:
            ret = []
            for term in atom.terms:
                assert isinstance(term, Constant)
                if term not in obj_to_var:
                    tt = set(term.type_tags)
                    tt = frozenset(tt - {OBJECT_TYPE})
                    var = Variable(term.name, type_tags=tt)
                    obj_to_var[term] = var
                    assert var not in variables
                    variables[var] = True
                ret.append(obj_to_var[term])
            return ret

        rule_body = [self.get_predicate(atom.name, get_params(atom), make=False) for atom in partial_state]
        rule_body += [self.get_predicate(g.name, get_params(g), make=False) for g in goals]

        return rule_body, variables
