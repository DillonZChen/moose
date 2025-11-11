from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable, List

from pddl.action import Action
from pddl.logic import Predicate
from pddl.logic.base import Not
from pddl.logic.functions import NumericFunction

from moose.planning import Literal
from moose.planning.strings import ClassicCondition, Condition, NumericCondition, aopt_to_string, aopt_to_stringB
from moose.precedence import PrecedenceValue
from moose.util.logging import mat_to_str

UG_SUFFIX = "__ug"
OBJECT_TYPE = "any_obj"


def to_ug_condition(condition: Condition) -> Condition:
    if isinstance(condition, Not):
        raise NotImplementedError("Negations not yet supported for goals.")
    if isinstance(condition, ClassicCondition):
        return Predicate(condition.name + UG_SUFFIX, *condition.terms)
    elif isinstance(condition, NumericCondition):
        operands = list(condition.operands)
        for i in range(len(operands)):
            op = operands[i]
            if isinstance(op, NumericFunction):
                operands[i] = NumericFunction(op.name + UG_SUFFIX, *op.terms)
        condition = deepcopy(condition)
        condition._operands = tuple(operands)
        return condition
    else:
        raise ValueError(f"Invalid condition type: {type(condition)}")


@dataclass
class Rule:
    def __init__(
        self,
        s_cond: Iterable[Literal],
        g_cond: Iterable[Literal],
        actions: List,
        precedence: PrecedenceValue,
    ):
        """Sort conditions and actions"""
        self._s_cond = sorted(s_cond, key=lambda x: str(x))
        self._g_cond = sorted([to_ug_condition(g) for g in g_cond], key=lambda x: str(x))
        self._actions = actions  # do **not** sort actions, their order is important
        self._vars = sorted({v for action in actions for v in action[1]}, key=lambda x: str(x))
        self._precedence = precedence

    @property
    def s_cond(self) -> list[Literal]:
        return self._s_cond

    @property
    def g_cond(self) -> list[Literal]:
        return self._g_cond

    @property
    def actions(self) -> list[Action]:
        return self._actions

    @property
    def precedence(self) -> PrecedenceValue:
        return self._precedence

    @property
    def n_conditions(self) -> int:
        return len(self.s_cond) + len(self.g_cond)

    @property
    def n_actions(self) -> int:
        return len(self.actions)

    def _get_atoms_as_str(self) -> list[str]:
        key = Rule.condition_to_string((self.s_cond, self.g_cond), delimiter=" ")
        anames = [aopt_to_string(a).replace("?", "") for a in self.actions]
        return anames + key.split(" ")

    def to_string(self) -> str:
        variables = " ".join([str(v) for v in self._vars])
        s_cond = " ".join([str(x) for x in self.s_cond])
        g_cond = " ".join([str(x) for x in self.g_cond])
        g_cond = g_cond.replace("__ug ", " ")
        actions = " ".join(aopt_to_stringB(a).replace("?", "") for a in self.actions)
        mat = []
        mat.append(("precedence", ":", f"{self.precedence}"))
        mat.append(("vars", ":", variables))
        mat.append(("s_cond", ":", s_cond))
        mat.append(("g_cond", ":", g_cond))
        mat.append(("actions", ":", actions))
        return mat_to_str(mat, rjust=[1, 0, 0], space=[1, 0, 0])

    def modify_precedence(self, modification: PrecedenceValue) -> None:
        self._precedence = self._precedence + modification

    def __eq__(self, other):
        if not isinstance(other, Rule):
            return False
        return self.to_string() == other.to_string()

    def __hash__(self):
        return hash(self.to_string())

    def __lt__(self, other):
        return str(self) < str(other)

    def __str__(self):
        return self.to_string()
