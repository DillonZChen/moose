from pddl.action import Action
from pddl.core import Domain
from pddl.logic import Predicate
from pddl.logic.base import Formula, Not
from pddl.logic.functions import Decrease, EqualTo, GreaterEqualThan, Increase, NumericFunction, NumericValue

from moose.planning import Literal, State, get_literals_list
from moose.planning.strings import ClassicCondition, NumericCondition, token_to_numeric_value

__all__ = ["regress_effects", "regress_action"]


def check_regression_support(domain: Domain) -> None:
    """Checks if `regress_effects` supports the fragment of PDDL exhibited in the domain."""

    for schema in domain.actions:
        pres = get_literals_list(schema.precondition)
        effs = set(get_literals_list(schema.effect))
        for pre in pres:
            if isinstance(pre, Predicate):
                continue

            if isinstance(pre, Not):
                if pre.argument not in effs:
                    raise NotImplementedError(
                        f"Negative preconditions which do not show up as add effects are not support yet.\n"
                        f"In this case, {pre} in the schema `{schema.name}` is a negative precondition but not an add."
                    )

            if isinstance(pre, NumericCondition) and (
                not isinstance(pre, GreaterEqualThan)
                or not isinstance(pre.operands[0], NumericFunction)
                or not isinstance(pre.operands[1], NumericValue)
            ):
                raise NotImplementedError(
                    f"Numeric preconditions not of the form `func(x1, ..., xn) >= c` are not supported.\n"
                    f"In this case, {pre} in the schema `{schema.name}` is a precondition but not of this form."
                )

    # All is well if we get here
    return


def regress_effects(state: State, precondition: Formula, effects: Formula) -> State:
    """Regress the effects of a state under the precondition and effects."""

    """Process states"""
    classic_conditions = set()
    numeric_conditions = {}
    for f in state:
        if isinstance(f, NumericCondition):
            function = f.operands[0]
            value = token_to_numeric_value(f.operands[1])
            assert function not in numeric_conditions

            # Ensure canonical form `function \comparator value`
            # Also assume all conditions are >= except for the goal which should be = 0
            assert isinstance(f, EqualTo | GreaterEqualThan)
            assert isinstance(function, NumericFunction)
            assert isinstance(value, float | int)

            numeric_conditions[function] = value
        elif isinstance(f, Literal):
            classic_conditions.add(f)
        else:
            raise NotImplementedError(f"Numeric condition not supported yet: {f}")

    """Process effects and preconditions"""
    effects = get_literals_list(effects)
    classic_effs = []
    numeric_effs = []
    preconditions = get_literals_list(precondition)
    classic_pres = []
    numeric_pres = []

    for eff in effects:
        if isinstance(eff, ClassicCondition):
            classic_effs.append(eff)
        elif isinstance(eff, Increase | Decrease):
            numeric_effs.append(eff)
        else:
            raise ValueError(f"Unknown effect type: {type(eff)}")

    for pre in preconditions:
        if isinstance(pre, ClassicCondition):
            classic_pres.append(pre)
        elif isinstance(pre, NumericCondition):
            numeric_pres.append(pre)
        else:
            raise ValueError(f"Unknown precondition type: {type(pre)}")

    """Classic PDDL"""
    for eff in classic_effs:
        if isinstance(eff, Predicate):
            classic_conditions -= frozenset({eff})
        # We do not handle deletes as we assume they appear in pre

    for pre in classic_pres:
        if isinstance(pre, Predicate):
            classic_conditions |= frozenset({pre})
        # We do not handle negative preconditions yet, see `check_regression_support`

    """Numeric PDDL"""
    for eff in numeric_effs:
        function = eff.operands[0]
        value = token_to_numeric_value(eff.operands[1])
        if isinstance(eff, Increase):
            if function not in numeric_conditions:
                numeric_conditions[function] = 0
            else:
                numeric_conditions[function] -= value
        # Again, let preconditions handle decrease effects, usually assume that we have a >= for decreases

    for pre in numeric_pres:
        function = pre.operands[0]
        value = token_to_numeric_value(pre.operands[1])
        if isinstance(pre, GreaterEqualThan):
            numeric_conditions[function] = value
        else:
            raise NotImplementedError(f"Numeric condition of type {type(pre)} not supported yet: {pre}")

    ret = classic_conditions | set(GreaterEqualThan(f, v) for f, v in numeric_conditions.items())
    # for k in sorted(ret, key=lambda x: str(x)): print(k)

    return ret


def regress_action(state: State, action: Action) -> State:
    return regress_effects(state, action.precondition, action.effect)
