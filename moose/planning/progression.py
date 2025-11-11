"""
Code modified from Till Hofmann's genfond.
Changed code to not handle FOND for simplicity, and added regression.
"""

from collections.abc import Collection

from pddl.action import Action
from pddl.logic import Predicate
from pddl.logic.base import And, Formula, Not
from pddl.logic.effects import When
from pddl.logic.functions import (
    Assign,
    BinaryFunction,
    Decrease,
    Divide,
)
from pddl.logic.functions import EqualTo as FunctionEqualTo
from pddl.logic.functions import (
    FunctionExpression,
    GreaterEqualThan,
    GreaterThan,
    Increase,
    LesserEqualThan,
    LesserThan,
    Minus,
    NumericFunction,
    NumericValue,
    Plus,
    ScaleDown,
    ScaleUp,
    Times,
)
from pddl.logic.predicates import EqualTo

from moose.planning import State
from moose.planning.strings import token_to_numeric_value


def eval_function_term(term: FunctionExpression, state: State) -> float | int:
    if isinstance(term, NumericValue):
        return token_to_numeric_value(term)
    elif isinstance(term, NumericFunction):
        for f in state:
            if not isinstance(f, FunctionEqualTo):
                continue
            if str(f.operands[0]) == str(term):
                return token_to_numeric_value(f.operands[1])
        # If a numeric fluent does not appear in the initial state of the PDDL, its value is assumed to be 0.
        return 0
    elif isinstance(term, Plus):
        return eval_function_term(term.operands[0], state) + eval_function_term(term.operands[1], state)
    else:
        raise ValueError("Unknown term type: {}".format(type(term)))


def check_formula(state: State, formula: Formula) -> bool:
    if isinstance(formula, And):
        return all(check_formula(state, subformula) for subformula in formula.operands)
    elif isinstance(formula, Not):
        return not check_formula(state, formula.argument)
    elif isinstance(formula, Predicate):
        return formula in state
    elif isinstance(formula, EqualTo):
        return formula.left == formula.right
    elif isinstance(formula, LesserThan):
        return eval_function_term(formula.operands[0], state) < eval_function_term(formula.operands[1], state)
    elif isinstance(formula, LesserEqualThan):
        return eval_function_term(formula.operands[0], state) <= eval_function_term(formula.operands[1], state)
    elif isinstance(formula, GreaterThan):
        return eval_function_term(formula.operands[0], state) > eval_function_term(formula.operands[1], state)
    elif isinstance(formula, GreaterEqualThan):
        return eval_function_term(formula.operands[0], state) >= eval_function_term(formula.operands[1], state)
    elif isinstance(formula, FunctionEqualTo):
        return eval_function_term(formula.operands[0], state) == eval_function_term(formula.operands[1], state)
    else:
        raise ValueError("Unknown formula type: {}".format(type(formula)))


def apply_action(state: State, action: Action) -> State:
    return apply_effects(state, action.effect)


def apply_effects(state: State, effects: Formula) -> State:
    new_state = apply_effects_to_state(state, effects)
    assert isinstance(new_state, Collection)
    assert all(isinstance(f, (Predicate, FunctionEqualTo)) for f in new_state)
    return new_state


def apply_effects_to_state(state: State, effects: Formula) -> State:
    assert all(isinstance(f, (Predicate, FunctionEqualTo)) for f in state)
    if isinstance(effects, And):
        for effect in effects.operands:
            state = apply_effects(state, effect)
        return state
    elif isinstance(effects, Predicate):
        return state | {effects}
    elif isinstance(effects, Not):
        return frozenset([f for f in state if f != effects.argument])
    elif isinstance(effects, When):
        if check_formula(state, effects.condition):
            return apply_effects(state, effects.effect)
        else:
            return state
    elif isinstance(effects, BinaryFunction):
        if isinstance(effects.operands[0], NumericFunction):
            fct = effects.operands[0]
            change = effects.operands[1]
        else:
            fct = effects.operands[1]
            change = effects.operands[0]
        change = token_to_numeric_value(change)
        assert isinstance(fct, NumericFunction)

        current_evals = filter(
            lambda f: isinstance(f, FunctionEqualTo) and (f.operands[0] == fct) | (f.operands[1] == fct),
            state,
        )
        if not current_evals:
            current_eval = FunctionEqualTo(fct, NumericValue(0))
        else:
            current_evals = list(current_evals)
            assert len(current_evals) == 1, current_evals
            current_eval = current_evals[0]
        current_value = token_to_numeric_value(current_eval.operands[1].value)

        if isinstance(effects, Assign):
            new_value = change
        elif isinstance(effects, Increase):  # NOTE problematic if values are floats
            new_value = current_value + change
        elif isinstance(effects, Decrease):  # NOTE problematic if values are floats
            new_value = current_value - change
        elif isinstance(effects, (Plus, Minus, Times, Divide, ScaleUp, ScaleDown)):
            raise NotImplementedError()
        else:
            raise ValueError(f"Unknown effect {effects} of type {type(effects)}")
        return frozenset([f for f in state if f != current_eval] + [FunctionEqualTo(fct, NumericValue(new_value))])
    else:
        raise ValueError("Unknown effect type: {}".format(type(effects)))


def get_num_vals(state: State) -> set[int]:
    return {f.operands[1].value for f in state if isinstance(f, FunctionEqualTo)}
