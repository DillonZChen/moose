import logging
from copy import deepcopy
from enum import Enum
from typing import Mapping

import pddl
from pddl.core import Domain, Problem

from moose.planning import get_domain, get_plan, get_problem, is_numeric_domain, new_problem
from moose.planning.progression import apply_action, check_formula
from moose.planning.strings import action_to_string, state_to_string, strings_to_actions
from moose.precedence import ExecutionStrategy
from moose.satisficing.executor import PolicyExecutor
from moose.util.managers import TimerContextManager

FailureDiagnosis = Mapping


class FailureReason(Enum):
    """Enum for failure reasons."""

    DEADEND = "deadend"
    NO_ACTION = "no_action"
    CYCLE = "cycle"


def diagnose_failure(
    domain: str | Domain, problem: str | Problem, policy_executor: PolicyExecutor
) -> FailureDiagnosis:
    domain = get_domain(domain)
    problem = get_problem(problem)

    reason = None
    partial_plan_str = policy_executor.get_partial_plan()
    problematic_index = len(partial_plan_str) - 1
    problematic_rule = None

    if policy_executor.detected_cycle():
        reason = FailureReason.CYCLE
    elif len(partial_plan_str) == 0:
        reason = FailureReason.NO_ACTION
        problematic_index = None
    else:
        partial_plan = strings_to_actions(domain, problem, partial_plan_str)

        state = deepcopy(problem.init)

        is_numeric = is_numeric_domain(domain)
        planner = "lmcut-numeric" if is_numeric else "lama-first"

        # collect plan trace
        states = []
        for i, action in enumerate(partial_plan):
            if not check_formula(state, action.precondition):
                s = state_to_string(state, delimiter="\n")
                msg = f"Precondition not satisfied.\n State:\n{s}\n Action:\n{action_to_string(action)}\nPre:\n{action.precondition}\n"
                raise RuntimeError(msg)
            state = apply_action(state, action)
            states.append(deepcopy(state))
        assert len(states) == len(partial_plan)

        # check failure due to no policy action
        end_problem = new_problem(domain, problem, states[problematic_index])
        with TimerContextManager("checking deadend at last action"):
            plan = get_plan(domain, end_problem, planner=planner)
        if plan is not None:
            reason = FailureReason.NO_ACTION
        else:
            reason = FailureReason.DEADEND

        # check failure due to earliest intermediate deadend
        if reason == FailureReason.DEADEND:
            for i in range(len(partial_plan) - 1):
                progressed_problem = new_problem(domain, problem, states[i])
                with TimerContextManager(f"checking deadend at action {i+1}/{len(partial_plan)}"):
                    plan = get_plan(domain, progressed_problem, planner=planner)
                if plan is None:
                    # then we have a deadend
                    problematic_index = i
                    reason = FailureReason.DEADEND
                    break

    assert reason is not None
    if problematic_index is not None:
        execution_strategy = policy_executor._policy.execution_strategy
        if execution_strategy in {ExecutionStrategy.GREEDY, ExecutionStrategy.GREEDY_LOOP}:
            fired_rules = []
            for rule in policy_executor.get_fired_rules():
                for _ in range(len(rule.actions)):
                    fired_rules.append(rule)
        elif execution_strategy == ExecutionStrategy.CONSERVATIVE:
            fired_rules = policy_executor.get_fired_rules()
        else:
            raise ValueError(f"Unknown policy type: {policy_executor.policy_type}")

        problematic_rule = fired_rules[problematic_index]

    diagnosis = {
        "reason": reason,
        "trace": partial_plan_str,
        "problematic_index": problematic_index,
        "problematic_rule": problematic_rule,
        "policy_executor": policy_executor,
        "domain": domain,
        "problem": problem,
    }

    diagnosis_results = "\n".join(
        f"\t{key}={value}" for key, value in diagnosis.items() if key not in {"domain", "problem"}
    )
    logging.info(f"Diagnosis results:\n{diagnosis_results}")

    return diagnosis
