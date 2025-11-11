#!/usr/bin/env python3

import logging

import pddl

from moose.options import Options
from moose.policy import Policy
from moose.satisficing.executors.mk3 import SQLitePolicyMk3
from moose.util.managers import TimerContextManager


def get_executor_from_options(options: Options):
    with TimerContextManager("parsing domain"):
        domain = pddl.parse_domain(options.domain_file)

    with TimerContextManager("parsing problem"):
        problem = pddl.parse_problem(options.problem_file)

    with TimerContextManager("loading policy"):
        policy = Policy.load(options.model_file)

    if len(policy) == 0:
        logging.warning("Policy is empty.")

    return SQLitePolicyMk3(
        domain=domain,
        problem=problem,
        policy=policy,
        precedence_strategy=options.precedence_strategy,
        bound=options.bound,
        disallow_cycles=options.disallow_cycles,
        verbosity=options.verbosity,
    )
