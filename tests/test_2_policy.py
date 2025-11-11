import logging
import os
from itertools import product

import pytest

from moose.core import InvalidPlanError, execute_policy
from moose.options import get_policy_parser

from .fixtures import (
    BENCHMARK_DIR,
    CLASSIC_DOMAINS,
    PLANS_DIR,
    PROBLEMS,
    get_domain_file,
    get_model_file,
    train_routine,
)


@pytest.mark.parametrize(
    "domain_name,problem_name",
    product(CLASSIC_DOMAINS, PROBLEMS),
)
def test_policy(domain_name: str, problem_name: str):
    os.makedirs(PLANS_DIR, exist_ok=True)

    domain_file = get_domain_file(domain_name)
    model_file = get_model_file(domain_name)
    if not os.path.exists(model_file):
        train_routine(domain_name)

    problem_file = f"{BENCHMARK_DIR}/{domain_name}/training/{problem_name}.pddl"
    plan_file = f"{PLANS_DIR}/{domain_name}-{problem_name}.plan"
    plan_args = [
        model_file,
        domain_file,
        problem_file,
        "--plan-file",
        plan_file,
        "-val",
    ]
    logging.info(f"Planning with cmd:\n\n\t./policy.py {' '.join(plan_args)}\n")

    parser = get_policy_parser()
    opts = parser.parse_args(plan_args)
    try:
        execute_policy(options=opts)
    except NotImplementedError:
        pytest.skip("Config not implemented")
    except InvalidPlanError:
        pytest.fail(f"Invalid plan for {domain_name=}, {problem_name=}")
