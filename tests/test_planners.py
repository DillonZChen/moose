import logging
from itertools import product

import pytest

from moose.core import execute_search
from moose.options import get_search_parser
from moose.planning.planners import PlannerNotFoundError

from .fixtures import (
    BENCHMARK_DIR,
    CLASSIC_DOMAINS,
    CLASSIC_PLANNERS,
    NUMERIC_DOMAINS,
    NUMERIC_PLANNERS,
    get_domain_file,
)


@pytest.mark.parametrize(
    "domain_name,problem_name,planner",
    list(product(CLASSIC_DOMAINS, ["p01"], CLASSIC_PLANNERS))
    + list(product(NUMERIC_DOMAINS, ["p01"], NUMERIC_PLANNERS)),
)
def test_classic_planners(domain_name: str, problem_name: str, planner: str):
    if planner in {"blind", "scorpion"}:
        pytest.skip()

    domain_file = get_domain_file(domain_name)
    problem_file = f"{BENCHMARK_DIR}/{domain_name}/training/{problem_name}.pddl"
    plan_args = [
        planner,
        domain_file,
        problem_file,
        "-val",
    ]
    logging.info(f"Planning with cmd:\n\n\t./search.py {' '.join(plan_args)}\n")

    parser = get_search_parser()
    opts = parser.parse_args(plan_args)
    try:
        execute_search(options=opts)
    except PlannerNotFoundError:
        pytest.skip(f"Planner binary for {planner} not found")
    except NotImplementedError:
        pytest.skip("Config not implemented")
