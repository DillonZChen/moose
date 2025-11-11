import logging
import os
import re
import shutil
import subprocess
import tempfile
from typing import Mapping, Optional, Union

import pddl
from pddl.core import Domain, Problem
from pddl.logic import Predicate
from pddl.logic.base import And, Formula, Not
from pddl.logic.functions import (
    Decrease,
    EqualTo,
    GreaterEqualThan,
    GreaterThan,
    Increase,
    LesserEqualThan,
    LesserThan,
)

from moose.planning.planners import get_plan_path, get_planner_cmd
from moose.util.logging import mat_to_str

__all__ = ["get_plan", "Plan", "State", "Literal"]

Plan = list[str]
Literal = Predicate | Not | EqualTo | GreaterEqualThan | GreaterThan | LesserEqualThan | LesserThan
State = frozenset[Literal]


def get_domain(domain: str | Domain) -> Domain:
    if isinstance(domain, str):
        domain = pddl.parse_domain(domain)
    assert isinstance(domain, Domain)
    return domain


def get_problem(problem: str | Problem) -> Problem:
    if isinstance(problem, str):
        problem = pddl.parse_problem(problem)
    assert isinstance(problem, Problem)
    return problem


def is_numeric_domain(domain: Domain) -> bool:
    return len(domain.functions) > 0


def new_problem(domain: Domain, problem: Problem, state: State, goal: Optional[Formula] = None) -> Problem:
    if goal is None:
        goal = problem.goal
    return Problem(
        name=f"p{hash(state)}{hash(goal)}",
        domain=domain,
        objects=problem.objects,
        init=state,
        goal=goal,
    )


def get_literals_list(formula: Formula) -> list[Literal]:
    if isinstance(formula, And):
        return list(formula.operands)
    elif isinstance(formula, Literal | Increase | Decrease):
        return [formula]
    else:
        raise ValueError(f"Unknown formula type: {type(formula)}")


def get_domain_statistics(domain: Domain) -> Mapping:
    ret = {
        "types": len(domain.types),
        "constants": len(domain.constants),
        "predicates": len(domain.predicates),
        "functions": len(domain.functions),
        "axioms": len(domain.derived_predicates),
        "actions": len(domain.actions),
    }
    # for action in domain.actions:
    #     ret[f"{action.name} pre"] = len(get_literals_list(action.precondition))
    #     ret[f"{action.name} eff"] = len(get_literals_list(action.effect))
    return ret


def get_problem_statistics(problem: Problem) -> Mapping:
    return {
        "objects": len(problem.objects),
        "init": len(problem.init),
        "goal": len(get_literals_list(problem.goal)),
    }


def dump_statistics(domain: Domain, problem: Problem, description_prefix: str) -> None:
    domain_stats = get_domain_statistics(domain)
    problem_stats = get_problem_statistics(problem)
    mat_str = mat_to_str(
        [("Domain", "", "")]
        + [("", k, v) for k, v in domain_stats.items()]
        + [("Problem", "")]
        + [("", k, v) for k, v in problem_stats.items()]
    )
    logging.info(f"{description_prefix}{mat_str}")


def get_plan(
    domain: Union[str, Domain],
    problem: Union[str, Problem],
    planner: str = "lmcut",
    timeout: Optional[int] = None,
    verbose: bool = False,
    clean_up: bool = True,
    k: int = 1,
    uid: Optional[str] = None,
) -> Optional[list[Plan]]:
    """Returns a list of plans or None if no plan exists.

    k >= 0 specifies the (max) number of optimal plans to find, where k=0 tries to find all plans.
    Top-k planning is only supported by symk.
    """
    if k != 1 and planner != "symk":
        raise ValueError("Top-k planning only works for symk.")

    if isinstance(domain, str):
        domain_name = os.path.basename(domain).split(".")[0]
    else:
        assert isinstance(domain, Domain)
        domain_name = domain.name
    if isinstance(problem, str):
        problem_name = os.path.basename(problem).split(".")[0]
    else:
        assert isinstance(problem, Problem)
        problem_name = problem.name

    # Do everything in a tmp dir
    current_dir = os.getcwd()
    tmp_dir = tempfile.mkdtemp()
    if uid is None:
        tmp_dir = os.path.normpath(f"{current_dir}/out/{domain_name}-{problem_name}")

    def get_tmp_domain_path() -> str:
        tmp_domain_path = f"{tmp_dir}/domain.pddl"
        if uid is not None:
            tmp_domain_path = f"{tmp_dir}/domain_{uid}.pddl.tmp"
        return tmp_domain_path

    def get_tmp_problem_path() -> str:
        tmp_problem_path = f"{tmp_dir}/problem.pddl"
        if uid is not None:
            tmp_problem_path = f"{tmp_dir}/problem_{uid}.pddl.tmp"
        return tmp_problem_path

    def clean_up_routine():
        if uid is not None:
            intermediate_file = f"{tmp_dir}/{uid}.sas"
            domain_file = get_tmp_domain_path()
            problem_file = get_tmp_problem_path()
            plan_file = get_plan_path(uid, tmp_dir=tmp_dir)
            for f in [intermediate_file, domain_file, problem_file, plan_file]:
                if os.path.exists(f):
                    os.remove(f)
            return
        if clean_up:
            shutil.rmtree(tmp_dir)
        os.chdir(current_dir)

    if uid is None:
        os.makedirs(tmp_dir, exist_ok=True)
        os.chdir(tmp_dir)

    try:
        if isinstance(domain, str):
            tmp_domain_path = domain
            if not os.path.isabs(tmp_domain_path):
                tmp_domain_path = f"{current_dir}/{tmp_domain_path}"
        else:
            tmp_domain_path = get_tmp_domain_path()
            with open(tmp_domain_path, "w") as f:
                content = str(domain)
                for o in domain.constants:
                    content = content.replace(f"?{o}", str(o))
                f.write(content)
        tmp_domain_path = os.path.normpath(tmp_domain_path)

        if isinstance(problem, str):
            tmp_problem_path = problem
            if not os.path.isabs(tmp_problem_path):
                tmp_problem_path = f"{current_dir}/{tmp_problem_path}"
        else:
            tmp_problem_path = get_tmp_problem_path()
            with open(tmp_problem_path, "w") as f:
                f.write(str(problem))
        tmp_problem_path = os.path.normpath(tmp_problem_path)

        if planner == "symk" and k != 1:
            planner = "symk-k"

        cmd = get_planner_cmd(planner, k, tmp_domain_path, tmp_problem_path, uid=uid, tmp_dir=tmp_dir)

        if verbose:
            if timeout is not None:
                raise NotImplementedError
            logging.info(f"Running planner with command:\n\n\t{' '.join(cmd)}\n")
            if subprocess.call(cmd) != 0:
                return clean_up_routine()
        else:
            logging.debug(f"Running planner with command:\n\n\t{' '.join(cmd)}\n")
            if timeout is not None:
                try:
                    run = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)
                except subprocess.TimeoutExpired:
                    logging.warning(f"Planner timed out after {timeout} seconds.")
                    return clean_up_routine()
            else:
                run = subprocess.run(cmd, capture_output=True, text=True, check=False)
            stdout = run.stdout
            stderr = run.stderr
            logging.debug(stdout)
            logging.debug(stderr)
            solved = any(substr in stdout for substr in ["Solution found", "Found Plan"])
            if not solved:
                return clean_up_routine()

        def collect_plan(plan_file: str) -> Plan:
            plan = []
            for line in open(plan_file, "r").readlines():
                if line.startswith(";"):
                    continue
                plan.append(line.strip())
            return plan

        if k != 1 or planner == "lama-anytime":
            plans = []
            for plan_file in sorted(os.listdir(tmp_dir)):
                if plan_file.startswith("sas_plan."):
                    plans.append(collect_plan(plan_file))
        else:
            plan_file = get_plan_path(uid, tmp_dir=tmp_dir)
            plans = [collect_plan(plan_file)]

        clean_up_routine()
        return plans
    except KeyboardInterrupt as e:
        clean_up_routine()
        raise e
