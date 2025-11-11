import os
from typing import Optional

__all__ = ["is_planner", "get_planner_cmd"]

PLANNER_DIR = os.path.normpath(f"{os.path.dirname(os.path.abspath(__file__))}/../../ext/planners")

CLASSIC_FD_DIR = f"{PLANNER_DIR}/scorpion"
CLASSIC_FD_BIN = f"{CLASSIC_FD_DIR}/scorpion.sif"

NUMERIC_FD_DIR = f"{PLANNER_DIR}/downward-numeric"
NUMERIC_FD_BIN = f"{NUMERIC_FD_DIR}/builds/release64/bin"

SYMK_DIR = f"{PLANNER_DIR}/symk"
SYMK_BIN = f"{SYMK_DIR}/symk.sif"

ENHSP_DIR = f"{PLANNER_DIR}/enhsp"
ENHSP_JAR = f"{ENHSP_DIR}/enhsp.jar"

PLANNERS = {
    "lama-first",
    "lama-anytime",
    "lmcut",
    "symk",
    "blind",
    "scorpion",
    "scorpion-ce",
    "lmcut-numeric",
    "enhsp-mq",
    "enhsp-hmrphj",
}

FD_PLANNERS = {
    "lama-first",
    "lama-anytime",
    "lmcut",
    "symk",
    "blind",
    "scorpion",
    "lmcut-numeric",
}


class PlannerNotFoundError(RuntimeError):
    def __init__(self, message: str):
        super().__init__(message)


def is_planner(planner: str) -> bool:
    return planner in PLANNERS


def check_downward_bin() -> None:
    if not os.path.exists(CLASSIC_FD_BIN):
        raise PlannerNotFoundError(f"Could not find Downward binary at {CLASSIC_FD_BIN}. Please build it.")


def check_downward_numeric_bin() -> None:
    if not os.path.exists(NUMERIC_FD_BIN):
        raise PlannerNotFoundError(f"Could not find Numeric Downward binary at {NUMERIC_FD_BIN}. Please build it.")


def check_symk_bin() -> None:
    if not os.path.exists(SYMK_BIN):
        raise PlannerNotFoundError(f"Could not find SYMK binary at {SYMK_BIN}. Please build it.")


def check_enhsp_jar() -> None:
    if not os.path.exists(ENHSP_JAR):
        msg = f"Could not find ENHSP JAR file at {ENHSP_JAR}. Please install it from\n\n"
        msg += f"\thttps://github.com/hstairs/jpddlplus/tree/socs24-width-and-mq\n"
        raise PlannerNotFoundError(msg)


def get_enhsp_cmd(planner: str, domain_pddl: str, problem_pddl: str) -> list[str]:
    return [
        f"java",
        "-jar",
        ENHSP_JAR,
        "-o",
        domain_pddl,
        "-f",
        problem_pddl,
        "-sp",
        "sas_plan",
        "-planner",
        planner,
    ]


def get_plan_path(uid: Optional[None], tmp_dir: Optional[str] = None) -> str:
    plan_file = "sas_plan"  # assume planners are run in tmp directories
    if uid is not None:
        plan_file = f"{uid}.plan"
    if tmp_dir is not None:
        plan_file = f"{tmp_dir}/{plan_file}"
    return plan_file


def get_planner_cmd(
    planner: str,
    k: int,
    domain_pddl: str,
    problem_pddl: str,
    uid: Optional[str] = None,
    tmp_dir: Optional[str] = None,
) -> list[str]:
    domain_problem = [domain_pddl, problem_pddl]
    if uid is not None:
        assert planner in {"lmcut", "lmcut-numeric"}
        sas_file = f"{uid}.sas"
        if tmp_dir is not None:
            sas_file = f"{tmp_dir}/{sas_file}"
        sas_file_cmd = ["--sas-file", sas_file]
    else:
        sas_file_cmd = []
    plan_file = get_plan_path(uid, tmp_dir=tmp_dir)
    plan_file_cmd = ["--plan-file", plan_file]
    match planner:
        # Classic -- SAT
        case "lama-first":
            check_downward_bin()
            cmd = [f"{CLASSIC_FD_DIR}/lama"] + domain_problem
        case "lama-anytime":
            check_downward_bin()
            cmd = [f"{CLASSIC_FD_DIR}/lama-seq"] + domain_problem
        # Classic -- OPT
        case "blind":
            check_downward_bin()
            cmd = [CLASSIC_FD_BIN] + domain_problem + ["--search", "astar(blind())"]
        case "lmcut":
            check_downward_bin()
            cmd = [CLASSIC_FD_BIN] + sas_file_cmd + plan_file_cmd + domain_problem + ["--search", "astar(lmcut())"]
        case "symk":
            check_symk_bin()
            cmd = [SYMK_BIN] + domain_problem + ["--search", "sym_bd()"]
        case "symk-k":
            check_symk_bin()
            if k == 0:
                k = "infinity"
            elif k < 0:
                raise ValueError(f"Invalid value for k: {k}. Must be a positive integer or 0.")
            cmd = (
                [SYMK_BIN]
                + domain_problem
                + ["--search", f"symq_bd(plan_selection=top_k(num_plans={k},dump_plans=false),quality=1)"]
            )
        case "scorpion":
            check_downward_bin()
            cmd = [
                CLASSIC_FD_BIN,
                "--transform-task",
                "preprocess-h2",
                "--alias",
                "scorpion",
            ] + domain_problem
        case "scorpion-ce":  # scorpion with conditional effects config described in the README
            check_downward_bin()
            cmd = [CLASSIC_FD_BIN, "--transform-task", "preprocess-h2"] + sas_file_cmd + plan_file_cmd + domain_problem
            cmd += [
                "--search",
                r"astar(scp_online([projections(sys_scp(max_time=100, max_time_per_restart=10, max_pdb_size=2M, max_collection_size=20M, pattern_type=interesting_non_negative, create_complete_transition_system=true),           create_complete_transition_system=true)], saturator=perimstar, max_time=100, max_size=1M, interval=10K, orders=greedy_orders()))",
            ]
        # Numeric -- SAT
        case "enhsp-mq":
            check_enhsp_jar()
            cmd = get_enhsp_cmd("sat-mq3h3n", domain_pddl, problem_pddl)
        case "enhsp-hmrphj":
            check_enhsp_jar()
            cmd = get_enhsp_cmd("sat-hmrphj", domain_pddl, problem_pddl)
        # Numeric -- OPT
        case "lmcut-numeric":
            check_downward_numeric_bin()
            cmd = [f"{NUMERIC_FD_DIR}/opt_lmcut.py"] + sas_file_cmd + plan_file_cmd + domain_problem
        case _:
            raise ValueError(f"Unknown planner: {planner}")

    return cmd
