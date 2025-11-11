#!/usr/bin/env python3

import argparse
import os

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description="Run LM-Cut")
    parser.add_argument("domain", help="domain file")
    parser.add_argument("problem", help="problem file")
    parser.add_argument("--plan-file", help="output plan file", default="sas_plan")
    parser.add_argument("--sas-file", help="classical sas", default="output.sas")
    parser.add_argument("--timeout", help="timeout in seconds", type=int, default=3600)
    args = parser.parse_args()

    heuristic = "lmcutnumeric(use_second_order_simple=true,bound_iterations=10,ceiling_less_than_one=true)"
    search = f"'astar({heuristic},max_time={args.timeout})'"

    cmd = [
        "python2",
        f"{_CUR_DIR}/fast-downward.py",
        "--build",
        "release64",
        "--plan-file",
        args.plan_file,
        "--sas_file",
        args.sas_file,
        args.domain,
        args.problem,
        "--search",
        search,
    ]

    os.system(" ".join(cmd))


if __name__ == "__main__":
    main()
