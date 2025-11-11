import logging
import os
import random
import subprocess
import sys
import time

import pddl
import termcolor as tc

from moose.diagnose import diagnose_failure
from moose.learning.policy_learner import PolicyLearner
from moose.optimal.transformer import Transformer
from moose.options import Options
from moose.planning import Plan, dump_statistics, get_plan
from moose.planning.planners import is_planner
from moose.policy import Policy
from moose.precedence import ExecutionStrategy
from moose.satisficing import get_executor_from_options
from moose.util.logging import init_logger, log_opts, mat_to_str
from moose.util.managers import TimerContextManager


class InvalidPlanError(Exception):
    """Exception raised for invalid plans."""


def train(options: Options) -> None:
    init_logger(log_level=logging.DEBUG if options.debug else logging.INFO)
    log_opts(desc="train", opts=options)

    random.seed(options.random_seed)
    total_time = time.time()
    learner = PolicyLearner(options)

    with TimerContextManager(tc.colored(f"initial learning step", "red", attrs=["bold"])) as timer:
        try:
            learner.learn()
            learner.policy.set_execution_strategy(ExecutionStrategy.GREEDY)
        except KeyboardInterrupt:
            logging.critical("Learning cancelled early by keyboard interrupt")
        time_synthesis = timer.get_time()

    rules_learned = len(learner.policy)
    if rules_learned == 0:
        logging.critical(f"No rules were learned.")
    else:
        logging.info(f"Distinct rules learned: {rules_learned}")

    # Log some stats
    total_time = time.time() - total_time

    rules_learned = len(learner.policy)
    max_condition_size = max(rule.n_conditions for rule in learner.policy.rules)
    max_actions_size = max(rule.n_actions for rule in learner.policy.rules)

    def format_time(t: float) -> str:
        return f"{t:.4f}"

    def percentage_time(t: float) -> str:
        return f"{t * 100 / total_time:.1f}%"

    stats = [
        ("*", "*", "*"),
        ("Rules learned", f"{rules_learned}"),
        ("Max condition size", f"{max_condition_size}"),
        ("Max actions sequence", f"{max_actions_size}"),
        ("Synthesis time", format_time(time_synthesis), percentage_time(time_synthesis)),
        ("Total time", format_time(total_time)),
        ("*", "*", "*"),
    ]
    logging.info("Training statistics:\n" + mat_to_str(stats, rjust=[0, 1, 1]))

    learner.save(options.save_file)
    if options.dump_policy:
        learner.dump()


def _handle_plan(options: Options, plan: Plan) -> None:
    if options.validate and options.plan_file is None:
        options.plan_file = "sas_plan"

    if options.plan_file:
        with open(options.plan_file, "w") as f:
            for action in plan:
                f.write(f"{action}\n")
        logging.info(f"Plan saved to {tc.colored(options.plan_file, 'cyan')}")

    if options.validate:
        output = subprocess.run(["which", "validate"], capture_output=True, text=True, check=False).stdout
        if len(output) == 0:
            logging.info("The command `validate` from VAL was not found. Skipping plan validation")
        else:
            with TimerContextManager("validating plan"):
                cmd = ["validate", options.domain_file, options.problem_file, options.plan_file]
                output = subprocess.run(cmd, capture_output=True, text=True, check=False).stdout
            logging.info(f"Validation output:\n{output.strip()}")
            if "Failed plans" in output:
                logging.error(tc.colored("INVALID PLAN", "red"))
                raise InvalidPlanError("Plan validation failed")


def execute_policy(options: Options) -> None:
    if options.verbosity >= 0:
        init_logger(log_level=logging.DEBUG if options.verbosity >= 4 else logging.INFO)
    log_opts(desc="plan", opts=options)

    model_file = options.model_file

    if options.dump_policy:
        policy = Policy.load(model_file)
        policy.dump()
        print(f"\n{len(policy)=}")
        return

    # Check paths
    domain_file = options.domain_file
    problem_file = options.problem_file
    assert domain_file is not None, "Domain file is required"
    assert problem_file is not None, "Problem file is required"
    assert os.path.exists(domain_file), f"Domain file {domain_file} does not exist"
    assert os.path.exists(problem_file), f"Problem file {problem_file} does not exist"
    assert os.path.exists(model_file), f"Model file {model_file} does not exist"

    # Execute policy
    policy_executor = get_executor_from_options(options=options)
    plan = policy_executor.solve()
    if plan is not None:
        logging.info(f"Plan found!")
        policy_executor.log_stats()
    elif options.diagnose_failure:
        diagnose_failure(domain_file, problem_file, policy_executor)

    if plan is None:
        logging.info("No plan found")
        sys.exit(1)

    # Optionally, write and/or validate plan
    _handle_plan(options, plan)

    # Good
    logging.info(tc.colored("GOOD", "green"))
    return


def execute_search(options: Options) -> None:
    if options.verbosity >= 0:
        init_logger(log_level=logging.DEBUG if options.verbosity >= 4 else logging.INFO)
    log_opts(desc="plan", opts=options)

    model_file = options.model_file

    if options.dump_policy:
        policy = Policy.load(model_file)
        policy.dump()
        print(f"\n{len(policy)=}")
        return

    # Check paths
    domain_file = options.domain_file
    problem_file = options.problem_file
    assert domain_file is not None, "Domain file is required"
    assert problem_file is not None, "Problem file is required"
    assert os.path.exists(domain_file), f"Domain file {domain_file} does not exist"
    assert os.path.exists(problem_file), f"Problem file {problem_file} does not exist"

    # Execute search
    if is_planner(planner=model_file):
        # Standalone planning
        plan = get_plan(domain=domain_file, problem=problem_file, planner=model_file, verbose=True)[0]
    else:
        # Planning with lifted rules as control knowledge for search
        assert os.path.exists(model_file), f"Model file {model_file} does not exist"
        transformer = Transformer(
            domain=pddl.parse_domain(options.domain_file),
            problem=pddl.parse_problem(options.problem_file),
            policy=Policy.load(options.model_file),
            options=options,
        )
        transformer.transform()

        new_domain, new_problem = transformer.get_transformed_instance()

        # Write transformed domain and problem
        if options.output_domain_file:
            with open(options.output_domain_file, "w") as f:
                f.write(str(new_domain))
            logging.info(f"Wrote transformed domain to {tc.colored(options.output_domain_file, 'cyan')}")
        if options.output_problem_file:
            with open(options.output_problem_file, "w") as f:
                f.write(str(new_problem))
            logging.info(f"Wrote transformed problem to {tc.colored(options.output_problem_file, 'cyan')}")
        if options.search is None:
            logging.info("Exiting: no planner was specified with `--search`")
            sys.exit(0)

        match options.encoding:
            case "disjpres":
                from nraxioms2disjpres.core import NrAxioms2DisjPresCompiler

                compiler = NrAxioms2DisjPresCompiler(new_domain)
                new_domain = compiler.compile()
            case "condeffs":
                raise NotImplementedError
            case "axioms":
                pass
            case _:
                raise ValueError(f"Unknown value {options.encoding=}")

        dump_statistics(new_domain, new_problem, description_prefix="Transformed instance statistics:\n")
        plan = get_plan(new_domain, new_problem, planner=options.search, verbose=True, clean_up=False)[0]

    if plan is None:
        logging.info("No plan found")
        sys.exit(1)

    # Optionally, write and/or validate plan
    _handle_plan(options, plan)

    # Good
    logging.info(tc.colored("GOOD", "green"))
    return
