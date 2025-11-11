import argparse
from argparse import RawDescriptionHelpFormatter as Formatter

import termcolor

c = lambda x: termcolor.colored(x, color="blue", on_color="on_white")

TRAIN_DESCRIPTION = "Moose Generalised Planner"
TRAIN_EPILOG = f"""
Example usage:
{c('./train.py benchmarks/ferry/domain.pddl benchmarks/ferry/training/ --save-file ferry.model')}
"""

POLICY_DESCRIPTION = "Moose Policy Execution"
POLICY_EPILOG = f"""
Example usages:

Planning via policy execution
{c('./policy.py ferry.model benchmarks/ferry/domain.pddl benchmarks/ferry/testing/p0_30.pddl -val')}

Dump saved policy
{c('./policy.py ferry.model --dump-policy')}
"""

SEARCH_DESCRIPTION = "Moose Search Execution"
SEARCH_EPILOG = f"""
Example usages:

Planning via policy guided search with axioms
{c('./search.py ferry.model benchmarks/ferry/domain.pddl benchmarks/ferry/testing/p0_30.pddl -val --search=symk')}

Planning via policy guided search with axioms compiled away as disjunctive preconditions
{c('./search.py ferry.model benchmarks/ferry/domain.pddl benchmarks/ferry/testing/p0_30.pddl -val --search=symk --encoding=disjpres')}

Translate only and serialise
{c('./search.py ferry.model benchmarks/ferry/domain.pddl benchmarks/ferry/testing/p0_30.pddl -od domain_new.pddl -op problem_new.pddl')}

Translate only and compile away axioms and serialise
{c('./search.py ferry.model benchmarks/ferry/domain.pddl benchmarks/ferry/testing/p0_30.pddl --no-axioms -od domain_new_na.pddl -op problem_new_na.pddl')}

Run LAMA-first by itself
{c('./search.py lama-first benchmarks/ferry/domain.pddl benchmarks/ferry/testing/p0_30.pddl')}

Dump saved policy
{c('./policy.py ferry.model --dump-policy')}
"""

Options = argparse.Namespace


def get_train_parser() -> argparse.ArgumentParser:
    # fmt: off
    parser = argparse.ArgumentParser(description=TRAIN_DESCRIPTION, epilog=TRAIN_EPILOG, formatter_class=Formatter)

    parser.add_argument("domain_file")
    parser.add_argument("training_dir", nargs="?", default=None)

    parser.add_argument("-t", "--num_workers", type=int, default=8,
                        help="Number of threads for synthesis.")
    parser.add_argument("-p", "--num-permutations", type=int, default=3,
                        help="Number of goal permutations per problem")
    parser.add_argument("-g", "--goal-max-size", type=int, default=1,
                        help="Maximum size of goals")
    parser.add_argument("-n", "--num-training", type=int, default=-1,
                        help="Number of training problems: -1 for all.")
    parser.add_argument("-v", "--num-validation", type=int, default=-1,
                        help="Number of validation problems: -1 for all.")
    parser.add_argument("-vb", "--validation-bound", type=int, default=30,
                        help="Bound for validation loops.")

    parser.add_argument("--random-seed", type=int, default=0, help="Random seed")
    parser.add_argument("--save-file", type=str, default=None)
    parser.add_argument("--dump-policy", action="store_true")
    parser.add_argument("-d", "--debug", action="store_true")

    return parser
    # fmt: on


def get_policy_parser() -> argparse.ArgumentParser:
    # fmt: off
    parser = argparse.ArgumentParser(description=POLICY_DESCRIPTION, epilog=POLICY_EPILOG, formatter_class=Formatter)

    # Input
    parser.add_argument("model_file")
    parser.add_argument("domain_file", nargs="?", default=None)
    parser.add_argument("problem_file", nargs="?", default=None)

    # Configuration
    parser.add_argument("--bound", type=int, default=0,
                        help="Bound for policy execution: 0 for none.")
    parser.add_argument("--policy-strategy", choices=["action", "macro"], default=None,
                        help="Policy execution strategy. Default is None, which uses the strategy that was learned.")
    parser.add_argument("--precedence-strategy", choices=["min", "max"], default=None,
                        help="Precedence search strategy. Default is None, which uses the strategy that was learned.")
    parser.add_argument("--disallow-cycles", action="store_true",
                        help="Terminate if a cycle is detected.")
    parser.add_argument("--diagnose-failure", action="store_true",
                        help="Diagnose failure if planning fails")

    # Output
    parser.add_argument("--dump-policy", action="store_true", help="Dump policy and exit without planning")
    parser.add_argument("--plan-file", help="Output plan file")
    parser.add_argument("-val", "--validate", action="store_true", help="Validate plan")
    parser.add_argument("-v", "--verbosity", type=int, default=0, help="Increase output verbosity")

    return parser
    # fmt: on


def get_search_parser() -> argparse.ArgumentParser:
    # fmt: off
    parser = argparse.ArgumentParser(description=SEARCH_DESCRIPTION, epilog=SEARCH_EPILOG, formatter_class=Formatter)

    # Input
    parser.add_argument("model_file")
    parser.add_argument("domain_file", nargs="?", default=None)
    parser.add_argument("problem_file", nargs="?", default=None)

    # Configuration
    parser.add_argument("--search", choices=["symk", "scorpion", "scorpion-ce", "lama-first"], default=None,
                        help="Planner for policy guided search. None if only output transformation.")
    parser.add_argument("--encoding", choices=["axioms", "disjpres", "condeffs"], default="axioms",
                        help="Method to encode search control")

    # Output
    parser.add_argument("-od", "--output-domain-file", default=None, help="Output domain file if specified.")
    parser.add_argument("-op", "--output-problem-file", default=None, help="Output problem file if specified.")
    parser.add_argument("--dump-policy", action="store_true", help="Dump policy and exit without planning")
    parser.add_argument("--plan-file", help="Output plan file")
    parser.add_argument("-val", "--validate", action="store_true", help="Validate plan")
    parser.add_argument("-v", "--verbosity", type=int, default=0, help="Increase output verbosity")

    return parser
    # fmt: on
