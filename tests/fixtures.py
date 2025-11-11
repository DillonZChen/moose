import logging
import os

from moose.core import train
from moose.options import get_train_parser

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))

# Training
N_PERMUTATIONS = 1
N_TRAIN = 20
N_VALIDATE = 20
N_TEST = 3

# Domains
CLASSIC_DOMAINS = ["ferry", "logistics", "miconic", "rovers", "satellite", "transport"]
NUMERIC_DOMAINS = [f"numeric-{d}" for d in ["minecraft"]]
ALL_DOMAINS = [
    "barman",
    "ferry",
    "gripper",
    "logistics",
    "miconic",
    "rovers",
    "satellite",
    "transport",
    "numeric-ferry",
    "numeric-miconic",
    "numeric-minecraft",
    "numeric-transport",
]
PROBLEMS = [f"p{i:02}" for i in range(1, N_TEST + 1)]
ENCODINGS = ["axioms", "disjpres"]

# Planners
CLASSIC_PLANNERS = {
    "blind",
    "lama-first",
    "scorpion",
    "lmcut",
    "symk",
}

NUMERIC_PLANNERS = {"lmcut-numeric", "enhsp-mq"}

# Directories
BENCHMARK_DIR = os.path.normpath(f"{_CUR_DIR}/../benchmarks")
MODEL_DIR = f"{_CUR_DIR}/_models"
PLANS_DIR = f"{_CUR_DIR}/_plans"


def get_domain_file(domain_name: str) -> str:
    return f"{BENCHMARK_DIR}/{domain_name}/domain.pddl"


def get_training_dir(domain_name: str) -> str:
    return f"{BENCHMARK_DIR}/{domain_name}/training"


def get_model_file(domain_name: str) -> str:
    return f"{MODEL_DIR}/{domain_name}-test.model"


def train_routine(domain_name: str):
    os.makedirs(MODEL_DIR, exist_ok=True)

    domain_file = get_domain_file(domain_name)
    training_dir = get_training_dir(domain_name)
    model_file = get_model_file(domain_name)

    if os.path.exists(model_file):
        os.remove(model_file)

    train_args = [
        domain_file,
        training_dir,
        "--num-training",
        str(N_TRAIN),
        "--num-permutations",
        str(N_PERMUTATIONS),
        "--save-file",
        model_file,
    ]
    logging.info(f"Training with cmd:\n\n\t./train.py {' '.join(train_args)}\n")

    parser = get_train_parser()
    opts = parser.parse_args(train_args)
    train(options=opts)
