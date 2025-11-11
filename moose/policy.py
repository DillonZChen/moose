import argparse
import logging
import os
import pickle
from typing import Optional

from pddl.core import Domain
from pddl.logic import Predicate

from moose.planning import State
from moose.precedence import ExecutionStrategy, PrecedenceStrategy, PrecedenceValue
from moose.rule import Rule


class Policy:
    def __init__(self, domain: Domain):
        self._domain = domain

        self._rules: set[Rule] = set()
        self._key_to_objects: dict[str, tuple[State, list[Predicate]]] = {}

        self._precedence_strategy = PrecedenceStrategy.MIN
        self._execution_strategy = None

        self._options = None

    @property
    def domain(self) -> Domain:
        return self._domain

    @property
    def precedence_strategy(self) -> str:
        return self._precedence_strategy

    @property
    def execution_strategy(self) -> Optional[str]:
        return self._execution_strategy

    @property
    def rules(self) -> list[Rule]:
        return self._rules

    @property
    def options(self) -> Optional[argparse.Namespace]:
        """Returns the options used for training."""
        return self._options

    @property
    def precedences(self) -> list[PrecedenceValue]:
        """Gets precedences sorted by precedence strategy."""
        return PrecedenceStrategy.sort_precedences(
            [rule.precedence for rule in self._rules], self._precedence_strategy
        )

    def set_train_options(self, options) -> None:
        self._options = options

    def set_execution_strategy(self, execution_strategy: ExecutionStrategy) -> None:
        logging.info(f"Setting policy execution strategy to `{execution_strategy}`.")
        match execution_strategy:
            case ExecutionStrategy.GREEDY:
                self._precedence_strategy = PrecedenceStrategy.MIN
            case ExecutionStrategy.GREEDY_LOOP:
                self._precedence_strategy = PrecedenceStrategy.MIN
            case ExecutionStrategy.CONSERVATIVE:
                self._precedence_strategy = PrecedenceStrategy.MAX
            case _:
                raise ValueError(f"Invalid execution strategy: {execution_strategy}")
        self._execution_strategy = execution_strategy

    def insert_rule(self, rule: Rule) -> bool:
        if rule in self._rules:
            return False
        self._rules.add(rule)
        return True

    def modify_precedence(self, rule: Rule, modification: PrecedenceValue) -> None:
        logging.info(f"Changing priority by {modification} for rule\n{rule}")
        self._rules.discard(rule)
        rule.modify_precedence(modification)
        self._rules.add(rule)
        logging.info(f"Rule after priority update is\n{rule}")

    def dump(self) -> None:
        print(str(self))

    def save(self, filename: str) -> None:
        dir_name = os.path.dirname(filename)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def save_readable(self, filename: str) -> None:
        dir_name = os.path.dirname(filename)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(filename, "w") as f:
            f.write(str(self))

    @staticmethod
    def load(filename: str) -> "Policy":
        with open(filename, "rb") as f:
            policy = pickle.load(f)
        return policy

    def __len__(self) -> int:
        return len(self._rules)

    def __str__(self) -> str:
        precedences = {p: [] for p in self.precedences}
        for rule in self._rules:
            precedences[rule.precedence].append(rule.to_string())
        ret = []
        for rules in precedences.values():
            for rule in rules:
                ret.append(rule)
        return "\n\n".join(map(str, ret))
