from dataclasses import dataclass
from enum import Enum


@dataclass
class PrecedenceValue:
    goal_size: int = 0
    p0: int = 0
    p1: int = 0
    cycle: int = 0

    def __add__(self, other: "PrecedenceValue") -> "PrecedenceValue":
        return PrecedenceValue(
            self.goal_size + other.goal_size, self.p0 + other.p0, self.p1 + other.p1, self.cycle + other.cycle
        )

    def __hash__(self) -> int:
        return hash((self.goal_size, self.p0, self.p1, self.cycle))

    def __str__(self) -> str:
        return str((self.goal_size, self.p0, self.p1, self.cycle))


class PrecedenceStrategy(Enum):
    MIN = "min"
    MAX = "max"

    @staticmethod
    def sort_precedences(precedences: list[PrecedenceValue], strategy: "PrecedenceStrategy") -> list[PrecedenceValue]:
        if strategy == PrecedenceStrategy.MIN:
            # # Try lowest h* value, tie breaking by larger goal size, then validation tiebreaker
            # return sorted(precedences, key=lambda x: (x.p0, -x.goal_size, x.p1))
            # Try largest goal sizes, tie breaking by lowest h* value, then validation tiebreaker
            return sorted(precedences, key=lambda x: (-x.goal_size, x.p0, x.p1))
        elif strategy == PrecedenceStrategy.MAX:
            return sorted(precedences, key=lambda x: (-x.p0, x.p1))
        else:
            raise ValueError(f"Invalid precedence strategy: {strategy}")


class ExecutionStrategy(Enum):
    GREEDY = "greedy"
    GREEDY_LOOP = "greedy-loop"
    CONSERVATIVE = "conservative"

    @classmethod
    def choices(cls):
        return [item for item in cls]

    @classmethod
    def parse(cls, value):
        if isinstance(value, cls):
            return value
        return cls(value)

    def __str__(self):
        return self.value
