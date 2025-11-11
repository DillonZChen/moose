import logging
import time

import termcolor as tc
from pddl.action import Action
from pddl.logic.base import Not
from pddl.logic.functions import Decrease, Increase, NumericFunction
from pddl.logic.predicates import Predicate
from tqdm import tqdm

from moose.planning import State, get_literals_list
from moose.planning.strings import NumericCondition, action_to_string, token_to_numeric_value
from moose.satisficing.executors.sqlite_executor import SQLitePolicy, to_sql_name


class FakeTqdmBar:
    def update(self, _: int) -> None:
        pass


class SQLitePolicyMk3(SQLitePolicy):
    def _init_impl(self):
        super()._init_impl()
        self._step = 1

    def get_init_state(self) -> State:
        init_state = self._problem.init

        # facts and numerics
        for f in init_state:
            if isinstance(f, NumericCondition):
                table = to_sql_name(f.operands[0])
                values = ", ".join(self.to_sql_terms(f.operands[0], constants=True)[:-1] + [str(f.operands[1])])
            else:
                table = to_sql_name(f)
                values = self.sql_terms_str(f, constants=True)
            self.exec(f"INSERT INTO {table} VALUES ({values})")

        # typing information
        for obj in self._problem.objects:
            f = self.to_type_predicate(obj)
            self.exec(f"INSERT INTO {to_sql_name(f)} VALUES ({self.sql_terms_str(f, constants=True)})")

        self.check_cycle()
        return init_state

    def get_numeric_value(self, function: NumericFunction) -> float:
        binding = [
            f"{obj} = {term}"
            for obj, term in zip(
                self.to_sql_terms(function, constants=True)[:-1],
                self.to_sql_terms(self._name_to_function[function.name], constants=False)[:-1],
            )
        ]
        sql_query = f"SELECT numeric_value FROM {to_sql_name(function, ug=False)}"
        if len(binding) > 0:
            sql_query += f" WHERE {' AND '.join(binding)}"
        state_value = self.exec(sql_query).fetchall()[0][0]
        return state_value

    def numeric_goal_achieved(self, function: NumericFunction) -> bool:
        state_value = self.get_numeric_value(function)
        goal_value = token_to_numeric_value(self._numeric_goals[function])
        return state_value == goal_value

    def numeric_goals_achieved(self) -> bool:
        for function in self._numeric_goals:
            if not self.numeric_goal_achieved(function):
                return False
        return True

    def classic_goal_achieved(self, goal: Predicate) -> bool:
        binding = [
            f"{obj} = {term}"
            for obj, term in zip(
                self.to_sql_terms(goal, constants=True),
                self.to_sql_terms(self._name_to_predicate[goal.name], constants=False),
            )
        ]
        sql_query = f"SELECT EXISTS(SELECT 1 FROM {to_sql_name(goal, ug=False)} WHERE {' AND '.join(binding)})"
        achieved = self.exec(sql_query).fetchall()[0][0]
        return bool(achieved)

    def apply_action_internal(self, action: Action) -> tuple[list[Predicate], list[Predicate]]:
        """Apply action to SQLite database state"""
        t = time.time()

        adds = []
        dels = []
        for effect in get_literals_list(action.effect):
            if isinstance(effect, Predicate):
                fact = effect
                adds.append(fact)
                self.add_fact(fact, ug=False)
            elif isinstance(effect, Not):
                fact = effect.argument
                dels.append(fact)
                self.del_fact(fact, ug=False)
            elif isinstance(effect, Increase):
                function = effect.operands[0]
                change = token_to_numeric_value(effect.operands[1])
                old_value = self.get_numeric_value(function)
                new_value = old_value + change
                self.set_value(function, new_value)
            elif isinstance(effect, Decrease):
                function = effect.operands[0]
                change = token_to_numeric_value(effect.operands[1])
                old_value = self.get_numeric_value(function)
                new_value = old_value - change
                self.set_value(function, new_value)
            else:
                raise NotImplementedError(f"Effect of type {type(effect)} not support yet: {effect}")

        self._profiling_statistics["apply_action"] += time.time() - t
        if self._verbosity >= 1:
            step = tc.colored(f"step {self._step}", "blue")
            logging.info(f"{step}: {action_to_string(action, plan_style=True)}")
            self._step += 1
        return adds, dels

    def _solve_impl(self) -> bool:
        init_state = self.get_init_state()

        unachieved_classic_goals = set(goal for goal in self._classic_goals if goal not in init_state)

        if self._verbosity >= 0:
            progress = tqdm(total=len(unachieved_classic_goals))
        else:
            progress = FakeTqdmBar()

        for goal in unachieved_classic_goals:
            self.add_fact(goal, ug=True)

        if self._verbosity >= 2:
            state_str = self.get_state_str(delimiter="\n\t")
            logging.info(f"current state:\n\t{state_str}")

        while True:
            if len(unachieved_classic_goals) == 0 and self.numeric_goals_achieved():  # TODO check numeric goal
                break
            if self.reached_bound():
                return False

            # Query and apply action
            actions = self.query()
            if actions is None:
                return False

            self.add_actions_to_plan(actions)
            for action in actions:
                adds, dels = self.apply_action_internal(action)
                if self.check_cycle():
                    return False
                for f in adds:
                    if f in unachieved_classic_goals:
                        self.del_fact(f, ug=True)
                        progress.update(1)
                        unachieved_classic_goals.remove(f)
                # Add deleted goals back e.g. Satellite pointing goals
                for f in dels:
                    if f in self._classic_goals and f not in unachieved_classic_goals:
                        self.add_fact(f, ug=True)
                        progress.update(-1)
                        unachieved_classic_goals.add(f)

                if self._verbosity >= 2:
                    state_str = self.get_state_str(delimiter="\n\t")
                    logging.info(f"current state:\n\t{state_str}")

        return True
