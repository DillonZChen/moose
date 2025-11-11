import logging
import sqlite3
import time
from abc import abstractmethod
from itertools import combinations, product
from typing import Any, Iterable, Optional, Union

from pddl.action import Action
from pddl.logic import Constant, Predicate
from pddl.logic.base import Not
from pddl.logic.functions import BinaryFunction, EqualTo, NumericFunction, NumericValue
from pddl.logic.terms import Term
from tqdm import tqdm

from moose.planning import Literal, State
from moose.planning.ground import ground_action
from moose.planning.progression import apply_action
from moose.planning.strings import NumericCondition, action_to_string, aopt_to_string, get_numeric_condition_symbol
from moose.precedence import ExecutionStrategy, PrecedenceValue
from moose.rule import OBJECT_TYPE, UG_SUFFIX, Rule
from moose.satisficing.executor import PolicyExecutor
from moose.util.logging import mat_to_str
from moose.util.managers import TimerContextManager

_DUMMY_TERM = "v"
_NUMERIC_VALUE_TERM = "numeric_value"


def to_sql_name(predicate: Any, ug: bool = False) -> str:
    if isinstance(predicate, Action):
        prefix = "a"
        name = predicate.name
    elif isinstance(predicate, NumericFunction):
        prefix = "n"
        name = predicate.name
    elif isinstance(predicate, Predicate):
        prefix = "p"
        name = predicate.name
    elif isinstance(predicate, Not):
        prefix = "p"
        name = predicate.argument.name
    elif isinstance(predicate, str):
        prefix = "p"
        name = predicate
    else:
        raise ValueError(f"Invalid predicate type: {type(predicate)}")
    ret = f"{prefix}_{name.replace('-', '_')}"
    if ug:
        ret += UG_SUFFIX
    return ret


class SQLitePolicy(PolicyExecutor):
    def _init_impl(self):

        domain = self._domain
        problem = self._problem
        policy = self._policy

        if domain.name != policy.domain.name:
            logging.critical(f"Domain name mismatch: {domain.name} != {policy.domain.name}.")

        # Statistics
        precedences = policy.precedences
        self._planning_statistics: dict[str, Any] = {
            "n_sql_executions": 0,
            "n_queries_by_precedence": {d: 0 for d in precedences},
            "n_sat_queries_by_precedence": {d: 0 for d in precedences},
            "n_sat_repeat_queries": 0,
            "n_unsat_repeat_queries": 0,
            "n_queries": 0,
        }
        self._profiling_statistics: dict[str, Any] = {
            "initialise": 0,
            "apply_action": 0,
            "policy_query": {d: 0 for d in precedences},
            "sat_query": 0,
            "unsat_query": 0,
        }

        # SQLite objects
        self._con = sqlite3.connect(":memory:")
        self._cur = self._con.cursor()

        # Data structures
        self._query_to_rule: dict[str, Rule] = {}
        self._query_to_schemata: dict[str, list[Action]] = {}
        self._query_to_var_mapping: dict[str, dict[Any, Any]] = {}
        self._constant_obj_names: set[str] = {o.name for o in domain.constants}
        self._pddl_term_to_sql_term: dict[str, str] = {}
        self._pred_term_i_to_sql_term: dict[tuple[str, int], str] = {}
        self._sql_term_to_pddl_term: dict[str, str] = {}
        self._sql_obj_to_pddl_obj: dict[str, Constant] = {
            self.to_sql_term(o.name): o for o in problem.objects | domain.constants
        }

        # Repetition
        self._prev_sat_query = None
        self._prev_action = None

        # Initialise
        with TimerContextManager("initialising policy executor") as timer:
            self._set_up_policy()
        self._profiling_statistics["initialise"] = timer.get_time()

    def _set_up_policy(self) -> None:
        domain = self._domain
        policy = self._policy

        # Predicates
        for literal in domain.predicates:
            for ug in [False, True]:
                sql_terms = self.to_sql_terms(literal, constants=False)
                table_terms = self.sql_terms_str(literal, constants=False, sql_typing=True)
                index_terms = self.sql_terms_str(literal, constants=False, sql_typing=False)
                sql_name = to_sql_name(literal, ug=ug)
                self.exec(f"CREATE TABLE {sql_name} ({table_terms})")
                self.exec(f"CREATE INDEX {sql_name}_index ON {sql_name}({index_terms})")
                for i, term in enumerate(sql_terms):
                    self._pred_term_i_to_sql_term[(sql_name, i)] = term

        # Numeric Functions
        for function in domain.functions:
            sql_terms = self.to_sql_terms(function, constants=False)
            table_terms = self.sql_terms_str(function, constants=False, sql_typing=True)
            index_terms = self.sql_terms_str(function, constants=False, sql_typing=False)
            sql_name = to_sql_name(function, ug=False)
            self.exec(f"CREATE TABLE {sql_name} ({table_terms})")
            self.exec(f"CREATE INDEX {sql_name}_index ON {sql_name}({index_terms})")
            for i, term in enumerate(sql_terms):
                self._pred_term_i_to_sql_term[(sql_name, i)] = term

        # Object types
        for obj_type in list(domain.types) + [OBJECT_TYPE]:
            type_predicate = self.to_type_predicate(obj_type)
            sql_name = to_sql_name(type_predicate, ug=False)

            self.exec(f"CREATE TABLE {sql_name} (name TEXT)")
            self.exec(f"CREATE INDEX {sql_name}_index ON {sql_name}(name)")
            self._pred_term_i_to_sql_term[(sql_name, 0)] = "name"

        # Rules
        self.rules: dict[PrecedenceValue, list[str]] = {d: [] for d in policy.precedences}
        for rule in policy.rules:
            # Set up components
            precedence = rule.precedence
            schemata = rule.actions
            classic_body = []
            numeric_body = []
            classic_goals = []
            numeric_goals = []
            for p in sorted(rule.s_cond, key=lambda x: str(x)):
                if isinstance(p, NumericCondition):
                    numeric_body.append(p)
                else:
                    classic_body.append(p)
            for g in sorted(rule.g_cond):
                if isinstance(g, NumericCondition):
                    numeric_goals.append(g)
                else:
                    classic_goals.append(g)

            body = []
            body += classic_body
            body += numeric_body
            body += classic_goals
            # body += numeric_goals

            # Collect terms
            terms = set()
            for literal in body:
                if isinstance(literal, BinaryFunction):
                    literal = literal.operands[0]
                terms.update(literal.terms)
            terms = sorted(terms)

            # Collect used predicates and functions
            sql_preds = (
                [to_sql_name(p) for p in classic_body]
                + [to_sql_name(p.operands[0]) for p in numeric_body]
                + [to_sql_name(p) for p in classic_goals]
                # + [to_sql_name(p.operands[0]) for p in numeric_goals]  # don't want to suffix numeric goals
            )

            # Add typing condition to body
            for term in terms:
                if term.name in self._constant_obj_names:
                    continue
                type_predicate = self.to_type_predicate(term)
                sql_name = to_sql_name(type_predicate, ug=False)

                body.append(type_predicate)
                sql_preds.append(sql_name)
                self._pred_term_i_to_sql_term[(sql_name, 0)] = "name"

            # Convert parameters to their sql equivalent
            term_to_sql = {}
            for i, literal in enumerate(body):
                for j, term in enumerate(self.to_sql_terms(literal, constants=False)):
                    sql_term = f"p{i}.{self._pred_term_i_to_sql_term[(sql_preds[i], j)]}"
                    term_to_sql[term] = sql_term

            # 1. head variables
            var_mapping = [[None for _ in instantiation] for _, instantiation in schemata]
            term_to_i_mapping = {}
            for sch_i, (schema_name, instantiation) in enumerate(schemata):
                for term_i, term in enumerate(self.to_sql_terms(instantiation, constants=False)):
                    if term in self._constant_obj_names | {_NUMERIC_VALUE_TERM}:
                        continue
                    sql_term = term_to_sql[term]
                    if sql_term not in term_to_i_mapping:
                        term_to_i_mapping[sql_term] = len(term_to_i_mapping)
                    var_mapping[sch_i][term_i] = term_to_i_mapping[sql_term]

            select_terms = term_to_i_mapping.keys()
            select_line = "SELECT DISTINCT " + ", ".join(select_terms)
            if len(select_terms) == 0:
                select_line = "SELECT DISTINCT 1"  # for rules with no free variables

            # 2. naming predicates
            from_line = "FROM\n\t" + ",\n\t".join(p + f" p{i}" for i, p in enumerate(sql_preds))

            # 3. unification
            where_line = []
            # equality constraints
            for i, literal in enumerate(body):
                for j, term in enumerate(self.to_sql_terms(literal, constants=False)):
                    sql_term = term_to_sql[term]
                    this_term = f"p{i}.{self._pred_term_i_to_sql_term[(sql_preds[i], j)]}"
                    if term == _NUMERIC_VALUE_TERM:
                        comparison = get_numeric_condition_symbol(literal)
                        lhs = this_term
                        rhs = literal.operands[1]
                    else:
                        comparison = "=="
                        if this_term != sql_term:
                            lhs = this_term
                            rhs = sql_term
                        elif term in self._constant_obj_names:
                            lhs = sql_term
                            rhs = f"'{term}'"
                        else:
                            continue
                    where_line.append(f"{lhs} {comparison} {rhs}")
            # inequality constraints
            for t1, t2 in combinations(sorted(select_terms, key=lambda x: str(x)), 2):
                where_line.append(f"{t1} != {t2}")
            where_line = "WHERE\n\t" + " AND\n\t".join(where_line)

            sql_query = "\n".join([select_line, from_line, where_line])
            sql_query += " LIMIT 1"

            self.rules[precedence].append(sql_query)
            self._query_to_rule[sql_query] = rule
            self._query_to_schemata[sql_query] = schemata
            self._query_to_var_mapping[sql_query] = var_mapping

            # print(rule.to_string(body_delimiter="\n\t"))
            # print(sql_query)
            # breakpoint()

        sql_rules_str = []
        for precedence, queries in self.get_rule_items():
            for query in queries:
                schemata = self._query_to_schemata[query]
                schemata_str = ", ".join(aopt_to_string(s) for s in schemata)
                sql_rules_str.append(f"{precedence} : {schemata_str}\n{query}\n")
        if self._verbosity >= 5:
            rules_str = "\n".join(sql_rules_str)
            logging.debug(f"SQL Rules:\n{rules_str}")

    def get_rule_items(self):
        return list(self.rules.items())

    def exec(self, cmd: str) -> Any:
        logging.debug(f"Executing sqlite cmd:\n{cmd}\n")
        try:
            self._planning_statistics["n_sql_executions"] += 1
            return self._cur.execute(cmd)
        except sqlite3.Error as e:
            logging.error(f"Error executing sqlite cmd:\n\n{cmd}\n\nCausing the error:\n\n{e}")
            raise e

    def to_type_predicate(self, term: Union[Term, str]) -> Predicate:
        if isinstance(term, str):
            return Predicate(f"{term}__t")
        elif isinstance(term, Term):
            tt = term.type_tag
            if tt is None:
                tt = OBJECT_TYPE
            return Predicate(f"{tt}__t", term)
        else:
            raise ValueError(f"Invalid term type for type predicate: {type(term)}")

    def to_sql_term(self, term: str) -> str:
        if term not in self._pddl_term_to_sql_term:
            sql_term = term.replace("-", "_")
            self._pddl_term_to_sql_term[term] = sql_term
            self._sql_term_to_pddl_term[sql_term] = term
        return self._pddl_term_to_sql_term[term]

    def to_sql_terms(self, input, constants: bool, sql_typing: bool = False) -> list[str]:
        # assert isinstance(input, (Literal, NumericFunction, Action, NumericCondition))

        if isinstance(input, Not):
            input = input.argument
        elif isinstance(input, NumericCondition):
            input = input.operands[0]

        terms = []
        if isinstance(input, Iterable):
            input_terms = input
        else:
            input_terms = input.terms
        for term in input_terms:
            terms.append(self.to_sql_term(term.name))

        if len(terms) == 0 and isinstance(input, Predicate):
            terms.append(_DUMMY_TERM)
        elif isinstance(input, NumericFunction):
            terms.append(_NUMERIC_VALUE_TERM)

        if constants:
            terms = [f"'{term}'" for term in terms]

        if sql_typing:
            terms = [f"{term} TEXT" for term in terms]
            if isinstance(input, NumericFunction):
                terms[-1] = terms[-1].replace(" TEXT", " REAL")

        return terms

    def sql_terms_str(self, literal: Literal | Action, constants: bool, sql_typing: bool = False) -> str:
        terms = self.to_sql_terms(literal, constants, sql_typing=sql_typing)
        ret = f"{', '.join(terms)}"
        return ret

    def get_state_str_list(self, pddl_format: bool = True) -> list[str]:
        state = set()
        for p, ug in product(self._predicates, [True, False]):
            for objs in self.exec(f"SELECT {self.sql_terms_str(p, constants=False)} FROM {to_sql_name(p, ug=ug)}"):
                pred_name = p.name
                if p.arity == 0:
                    state.add(pred_name)
                    continue
                if ug:
                    pred_name += UG_SUFFIX
                if pddl_format:
                    fact = "(" + " ".join([pred_name] + [self._sql_term_to_pddl_term[o] for o in objs]) + ")"
                else:
                    fact = f"{pred_name}({','.join(self._sql_term_to_pddl_term[o] for o in objs)})"
                state.add(fact)

        for f in self._functions:
            for objs in self.exec(f"SELECT {self.sql_terms_str(f, constants=False)} FROM {to_sql_name(f, ug=False)}"):
                if pddl_format:
                    fact = "(" + " ".join([f.name] + [self._sql_term_to_pddl_term[o] for o in objs[:-1]]) + ")"
                else:
                    fact = f"{f.name}({','.join(self._sql_term_to_pddl_term[o] for o in objs[:-1])})"
                fact = f"{fact} == {objs[-1]}"
                state.add(fact)

        return sorted(state)

    def get_state_str(self, pddl_format: bool = True, delimiter: str = ",") -> str:
        state_str = self.get_state_str_list(pddl_format)
        return delimiter.join(state_str)

    def get_state_pddl(self) -> State:
        state_str = self.get_state_str_list(pddl_format=False)
        str_to_pred = {p.name: p for p in self._domain.predicates}
        str_to_func = {f.name: f for f in self._domain.functions}
        str_to_obj = {o.name: o for o in self._domain.constants | self._problem.objects}
        state = set()
        # goals = set()
        for fact_str in state_str:
            # print(fact_str)
            header = fact_str.split("(")[0]
            terms = fact_str.split("(")[1].split(")")[0].split(",")
            is_goal = header.endswith(UG_SUFFIX)
            header = header.replace(UG_SUFFIX, "")
            if header in str_to_pred:
                fact = str_to_pred[header](*tuple(str_to_obj[t] for t in terms))
            else:
                value = float(fact_str.split("==")[1].replace(" ", ""))
                fact = EqualTo(str_to_func[header](*tuple(str_to_obj[t] for t in terms)), NumericValue(value))
            if is_goal:
                # goals.add(fact)
                continue
            else:
                state.add(fact)
        return frozenset(state)

    def check_cycle(self) -> bool:
        if not self._disallow_cycles:
            return False
        state_str = self.get_state_str()
        if state_str in self._seen:
            self._detected_cycle = True
            return True
        else:
            self._seen.add(state_str)
            return False

    def add_fact(self, fact: Predicate, ug: bool) -> None:
        self.exec(f"INSERT INTO {to_sql_name(fact, ug=ug)} VALUES ({self.sql_terms_str(fact, constants=True)})")

    def del_fact(self, fact: Predicate | NumericFunction, ug: bool) -> None:
        if len(fact.terms) == 0 and isinstance(fact, NumericFunction):
            self.exec(f"DELETE FROM {to_sql_name(fact, ug=ug)}")
        elif len(fact.terms) == 0:
            self.exec(f"DELETE FROM {to_sql_name(fact, ug=ug)} WHERE {_DUMMY_TERM} = '{_DUMMY_TERM}'")
        else:
            sql_name = to_sql_name(fact, ug=ug)
            objs = [self.to_sql_term(o.name) for o in fact.terms]
            terms = [self._pred_term_i_to_sql_term[(sql_name, i)] for i in range(len(fact.terms))]
            where_line = " AND ".join(f"{o} = '{v}'" for o, v in zip(terms, objs))
            self.exec(f"DELETE FROM {sql_name} WHERE {where_line}")

    def set_value(self, function: NumericFunction, value: float) -> None:
        self.del_fact(function, ug=False)
        table = to_sql_name(function, ug=False)
        values = ", ".join(self.to_sql_terms(function, constants=True)[:-1] + [str(value)])
        self.exec(f"INSERT INTO {table} VALUES ({values})")

    def _log_failed_query_action(self) -> None:
        db_state = self.get_state_str_list()
        db_state_str = "\n".join(db_state)
        actions = self._plan
        actions = "\n".join(actions) if actions else "None"
        logging.warning(f"No actions found at state\n{db_state_str}\n\nafter executing actions\n{actions}\n")

    def get_actions_from_queries(self, query: str, instantiations: list[tuple[str]]) -> Optional[list[Action]]:
        schemata = self._query_to_schemata[query]
        var_mapping = self._query_to_var_mapping[query]

        for row in instantiations:
            actions = []
            if row == (1,):
                terms = []  # occurs for nullary (macro) actions
            else:
                terms = [self._sql_obj_to_pddl_obj[o] for o in row]

            for sch_i, (schema_name, parameters) in enumerate(schemata):
                action_terms = []
                for term_i, term in enumerate(parameters):
                    if term.name in self._constant_obj_names:
                        action_terms.append(term)
                    else:
                        action_terms.append(terms[var_mapping[sch_i][term_i]])

                action = ground_action(self._domain, self._name_to_schema[schema_name], tuple(action_terms))
                actions.append(action)
            return actions
        return None

    def query_single(self, query: str) -> list[Action] | None:
        distance = self._query_to_rule[query].precedence

        t = time.time()
        instantiations = self.exec(query).fetchall()
        t = time.time() - t

        actions = self.get_actions_from_queries(query, instantiations)
        self._planning_statistics["n_queries_by_precedence"][distance] += 1

        if actions is not None:
            rule = self._query_to_rule[query]
            if self._verbosity >= 3:
                logging.info(f"Fired rule: {rule}")
            self._planning_statistics["n_sat_queries_by_precedence"][distance] += 1
            self._fired_rules.append(rule)
            self._profiling_statistics["sat_query"] += t
            self._prev_sat_query = query if not rule.precedence.cycle else None
        else:
            self._profiling_statistics["unsat_query"] += t
        self._profiling_statistics["policy_query"][distance] += t

        # if actions is not None and actions[0] == self._prev_action:
        #     actions = None
        # elif actions is not None and actions[0] != self._prev_action:
        #     self._prev_action = actions[0]

        return actions

    def query(self) -> Optional[list[Action]]:
        """Generates actions from the database representing the current state.

        Returns:
            Optional[list[Action]]: List of actions or None if no actions found.
        """

        exec_strategy = self._policy.execution_strategy

        def ret_actions(query, actions):
            if exec_strategy in {ExecutionStrategy.CONSERVATIVE, ExecutionStrategy.GREEDY_LOOP}:
                # if True:
                actions = [actions[0]]
            self._prev_action = actions[0]
            self._plan_to_fired_rules += [query for _ in range(len(actions))]
            return actions

        # Try execute previous query if exists
        if self._prev_sat_query is not None:
            query = self._prev_sat_query
            actions = self.query_single(query)
            if actions is None:
                self._planning_statistics["n_unsat_repeat_queries"] += 1
            else:
                self._planning_statistics["n_sat_repeat_queries"] += 1
                return ret_actions(query, actions)

        self._prev_action = None
        queries = [q for _, query_list in self.get_rule_items() for q in query_list if q != self._prev_sat_query]

        for query in queries:
            actions = self.query_single(query)
            if actions is not None:
                return ret_actions(query, actions)

        # If we get here, then no actions were found.
        self._log_failed_query_action()
        return None

    def add_action_to_plan(self, action: Action) -> None:
        action_str = action_to_string(action, plan_style=True)
        self._plan.append(action_str)

    def add_actions_to_plan(self, actions: list[Action]) -> None:
        for action in actions:
            self.add_action_to_plan(action)

    def apply_action(self, state: State, action: Action) -> State:
        t = time.time()
        new_state = apply_action(state, action)
        self.add_action_to_plan(action)
        self._profiling_statistics["apply_action"] += time.time() - t
        return new_state

    @abstractmethod
    def _solve_impl(self) -> None:
        raise NotImplementedError

    def dump_planning_stats(self) -> None:
        stats = []
        stats.append(("*", "*"))
        stats.append(("plan_length", len(self._plan)))
        stats.append(("planning_time", f"{self._planning_time}s"))
        stats.append(("n_sql_executions", self._planning_statistics["n_sql_executions"]))

        n_sat_repeat_queries = self._planning_statistics["n_sat_repeat_queries"]
        n_unsat_repeat_queries = self._planning_statistics["n_unsat_repeat_queries"]
        n_repeat_queries = n_sat_repeat_queries + n_unsat_repeat_queries
        ratio_sat_repeat_queries = n_sat_repeat_queries / n_repeat_queries if n_repeat_queries > 0 else "nan"
        stats.append(("*", "*"))
        stats.append(("n_repeat_queries", n_repeat_queries))
        stats.append(("n_sat_repeat_queries", n_sat_repeat_queries))
        stats.append(("ratio_sat_repeat_queries", f"{ratio_sat_repeat_queries}%"))

        n_queries = sum(self._planning_statistics["n_queries_by_precedence"].values())
        n_sat_queries = sum(self._planning_statistics["n_sat_queries_by_precedence"].values())
        ratio_sat_queries = n_sat_queries / n_queries
        stats.append(("*", "*"))
        stats.append(("n_queries", n_queries))
        stats.append(("n_sat_queries", n_sat_queries))
        stats.append(("ratio_sat_queries", f"{ratio_sat_queries}%"))
        stats.append(("*", "*"))

        # stats.append(("*" * max(len(v[0]) for v in stats), ""))
        # for d in sorted(set(self._policy.precedences)):
        #     n_queries_d = self._planning_statistics["n_queries_by_precedence"][d]
        #     n_queries_d = 1 if n_queries_d == 0 else n_queries_d
        #     n_sat_queries_d = self._planning_statistics["n_sat_queries_by_precedence"][d]
        #     ratio_sat_queries_d = n_sat_queries_d / n_queries_d
        #     stats.append((d, ratio_sat_queries_d))

        logging.info(f"Planning statistics:\n{mat_to_str(stats)}")

    def dump_profiling_stats(self):
        tot_t = self._planning_time

        t_policy_query_by_d = self._profiling_statistics["policy_query"]
        self._profiling_statistics["policy_query"] = sum(t_policy_query_by_d.values())

        stats = [("*", "*", "*")]
        stats += [
            (k, f"{v:.5f}", f"{100*v/tot_t:.5f}%")
            for k, v in self._profiling_statistics.items()
            if isinstance(v, float)
        ]
        stats.append(("*", "*", "*"))
        stats += [(f"{d}", f"{v:.5f}", f"{100*v/tot_t:.5f}%") for d, v in t_policy_query_by_d.items()]
        stats.append(("*", "*", "*"))

        logging.info(f"SQL profiling statistics:\n{mat_to_str(stats, rjust=[0, 1, 1])}")
