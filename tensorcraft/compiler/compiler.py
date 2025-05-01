"""Compiler class of tensorcraft."""

import importlib.resources
import logging
from typing import cast

import lark
import lark.ast_utils
import networkx as nx

from tensorcraft.compiler.model import (
    AssignmentType,
    Program,
    TensorExpression,
    TensorVariable,
)

log = logging.getLogger(__name__)


class Compiler:
    """A class for compiling code using a tensor notation."""

    def __init__(self) -> None:
        """Initialize the Compiler class."""
        try:
            self._grammar = (
                importlib.resources.files("tensorcraft.compiler")
                .joinpath("grammar.lark")
                .read_text()
            )
        except FileNotFoundError:
            log.error("Grammar file not found.")

        log.info("Grammar file loaded successfully.")
        log.debug(f"Grammar file content:\n{self._grammar}")

        self._parser = lark.Lark(self._grammar, parser="lalr", propagate_positions=True)
        log.info("Parser object created successfully.")

    def compile(self, code: str) -> Program:
        """Compile the given code and return the abstract syntax tree (AST).

        Args:
            code (str): The code to be compiled.

        Returns
        -------
            lark.Tree: The abstract syntax tree (AST) of the compiled code.
        """
        log.info(f"Compiling code:\n{code}")

        try:
            ast = self._parser.parse(code)
        except Exception as e:
            log.error(f"Invalid program: {e}")
            raise ValueError(f"Invalid program: {e}")
        program = cast(Program, ProgramTransformer(code).transform(ast))
        return program


class ProgramTransformer(lark.Transformer):
    """
    Transformer class for converting tensor expressions into a directed graph.

    It processes the tree from left to right, bottom to top.

    Parameters
    ----------
    code : str
        The code containing the tensor expressions.

    Attributes
    ----------
    _code : list
        The code split into lines.
    _current_tensor_index_vars : list
        The current tensor index variables.
    _current_exp_variables : list
        The current tensor expression variables.
    _current_exp_index_vars : list
        The current tensor expression index variables.
    _variables : dict
        The variables in the code.
    _derived_variables : set
        The derived variables in the code.
    """

    def __init__(self, code: str):
        super().__init__()
        self._code = code.split("\n")

        self._op_graph = nx.DiGraph()
        self._op_count_dict: dict[str, int] = {}
        self._op_count = 0

        self._current_tensor_multi_index: list[str] = []

        self._current_exp_variables: list[str] = []
        self._current_exp_multi_idx_exp: list[list[str]] = []
        self._current_exp_index_vars: set[str] = set()

        self._variables: dict[str, TensorVariable] = {}

        self._derived_variables: set[str] = set()

    def start(self, tensor_ops: list[TensorExpression]) -> Program:
        """
        Convert the tensor expressions into a directed graph.

        Parameters
        ----------
        tensor_expressions : list[TensorExpression]
            The list of tensor expressions.

        Returns
        -------
        Programm
            The program containing the tensor expressions and variable information.

        """
        G = nx.DiGraph()

        current_variable: dict[str, str | int] = {}

        input_variables = set(self._variables.keys()) - self._derived_variables

        for var in input_variables:
            self._variables[var].lines.insert(0, -1)
            current_variable[var] = var
            G.add_node(var)

        max_loop_depth = 0
        max_loop_depth_line = 0

        max_op_count = 0
        max_op_count_line = 0

        for op in tensor_ops:
            if max_loop_depth < op.loop_count:
                max_loop_depth_line = op.line
                max_loop_depth = op.loop_count

            if max_op_count < op.op_count:
                max_op_count_line = op.line
                max_op_count = op.op_count

            G.add_node(op.line, op=op)
            for input, _ in op.inputs:
                G.add_edge(current_variable[input], op.line)

            current_variable[op.output[0]] = op.line

        ops_dict = {op.line: op for op in tensor_ops}

        program = Program(
            G,
            list(input_variables),
            self._variables,
            ops_dict,
            (max_loop_depth, max_loop_depth_line),
            (max_op_count, max_op_count_line),
        )

        return program

    @lark.v_args(meta=True)
    def assign(self, meta: lark.tree.Meta, expr: list[lark.Tree]) -> TensorExpression:
        """Process assignment tensor expression."""
        return self._handle_tensor_exp(AssignmentType.ASSIGN, meta, expr)

    @lark.v_args(meta=True)
    def add_assign(
        self, meta: lark.tree.Meta, expr: list[lark.Tree]
    ) -> TensorExpression:
        """Process add and assign tensor expression."""
        return self._handle_tensor_exp(AssignmentType.ADD, meta, expr)

    @lark.v_args(meta=True)
    def sub_assign(
        self, meta: lark.tree.Meta, expr: list[lark.Tree]
    ) -> TensorExpression:
        """Process substract and assign tensor expression."""
        return self._handle_tensor_exp(AssignmentType.SUB, meta, expr)

    @lark.v_args(meta=True)
    def mul_assign(
        self, meta: lark.tree.Meta, expr: list[lark.Tree]
    ) -> TensorExpression:
        """Process multiply and assign tensor expression."""
        return self._handle_tensor_exp(AssignmentType.MUL, meta, expr)

    @lark.v_args(meta=True)
    def div_assign(
        self, meta: lark.tree.Meta, expr: list[lark.Tree]
    ) -> TensorExpression:
        """Process divide and assign tensor expression."""
        return self._handle_tensor_exp(AssignmentType.DIV, meta, expr)

    def _handle_tensor_exp(
        self, assign_op: AssignmentType, meta: lark.tree.Meta, expr: list[lark.Tree]
    ) -> TensorExpression:
        """
        Process a tensor expression and returns its dependencies.

        Parameters
        ----------
        meta : lark.Token
            The metadata of the tensor expression.
        expr : str
            The tensor expression.

        Returns
        -------
        TensorExpression
            The dependencies of the tensor expression.
        """
        self._derived_variables.add(self._current_exp_variables[0])

        for var in set(self._current_exp_variables):
            self._variables[var].lines.append(meta.line)

        op = TensorExpression(
            meta.line,
            self._code[int(meta.line) - 1],
            [
                (var, idx_tuple)
                for var, idx_tuple in zip(
                    self._current_exp_variables[1:], self._current_exp_multi_idx_exp[1:]
                )
            ],
            (self._current_exp_variables[0], self._current_exp_multi_idx_exp[0]),
            assign_op,
            loop_count=len(self._current_exp_index_vars),
            op_graph=self._op_graph,
            op_count=self._op_count,
        )

        self._op_count = 0
        self._current_exp_variables = []
        self._current_exp_multi_idx_exp = []
        self._current_exp_index_vars = set()
        self._op_graph = nx.DiGraph()
        self._op_count_dict = {}
        return op

    def add(self, children: list[lark.Tree]) -> str:
        """Process an addition operation."""
        return self._op_handler(children, "+")

    def sub(self, children: list[lark.Tree]) -> str:
        """Process a subtraction operation."""
        return self._op_handler(children, "-")

    def prod(self, children: list[lark.Tree]) -> str:
        """Process a multiplication operation."""
        return self._op_handler(children, "*")

    def div(self, children: list[lark.Tree]) -> str:
        """Process a division operation."""
        return self._op_handler(children, "/")

    def equal(self, children: list[lark.Tree]) -> str:
        """Process an equality operation."""
        return self._op_handler(children, "==")

    def nequal(self, children: list[lark.Tree]) -> str:
        """Process a non-equality operation."""
        return self._op_handler(children, "!=")

    def bool_and(self, children: list[lark.Tree]) -> str:
        """Process a boolean AND operation."""
        return self._op_handler(children, "&&")

    def bool_or(self, children: list[lark.Tree]) -> str:
        """Process a boolean OR operation."""
        return self._op_handler(children, "||")

    def gt(self, children: list[lark.Tree]) -> str:
        """Process a greater than operation."""
        return self._op_handler(children, ">")

    def lt(self, children: list[lark.Tree]) -> str:
        """Process a less than operation."""
        return self._op_handler(children, "<")

    def ge(self, children: list[lark.Tree]) -> str:
        """Process a greater than or equal operation."""
        return self._op_handler(children, ">=")

    def le(self, children: list[lark.Tree]) -> str:
        """Process a less than or equal operation."""
        return self._op_handler(children, "<=")

    def pow(self, children: list[lark.Tree]) -> str:
        """Process a power operation."""
        return self._op_handler(children, "^")

    def parenthesis(self, children: list[lark.Tree]) -> lark.Tree:
        """Process a parenthesis operation."""
        return children[0]

    def ceil(self, children: list[lark.Tree]) -> str:
        """Process a ceil operation."""
        return self._op_handler(children, "ceil")

    def floor(self, children: list[lark.Tree]) -> str:
        """Process a floor operation."""
        return self._op_handler(children, "floor")

    def sin(self, children: list[lark.Tree]) -> str:
        """Process a sin operation."""
        return self._op_handler(children, "sin")

    def cos(self, children: list[lark.Tree]) -> str:
        """Process a cos operation."""
        return self._op_handler(children, "cos")

    def tan(self, children: list[lark.Tree]) -> str:
        """Process a tan operation."""
        return self._op_handler(children, "tan")

    def asin(self, children: list[lark.Tree]) -> str:
        """Process a asin operation."""
        return self._op_handler(children, "asin")

    def acos(self, children: list[lark.Tree]) -> str:
        """Process a acos operation."""
        return self._op_handler(children, "acos")

    def atan(self, children: list[lark.Tree]) -> str:
        """Process a atan operation."""
        return self._op_handler(children, "atan")

    def sinh(self, children: list[lark.Tree]) -> str:
        """Process a sinh operation."""
        return self._op_handler(children, "sinh")

    def cosh(self, children: list[lark.Tree]) -> str:
        """Process a cosh operation."""
        return self._op_handler(children, "cosh")

    def tanh(self, children: list[lark.Tree]) -> str:
        """Process a tanh operation."""
        return self._op_handler(children, "tanh")

    def asinh(self, children: list[lark.Tree]) -> str:
        """Process a asinh operation."""
        return self._op_handler(children, "asinh")

    def acosh(self, children: list[lark.Tree]) -> str:
        """Process a acosh operation."""
        return self._op_handler(children, "acosh")

    def atanh(self, children: list[lark.Tree]) -> str:
        """Process a atanh operation."""
        return self._op_handler(children, "atanh")

    def exp(self, children: list[lark.Tree]) -> str:
        """Process a exp operation."""
        return self._op_handler(children, "exp")

    def log(self, children: list[lark.Tree]) -> str:
        """Process a log operation."""
        return self._op_handler(children, "log")

    def log2(self, children: list[lark.Tree]) -> str:
        """Process a log2 operation."""
        return self._op_handler(children, "log2")

    def log10(self, children: list[lark.Tree]) -> str:
        """Process a log10 operation."""
        return self._op_handler(children, "log10")

    def sqrt(self, children: list[lark.Tree]) -> str:
        """Process a sqrt operation."""
        return self._op_handler(children, "sqrt")

    def abs(self, children: list[lark.Tree]) -> str:
        """Process a abs operation."""
        return self._op_handler(children, "abs")

    def _op_handler(self, children: list[lark.Tree], op: str) -> str:
        if isinstance(children[0], lark.Tree):
            op_0 = children[0].children[0]
        else:
            op_0 = children[0]

        if op not in self._op_count_dict:
            self._op_count_dict[op] = 0
        else:
            self._op_count_dict[op] += 1

        node_name = f"{op} {self._op_count_dict[op]}"

        self._op_graph.add_node(node_name)

        self._op_graph.add_edge(op_0, node_name)
        if len(children) > 1:
            self._op_graph.add_edge(children[1], node_name)
        self._op_count += 1
        return node_name

    def function(self, children: list[lark.Tree]) -> lark.Tree:
        """Process a function operation."""
        return children[0]

    def operand(self, children: list[lark.Token]) -> str:
        """
        Process an operand and adds it to the current tensor expression variables.

        Parameters
        ----------
        children : list
            The children of the operand.

        Return
        ------
        str
            Operand name.

        """
        if isinstance(children[0], lark.Tree):
            if children[0].data == "pos_tensor":
                var_name = children[0].children[0][0]

                index_vars = ",".join(children[0].children[0][1])
                operand_name = f"{var_name}[{index_vars}]"

                neg = False
            elif children[0].data == "neg_tensor":
                var_name = children[0].children[0][0]

                index_vars = ",".join(children[0].children[0][1])
                operand_name = f"{var_name}[{index_vars}]"
                neg = True
            elif children[0].data == "pos_indexvar":
                operand_name = children[0].children[0]
                neg = False
            elif children[0].data == "neg_indexvar":
                operand_name = f"{children[0].children[0]}"
                neg = True
            self._op_graph.add_node(operand_name, neg=neg)
        elif isinstance(children[0], lark.Token):
            operand_name = children[0].value
            self._op_graph.add_node(operand_name)

        return cast(str, operand_name)

    def tensor(self, children: list[lark.Tree]) -> tuple[str, list[str]]:
        """
        Process a tensor and adds it to the current variables.

        Parameters
        ----------
        children : list
            The children of the tensor.

        Return
        ------
        str
            Node string id.
        list[str]
            List with related index variables to the tensor.

        """
        name = children[0].value  # type: ignore[attr-defined]
        order = len(self._current_tensor_multi_index)

        if name in self._variables:
            prev_order = self._variables[name].order
            if prev_order != order:
                raise ValueError(
                    f"Variable {name} has different order. Previous {prev_order} != {order}."
                )
        else:
            self._variables[name] = TensorVariable(name, order, [])

        self._current_exp_multi_idx_exp.append(self._current_tensor_multi_index)
        self._current_tensor_multi_index = []

        self._current_exp_variables.append(name)

        return name, self._current_exp_multi_idx_exp[-1]

    def var_idx(self, children: list[lark.Token]) -> str:
        """
        Process an index variable and adds it to the current tensor index variables.

        Parameters
        ----------
        children : list
            The children of the index variable.

        Return
        ------
        str
            Token string id.

        """
        result = cast(str, children[0].value)
        self._current_exp_index_vars.update(result)
        self._current_tensor_multi_index.append(result)
        return result

    def scalar_idx(self, children: list[lark.Token]) -> str:
        """
        Process a scalar index and adds it to the current tensor index variables.

        Parameters
        ----------
        children : list
            The children of the scalar index.

        Return
        ------
        str
            Node string id.

        """
        result = cast(str, children[0].value)
        self._current_tensor_multi_index.append(result)
        return result

    def joined_idx(self, children: list[lark.Token]) -> str:
        """
        Process a joined index and adds it to the current tensor index variables.

        Parameters
        ----------
        children : list
            The children of the joined index.

        Return
        ------
        str
            Joined index string.

        """
        self._current_exp_index_vars.update(child.value for child in children)
        result = "".join(child.value for child in children)
        self._current_tensor_multi_index.append(result)
        return result
