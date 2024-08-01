"""Compiler class of tensorcraft."""

import importlib.resources
import logging

import lark
import lark.ast_utils
import networkx as nx

from tensorcraft.compiler.model import Program, TensorExpression, TensorVariable

log = logging.getLogger("tensorcraft")


class Compiler:
    """A class for compiling code using a tensor notation."""

    def __init__(self):
        """Initialize the Compiler class."""
        try:
            self._grammar = importlib.resources.read_text(
                "tensorcraft.compiler", "grammar.lark"
            )
        except FileNotFoundError:
            log.error("Grammar file not found.")

        log.info("Grammar file loaded successfully.")
        log.debug(f"Grammar file content:\n{self._grammar}")

        self._parser = lark.Lark(self._grammar, parser="lalr", propagate_positions=True)
        log.info("Parser object created successfully.")

    def compile(self, code: str) -> lark.Tree:
        """Compile the given code and return the abstract syntax tree (AST).

        Args:
            code (str): The code to be compiled.

        Returns
        -------
            lark.Tree: The abstract syntax tree (AST) of the compiled code.
        """
        log.info(f"Compiling code:\n{code}")

        ast = self._parser.parse(code)
        return ast


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

        self._current_tensor_index_vars: list[str] = []

        self._current_exp_variables: list[str] = []
        self._current_exp_index_vars: list[list[str]] = []

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

        for op in tensor_ops:
            G.add_node(op.line, op=op)
            for input, _ in op.inputs:
                G.add_edge(current_variable[input], op.line)

            current_variable[op.output[0]] = op.line

        ops_dict = {op.line: op for op in tensor_ops}

        program = Program(G, ops_dict, self._variables, list(input_variables))

        return program

    @lark.v_args(meta=True)
    def tensorexp(self, meta: lark.tree.Meta, expr: list[lark.Tree]):
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

        unique_idx_vars = set()

        for idx_vars in self._current_exp_index_vars:
            unique_idx_vars.update(idx_vars)

        op = TensorExpression(
            meta.line,
            self._code[int(meta.line) - 1],
            [
                (var, idx_tuple)
                for var, idx_tuple in zip(
                    self._current_exp_variables[1:], self._current_exp_index_vars[1:]
                )
            ],
            (self._current_exp_variables[0], self._current_exp_index_vars[0]),
            loop_count=len(unique_idx_vars),
            op_count=len(self._current_exp_variables) - 1,
        )

        self._current_exp_variables = []
        self._current_exp_index_vars = []
        return op

    def tensor(self, children: list[lark.Tree]):
        """
        Process a tensor and adds it to the current variables.

        Parameters
        ----------
        children : list
            The children of the tensor.

        """
        name = children[0].value
        order = len(self._current_tensor_index_vars)

        if name in self._variables:
            prev_order = self._variables[name].order
            if prev_order != order:
                raise ValueError(
                    f"Variable {name} has different order. Previous {prev_order} != {order}."
                )
        else:
            self._variables[name] = TensorVariable(name, order, [])

        self._current_exp_index_vars.append(self._current_tensor_index_vars)
        self._current_tensor_index_vars = []

        self._current_exp_variables.append(name)

    def indexvar(self, children: list[lark.Token]):
        """
        Process an index variable and adds it to the current tensor index variables.

        Parameters
        ----------
        children : list
            The children of the index variable.

        """
        self._current_tensor_index_vars.append(children[0].value)
