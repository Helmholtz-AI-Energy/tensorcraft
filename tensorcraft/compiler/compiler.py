"""Compiler class of tensorcraft."""

import importlib.resources
import logging
from typing import Any

import lark
import lark.ast_utils
import networkx as nx

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


class DataGraphTransformer(lark.Transformer):
    """
    Transformer class for converting tensor expressions into a directed graph.

    Parameters
    ----------
    code : str
        The code containing the tensor expressions.

    Attributes
    ----------
    _code : list
        The code split into lines.
    _current_variables : list
        The current variables being processed.
    _derived_variables : set
        The set of derived variables.
    _all_variables : set
        The set of all variables.

    """

    def __init__(self, code: str):
        super().__init__()
        self._code = code.split("\n")
        self._current_variables: list[Any] = []
        self._derived_variables: set[Any] = set()
        self._all_variables: set[Any] = set()

    def start(self, tensor_expressions: list[lark.Tree]):
        """
        Convert the tensor expressions into a directed graph.

        Parameters
        ----------
        tensor_expressions : list
            The list of tensor expressions.

        Returns
        -------
        G : networkx.DiGraph
            The directed graph representing the tensor expressions.

        """
        G = nx.DiGraph()

        current_variable: dict[str, str] = {}

        input_variables = self._all_variables - self._derived_variables
        for var in input_variables:
            current_variable[var] = var
            G.add_node(var)

        for tensor_expression in tensor_expressions:
            exp = tensor_expression[0]
            output = tensor_expression[1]
            deps = tensor_expression[2]

            G.add_node(exp)
            for dep in deps:
                G.add_edge(current_variable[dep], exp)

            current_variable[output] = exp
            print(current_variable)

        return G

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
        deps : tuple
            The dependencies of the tensor expression.

        """
        deps = (
            self._code[int(meta.line) - 1],
            self._current_variables[0],
            set(self._current_variables[1:]),
        )

        self._derived_variables.add(self._current_variables[0])
        self._all_variables.update(self._current_variables)
        self._current_variables = []
        return deps

    def tensor(self, children: list[lark.Tree]):
        """
        Process a tensor and adds it to the current variables.

        Parameters
        ----------
        children : list
            The children of the tensor.

        """
        self._current_variables.append(children[0])
