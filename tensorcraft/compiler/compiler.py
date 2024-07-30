"""Compiler class of tensorcraft."""

import importlib.resources
import logging

import lark

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

        self._parser = lark.Lark(self._grammar, parser="lalr")
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
