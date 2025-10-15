"""Calculator tool for safely evaluating math expressions."""

from __future__ import annotations

import ast
import operator
from typing import Callable, Dict, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class _CalculatorToolInput(BaseModel):
	"""Input schema for the calculator tool."""

	expression: str = Field(
		..., description="The math expression to evaluate (supports +, -, *, /, //, %, **, and parentheses). Use . for decimal points."
	)


class CalculatorTool(BaseTool):
	"""Tool that safely evaluates math expressions using Python's AST module."""

	name: str = "Calculator"
	description: str = (
		"Safely evaluate arithmetic expressions with operators +, -, *, /, //, %, **, and parentheses. Use . for decimal points."
	)
	args_schema: Type[BaseModel] = _CalculatorToolInput

	_ALLOWED_BINARY_OPERATORS: Dict[type, Callable[[float, float], float]] = {
		ast.Add: operator.add,
		ast.Sub: operator.sub,
		ast.Mult: operator.mul,
		ast.Div: operator.truediv,
		ast.FloorDiv: operator.floordiv,
		ast.Mod: operator.mod,
		ast.Pow: operator.pow,
	}
	_ALLOWED_UNARY_OPERATORS: Dict[type, Callable[[float], float]] = {
		ast.UAdd: operator.pos,
		ast.USub: operator.neg,
	}

	def _run(self, expression: str) -> str:  # pragma: no cover - 
		"""Evaluate the provided math expression and return the result as a string."""

		try:
			parsed = ast.parse(expression, mode="eval")
			result = self._evaluate(parsed.body)
			return str(result)
		except ZeroDivisionError as exc:  # pragma: no cover - guardrail
			raise ValueError("Division by zero is not allowed.") from exc
		except Exception as exc:  # pragma: no cover - general safety
			raise ValueError("Invalid expression provided.") from exc

	def _evaluate(self, node: ast.AST) -> float:
		"""Recursively evaluate an AST node representing a safe arithmetic expression."""

		if isinstance(node, ast.Constant):
			value = ast.literal_eval(node)
			if isinstance(value, (int, float)) and not isinstance(value, bool):
				return value
			raise ValueError("Only numeric literals are allowed.")

		if isinstance(node, ast.BinOp):
			operator_type = type(node.op)
			operator_fn = self._ALLOWED_BINARY_OPERATORS.get(operator_type)
			if operator_fn is None:
				raise ValueError(f"Operator {operator_type.__name__} is not allowed.")
			left_value = self._evaluate(node.left)
			right_value = self._evaluate(node.right)
			return operator_fn(left_value, right_value)

		if isinstance(node, ast.UnaryOp):
			operator_type = type(node.op)
			operator_fn = self._ALLOWED_UNARY_OPERATORS.get(operator_type)
			if operator_fn is None:
				raise ValueError(f"Unary operator {operator_type.__name__} is not allowed.")
			operand_value = self._evaluate(node.operand)
			return operator_fn(operand_value)

		if isinstance(node, ast.Expr):
			return self._evaluate(node.value)

		raise ValueError("Unsupported expression component encountered.")
