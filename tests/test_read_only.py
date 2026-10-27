"""Test that scripts/ and src/ contain no HTTP method other than GET."""

import ast
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

FORBIDDEN_HTTP_METHODS = {"post", "put", "patch", "delete", "head", "options"}


def find_py_files(*directories: str) -> list[str]:
    """Find all python files under given directory names relative to repo root."""
    py_files = []
    for dir_name in directories:
        target_dir = os.path.join(REPO_ROOT, dir_name)
        if not os.path.exists(target_dir):
            continue
        for root, _, files in os.walk(target_dir):
            for file in files:
                if file.endswith(".py"):
                    py_files.append(os.path.join(root, file))
    return py_files


def test_no_http_methods_other_than_get_in_scripts_and_src():
    """Assert that scripts/ and src/ do not invoke any HTTP method other than GET."""
    files_to_check = find_py_files("scripts", "src")
    assert files_to_check, "No Python files found in scripts/ or src/ to check!"

    violations = []

    for file_path in files_to_check:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source, filename=file_path)

        for node in ast.walk(tree):
            # Check attribute method calls like session.post(), requests.post()
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute):
                    method_name = func.attr.lower()
                    if method_name in FORBIDDEN_HTTP_METHODS:
                        violations.append(
                            f"{os.path.relpath(file_path, REPO_ROOT)}:L{node.lineno} "
                            f"forbidden HTTP method call: .{func.attr}()"
                        )
                elif isinstance(func, ast.Name):
                    if func.id.lower() in FORBIDDEN_HTTP_METHODS:
                        violations.append(
                            f"{os.path.relpath(file_path, REPO_ROOT)}:L{node.lineno} "
                            f"forbidden HTTP function call: {func.id}()"
                        )

            # Check generic requests.request("POST", ...) calls
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "request" and node.args:
                    first_arg = node.args[0]
                    if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                        method_val = first_arg.value.upper()
                        if method_val != "GET":
                            violations.append(
                                f"{os.path.relpath(file_path, REPO_ROOT)}:L{node.lineno} "
                                f"forbidden HTTP request method: '{method_val}'"
                            )

    assert not violations, (
        f"Detected forbidden HTTP methods (only GET is permitted):\n"
        + "\n".join(violations)
    )

