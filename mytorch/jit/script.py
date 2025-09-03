import inspect
import ast


class ASTAnalyzer:
    def __init__(self, global_vars):
        self.global_vars = global_vars

    def _dispatch_analyze_stmt(self, stmt: ast.stmt):
        if isinstance(stmt, ast.FunctionDef):
            self._analyze_function_def(stmt)
        elif isinstance(stmt, ast.Assign):
            self._analyze_assign(stmt)

    def _dispatch_analyze_expr(self, root_stmt: ast.expr):
        if isinstance(root_stmt, ast.Name):
            self._analyze_name(root_stmt)
        if isinstance(root_stmt, ast.Call):
            self._analyze_call(root_stmt)

    def _analyze_name(self, root_stmt: ast.Name): ...

    def _analyze_call(self, root_stmt: ast.Call):
        for arg in root_stmt.args:
            self._dispatch_analyze_expr(arg)

    def _analyze_assign(self, root_stmt: ast.Assign):
        targets = root_stmt.targets
        for target in targets:
            self._dispatch_analyze_expr(target)
        self._dispatch_analyze_expr(root_stmt.value)

    def _analyze_function_def(self, root_stmt: ast.FunctionDef):
        name = root_stmt.name
        for stmt in root_stmt.body:
            self._dispatch_analyze_stmt(stmt)

    def _analyze_func(self, func):
        tree = ast.parse(inspect.getsource(func))
        for stmt in tree.body:
            self._dispatch_analyze_stmt(stmt)


def script(func):
    def wrapper(*args, **kwargs):
        ASTAnalyzer(func.__globals__)._analyze_func(func)

    return wrapper
