import inspect
import ast


def script(func):
    def wrapper(*args, **kwargs):
        tree = ast.parse(inspect.getsource(func))
        print(ast.dump(tree, indent=4))

    return wrapper
