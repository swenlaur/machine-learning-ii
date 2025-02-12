import inspect
from IPython.display import display
from IPython.display import Markdown

from types import CodeType


def display_source(function: CodeType):
    """
    Displays source code of the function with syntactical highlighting.
    """
    display(Markdown(f"```python\n{inspect.getsource(function)}\n```"))
