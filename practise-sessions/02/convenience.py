import itertools as it
from pandas import Series
from pandas import DataFrame
from typing import Dict, List, Any
from types import CodeType

from IPython.display import display_html
from IPython.display import display
from IPython.display import Markdown


def head(df: DataFrame, *args, **kwargs) -> DataFrame:
    """
    Convenience function for R users 
    """ 
    
    return df.head( *args, **kwargs)


def tail(df: DataFrame, *args, **kwargs) -> DataFrame:
    """
    Convenience function for R users 
    """    

    return df.tail( *args, **kwargs)


def dim(df: DataFrame) -> DataFrame:
    """
    Convenience function for R users 
    """ 
    
    return df.shape



def combine_categories(cat_dict:Dict[str, List[Any]]) -> DataFrame:
    """
    Creates a data frame that contains all possible combinations of list elements
    """    

    return DataFrame(list(it.product(*cat_dict.values())), columns = cat_dict.keys())



def reset_column_index(df: DataFrame, level: List[Any], drop: bool=True, inplace: bool=False):

    """
    Removes specified levels from the column index. Works analogously to reset_index
    """
    
    if inplace:
        if drop:
            df.columns = df.columns.droplevel(level)
        else:
            raise NotImplementedError
        return df
    else:
        if drop:
            result = df.copy()
            result.columns = df.columns.droplevel(level)
        else:
            result = df.stack(level)
        return result


def mdisplay(dfs: List[DataFrame], names:List[str]=[]):
    """
    Displays several data frames side by side
    
    Adapded form
    https://stackoverflow.com/questions/38783027/jupyter-notebook-display-two-pandas-tables-side-by-side
    """
    
    html_str = ''
    if names:
        html_str += ('<tr>' + 
                     ''.join(f'<td style="text-align:center">{name}</td>' for name in names) + 
                     '</tr>')
    html_str += ('<tr>' + 
                 ''.join(f'<td style="vertical-align:top"> {df.to_html(index=False)}</td>' 
                         for df in dfs) + 
                 '</tr>')
    html_str = f'<table>{html_str}</table>'
    html_str = html_str.replace('table','table style="display:inline"')
    display_html(html_str, raw=True)
    
    
def combine_columns(df: DataFrame, cols:List[str], sep:str="|") -> Series:
    """
    Combines two or more dataframe string columns into a singe column
    """
    assert len(cols) > 1, 'There must be at least two columns to combine'
    return df[cols[0]].str.cat([df[col] for col in cols[1:]], sep=sep)


def display_source(function: CodeType):
    """
    Displays source code of the function with syntactical highlighting.
    """
    display(Markdown(f"```python\n{inspect.getsource(function)}\n```"))