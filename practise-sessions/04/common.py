import numpy as np
from pandas import Series
from pandas import DataFrame
import numpy.random as random

from typing import List, Tuple


def ngrams(text:str, n:int=2) -> List[str]:
    '''
    Returns a list of ngrams for the text 
    
    Inspired by http://locallyoptimal.com/blog/2013/01/20/elegant-n-gram-generation-in-python/
    '''
    
    return [''.join(x) for x in zip(*[text[i:] for i in range(n)])]