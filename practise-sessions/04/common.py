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

def convert_to_grayscale(image: np.array)-> np.array:
    """Converts RGB image to grayscale image preserving perceived luminocity based on Y_601 formula""" 
    return 0.2989 * image[:,:,0] + 0.5870 * image[:,:,1] + 0.1140 * image[:,:,2]


def extract_texture_vectors(bw_image: np.array, x: np.array, y:np.array, m: int, n: int, d: int) -> DataFrame:
    """
    Extracts square (2d + 1) x (2d + 1) texture patches around center points given by vectors x an y
    and flattens them into feature vectors using C-style row-by-row flattening procedure.
    """
    
    index = (d <= x) & (x < m - d) & (d <= y) & (y < n - d)
    xm = x[index]
    ym = y[index]
    
    x0 = xm - d
    x1 = xm + d + 1
    y0 = ym - d
    y1 = ym + d + 1
    
    r = len(xm)
    textures = np.empty([r, (2 * d + 1)**2])
    for i in range(r):
        textures[i, :] = bw_image[x0[i]:x1[i], y0[i]:y1[i]].flatten()
        
    columns = ['x{:02d}'.format(i+1) for i in range((2*d+1)**2)]   
    return (DataFrame(textures, columns = columns)
            .assign(xm = Series(xm).astype(int))
            .assign(ym = Series(ym).astype(int))
            [['xm', 'ym'] + columns])