import numpy as np
import pandas as pd
import ipyvolume as ipv
import scipy.stats as stats
from plotnine import geom_path, aes

from pandas import Series
from pandas import DataFrame
import numpy.random as random

from typing import List, Tuple



def geom_ellipse(mean: np.array, cov: np.array, data: np.array=None, q: float=0.95, **kwargs):
    """
    Draws confidence envelope for a multivariate normal distribution
    
    The ellipse can be specified by mean and covariance. Use stats_ellipse(level=q) to draw
    the empirical confidence envelopes for the data.
    """

    # Radius that covers q-fraction of white gaussian noise  
    r = np.sqrt(stats.chi2.ppf(q=q, df=2))
    
    # Eigen-directions of a covariance matrix
    try:
        L, W = np.linalg.eigh(cov)
    except:
        return geom_path(aes(x = 'x', y = 'y'), data = DataFrame(columns=['x', 'y']))
    
    # Properly scaled eigen-directions
    W[0, :] = W[0, :] * r * np.sqrt(L[0]) 
    W[1, :] = W[1, :] * r * np.sqrt(L[1]) 
   
    theta = np.linspace(0, 2 * np.pi, 100)
   
    return geom_path(aes(x = 'x', y = 'y'), data = DataFrame()
                     .assign(x = mean[0] + np.sin(theta) * W[0, 0] + np.cos(theta) * W[1, 0])
                     .assign(y = mean[1] + np.sin(theta) * W[0, 1] + np.cos(theta) * W[1, 1]), **kwargs)


def convert_to_grayscale(image: np.array) -> np.array:
    """Converts RGB image to grayscale image preserving perceived luminocity based on Y_601 formula"""

    assert len(image.shape) == 3 and image.shape[2] == 3, "Image must be m x n x 3 dimensional RGB array"
    return 0.2989 * image[:,:,0] + 0.5870 * image[:,:,1] + 0.1140 * image[:,:,2]

def luma(cspace: np.array) -> np.array:
    """Computes luminocity for the colorspace vector according to Y_601 formula"""
    
    assert isinstance(cspace, DataFrame), "Colorspace must be a dataframe"
    assert all(np.isin(['R', 'G', 'B'], cspace.columns)), "Colorspace must contain RGB columns"
    return 0.2989 * cspace['R'] + 0.5870 * cspace['G'] + 0.1140 * cspace['B']


def image_to_colorspace(image: np.array, sample_count: int = None) -> DataFrame:
    """
    Converts RGB image to a dataframe where each row corresponds to single pixel.

    If sample_count is set then only random subset of rows is returned. 
    """
    
    assert len(image.shape) == 3 and image.shape[2] == 3, "Image must be m x n x 3 dimensional RGB array"
    if sample_count:
        return DataFrame({'R':image[:,:,0].flatten(),'G':image[:,:, 1].flatten(),'B':image[:,:, 2].flatten()}).sample(n = sample_count, replace=False)
    else: 
        return DataFrame({'R':image[:,:,0].flatten(),'G':image[:,:, 1].flatten(),'B':image[:,:, 2].flatten()})

def colorspace_to_image(cspace: DataFrame, m: int, n: int) -> np.array:
    """Converts colorspace vector back to RGB image. Colorspace can have extra columns"""
    
    assert isinstance(cspace, DataFrame), "Colorspace must be a dataframe"
    assert len(cspace) == m * n, 'Image dimensions must match'
    assert all(np.isin(['R', 'G', 'B'], cspace.columns)), "Colorspace must contain RGB columns"
    
    result = np.empty([m,n,3])
    result[:,:, 0] = cspace['R'].values.reshape(m, n)
    result[:,:, 1] = cspace['G'].values.reshape(m, n)
    result[:,:, 2] = cspace['B'].values.reshape(m, n)
    return result

def show_colorspace(cspace: np.array, clip=True, size = 0.5, marker='sphere', **kwargs) -> None:
    """
    Visualise colorspace vector as an interactive 3D figure. Colorspace can have extra columns
     
    By default RGB channels are clipped to the range [0,1]. 
    Extra arguments can be used to control the appearance of ipyvolume.scatter
    """

    assert isinstance(cspace, DataFrame), "Colorspace must be a dataframe"
    assert all(np.isin(['R', 'G', 'B'], cspace.columns)), "Colorspace must contain RGB columns"

    fig = ipv.figure()
    if clip:
        ipv.scatter(cspace.loc[:, 'R'].values, cspace.loc[:, 'G'].values, cspace.loc[:, 'B'].values, 
                    color=np.clip(cspace[['R', 'G', 'B']].values, 0, 1), s=size, marker=marker, *kwargs)
    else:
        ipv.scatter(cspace.loc[:, 'R'].values, cspace.loc[:, 'G'].values, cspace.loc[:, 'B'].values, 
                    color=cspace[['R', 'G', 'B']].values, s=size, marker=marker, *kwargs)
    ipv.show()
    
    
def luma_decomposition(cspace: DataFrame) -> DataFrame:    
    """Decomposes RGB representation to a naive luma and chroma representation depicted by the formula RGB = luma * rRGB. """
    
    assert isinstance(cspace, DataFrame), "Colorspace must be a dataframe"
    assert all(np.isin(['R', 'G', 'B'], cspace.columns)), "Colorspace must contain RGB columns"

    return (cspace
            .assign(luma = lambda df: luma(df))
            .assign(rR = lambda df: df['R']/df['luma'])
            .assign(rG = lambda df: df['G']/df['luma'])
            .assign(rB = lambda df: df['B']/df['luma']))

def rgb_reconstruction(lumaspace: DataFrame) -> DataFrame:
    """Reconstructs RGB representation from the naive luma and chroma representation by the formula RGB = luma * rRGB. """
    
    assert isinstance(lumaspace, DataFrame), "Colorspace must be a dataframe"
    assert all(np.isin(['luma', 'rR', 'rG', 'rB'], lumaspace.columns)), "Lumaspace must contain RGB columns"    
    return (lumaspace
            .assign(R = lambda df: df['luma'] * df['rR'])
            .assign(G = lambda df: df['luma'] * df['rG'])
            .assign(B = lambda df: df['luma'] * df['rB']))
    