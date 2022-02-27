import numpy as np
from pandas import Series
from pandas import DataFrame
import numpy.random as random

from typing import List, Tuple


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def logit(x:Series, w:Series):
    """
    Labels output according to logit model pr[y=1| sigmoid(w*x)]
    
    You can omit trailing zeroes in w by specifying only first non-zero coefficients  
    """
    return random.rand() <= sigmoid(np.dot(x.iloc[0:len(w)] - 0.5, w))


def data_sampler(n:int, k:int,  f:callable) -> DataFrame:
    """
    Data generator that generates n x k feature matrix and a target vector
    
    Returns a data frame with columns x_1, ..., x_k, y where y is computed
    using ramdomised labelling function f.
    """
    
    columns = ['x_{}'.format(num) for num in range(1, k + 1)]
    return (DataFrame(random.rand(n, k), columns = columns)
            .transform(lambda x: x >= 0.5)
            .assign(y = lambda df: df.apply(f, axis=1)))


def empirical_risk(clf, X: DataFrame, y: Series) -> float:
    """
    Computes empirical risk for standard symmetrical loss function
    """
    return sum(clf.predict(X) != y) / len(X) * 100


class MajorityVoting:
    """
    Classifier that implements majority voting for each input 
    
    Data is divided into subgroups based on feature values.
    The output for each input is computed as a majority vote
    """
    
    def __init__(self, features:List[str]=None):
        if features:
            self.features = list(features)
        else:
            self.features = None
    
    def set_params(features: List[str]) -> None:
        self.features = features
    
    def fit(self, X: DataFrame, y: Series) -> None:
        
        if self.features is None:
            self.features = list(X.columns.values)

        data = X.assign(y = y)
        pred = data.groupby(self.features).aggregate(['count', 'sum'])
        pred.columns = pred.columns.droplevel(0)
        self.pred = DataFrame({'prediction':(pred['sum']/pred['count'] >= 0.5)})
    
    def predict(self, X: DataFrame) -> np.array:
        
        return (X[self.features]
                .join(self.pred, on=self.features, how='left')['prediction']
                .fillna(True)
                .values)    

    
def crossvalidation_splits(X: DataFrame, y: Series, k: int=10) -> Tuple[List[int], List[int]]:
    """
    Generator that computes splits for k-fold crossvalidation. Works only if the dataset size is a multiple of k.
    """
    
    assert len(X) == len(y), 'Data matrix and the target vector must match'
    assert len(X) % k == 0,  'Crossvalidation is unimplemented for cases n != k * m'  
    assert X.index.equals(y.index), 'Indices of the data matrix and target vector must match'

    n = len(X)
    m = int(n/k)
    samples = np.random.permutation(X.index)
    folds = [samples[start: start + m] for start in range(0, n, m)]
    
    for i in range(k):
        training_index = [x for x in range(k) if x != i]
        training_samples = np.concatenate([folds[i] for i in training_index])
        test_samples = folds[i]
        yield (i, training_samples, test_samples)