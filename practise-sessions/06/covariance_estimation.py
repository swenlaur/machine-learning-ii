import numpy as np

def near_psd_matrix(x:np.array, epsilon:float=0):
    '''
    Calculates the nearest postive semi-definite matrix for a correlation/covariance matrix

    Parameters
    ----------
    x : array_like
      Covariance/correlation matrix
    epsilon : float
      Eigenvalue limit (usually set to zero to ensure positive definiteness)

    Returns
    -------
    near_cov : array_like
      closest positive definite covariance/correlation matrix

    Notes
    -----
    Document source
    http://www.quarchome.org/correlationmatrix.pdf
    
    Source is directly copied form 
    https://stackoverflow.com/questions/10939213/how-can-i-calculate-the-nearest-positive-semi-definite-matrix
    '''

    if min(np.linalg.eigvals(x)) > epsilon:
        return x

    # Removing scaling factor of covariance matrix
    n = x.shape[0]
    var_list = np.array([np.sqrt(x[i,i]) for i in range(n)])
    y = np.array([[x[i, j]/(var_list[i]*var_list[j]) for i in range(n)] for j in range(n)])

    # getting the nearest correlation matrix
    eigval, eigvec = np.linalg.eig(y)
    val = np.matrix(np.maximum(eigval, epsilon))
    vec = np.matrix(eigvec)
    T = 1/(np.multiply(vec, vec) * val.T)
    T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)) )))
    B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
    near_corr = B*B.T    

    # returning the scaling factors
    near_cov = np.array([[near_corr[i, j]*(var_list[i]*var_list[j]) for i in range(n)] for j in range(n)])
    return near_cov


def pairwise_cov_matrix(X:np.array, psd_correction:bool=False, ridge_coeff:float=0)-> np.array:
    """
    Computes covariance matrix in the presence of missing values using pairwise covariance estimates.
    
    Arguments:
    X -- the data matrix where observations are in the rows
    psd_correction -- if set makes the covariance matrix positively semi definite
    ridge_coeff -- coefficient to be added to the main diagonal (before psd correction)
    
    Return value:
    Covariance matrix that may contain nan values if the number of matching observations is too low
    for particular column pair. If psd_correction is set return matrix that is positively semidefinite
    by finding the nearest PSD matrix for the original covariance matrix.  
    
    This implementation is the closest match to GNU R function cov(X, use = "pairwise"). 
    Note that the pandas implementation of covariance matrix is different. 
    """
    
    cov_matrix = np.ma.cov(np.ma.array(X, mask=np.isnan(X)), rowvar=False, allow_masked=True)

    if np.ma.is_masked(cov_matrix):
        raise Exception("Too many missing values")
    
    cov_matrix = np.ma.filled(cov_matrix.astype(float), np.nan)
    cov_matrix = 0.5 * (cov_matrix + cov_matrix.T) 
    cov_matrix[np.diag_indices_from(cov_matrix)] += ridge_coeff

    if np.linalg.matrix_rank(cov_matrix) != cov_matrix.shape[0]:
        raise Exception("Covariance matrix is linearly dependent. Increase ridge regularisation parameter")
    
    return near_psd_matrix(cov_matrix) 