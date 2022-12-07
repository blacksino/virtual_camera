import numpy as np
    
def dist2(x = None,c = None):
    """
    calculate the distance between x and c
    :param x:
    :param c:
    :return:
    """
    x_rows,x_cols = x.shape
    c_rows,c_cols = c.shape
    if x_cols != c_cols:
        raise Exception('Data dimension does not match dimension of centres')
    
    n2 = (np.ones((c_rows,1)) * np.sum((x ** 2).T, 0)).T + np.ones((x_rows,1)) * np.sum((c ** 2).T, 0) - 2.0 * (x @ c.T)


    return n2