# coding: utf-8
import numpy as np


def make_projective(src, dst):
    '''
    Determine parameters of 2D projective transformation in the order:
        a0, a1, a2, b0, b1, b2, c0, c1
    where the transformation is:
        X = (a0+a1*x+a2*y) / (1+c0*x+c1*y)
        Y = (b0+b1*x+b2*y) / (1+c0*x+c1*y)
    It is capable of determining the over-, well- and under-determined result
    with the least-squares method.
    Source and destination coordinates must be Nx2 matrices (x, y).
    '''

    rows = src.shape[0]
    A = np.zeros((rows*2, 8))
    A[:rows,0] = 1
    A[:rows,1] = src[:,0]
    A[:rows,2] = src[:,1]
    A[rows:,3] = 1
    A[rows:,4] = src[:,0]
    A[rows:,5] = src[:,1]
    A[:rows,6] = - dst[:,0] * src[:,0]
    A[:rows,7] = - dst[:,0] * src[:,1]
    A[rows:,6] = - dst[:,1] * src[:,0]
    A[rows:,7] = - dst[:,1] * src[:,1]
    b = np.zeros((rows*2,))
    b[:rows] = dst[:,0]
    b[rows:] = dst[:,1]
    return np.linalg.lstsq(A, b)[0]

def projective_transform(coords, params, inverse=False):
    '''
    Applies projective transformation to a Nx2 coordinate matrix (x, y) with
    parameters returned by ``make_projective``.
    '''

    a0, a1, a2, b0, b1, b2, c0, c1 = params
    x = coords[:,0]
    y = coords[:,1]
    out = np.zeros(coords.shape)
    if inverse:
        out[:,0] = (a2*b0-a0*b2+(b2-b0*c1)*x+(a0*c1-a2)*y) \
            / (a1*b2-a2*b1+(b1*c1-b2*c0)*x+(a2*c0-a1*c1)*y)
        out[:,1] = (a0*b1-a1*b0+(b0*c0-b1)*x+(a1-a0*c0)*y) \
            / (a1*b2-a2*b1+(b1*c1-b2*c0)*x+(a2*c0-a1*c1)*y)
    else:
        out[:,0] = (a0+a1*x+a2*y) / (1+c0*x+c1*y)
        out[:,1] = (b0+b1*x+b2*y) / (1+c0*x+c1*y)
    return out
