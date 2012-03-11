# coding: utf-8
import numpy as np


def make_affine(src, dst, explicit=False):
    '''
    Determine parameters of 2D affine transformation in the order:
        a0, a1, a2, b0, b1, b2
    where the transformation is:
        X = a0+a1*x+a2*y
        Y = b0+b1*x+b2*y
    It is capable of determining the over-, well- and under-determined result
    with the least-squares method.
    Source and destination coordinates must be Nx2 matrices (x, y).

    If explicit is True explicit parameters are returned in the order:
        a0, a1, mx, my, alpha, beta
    where the transformation is:
        X = mx*x*cos(alpha)-my*y*sin(alpha+beta)
        Y = mx*x*sin(alpha)+my*y*cos(alpha+beta)
    '''

    rows = src.shape[0]
    A = np.zeros((rows*2, 6))
    A[:rows,0] = 1
    A[:rows,1] = src[:,0]
    A[:rows,2] = src[:,1]
    A[rows:,3] = 1
    A[rows:,4] = src[:,0]
    A[rows:,5] = src[:,1]
    b = np.zeros((rows*2,))
    b[:rows] = dst[:,0]
    b[rows:] = dst[:,1]
    params = np.linalg.lstsq(A, b)[0]
    if explicit:
        a0, a1 = params[:0], params[:3]
        alpha = math.atan2(params[4], params[1])
        beta = math.atan2(params[5], -params[2]) - alpha
        mx = params[1] / math.cos(alpha)
        mx = params[5] / math.cos(alpha+beta)
        return a0, a1, mx, my, alpha, beta
    else:
        return params

def affine_transform(coords, params, inverse=False):
    '''
    Applies projective transformation to a Nx2 coordinate matrix (x, y) with
    parameters returned by ``make_affine`` with explicit=False.
    '''

    a0, a1, a2, b0, b1, b2 = params
    x = coords[:,0]
    y = coords[:,1]
    out = np.zeros(coords.shape)
    if inverse:
        out[:,0] = (a2*(y-b0)-b2*(x-a0)) / (a2*b1-a1*b2)
        out[:,1] = (b1*(x-a0)-a1*(y-b0)) / (a2*b1-a1*b2)
    else:
        out[:,0] = a0+a1*x+a2*y
        out[:,1] = b0+b1*x+b2*y
    return out

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
