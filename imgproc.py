# coding: utf-8
import numpy as np
import math


def geodesic_dilation(img, mask_off, size=(3, 3)):
    '''
    Geodesic dilation of image.

    :param img: numpy.ndarray
        NxMx1 marker image
    :param mask_off: int, float
        offset between mask and marker image
    :param size: tuple of ints
        size of the structuring element
    '''

    mask = img + mask_off
    max_val = img.max()
    prev_img = None
    while True:
        img = nd.grey_dilation(img, size)
        img = np.minimum(img, mask)
        # no changes
        if np.all(img==prev_img):
            break
        prev_img = img
    # subtraction
    img = img - (max_val-mask_off)
    img[img<0] = 0
    return img

def hough(coords, rho_resolution=1, theta=None):
    '''
    Hough transform.

        rho = x*cos(theta) + y*sin(theta)

    :param coords: iterable
        yielding (x, y) coordinate tuples
    :param rho_resolution: int, float
        spacing of the Hough transform bins along the rho axis, default is 1
    :param theta: numpy.ndarray
        specify a vector of Hough transform theta values as radians,
        default is np.arange(-90, 90, 1) as radians

    :returns: hough space matrix [rhos:thetas], rho values, theta values
    '''

    if theta is None:
        theta = np.arange(-90, 90, 1)
        theta = theta / 360. * 2 * math.pi
    rho_resolution = float(rho_resolution)
    rho_max = math.ceil(math.sqrt(np.max(coords[:,0])**2+np.max(coords[:,1])**2))
    rho_num = rho_max / rho_resolution
    # hough space matrix as rho:theta
    hough_space = np.zeros((math.ceil(rho_num)+1, len(theta)))
    # index for accessing hough space matrix
    theta_idx = np.arange(len(theta))
    rho_idx = np.zeros(theta.shape, 'int32')
    theta_cos = np.cos(theta)
    theta_sin = np.sin(theta)
    for x, y in coords:
        rho = np.abs(x*theta_cos + y*theta_sin)
        np.round(rho / rho_resolution, out=rho_idx)
        hough_space[rho_idx,theta_idx] += 1
    return hough_space, np.arange(0, rho_max, rho_resolution), theta

def hough_peaks(hough_space, rho, theta, num=1):
    '''
    Identify peaks in Hough transform.

    :param hough_space: numpy.ndarray
        hough space matrix returned by `hough_lines`
    :param rho: numpy.ndarray
        rho values returned by `hough_lines`
    :param theta: numpy.ndarray
        theta values returned by `hough_lines`
    :param num: int
        maximum number of peaks to identify, default is 1

    :returns: rho values, theta values of peaks, each with length `num`
    '''

    hsorted = np.argsort(hough_space.ravel())
    rho_idx, theta_idx = np.unravel_index(hsorted[-num:], hough_space.shape)
    return rho[rho_idx], theta[theta_idx]

def surface_normals(img, size=2):
    '''
    Determines the normal unit vector in a image.

    Uses the local neighbourhood of each pixel to determine the adjusted
    hesse normal form of the surface:
        ax + by + cz + [d] = 0

    :param img: numpy.ndarray
        NxMx1 image
    :param size: int
        determines local neighbourhood (2*size + 1)

    :returns: numpy.ndarray
        NxMx3 image, each channel representing one component of the normal
        vector [a, b, c] with length 1
    '''

    # coordinates within mask
    xy =  np.mgrid[-size:size+1,-size:size+1]
    # number of equations for each mask
    eq_num = (2*size + 1)**2

    # over determined surface equation: A*[a, b, c] = b
    A = np.vstack([xy[0].flatten(), xy[1].flatten(), np.zeros((eq_num, ))]).T
    b = np.ones((eq_num, ))

    out = np.zeros((img.shape[0], img.shape[1], 3))
    for r in xrange(size, img.shape[0]-size):
        for c in xrange(size, img.shape[1]-size):
            z = img[r-size:r+size+1,c-size:c+size+1].flatten()
            A[:,2] = z
            out[r,c] = np.linalg.lstsq(A, b)[0]
    # norm all vectors
    norms = np.apply_along_axis(np.linalg.norm, 2, out)
    for i in xrange(3):
        out[:,:,i] = out[:,:,i] / norms

    return out
