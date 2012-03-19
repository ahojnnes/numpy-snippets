# coding: utf-8
import numpy as np
import math


def hough(coords, rho_resolution=1, theta=None):
    '''
    Hough transform.

        rho = x*cos(theta) + y*sin(theta)

    :param coords: iterable yielding (x, y) coordinate tuples
    :param rho_resolution: spacing of the Hough transform bins along the rho axis
    :param theta: specify a vector of Hough transform theta values as radians

    :returns: hough space matrix [rhos:thetas], rho values, theta values
    '''

    if theta is None:
        theta = np.arange(-90, 90, 1)
        theta = theta / 360. * 2 * math.pi
    rho_resolution = float(rho_resolution)
    rho_min = math.floor(math.sqrt(np.min(coords[:,0])**2 + np.min(coords[:,1])**2))
    rho_max = math.ceil(math.sqrt(np.max(coords[:,0])**2 + np.max(coords[:,1])**2))
    rho_num = (rho_max - rho_min) / rho_resolution
    # hough space matrix as rho:theta
    hough_space = np.zeros((math.ceil(rho_num)+1, len(theta)))
    # index for accessing hough space matrix
    theta_idx = np.arange(len(theta))
    rho_idx = np.zeros(theta.shape, 'int32')
    for x, y in coords:
        rho = np.abs(x*np.cos(theta) + y*np.sin(theta))
        np.round((rho - rho_min) / rho_resolution, out=rho_idx)
        hough_space[rho_idx,theta_idx] += 1
    return hough_space, np.arange(0, rho_max, rho_resolution), theta

def hough_peaks(hough_space, rho, theta, num=1):
    '''
    Identify peaks in Hough transform.

    :param hough_space: hough space matrix returned by `hough_lines`
    :param rho: rho values returned by `hough_lines`
    :param theta: theta values returned by `hough_lines`
    :param num: maximum number of peaks to identify, default is 1

    :returns: rho values, theta values of peaks, each with length `num`
    '''

    hsorted = np.argsort(hough_space.ravel())
    rho_idx, theta_idx = np.unravel_index(hsorted[-num:], hough_space.shape)
    return rho[rho_idx], theta[theta_idx]
