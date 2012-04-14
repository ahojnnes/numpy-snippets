# coding: utf-8
import numpy as np
import math


class LinearLeastSquares2D(object):

    '''
    2D linear least squares using the hesse normal form:
        d = x*sin(theta) + y*cos(theta)
    which allows you to have vertical lines.
    '''

    def fit(self, data):
        data_mean = data.mean(axis=0)
        x0, y0 = data_mean
        if data.shape[0] > 2: # over determined
            u, v, w = np.linalg.svd(data-data_mean)
            vec = w[0]
            theta = math.atan2(vec[0], vec[1])
        elif data.shape[0] == 2: # well determined
            theta = math.atan2(data[1,0]-data[0,0], data[1,1]-data[0,1])
        theta = (theta + math.pi * 5 / 2) % (2*math.pi)
        d = x0*math.sin(theta) + y0*math.cos(theta)
        return d, theta

    def residuals(self, model, data):
        d, theta = model
        dfit = data[:,0]*math.sin(theta) + data[:,1]*math.cos(theta)
        return np.abs(d-dfit)

    def is_degenerate(self, sample):
        return False


def ransac(data, model_class, min_samples, threshold, max_trials=1000):
    '''
    Fits a model to data with the RANSAC algorithm.

    :param data: numpy.ndarray
        data set to which the model is fitted, must be of shape NxD where
        N is the number of data points and D the dimensionality of the data
    :param model_class: object
        object with the following methods implemented:
         * fit(data): return the computed model
         * residuals(model, data): return residuals for each data point
         * is_degenerate(sample): return boolean value if sample choice is
            degenerate
        see LinearLeastSquares2D class for a sample implementation
    :param min_samples: int
        the minimum number of data points to fit a model
    :param threshold: int or float
        maximum distance for a data point to count as an inlier
    :param max_trials: int, optional
        maximum number of iterations for random sample selection, default 1000

    :returns: tuple
        best model returned by model_class.fit, best inlier indices
    '''

    best_model = None
    best_inlier_num = 0
    best_inliers = None
    data_idx = np.arange(data.shape[0])
    for _ in xrange(max_trials):
        sample = data[np.random.randint(0, data.shape[0], 2)]
        if model_class.is_degenerate(sample):
            continue
        sample_model = model_class.fit(sample)
        sample_model_residua = model_class.residuals(sample_model, data)
        sample_model_inliers = data_idx[sample_model_residua<threshold]
        inlier_num = sample_model_inliers.shape[0]
        if inlier_num > best_inlier_num:
            best_inlier_num = inlier_num
            best_inliers = sample_model_inliers
    if best_inliers is not None:
        best_model = model_class.fit(data[best_inliers])
    return best_model, best_inliers

def test():
    x = np.mgrid[-5:5:200j]
    y = np.mgrid[3:10:200j]
    data = np.vstack((x.ravel(), y.ravel())).T
    data += np.random.normal(size=data.shape)

    # generate some faulty data
    data[0,:] = (3, 20)
    data[1,:] = (4, 21)
    data[2,:] = (5, 22)
    data[3,:] = (5, 24)
    data[4,:] = (-2, -24)
    data[5,:] = (-3, -23)

    model, inliers = ransac(data, LinearLeastSquares2D(), 2, 2)

    #: plot results
    import pylab

    pylab.plot(data[:,0], data[:,1], '.r', label='outliers')
    pylab.plot(data[inliers][:,0], data[inliers][:,1], '.b', label='inliers')

    x = np.arange(-7, 8)
    dr, thetar = model
    y_ransac = (dr - x*math.sin(thetar)) / math.cos(thetar)
    ds, thetas = LinearLeastSquares2D().fit(data)
    y_simple = (ds - x*math.sin(thetas)) / math.cos(thetas)
    pylab.plot(x, y_simple, '-r', label='least-squares solution of all points')
    pylab.plot(x, y_ransac, '-b', label='RANSAC solution')

    pylab.legend(loc=4)
    pylab.show()

if __name__ == '__main__':
    test()
