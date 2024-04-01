import numpy as np
from scipy.stats import shapiro
from scipy.ndimage import gaussian_filter
import xarray as xr

def sec_since(time):
    return (time - time[0])/np.timedelta64(1,'s')


def mad(data, axis=-1):
    # Median absolute deviation
    median = np.nanmedian(data, axis=axis)
    return np.nanmedian(np.abs(data - median))  


def uneven_time_derivative(np_time, data, flagnans=True):
    # Calculate unevenly spaced time derivative using numpy gradient
    t_seconds = (np_time - np_time[0])/np.timedelta64(1,'s')
    if flagnans:
        nanx = ~np.isnan(data)
        grad_data = np.full_like(data, np.nan)
        grad_data[nanx] = np.gradient(data[nanx], t_seconds[nanx])
    else:
        grad_data = np.gradient(data, t_seconds)
    return grad_data


def pca_angle(x,y):
    top = np.sum(x*y)
    bot = np.sum(x**2)
    theta = np.arctan(top/bot)
    return theta


def point_distance(x,y):
    return np.sqrt(x**2 + y**2)


def ellipse_distance(x, y, x_ax, y_ax):
    ##### Do we need theta in here??? ######
    theta_2d = np.arctan(y/x)
    x_2 = np.abs((x_ax * y_ax) / np.sqrt(y_ax**2 + x_ax**2 * (np.tan(theta_2d)**2)))
    y_2 = np.sqrt(1 - (x_2/x_ax)**2) * y_ax
    e_dis = np.sqrt(x_2**2 + y_2**2)
    return e_dis


def universal_thresh(data, sig):
    n = len(data)
    abmax = np.sqrt(2*np.log(n)) * sig
    return abmax 


def calc_derivatives(time, data):
    dt1 = uneven_time_derivative(time, data)
    dt2 = uneven_time_derivative(time, dt1)
    return dt1, dt2


def sw_normal_test(data, sw_thresh=0.999, p_thresh=0.05, verbose=True):
    pass_test = False
    sw_statistic, p_value = shapiro(data)
    if verbose:
        print('SW stat: ' + str(np.around(sw_statistic, 3)) + ', p value: '\
            + str(np.around(p_value, 4)))
    if (p_value > p_thresh) | (sw_statistic > sw_thresh):
        pass_test = True
    return pass_test


def sample_norm(data, npoints=1000):
    d_mean = np.nanmean(data)
    d_std = np.nanstd(data)
    return np.random.normal(d_mean, d_std, npoints)


def ellipse_formula(n=100):
    nline = np.linspace(0,2*np.pi,n)
    xline = np.cos(nline)
    yline = np.sin(nline)
    return xline, yline


def flag_corr(w, c, corx):
    w_c = w.copy()
    w_c[c < corx] = np.nan
    return w_c


def nan_gauss(data, sigma, axis=None):
    V=data.copy()
    V[np.isnan(data)]=0
    VV=gaussian_filter(V,sigma=sigma, axes=axis)
    W=0*data.copy()+1
    W[np.isnan(data)]=0
    WW=gaussian_filter(W,sigma=sigma, axes=axis)
    return VV/WW

def nan_gauss_xr(data_xr, sigma, axis=None):
    data = nan_gauss(data_xr.values, sigma, axis=axis)
    return xr.DataArray(data=data, coords=data_xr.coords)


# def ellipse_threshold(x,y,n=100):
#     x1,y1 = ellipse_formula()
#     x_thresh = universal_thresh(x)
#     y_thresh = universal_thresh(y)
#     return x1*x_thresh, y1*y_thresh





# def ks_normal_test(data, thresh=0.05):
    # pass_test = False
    # ks_statistic, p_value = kstest(data, 'norm')
    # print(np.around(ks_statistic, 3), np.around(p_value, 9))
    # if p_value > thresh:
    #     pass_test = True
    # return pass_test

