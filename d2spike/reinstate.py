import numpy as np
from gptide import cov
from gptide import GPtideScipy
from gptide import mle
from scipy.stats import norm

from d2spike.utils import sec_since
from d2spike.inspect import plot_reinstate

# This module is not tested or implemented yet

def reinstate_thresh(data, orig_data, thresh_ix):   
    idx = (thresh_ix.values) & np.isnan(data)    
    data[idx] = orig_data.values[idx]
    return data


def out_score(point, mu, var):
    if point >= mu:
        score = 2*(1-norm(loc=mu, scale=np.sqrt(var)).cdf(point))
    elif point < mu:
        score = 2*norm(loc=mu, scale=np.sqrt(var)).cdf(point)
    return score


def calc_outscores(data, data_raw, y_mu, y_cond, out_thresh):
    nangp = np.isnan(data)
    arr_score = np.full_like(data, np.nan)

    for ix in range(len(data)):
        if nangp[ix]:
            point_dist = data_raw[ix]
            arr_score[ix] = out_score(point_dist, y_mu[ix], y_cond[ix])
    
    data[arr_score > out_thresh] = data_raw[arr_score > out_thresh]
    return data, arr_score, nangp


def gp_mu_cond(time, y_data, nan_yy, soln, covfunc):
    # Initiate an updated GP with all data points and solved parameters
    GP2 = GPtideScipy(time[nan_yy], time, soln['x'][0], covfunc, soln['x'][1:])

    # Get the mean and var at each point
    y_mu = GP2(y_data[nan_yy])

    # Get the conditional variance at each point
    y_cond = GP2._calc_err()
    return y_mu, y_cond


def calc_weighted_outlier_threshold(corr_data):
    # Weight outlier scores by correlation data
    y = np.array([1, 0.03])
    x = np.array([1.0, 100.0])
    beta = np.polyfit(x, np.log(y), 1)

    w_thresh = np.exp(beta[1]) * np.exp(beta[0]*corr_data)

    # x_fit = np.arange(1,101)
    # y_fit = np.exp(beta[1]) * np.exp(beta[0]*x_fit)
    # plt.plot(x, y, 'o')
    # plt.plot(x_fit, y_fit)

    return w_thresh


def gp_reinstate(y_data, y_orig, y_mu, y_cond, weighted_thresh):

    # Invert the uncertainty
    y_cinv = (np.max(y_cond)*1.05 - y_cond)

    y_data_np, arr_score, nangp = calc_outscores(y_data.flatten(), y_orig.flatten(),\
                                                 y_mu, y_cinv, weighted_thresh)
    
    return y_data_np, arr_score, nangp


def gp_mle_sol(time, data, verbose=True):
    # Get key vars
    nan_yy = ~np.isnan(data)
    time =  sec_since(time)[:,None]
    y_data = data[:,None]
    
    # Fit the GP parameters
    covfunc = cov.matern52_1d
    try:
        soln = mle(time[nan_yy], y_data[nan_yy], covfunc, [0.1,0.1], 0.01, verbose=False)
    except:
        soln = mle(time[nan_yy], y_data[nan_yy], covfunc, [0.1,0.1], 0.001, verbose=False)
    if verbose:
        print('GP mle fit: ' + str(soln['success']))
    return soln, covfunc, time, y_data, nan_yy


def gp_Reinstate_Loop(time, data, raw_data, corr_data, soln=None, plot=False, verbose=True):

    print('Warning, reinstatement functions not thoroughly tested!')
    
    # Get key vars
    nan_yy = ~np.isnan(data)
    time =  sec_since(time)[:,None]
    y_data = data[:,None]
    y_raw = raw_data[:,None]
    
    # Fit the GP parameters
    covfunc = cov.matern52_1d
    if not soln:
        try:
            soln = mle(time[nan_yy], y_data[nan_yy], covfunc, [0.1,0.1], 0.01, verbose=False)
        except:
            soln = mle(time[nan_yy], y_data[nan_yy], covfunc, [0.1,0.1], 0.001, verbose=False)
        if verbose:
            print('GP mle fit: ' + str(soln['success']))

    # Get mean and var
    y_mu, y_cond = gp_mu_cond(time, y_data, nan_yy, soln, covfunc)

    # Get the correlation weighted threshold (only once)
    weight_thresh = calc_weighted_outlier_threshold(corr_data)

    # Do the first re-instatement
    y_data_np, arr_score, nangp = gp_reinstate(y_data, y_raw, y_mu, y_cond, weight_thresh)

    if plot:
        fig, ax = plot_reinstate(time, y_data, y_raw, corr_data, nan_yy, y_mu,\
                                 y_cond, weight_thresh, arr_score, nangp)

    while np.sum(arr_score > weight_thresh) > 0:
        # Put selected data back in 
        nan_yy = ~np.isnan(y_data_np)

        y_mu, y_cond = gp_mu_cond(time, y_data_np[:,None], nan_yy, soln, covfunc)

        # Do the first re-instatement
        y_data_np, arr_score, nangp = gp_reinstate(y_data_np, y_raw, y_mu, y_cond, weight_thresh)   

        if plot:
            fig, ax = plot_reinstate(time, y_data_np, y_raw, corr_data,\
                                                           nan_yy, y_mu, y_cond, weight_thresh, arr_score, nangp)
    
    return y_data_np