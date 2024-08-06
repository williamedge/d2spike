import numpy as np
from afloat.pca import rotate_2D

from d2spike.utils import (mad, uneven_time_derivative, pca_angle, point_distance,\
                           ellipse_distance, universal_thresh, calc_derivatives,\
                           sw_normal_test)


def qc0_Flags(data, thresh=10, val=None):
    '''
    Flagging level 0 for despike toolbox. See Wahl (2003).

    Parameters
    ----------
    data : array (numpy) or DataArray (xarray)
        The **stationary** time series data (i.e. lowpass filtered or similar)
    thresh : float or int, optional
        A threshold flag as a multiplier of median absolute deviation (MAD)
        (default is 10).

    Returns
    -------
    data_f0
        array with flagged data set to NaN
    '''
    data_f0 = data.copy()
    if val:
        data_f0[np.abs(data) > val] = np.nan
    else:
        mad_data = mad(data)
        S = 1.483 * mad_data
        data_f0[np.abs(data) > thresh * mad_data] = np.nan
    return data_f0


def gen_fullout_dict(time, data, max_loops):
    output_full = dict()
    output_full['looped_data'] = np.full((len(time), max_loops+2), np.nan)
    output_full['looped_data'][:,0] = data
    output_full['flag_pair'] = np.zeros((len(time), max_loops+1))
    output_full['univ_criteria'] = np.full((max_loops+1,3,2), np.nan)
    output_full['theta_pair3'] = np.full((max_loops+1), np.nan)
    output_full['intensity'] = np.full((max_loops+1), np.nan)
    return output_full


def trim_fulloutput(output_full, n_lop):
    output_full['looped_data'] = output_full['looped_data'][:,:n_lop+1]
    output_full['flag_pair'] = output_full['flag_pair'][:,:n_lop]
    output_full['univ_criteria'] = output_full['univ_criteria'][:n_lop,:,:]
    output_full['theta_pair3'] = output_full['theta_pair3'][:n_lop]
    output_full['intensity'] = output_full['intensity'][:n_lop]
    return output_full


def despike_GN23(time, data, max_loops=1000, full_output=False, skip_pair=[-1],\
                 sw_thresh=0.98, method='universal_thresh', verbose=True):
    '''
    Goring-Nikora despike fucntion. Two options for setting the size of the ellipse; 
    median absolute deviation, or universal threshold

    Parameters
    ----------
    time : numpy datetime64 array
    
    data : array (numpy) or DataArray (xarray)
        The **stationary** time series data (i.e. lowpass filtered or similar)
    max_loops : int, optional
        Maximum number of despike iterations to do before returning data.
    full_output : bool, optional
        To return dictionary of detailed despike info to inspect (default is off).
    skip_pair : int, optional
        To skip a phase-space pair for the de-spiking method, either 0, 1, or 2.
        Default is no skips (any value other than 0,1,2).
    sw_thresh : float or array, optional between 0 and 1
        Sets the tolerance for how normal the final distributions has to be.
        Closer to 1 means closer to normal. Can be inspected with `plot_marginals`.
    method : str 'mad' or 'universal_thresh', optional
        Sets the method to determine the ellipse boundary
    verbose : bool, optional (default True)
        print info while running the function

    Returns
    -------
    dt0
        flagged array with spikes set to NaN
    output_full
        dict of flagging variables to inspect
    '''
    dt0 = data.copy()

    # Set intensity (this shouldn't be user controlled and is only activated
    # when a non-Gaussian finishing point is detected by sw_thresh)
    intense = 1
    sw_result = False

    # Create full output vars dict
    if full_output:
        output_full = gen_fullout_dict(time, data, max_loops)

    # Initiale while loop until max loops or no new flags
    n_out  = 999
    n_lop = 0
    while (n_out!=0) & (n_lop <= max_loops) & (len(dt0[~np.isnan(dt0)]) > 1):

        # Remove any nans before calcs
        nanx = np.isnan(dt0)
        dt0 = dt0[~nanx]
        flg_loop = np.full(dt0.shape, False)
        
        # Dont do anything if all nans
        if len(dt0) > 1:
            # print(dt0)

            # Get derivatives
            dt1 = uneven_time_derivative(time[~nanx], dt0)
            dt2 = uneven_time_derivative(time[~nanx], dt1)
            
            # Loop through each array and flag outliers
            for ix,(dxx,dyy) in enumerate([[dt0,dt1],[dt1,dt2],[dt0,dt2]]):

                # Skip a phase-space pair if chosen
                if ix not in skip_pair:

                    # Check for rotation for the third pair
                    if ix==2:
                        # Rotate before flag
                        theta_u_u2 = pca_angle(dyy, dxx)
                        dxx, dyy = rotate_2D(dxx, dyy, theta_u_u2)
                        if full_output:
                            output_full['theta_pair3'][n_lop] = theta_u_u2
                        
                    # Now calculate the point and ellipse distances
                    pd = point_distance(dxx, dyy)
                    if method == 'universal_thresh':
                        sig_xx = np.std(dxx)
                        sig_yy = np.std(dyy)
                    elif method == 'mad':
                        sig_xx = 1.483 * mad(dxx)
                        sig_yy = 1.483 * mad(dyy)
                    else:
                        raise ValueError('Method unknown')
                    ed = intense * ellipse_distance(dxx, dyy,\
                                        universal_thresh(dxx, sig_xx),\
                                        universal_thresh(dyy, sig_yy))

                    if full_output:
                        output_full['univ_criteria'][n_lop,ix,:] = [universal_thresh(dxx, sig_xx),\
                                                                    universal_thresh(dyy, sig_yy)]
                        # Record pair flagged (gets overwritten by higher pairs)
                        pair_temp = output_full['flag_pair'][~nanx,n_lop]
                        pair_temp[pd>ed] = ix+1
                        output_full['flag_pair'][~nanx,n_lop] = pair_temp

                    # Flag outliers
                    flg_loop = flg_loop | (pd > ed)
                    
            # Flag data and put back into full time series
            dt0_new = np.full_like(data, np.nan)
            # Few extra lines because you can't assign with a double slice
            dt0_temp = dt0_new[~nanx]
            dt0_temp[~flg_loop] = dt0[~flg_loop]
            dt0_new[~nanx] = dt0_temp
            dt0 = dt0_new
            
            if full_output:
                output_full['looped_data'][:,n_lop+1] = dt0
                output_full['intensity'][n_lop] = intense

            # Detect a non-Gaussian finish point
            if np.sum(flg_loop)==0:
                # Run Shapiro-Wilks test
                dt1, dt2 = calc_derivatives(time, dt0)
                vars = [dt0, dt1, dt2]
                if isinstance(sw_thresh, float):
                    sw_result = [sw_normal_test(var[~np.isnan(var)],\
                                                sw_thresh=sw_thresh, verbose=verbose)\
                                for var in vars]
                elif len(sw_thresh) == 3:
                    sw_result = [sw_normal_test(var[~np.isnan(var)],\
                                                sw_thresh=sw_t, verbose=verbose)\
                                                for var, sw_t in zip(vars, sw_thresh)]
                else:
                    raise ValueError('Shapiro Wilks threshold value is bad.')

        else:
            dt0 = np.full_like(data, np.nan)

        # Determine whether to exit loop
        if (np.sum(flg_loop)==0) & np.all(sw_result):
            # Successful exit
            n_out = np.sum(flg_loop)
        elif (np.sum(flg_loop)==0) & ~np.all(sw_result):
            # Increase intensity
            if (intense==1.0) & verbose:
                print('Non-Gaussian finishing point detected... increasing intensity to ')
            intense += -0.05
            if verbose:
                print(intense)
        else:
            # Continue looping
            n_out = np.sum(flg_loop)
            intense = 1.0
        n_lop += 1
    
    if verbose:
        print(n_lop)

    # Trim the full output
    if full_output:
        output_full = trim_fulloutput(output_full, n_lop)
        return dt0, output_full
    else:
        return dt0, []
