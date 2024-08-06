import numpy as np
# from d2spike.despike import DataArray



def round_one(beam_data, orig_data, gf_sig, re_val, sw_vals, skip_pair, verbose, full_output, max_z, max_t):

    raise ValueError('Function not implemented yet!!')
    # Use a 2D Gaussian filter to find the mean values (works much better than 1D lowpass with heavy spiking)
    # w_gf = xr.DataArray(data=nan_gauss(beam_data.values, gf_sig), coords=beam_data.coords)
    w_gf = beam_data.floatda.gaussian_filter(gf_sig)

    # Subtract the background values and despike
    w_gn = (beam_data - w_gf).copy()
    for ii, wd in enumerate(w_gn.T):
        w_gn[:,ii], _ = wd.floatda.despike_gn23(full_output=full_output,\
            sw_thresh=sw_vals, skip_pair=skip_pair, verbose=verbose)

    # Call 2D indexing reinstatement
    re_ix = np.abs(orig_data - w_gf) < re_val
    w_gn = w_gn.floatda.reinstate_threshold((orig_data - w_gf), re_ix)

    # Interpolate gaps of 1 (timestep and spatial bin)
    w_int = w_gn + w_gf.T
    w_int = w_int.interpolate_na(dim='height', method='cubic', max_gap=max_z)
    w_int = w_int.interpolate_na(dim='time', method='cubic', max_gap=np.timedelta64(max_t,'ms'))
    return w_int



def Full_Pipe(beam_data, corr_data, corrflag=45, qc0_val=0.5, gf_sig=[2,2], re1=0.05, re2=0.01,\
                sw_vals=0.5, skip_pair=[-1], verbose=False, full_output=False, max_z=0.07, max_t=600):

    '''
    Call the full de-spiking pipeline from raw data up to small gap interpolation.
    Large gap interpolation is handled later. 
    '''

    raise ValueError('Function not implemented yet!!')

    # Flag data below a correlation threshold (and initiate the despike class)
    # w_c = flag_corr(beam_data, corr_data, corrflag).T
    w_c = beam_data.floatda.qc0_lowcorr(corr_data, corrflag)

    # Also flag values that are physically unreasonable
    w_c = w_c.floatda.qc0_flags(val=qc0_val)

    # Flag round 1
    w_int1 = round_one(w_c, beam_data, gf_sig, re_val=re1, sw_vals=sw_vals, skip_pair=skip_pair,\
                        verbose=verbose, full_output=full_output, max_z=max_z, max_t=max_t)

    # Flag round 2
    w_int2 = round_one(w_int1, beam_data, gf_sig, re_val=re2, sw_vals=sw_vals, skip_pair=skip_pair,\
                        verbose=verbose, full_output=full_output, max_z=max_z, max_t=max_t)

    return w_int2