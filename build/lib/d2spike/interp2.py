import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import CloughTocher2DInterpolator

from scipy.spatial import Delaunay
from scipy.interpolate import CloughTocher2DInterpolator

def divide_data(data_shape, n_ints=20000):
    # Divide into blocks of about 20k
    div_int = int((data_shape[0] * data_shape[1])/n_ints)
    return np.linspace(0, data_shape[0], div_int).astype('int')

def ct_interpolate(data):
    hh, ww = data.shape
    xx, yy = np.meshgrid(np.arange(ww), np.arange(hh))
    # xx = xx

    known_x = xx[~np.isnan(data)]
    known_y = yy[~np.isnan(data)]
    unknown_x = xx[np.isnan(data)]
    unknown_y = yy[np.isnan(data)]

    tri = Delaunay(np.stack([known_x, known_y]).T)
    interpolator = CloughTocher2DInterpolator(tri, data[~np.isnan(data)])
    values_mesh = interpolator(np.stack([unknown_x, unknown_y]).T)   
    return values_mesh

def full_2dinterp(one_beam, max_z=0.07, max_t=600):
    ind_arr = divide_data(one_beam.shape)
    beam_int = one_beam.copy()
    for ixx, iyy in zip(ind_arr[:-1], ind_arr[1:]):
        int_vals = ct_interpolate(one_beam[ixx:iyy,:])
        beam_int[ixx:iyy,:][np.isnan(one_beam[ixx:iyy,:])] = int_vals
    return beam_int