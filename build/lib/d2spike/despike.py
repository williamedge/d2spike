import xarray as xr
from afloat.timeseries import quick_butter
from d2spike.inspect import plot_Uts, plot_Marginals, plot_Ellipses, plot_Explore_Flags
from d2spike.despike_GN import qc0_Flags, despike_GN23
from d2spike.reinstate import gp_Reinstate_Loop, reinstate_thresh
from d2spike.utils import nan_gauss, flag_corr
from d2spike.interp2 import full_2dinterp
# from d2spike.full_stack import Full_Pipe


@xr.register_dataarray_accessor("floatda")
class DataArray():
    def __init__(self, da):
        if 'units' in da.attrs:
            self.units = da.attrs['units']
        else:
            self.units = '?'
        dims = da.dims
        if not dims[0].lower() == 'time':
            raise(Exception("First dimension must be time"))
        self._obj = da

    @property
    def _da(self):
        return self._obj

    @property
    def dims(self):
        return self._obj.dims

    @property
    def other_dims(self):
        return [dim for dim in self._obj.dims if (not dim.lower()=='time')]

    @property
    def coords(self):
        return self._obj.coords

    def __repr__(self):
        return self._da.__repr__()

    @property
    def _despike(self):
        return Despike(self._da.time.values, self._da.values, units=self.units, other_dims=self.other_dims)

    def lowpass(self, T_cut_seconds, **kwargs):
        xr_result = xr.DataArray(self._despike.lowpass(T_cut_seconds, **kwargs),\
                                 coords=self.coords,\
                                 dims=self.dims)
        return xr_result
    
    def qc0_lowcorr(self, corr_data, corrflag, **kwargs):
        xr_result = xr.DataArray(self._despike.qc0_lowcorr(corr_data, corrflag, **kwargs),\
                                 coords=self.coords,\
                                 dims=self.dims)
        return xr_result    
    
    def qc0_flags(self, **kwargs):
        xr_result = xr.DataArray(self._despike.qc0_flags(**kwargs),\
                                 coords=self.coords,\
                                 dims=self.dims)
        return xr_result    
 
    def gaussian_filter(self, filter_var, **kwargs):
        xr_result = xr.DataArray(self._despike.gaussian_filter(filter_var, **kwargs),\
                                 coords=self.coords,\
                                 dims=self.dims)
        return xr_result
    
    def despike_gn23(self, **kwargs):
        result, fullout = self._despike.despike_gn23(**kwargs)
        xr_result = xr.DataArray(result,\
                                 coords=self.coords,\
                                 dims=self.dims)
        return xr_result, fullout

    def reinstate_threshold(self, orig_data, index, **kwargs):
        result = self._despike.reinstate_threshold(orig_data, index, **kwargs)
        xr_result = xr.DataArray(result,\
                                 coords=self.coords,\
                                 dims=self.dims)
        return xr_result
        
    def gp_reinstate_loop(self, **kwargs):
        result = self._despike.gp_reinstate_loop(**kwargs)
        xr_result = xr.DataArray(result,\
                                 coords=self.coords,\
                                 dims=self.dims)
        return xr_result

    def interp_2D(self, **kwargs):
        result = self._despike.interp_2D(**kwargs)
        xr_result = xr.DataArray(result,\
                                 coords=self.coords,\
                                 dims=self.dims)
        return xr_result
           
    def plot_uts(self, **kwargs):
        return self._despike.plot_uts(**kwargs)
    
    def plot_marginals(self, **kwargs):
        return self._despike.plot_marginals(**kwargs)

    def plot_ellipses(self, full_out, **kwargs):
        return self._despike.plot_ellipses(full_out, **kwargs)

    def plot_explore_flags(self, full_out, **kwargs):
        return self._despike.plot_explore_flags(full_out, **kwargs)


    # # Special function for full de-spiking
    # def full_pipe(self, corr_data, **kwargs):
    #     result = Full_Pipe(self, corr_data, **kwargs)
    #     return result
    

class Despike():
    def __init__(self, time, data, other_dims=[], **kwargs):
        # quick_validate(time, data)
        units = kwargs.pop("units", '?')
        s = data.shape
        n_other_dims = len(s) - 1
        assert n_other_dims==len(other_dims), "If data is multidimensional, must specify names of other dimensions"
        assert len(time) == s[0], "Length of time must equal first dimension of data" 
        self.time = time
        self.data = data
        self.units = units

    def lowpass(self, T_cut_seconds, **kwargs):
        return quick_butter(self.time, self.data, T_cut_seconds, **kwargs)

    def qc0_lowcorr(self, corr_data, corrflag, **kwargs):
        return flag_corr(self.data, corr_data, corrflag, **kwargs) 
       
    def qc0_flags(self, **kwargs):
        return qc0_Flags(self.data, **kwargs)    

    def gaussian_filter(self, filter_var, **kwargs):
        return nan_gauss(self.data, filter_var, **kwargs) 
    
    def despike_gn23(self, **kwargs):
        return despike_GN23(self.time, self.data, **kwargs)  

    def reinstate_threshold(self, orig_data, index, **kwargs):
        return reinstate_thresh(self.data, orig_data, index, **kwargs)
    
    def gp_reinstate_loop(self, **kwargs):
        return gp_Reinstate_Loop(self.time, self.data, **kwargs)
    
    def interp_2D(self, **kwargs):
        return full_2dinterp(self.data, **kwargs)
        
    def plot_uts(self, **kwargs):
        return plot_Uts(self.time, self.data, **kwargs)
    
    def plot_marginals(self, **kwargs):
        return plot_Marginals(self.time, self.data, **kwargs)
    
    def plot_ellipses(self, full_out, **kwargs):
        return plot_Ellipses(self.time, self.data, full_out, **kwargs)
    
    def plot_explore_flags(self, full_out, **kwargs):
        return plot_Explore_Flags(self.time, self.data, full_out, **kwargs)
    
    # # Special function for full de-spiking
    # def full_pipe(self, corr_data, **kwargs):
    #     return Full_Pipe(self.data, corr_data, **kwargs)