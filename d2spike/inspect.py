import numpy as np
import matplotlib.pyplot as plt
from afloat.pca import rotate_2D
from matplotlib.lines import Line2D
import seaborn as sns
from matplotlib import cm as cmp
from wutils.plotnice import vert_stack, basic_ts

from d2spike.utils import (calc_derivatives,\
                           sw_normal_test,\
                           sample_norm,\
                           ellipse_formula)

def plot_Uts(time, dt0, index=True, ax=None, text=True):
    if index:
        xd = np.arange(len(time))
    else:
        xd = time
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10,1.8))
    else:
        fig = None
    for txx in xd[np.isnan(dt0)]:
        ax.axvline(txx, ymin=0, ymax=0.1, c='lightgrey', lw=0.25)
        ax.axvline(txx, ymin=0.9, ymax=1.0, c='lightgrey', lw=0.25)
    ax.plot(xd, dt0, lw=0.75)
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_xlim(xd[0], xd[-1])
    prc = int(100*(np.sum(np.isnan(dt0)) / len(dt0)))
    if text:
        ax.text(0.01, 0.77, str(np.sum(np.isnan(dt0))) + '/' + str(len(dt0)) + ' (' + str(prc) + '%)',\
                transform=ax.transAxes)
    return fig, ax


def plot_nanloops(nanloops, size=(8,2)):
    # Diagnostic plot: nans
    fig, ax = plt.subplots(1, 1, figsize=size)
    plt.pcolormesh(nanloops.T)
    return fig, ax


def plot_looppairs(flag_pair, size=(8,2)):
    # Diagnostic plot: flag pair
    fig, ax = plt.subplots(1, 1, figsize=size)
    plt.pcolormesh(flag_pair.T, cmap='magma')
    return fig, ax


def plot_Marginals(time, data, size=(14,4), verbose=True, nbins=200, sw_thresh=0.98):
    fig, ax = plt.subplots(1, 3, figsize=size, gridspec_kw={'wspace':0.04})
    dt1, dt2 = calc_derivatives(time, data)
    for x, vr in zip(ax, [data, dt1, dt2]):
        pltbins = np.linspace(-np.nanmax(np.abs(vr)), np.nanmax(np.abs(vr)), nbins)
        sns.histplot(vr, ax=x, kde=True, kde_kws={'bw_adjust':0.25},\
                    stat='probability', bins=pltbins)
        x.lines[0].set_color('royalblue')
        x.set_ylabel('')
        x.grid()
        x.set_yticklabels('')
        vr_norm = sample_norm(vr, npoints=int(1e5))
        sns.histplot(vr_norm, ax=x, kde=True, stat='probability', bins=pltbins)
        x.set_ylabel('')
        test = sw_normal_test(vr[~np.isnan(vr)], sw_thresh=sw_thresh, verbose=verbose)
        x.text(0.02, 0.9, 'Test passed: ' + str(test), transform=x.transAxes)    
    return fig, ax


def plot_Ellipses(time, data, output_full,\
                  size=(14,4), lim_scale=1.5):
    
    data_orig = output_full['looped_data'][:,0]
    univ_criteria = output_full['univ_criteria']
    sig_3 = output_full['theta_pair3']
    all_intense = output_full['intensity']

    dt1_old, dt2_old = calc_derivatives(time, data_orig)
    dt1_new, dt2_new = calc_derivatives(time, data)

    fig, ax = plt.subplots(1, 3, figsize=(14,4), gridspec_kw={'wspace':0.07})

    alphas = np.logspace(np.log(0.5), np.log(1), len(all_intense))
    for ix, (uc, s3, ai, al) in enumerate(zip(univ_criteria, sig_3, all_intense, alphas)):
        # Plot ellipses
        for x, ui in zip(ax, uc):
            x1,y1 = ellipse_formula()
            x.plot(ai*x1*ui[0], ai*y1*ui[1], c='k', lw=0.5, alpha=al)

    for x, dt in zip(ax, [[data_orig,dt1_old],[dt1_old,dt2_old],[data_orig,dt2_old]]):
        if x==ax[-1]:
            dt[0], dt[1] = rotate_2D(dt[0], dt[1], s3)
        sns.scatterplot(x=dt[0], y=dt[1], ax=x, s=2, color='r')

    for x, dt in zip(ax, [[data,dt1_new],[dt1_new,dt2_new],[data,dt2_new]]):
        sns.scatterplot(x=dt[0], y=dt[1], s=2, ax=x, color='cornflowerblue')
        x.set_ylim(-2*np.nanmax(np.abs(y1*np.nanmax(univ_criteria[-1,:,1]))),\
                2*np.nanmax(np.abs(y1*np.nanmax(univ_criteria[-1,:,1]))))
        x.set_xlim(-2*np.nanmax(np.abs(x1*np.nanmax(univ_criteria[-1,:,0]))),\
                2*np.nanmax(np.abs(x1*np.nanmax(univ_criteria[-1,:,0]))))
        if x!=ax[0]:
            x.set_yticklabels('')
    return fig, ax



def plot_Explore_Flags(time, data, output_full, indexes=None, size=(14,2.5)):

    dt0_full = output_full['looped_data']
    pair_full = output_full['flag_pair']

    if indexes is not None:
        tx = indexes
    else:
        tx = [0, len(time)]
    txix = (time > time[tx[0]]) & (time <= time[tx[1]])

    fig, ax = plt.subplots(1, 1, figsize=(14,2.5))

    ax.plot(time[txix], dt0_full[txix,0], lw=2, c='grey')
    ax.plot(time[txix], data[txix], c='k', lw=2)
    ax.scatter(time[txix], data[txix], c='k', s=10, zorder=2)

    clm, sym = plot_flagpairs(ax, time, dt0_full, pair_full, txix)

    ax.set_xlim(time[txix][0], time[txix][-1])
    ax.set_ylim(-np.nanmax(np.abs(dt0_full[txix,0]))*1.1,\
                np.nanmax(np.abs(dt0_full[txix,0]))*1.1)
    ax.grid()
    sm = plt.cm.ScalarMappable(cmap=cmp.jet)
    sm.set_array([])
    cbar = plt.colorbar(sm, pad=0.01, ticks=[0.1,0.9])
    cbar.ax.set_yticklabels(['Earlier','Later'])
    cbar.ax.tick_params(width=0)

    handles = []
    for sy, lb in zip(sym, ['Pair 1','Pair 2','Pair 3']):
        handles.extend([Line2D([0],[0], label=lb, marker=sy, markersize=6, 
                markeredgecolor='k', markerfacecolor='k', linestyle='')])
    ax.legend(handles=handles)
    return fig, ax


def plot_flagpairs(ax, time, dt0_full, pair_full, txix):
    # Need to fix, row 0 of full outut has nans
    clm = cmp.jet(np.linspace(0, 1, dt0_full.shape[1]))
    sym = ['^','*','s']
    for ix, r0 in enumerate(dt0_full[txix,:].T):
        nx = np.isnan(r0) & ~np.isnan(dt0_full[txix,ix-1])
        if np.sum(nx)>0:
            flx = pair_full[txix,ix-1][nx]
            for xcx in range(3):
                ax.scatter(time[txix][nx][flx==xcx+1], dt0_full[txix,0][nx][flx==xcx+1],\
                            color=clm[ix], marker=sym[xcx], zorder=1)
    return clm, sym
  

def plot_reinstate(time, y_data, y_orig, y_corr, nan_yy, y_mu, y_cinv, weighted_thresh, arr_score, nangp):

    # Diagnostic plot: GP reinstatement
    fig, ax = plt.subplots(2, 1, figsize=(14,3), gridspec_kw={'hspace':0.05, 'height_ratios':[2,1]})
    ax[0].plot(time, y_mu,'k--')
    ax[0].plot(time[nan_yy], y_data[nan_yy], 'k.')
    ax[0].fill_between(time.flatten(),\
                    y1=y_mu.flatten() - np.sqrt(y_cinv),\
                    y2=y_mu.flatten() + np.sqrt(y_cinv),\
                    color='grey', alpha=0.3)
    ax[0].grid()

    # clm, sym = plot_flagpairs(ax[0], time, w_st4_orxx, output['looped_data'],\
    #                           output['flag_pair'], txix)

    for ix in range(len(time)):
        if nangp[ix]:
            ax[1].scatter(time[ix], arr_score[ix], color='royalblue')
            ax[1].scatter(time[ix], y_corr[ix]/100, color='k', s=5)
            if arr_score[ix] > weighted_thresh[ix]:
                ax[1].axvline(time[ix], ymin=0, ymax=1, c='green', lw=1)
                ax[0].scatter(time[ix], y_orig[ix], color='royalblue', s=30)
            # if w_st4_corr[ix] >= 60:
    ax[0].scatter(time[y_corr>50], y_orig[y_corr>50], c=y_corr[y_corr>50], cmap=cmp.magma, s=10)
    ax[1].set_ylim(0,1)
    ax[1].set_yticks(np.arange(0,1.01,0.1))
    ax[1].grid()
    ax[1].set_yticklabels('')

    for x in ax:
        x.set_xlim(time[0], time[-1])

    return fig, ax


# def plot_allts(time, dt0, index=True):
#     dt1, dt2 = calc_derivatives(time, dt0)

#     if index:
#         xd = np.arange(len(time))
#     else:
#         xd = time

#     fig, ax = plt.subplots(3, 1, figsize=(10,5), gridspec_kw={'hspace':0.07})
#     for x, dt in zip(ax, [dt0, dt1, dt2]):
#         for txx in xd[np.isnan(dt0)]:
#             x.axvline(txx, ymin=0, ymax=0.1, c='lightgrey', lw=0.25)
#             x.axvline(txx, ymin=0.9, ymax=1.0, c='lightgrey', lw=0.25)
#         x.plot(xd, dt, lw=0.75)
#         x.set_title('')
#         x.set_xlabel('')
#         x.set_xlim(xd[0], xd[-1])
#         if x == ax[0]:
#             x.text(0.01, 0.77, str(np.sum(np.isnan(dt))) + '/' + str(len(dt)),\
#                     transform=ax[0].transAxes)
#         if x != ax[-1]:
#             x.set_xticklabels('')
#     return fig, ax


def compare_plt(w1, w2, tx, hx):
    fig, ax = vert_stack(2, hsize=14, vsize=3)
    w1.isel(time=tx, height=hx).plot(cmap='PuOr', vmin=w1.min()/5, vmax=w1.max()/5,\
        ax=ax[0], center=0, cbar_kwargs={'pad':0.01})
    w2.isel(time=tx, height=hx).plot(cmap='PuOr', vmin=w1.min()/5, vmax=w1.max()/5,\
        ax=ax[1], center=0, cbar_kwargs={'pad':0.01})
    basic_ts(w1.time[tx], ax)
    print('Top plot: ' + str(perc_nan(w1)))
    print('Bottom plot: ' + str(perc_nan(w2)))


def perc_nan(w):
    return np.around(100*np.sum(np.isnan(w.values)) / len(w.values.flatten()), 1)


