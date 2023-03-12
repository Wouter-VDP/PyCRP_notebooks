# File with plotting and analysis helper functions to 
# Wouter Van De Pontseele, Jan, 31, 2023
import os
print("Working directory:",os.getcwd())

import re
from os.path import join
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pycrp import Event
from scipy.signal import savgol_filter
#from tsmoothie.smoother import *

class RQ_helper:
    def __init__(self, RQ_data, config, series, output_dir):
        self.config = config
        self.RQ_data = RQ_data
        self.event = Event(fs=config.Fs, series=series, ADC2uA=config.ADC2uA)
        self.output_dir = output_dir
        self.series = series
        self.colormap = plt.cm.tab10 #jet
        print("RQ helper is initialised")
            
            
    # Plot 2D variables
    # var: dict with 'name','label','range','bins','norm','log' (optional)
    # config_2d: 'type', 'mask' (optional), 'mask_lab' (optional)
    def plot_pairs(self, var1, var2, mask_arr, mask_labs, config_2d, savefig=True):
        fig, ax = plt.subplots(figsize=(13,5), ncols=3,
                               constrained_layout=True,
                               gridspec_kw={'width_ratios': [2,2,3]})
        # 1D plots
        for var_i,ax_i in zip([var1,var2],[ax[0],ax[1]]):
            for mask_str, mask_lab in zip(mask_arr,mask_labs):
                mask = self.RQ_data[mask_str]
                _ = ax_i.hist(self.RQ_data[var_i['name']][mask], range=var_i['range'], 
                              bins=var_i['bins'], label=mask_lab, alpha=0.5, density=var_i['norm'])

            ax_i.set_xlabel(var_i['label'])
            ax_i.set_xlim(var_i['range'])
            ax_i.set_ylabel('# Pulses')
            if var_i['norm']:
                ax_i.set_ylabel('Area Normalised')
            if ("log" in var_i) and var_i['log']:
                ax_i.set_xscale("log")     
        ax[0].legend()

        # 2D plot
        if config_2d['type']=='hist':
            mask_2D_lab = 'All triggered pulses'
            mask_2D = (self.RQ_data['trig_ch']==1)
            if 'mask' in config_2d:
                mask_2D = config_2d['mask']
            if 'mask_lab'in config_2d:
                mask_2D_lab = config_2d['mask_lab']

            ax[2].hist2d(self.RQ_data[var1['name']][mask_2D],self.RQ_data[var2['name']][mask_2D],
                         range=(var1['range'],var2['range']), bins=(int(var1['bins']/10),int(var2['bins']/10)),
                         cmap='coolwarm',norm=mpl.colors.LogNorm()
                        )
            ax[2].legend(title=mask_2D_lab+" (Log colour scale)")

        elif config_2d['type']=='scatter':
            for mask_str, mask_lab in zip(mask_arr,mask_labs):
                mask = self.RQ_data[mask_str]
                dot_size = max(1,int(250/np.sqrt(sum(mask))))
                print(mask_str,dot_size)
                _ = ax[2].scatter(self.RQ_data[var1['name']][mask],self.RQ_data[var2['name']][mask],
                                 alpha=0.25,s=dot_size,label=mask_lab)
                ax[2].set_xlim(var1['range'])
                ax[2].set_ylim(var2['range'])
        else:
            print("Unknown 2D plot type")
        ax[2].set_xlabel(var1['label'])
        ax[2].set_ylabel(var2['label'])

        if savefig:
            file_name = self.series+"-"+var1['name']+"-"+var2['name']
            fig.savefig(self.output_dir+file_name+".pdf")
        return fig, ax


    # Wrapper to plot a set of traces
    def plot_traces(self, mask, mask_lab, 
                    pre_trig=None, post_trig=None, 
                    nsmooth=1, plot_data=0.5, plot_mean=0,
                    max_nevents=100, savefig=True, fig=None, ax=None, output_dir=None):
        
        if pre_trig == None:
            pre_trig = self.config.POST_TRIG/8
        if post_trig == None:
            post_trig = self.config.POST_TRIG
        pre_trig = int(pre_trig)
        post_trig = int(post_trig)

        num_traces = sum(mask)
        if (num_traces<=max_nevents) & (num_traces>0):
            print("Raw data directory:",self.config.data_dir)
            traces = self.event.get_raw_events_MIT(self.RQ_data, 
                                                   mask=mask, 
                                                   pre_trig=pre_trig, 
                                                   post_trig=post_trig, 
                                                   negative=False, 
                                                   maxNumEvents=max_nevents,
                                                   rawdir=self.config.data_dir)

            time_axis = np.arange(-pre_trig,post_trig)/self.config.Fs*1e3 # in ms

            if fig==None or ax==None:
                fig, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
            ax.set_prop_cycle(plt.cycler('color', self.colormap(np.linspace(0, 1, num_traces))))

            if plot_data>0:
                ax.plot(time_axis, traces, alpha=plot_data, lw=0.5)
            if nsmooth>1:
                # operate smoothing
                #smoother = ConvolutionSmoother(window_len=nsmooth, window_type='hanning', copy=False)
                #smoother.smooth(traces.T)
                for trace in traces.T:
                    filtered_trace = savgol_filter(trace, nsmooth, 3)
                    ax.plot(time_axis, filtered_trace, alpha=0.5, lw=0.5)
            if plot_mean>0:
                ax.plot(time_axis,traces.mean(axis=1), label='Average', color='k', alpha=plot_mean)
                ax.legend()

            ax.set_xlabel('Time [ms]')
            ax.set_ylabel(r'TES current [$\mu$A]')
            ax.set_title(mask_lab)
            ax.grid(alpha=0.2)
            ax.set_xlim(min(time_axis),max(time_axis))

            if savefig:
                if output_dir==None:
                    output_dir = self.output_dir
                file_name = re.sub('[^A-Za-z0-9]+', '_', mask_lab)
                fig.savefig(output_dir+file_name+".pdf")

            return time_axis, traces
        else:
            print("Maximum number of traces per plot exceeded:", num_traces)
            return None,None
        

    # Calculate the pulse rates using an exponential fit and return the fitted rate in Hz
    def get_pulse_rate(self, mask, mask_lab, duration, nbins=None, nskip=None, savefig=True, fig=None, ax=None):
        num_peaks = sum(mask)
        time_diffs = np.diff(self.RQ_data["trig_loc_total"][mask])/self.config.Fs

        if fig==None or ax==None:
            fig,ax = plt.subplots(ncols=1, figsize=(6.2,3),constrained_layout=True)
        if nbins==None:
            nbins = int(sum(mask)**(1/3))
        if nskip==None:
            nskip = max(int(nbins/10),1)

        x_min = 0
        x_max = duration/num_peaks*4
        width = (x_max/nbins)

        bins, edges = np.histogram(time_diffs, range=(x_min,x_max), bins=nbins)
        edges_mid = (edges[1:]+edges[:-1])/2
        ax.errorbar(edges_mid, bins, yerr=np.sqrt(bins), xerr=width/2, fmt='.',label='Data')

        # Fit exponential
        p = np.polyfit(edges_mid[nskip:], np.log(bins[nskip:]), 1)
        a = np.exp(p[1])
        y_ftited = np.exp(p[1]) * np.exp(p[0] * edges_mid)
        ax.plot(edges_mid,y_ftited, label='Exp fit: {:.2f}Hz'.format(abs(p[0])))
                
        ax.set_xlabel(r'$\Delta$t [s]')    
        ax.set_ylabel('Pulse count')
        ax.set_ylim(bottom=0)
        ax.set_title('{} rate: Time constant'.format(mask_lab))
        ax.legend()
        ax.grid(alpha=0.2)

        if savefig:
            file_name = re.sub('[^A-Za-z0-9]+', '_', mask_lab)
            fig.savefig(self.output_dir+file_name+"_rate.pdf")

        return abs(p[0])

# Subsample a mask    
def sample_from_mask(mask,n_draw):
    if n_draw>sum(mask):
        print("Returning original mask with",sum(mask),"traces.")
        return mask
    else:
        return_mask = np.zeros_like(mask)
        indices = np.where(mask)
        sampled_indices = np.random.choice(indices[0], n_draw,replace=False)
        return_mask[sampled_indices]=True
        return return_mask
    
# Get total data taking duration given a mask
def get_duration(trigger_locations, mask, sampling_rate):
    return np.ptp(trigger_locations[mask])/sampling_rate


# Compare a variable for different sources using a histogram
def plot_var_comparison(source_dict, field, mask, x_range, bins, log_flag=False, save_fig=False, output_dir=None):
    fig, ax  = plt.subplots(figsize=(12,4), constrained_layout=True)
    fig_sub, ax_sub  = plt.subplots(figsize=(12,4), constrained_layout=True)

    for k,v in source_dict.items():
        data = v['data'][field][v['data'][mask]]

        vals, edges = np.histogram(data, bins=bins, range = x_range)
        mids = (edges[1:]+edges[:-1])/2
        vals_abs = (vals)*v['normalisation']
        vals_plus = (vals+np.sqrt(vals))*v['normalisation']
        vals_minus = (vals-np.sqrt(vals))*v['normalisation']

        ax.step(mids,vals_abs, where='mid', label=k, color=v['color'], alpha=0.8, lw=1)
        ax.fill_between(mids, vals_minus, vals_plus, alpha=0.2, step="mid", color=v['color'])
        
        # Background subtracted:
        data_bkgd = source_dict['Background']['data'][field][source_dict['Background']['data'][mask]]
        vals_bkgd, edges = np.histogram(data_bkgd, bins=bins, range = x_range)
        vals_bkgd_abs = (vals_bkgd)*source_dict['Background']['normalisation']

        if k=="Background":
            ax_sub.axhline(0, linestyle='--', label="Background", alpha=0.5, color=v['color'])
        else:
            # TODO his is incorrect, but I don't know how to do it properly for now.
            vals_sub = vals_abs-vals_bkgd_abs
            vals_sub_plus = vals_sub + np.sqrt(vals_bkgd_abs+vals_abs)
            vals_sub_minus = vals_sub - np.sqrt(vals_bkgd_abs+vals_abs)
            
            ax_sub.step(mids,vals_sub, where='mid', label=k, color=v['color'], alpha=0.8, lw=1)
            ax_sub.fill_between(mids, vals_sub_minus, vals_sub_plus, alpha=0.2, step="mid", color=v['color'])

    ax.grid(alpha=0.2)
    ax.legend()
    ax.set_xlabel(field)
    ax.set_ylabel("# Pulses/day/bin")
    ax.set_xlim(*x_range)

    ax_sub.grid(alpha=0.2)
    ax_sub.legend()
    ax_sub.set_xlabel(field)
    ax_sub.set_ylabel("# Pulses/day/bin [Background Subtracted]")
    ax_sub.set_xlim(*x_range)

    if log_flag:
        ax.set_yscale("log")

    if save_fig:
        file_name = re.sub('[^A-Za-z0-9]+', '_', f"{field}_{mask}")
        fig.savefig(join(output_dir, file_name+"_hist.pdf"))
        fig_sub.savefig(join(output_dir, file_name+"_hist_bkgd_sub.pdf"))

    return (fig,ax), (fig_sub,ax_sub)