# File with plotting and analysis helper functions to 
# Wouter Van De Pontseele, Jan, 31, 2023
import os
print("Working directory:",os.getcwd())

import re
import numpy as np
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
        self.colormap = plt.cm.tab10 #jet
        print("RQ helper is initialised")
            
            
    # Plot 2D variables
    # var: dict with 'name','label','range','bins','norm','log' (optional)
    # config_2d: 'type', 'mask' (optional), 'mask_lab' (optional)
    def plot_pairs(self, var1, var2, mask_arr, mask_labs, config_2d):
        fig, ax = plt.subplots(figsize=(13,5), ncols=3,
                               constrained_layout=True,
                               gridspec_kw={'width_ratios': [2,2,3]})
        num_items = len(mask_arr[0])
        # 1D plots
        for var_i,ax_i in zip([var1,var2],[ax[0],ax[1]]):
            for mask, mask_lab in zip(mask_arr,mask_labs):
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
            for mask, mask_lab in zip(mask_arr,mask_labs):
                _ = ax[2].scatter(self.RQ_data[var1['name']][mask],self.RQ_data[var2['name']][mask],
                                 alpha=0.25,s=num_items/sum(mask),label=mask_lab)
                ax[2].set_xlim(var1['range'])
                ax[2].set_ylim(var2['range'])
        else:
            print("Unknown 2D plot type")
        ax[2].set_xlabel(var1['label'])
        ax[2].set_ylabel(var2['label'])


        file_name = var1['name']+"-"+var2['name']
        fig.savefig(self.output_dir+file_name+".pdf")
        return fig, ax


    # Wrapper to plot a set of traces
    def plot_traces(self, mask, mask_lab, 
                    pre_trig=None, post_trig=None, 
                    nsmooth=1, plot_data=0.5, plot_mean=0,
                    max_nevents=100, fig=None, ax=None):
        
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
                fig, ax = plt.subplots(figsize=(8,5), constrained_layout=True)
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

            file_name = re.sub('[^A-Za-z0-9]+', '_', mask_lab)
            fig.savefig(self.output_dir+file_name+".pdf")
            fig.show()
            return time_axis, traces
        else:
            print("Maximum number of traces per plot exceeded:", num_traces)
            return None,None
        
def sample_from_mask(mask,n_draw):
    if n_draw>sum(mask):
        print("Returning origianl mask with",sum(mask),"traces.")
        return mask
    else:
        return_mask = np.zeros_like(mask)
        indices = np.where(mask)
        sampled_indices = np.random.choice(indices[0], n_draw,replace=False)
        return_mask[sampled_indices]=True
        return return_mask