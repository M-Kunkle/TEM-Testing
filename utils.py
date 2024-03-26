# -*- coding: utf-8 -*-
"""
Utility functions for programmer quality of life (aka im lazy)

1/26/24: Adding some of Ethan's functions as they are simple and handy

@author: mkrxp
"""
import numpy as np
import matplotlib.pyplot as plt
import signals as sigs
from IPython import get_ipython
import os
from pathlib import Path
import csv

# some universal plot styling
plt.style.use('ggplot')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['lines.linewidth'] = 1

def kunkle_clear():
    ''' Python equivalent of matlab clear all and clc '''
    plt.close('all')
    clear = lambda: os.system('cls')  # On Windows System
    clear()
    get_ipython().magic('reset -sf')
    
    
def signal_to_ascii(t, wfm, filename):
    ''' converts a wfm_generate t,sig to ascii to import to CST '''
    np.savetxt(filename+'.txt', np.c_[t,wfm])


def tictoc():
    ''' Copy pasted stack overflow implementation of MATLAB tic-toc '''
    from time import perf_counter_ns
    hashmap = {}
    
    def tic(key: str=None) -> None:
        """named tic method"""
        hashmap[key] = perf_counter_ns()
        return None
    
    def toc(key: str=None) -> float:
        """named toc method"""
        initial_ns = hashmap[key]
        current_ns = perf_counter_ns()
        elapsed_ns = current_ns - initial_ns
        elapsed_s  = elapsed_ns / (10**9)  # convert ns to s
        #print(f"Elapsed time is {elapsed_s} seconds.")
        return elapsed_s  
    
    return tic, toc

def indices(list, filtr=lambda x: bool(x)):
    ''' My version of MATLAB find() '''
    return [i for i,x in enumerate(list) if filtr(x)]

def signal_plot(t, wfm, start_freq, stop_freq, unit='V', norm=True, titlestr = ''):
    ''' Quick subplot for seeing time/frequency domain simultaneously '''
    f,s = sigs.generate_spectrum(t,wfm,1,norm)
    ylabelstr = 'Amplitude (' + unit + ')' 
    if norm:
        ylabelstr2 = 'Normalized Magnitude (dB)'
    else:
        ylabelstr2 = 'Magnitude (dB)'
        
    td_title = titlestr + 'Time Domain Signal'
    fd_title = titlestr + 'Frequency Domain Signal'
        
    quick_plot(t*1e9, wfm, title=td_title, xlab='Time (ns)', ylab=ylabelstr)
    quick_plot(f/1e9, s, title=fd_title, xlab='Frequency (GHz)', ylab=ylabelstr2, xlim=(start_freq/1e9, stop_freq/1e9), ylim=(-100, 5))
    
def quick_plot(x, y, title=None, xlim=None, ylim=None, xlab=None, ylab=None):
    ''' Quick plot template '''
    fig = plt.figure()
    plt.plot(x,y)
    ax = plt.gca()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.grid(True)
    plt.show()
    return fig
    
def dummy_signal_array(num_elements=4, freq_lb=2e9, freq_ub=4e9, fs=125e9, pw=20e-9, plot=False, same=True):
    ''' Dummy array used for the purpose of testing multidimensional array functions '''
    # constants
    dt = 1/fs
    t = np.arange(0, pw, dt)
    size = (num_elements, num_elements, len(t))
    center = np.floor((freq_ub + freq_lb) / 2)
    
    if same:
        # if all inputs are meant to be identical
        arr = np.zeros(size)
        default_wfm = sigs.wfm_generate(2, freq_lb, freq_ub, fs, pw)[1]
        for i in range(size[0]):
            for j in range(size[1]):
                arr[i,j,:] = default_wfm
        
    else:
        # setup rng for variety
        rng = np.random.default_rng(seed=216)
        types = rng.integers(1,5,(num_elements, num_elements))
        starts = np.linspace(freq_lb, center-1, num_elements**2).reshape(num_elements, num_elements)
        stops = np.linspace(center, freq_ub, num_elements**2).reshape(num_elements, num_elements)
        rng.shuffle(starts)
        rng.shuffle(stops)
        
        # generate array
        arr = np.zeros(size)
        for i in range(size[0]):
            for j in range(size[1]):
                wfm_type = types[i,j]
                arr[i,j,:] = sigs.wfm_generate(wfm_type, starts[i,j], stops[i,j], fs, pw)[1]
    if plot:
        for i in range(size[0]):
            for j in range(size[1]):            
                signal_plot(t, arr[i,j,:], freq_lb, freq_ub)    
    return t, arr

def timetable(t):
    ''' prints out the timetable results '''
    labels = [['setup', t[0]], ['tf loop', t[1]], ['itr loop', t[2]], ['plot', t[3]], ['total', t[4]]]
    print('|' + '-'*13 + 'TIMES' + '-'*14 + '|')
    for row in labels:
        print('| {:8} | {:12} seconds|'.format(*row))

def dummy_oscope_return(filename):
    ''' queries oscope and parses the return file '''
    testdata = Path('Measured Data')
    inputdata = np.loadtxt(testdata / filename, usecols=(3,4), delimiter=',')
    time = np.around([x-inputdata[0,0] for x in inputdata[:,0]], decimals=12)
    data = inputdata[:,1]
    return time, data

def dummy_send_awg(t, sig):
    ''' creates the awg signal file and sends it '''
    
    # i dont have the file format yet
    print('awg')
    
def array_to_csv(filename, label, *argv):
    ''' auto conversion from numpy array to csv file '''
    
    max_len = len(max(argv, key=len))
    data = [np.pad(x, pad_width=(0, max_len-len(x)), mode='constant') for x in argv]
    arr = np.column_stack(data)
    np.savetxt(filename, arr, delimiter=',', header=label)
    
def lin_2_db(lin_arr):
    return 20*np.log10(abs(lin_arr))

def inter_plot(x,y):
    for idx,i in enumerate(y):
        plt.subplot(int('42' + str(idx+1)))
        plt.plot(x,i)
        
def normalize(y, rescale=1):
    return rescale*(y / (max(abs(y))))

def time_calc(t, wfm, spacing, distance, pulse_width):
    c = 299792458
    rx_time = 2*distance / c
    coupling_time = spacing / c
    coupling_end = pulse_width + coupling_time
    rx_end = pulse_width + rx_time
    cutoff = (coupling_end + rx_time) / 2
    
    fig, ax = plt.subplots()

    ax.plot(t, wfm)
    ax.vlines(x=coupling_end, ymin=-1, ymax=1.0, color='b', linestyles='solid', label='COUPLING END: '+str(coupling_end*1e9))
    ax.vlines(x=cutoff, ymin=-1, ymax=1.0, color='r', linestyles='solid', label='CUTOFF: '+str(cutoff*1e9))
    ax.vlines(x=rx_time, ymin=-1, ymax=1.0, color='g', linestyles='dashdot', label='RX_0: '+str(rx_time*1e9))
    ax.vlines(x=rx_end, ymin=-1, ymax=1.0, color='y', linestyles='dashdot', label='RX_END: '+str(rx_end*1e9))
    ax.legend()
    
    plt.show()
    
    cutoff = (coupling_end + rx_time) / 2
    return cutoff

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

'''
ETHANS FUNCTIONS
'''

def read_csv(filepath):
    first_col = []
    sec_col = []
    with open(filepath, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row if present

        for row in csv_reader:
            first_col.append(float(row[3]))
            sec_col.append(float(row[4]))
    return first_col, sec_col  

# edited version for CST files
def read_cst_csv(filepath, skiprows=0):
    first_col = []
    sec_col = []
    with open(filepath, 'r') as file:
        csv_reader = csv.reader(file)
        for i in range(0,skiprows):
            next(csv_reader)  # Skip the header row if present

        for row in csv_reader:
            curr_row = row
            first_col.append(float(curr_row[0]))
            sec_col.append(float(curr_row[1]))
    return first_col, sec_col  


if __name__ == "__main__":
    tic,toc = tictoc()
    tic()
    #xx,zz = dummy_oscope_return()
    
    x,y = sigs.wfm_generate(2, 2e9, 4e9, 50e9, 10e-9,sample_count=5000)
    zzz = time_calc(x,y,1.5,4,10e-9)
    
    # y = [0] * 8
    # for i in range(8):
    #     t,y[i] = sigs.wfm_generate(2, 2e9, 4e9, 50e9, 20e-9)
    
    # inter_plot(t, y)
    # wfm_int = np.interp(xx, t, wfm)
    # signal_plot(t,wfm,2e9, 4e9)
    
    # signal_plot(xx,wfm_int,2e9, 4e9)
    # vv = sigs.reverse_signal(wfm_int)
    # signal_plot(xx,vv,2e9, 4e9)
    
    # array_to_csv('testing123.csv', 'tasst, ff, gg', xx, vv, zz)
    
    # signal_plot(xx,zz,.99e9, 1.01e9)
    # vv = sigs.reverse_signal(zz)
    # signal_plot(xx,vv,.99e9, 1.01e9)
    zzzzz = toc()