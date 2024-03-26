# -*- coding: utf-8 -*-
"""
Richard Testing

Created on Wed Feb 28 13:07:30 2024

@author: mkrxp
"""

import numpy as np
import signals as sigs
import utils as ut
import auxiliary as aux
from datetime import datetime
import pyvisa
import interface
import time
import sys
import csv
import scipy

ITERATIONS = 1
START_FREQ = .7e9
STOP_FREQ = 2e9
FS = 50e9
AMPLITUDE = .5 # 250mV 
WFM_TYPE = 2
PULSE_WIDTH = 10e-9
SIGNAL_SAMPLES = 5000

''' Device IDs: '''
TEK_AWG70001B = 'B020541'
TEK_DPO71304SX = 'B322209'
KEY_FIELDFOX = 'MY61131330'
TEK_MSO = 'C014452'


def read_rich_csv(filepath, skiprows=0):
    first_col = []
    with open(filepath, 'r') as file:
        csv_reader = csv.reader(file)
        for i in range(0,skiprows):
            next(csv_reader)  # Skip the header row if present

        for row in csv_reader:
            curr_row = row[0].split()
            first_col.append(float(curr_row[0]))
    return first_col


# t = np.linspace(0,5562*(1/FS),5562)

# connect with awg and oscope
rm = pyvisa.ResourceManager()
awg_address = interface.get_address(TEK_AWG70001B, rm)
oscope_address = interface.get_address(TEK_DPO71304SX, rm)
scope, rm = interface.connect_scope(oscope_address, rm)
awg = interface.awg_connect(awg_address)
interface.trigger_setup_manual(scope, ch=3, level=.2, slope='RISE', rec_length=5000)
interface.vertical_scale_reset(scope, ch=1, ycenter=0, ydiv=0.00625, yoffset=0)
interface.vertical_scale_reset(scope, ch=2, ycenter=0, ydiv=0.05, yoffset=0)
interface.vertical_scale_reset(scope, ch=3, ycenter=0, ydiv=0.2, yoffset=0)
# 
# t,wfm = sigs.wfm_generate(2, START_FREQ, STOP_FREQ, FS, PULSE_WIDTH, SIGNAL_SAMPLES)

# wfm = np.asarray(read_rich_csv('C:\\Users\\mkrxp\\Documents\\RichardWaveforms\\EXCITATIONS\\EXCITATIONS 2\\0207-NB_Average.csv'))
# t = np.linspace(0, 50e-9, step=(1/FS))
# interface.awg_upload_wfm(awg,wfm, 0.5, FS, 'current_iter.seqx')
# interface.capture_single_sequence(scope)
# interface.awg_fire(awg)

# tic()
t, sig_int = sigs.wfm_generate(2, START_FREQ, STOP_FREQ, FS, PULSE_WIDTH, sample_count=5000)
# sig_int = -sig_int
interface.awg_upload_wfm(awg, sig_int, .5, FS, 'current_iter.seqx')
interface.capture_single_sequence(scope)
interface.awg_fire(awg)

xx,zz,_ = interface.get_waveform_from_scope(scope, ch=1)
ut.signal_plot(xx,np.squeeze(zz),0,3e9,unit='V')
cc,vv,_ = interface.get_waveform_from_scope(scope, ch=2)
ut.signal_plot(cc,np.squeeze(vv),0,3E9,unit='V')

filtzz = sigs.wfm_filter(xx, np.squeeze(zz), .7e9, 2e9)
# filtTEM = sigs.wfm_filter(cc, np.squeeze(vv), .7e9, 2e9)

f,fft_dut = sigs.rdwfft(xx, filtzz)
ff,fft_tem = sigs.rdwfft(cc, np.squeeze(vv))
transfer_func = np.divide(fft_dut,fft_tem)
# tf_conj = np.conj(filtzz)
new_sig = sigs.rdwifft(f, transfer_func, xx)
ut.quick_plot(f, 20*np.log10(abs(transfer_func)),xlim=(.7e9,2e9),ylim=(-50,0))
ut.signal_plot(xx,np.flipud(new_sig),.2e9,3e9)
tf_f, tf_fd = sigs.rdwfft(xx, np.flipud(new_sig))

nbfreq = ff[np.argmax(abs(transfer_func[1:200]))]
nbt, nbsig = sigs.wfm_generate(1, nbfreq-nbfreq*.005, nbfreq+nbfreq*.005, FS, 25e-9, SIGNAL_SAMPLES)
ut.signal_plot(nbt,nbsig,.2e9,3e9)

filename = 'transferfunction4'
fn2 = 'narrowband4'

# csv_file = f'D:\\Mohamed Testing 3-21-2024\\{filename}.csv'
# csv_header = ','.join(('Time (s)','Amplitude (V)'))
# ut.array_to_csv(csv_file, csv_header, xx, np.squeeze(zz))
# csv_file = f'C:\\Users\\mkrxp\\Documents\\RichardWaveforms\\MeasuredData_02282024_DUT3\\{filename}_TEM.csv'
# csv_header = ','.join(('Time (s)','Amplitude (mV)'))
# ut.array_to_csv(csv_file, csv_header, cc, np.squeeze(vv))
csv_file = f'D:\\Mohamed Testing 3-21-2024\\Signals\\{filename}.csv'
csv_header = ','.join(('Time (s)','Amplitude (V)'))
ut.array_to_csv(csv_file, csv_header, xx, np.flipud(new_sig))
csv_file = f'D:\\Mohamed Testing 3-21-2024\\Signals\\{fn2}.csv'
csv_header = ','.join(('Time (s)','Amplitude (V)'))
ut.array_to_csv(csv_file, csv_header, nbt, nbsig)
awg.close()
scope.close()
del awg, scope