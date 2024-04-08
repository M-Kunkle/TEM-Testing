# -*- coding: utf-8 -*-
"""
Transfer Function Code

Created on Tue Mar 26 14:50:16 2024

@author: mkrxp
"""

import numpy as np
import signals as sigs
import utils as ut
import pyvisa
import interface
from pathlib import Path
import time

''' Working Directory '''
tfdir = Path('Results 03_27_2024\Transfer Funcs')

''' Signal Constants '''
START_FREQ = .7e9
STOP_FREQ = 2e9
FS = 50e9
AMPLITUDE = .5 # 250mV 
WFM_TYPE = 2
PULSE_WIDTH = 10e-9
SIGNAL_SAMPLES = 5000
START_FILT = .9e9
STOP_FILT = 2e9

''' Device IDs: '''
TEK_AWG70001B = 'B020541'
TEK_DPO71304SX = 'B322209'
KEY_FIELDFOX = 'MY61131330'
TEK_MSO = 'C014452'

''' Connect with AWG and Oscope '''
rm = pyvisa.ResourceManager()
awg_address = interface.get_address(TEK_AWG70001B, rm)
oscope_address = interface.get_address(TEK_DPO71304SX, rm)
scope, rm = interface.connect_scope(oscope_address, rm)
awg = interface.awg_connect(awg_address)
# scope.write('CLEAR ALL')

interface.trigger_setup_manual(scope, ch=2, level=.2, slope='RISE', rec_length=5000)
interface.vertical_scale_reset(scope, ch=3, ycenter=0, ydiv=0.00625, yoffset=0)
interface.vertical_scale_reset(scope, ch=4, ycenter=0, ydiv=0.05, yoffset=0)
interface.vertical_scale_reset(scope, ch=2, ycenter=0, ydiv=0.2, yoffset=0)

''' Generate Sinc and Activate AWG'''
t, sig_int = sigs.wfm_generate(2, START_FREQ, STOP_FREQ, FS, PULSE_WIDTH, sample_count=5000)
interface.awg_upload_wfm(awg, sig_int, .5, FS, 'current_iter.seqx')
interface.capture_single_sequence(scope)
interface.awg_fire(awg)
time.sleep(1)

''' Download and Plot Oscope Waveforms'''
DUT_t, DUT_wfm,_ = interface.get_waveform_from_scope(scope, ch=3)
# ut.signal_plot(DUT_t,np.squeeze(DUT_wfm),0,3e9,unit='V',titlestr='Oscope DUT ')
TEM_t, TEM_wfm,_ = interface.get_waveform_from_scope(scope, ch=4)
ut.signal_plot(TEM_t,np.squeeze(TEM_wfm),0,3E9,unit='V',titlestr='Oscope TEM ')

''' Filter Waveform '''
filt_wfm = sigs.wfm_filter(DUT_t, np.squeeze(DUT_wfm), START_FILT, STOP_FILT)
ut.signal_plot(DUT_t,np.squeeze(DUT_wfm),0,3e9,unit='V',titlestr='Filtered DUT ')

''' Calculate Transfer Function '''
f,fft_dut = sigs.rdwfft(DUT_t, filt_wfm)
ff,fft_tem = sigs.rdwfft(TEM_t, np.squeeze(TEM_wfm))
transfer_function = np.divide(fft_dut,fft_tem)
new_sig = sigs.rdwifft(f, transfer_function, DUT_t)
ut.quick_plot(f, 20*np.log10(abs(transfer_function)),title='Transfer Function',xlim=(.7e9,2e9),ylim=(-50,0))
ut.signal_plot(DUT_t,np.flipud(new_sig),.2e9,3e9, titlestr='TF Signal ')
tf_f, tf_fd = sigs.rdwfft(DUT_t, np.flipud(new_sig))

''' Locate Narrowband Frequency and Generate Narrowband Signal '''
nbfreq = ff[np.argmax(abs(transfer_function[1:200]))]
nbt, nbsig = sigs.wfm_generate(1, nbfreq-nbfreq*.005, nbfreq+nbfreq*.005, FS, 25e-9, SIGNAL_SAMPLES)
ut.signal_plot(nbt,nbsig,.2e9,3e9, titlestr='Narrowband ')

''' Create File Names'''
dut_num = input('Enter DUT Number: ')
fn1 = 'transferfunction' + dut_num + '.csv'
fn2 = 'narrowband' + dut_num + '.csv'

''' Save Data '''
csv_header = ','.join(('Time (s)','Amplitude (V)'))
ut.array_to_csv(tfdir / fn1, csv_header, DUT_t, np.flipud(new_sig))
ut.array_to_csv(tfdir / fn2, csv_header, nbt, nbsig)

''' Clear Instruments'''
awg.close()
scope.close()
rm.close()
del awg, scope, rm