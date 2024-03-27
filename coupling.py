# -*- coding: utf-8 -*-
"""
Coupling Excitation Code

Created on Tue Mar 26 16:35:38 2024

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
coupdir = Path('Results 03_27_2024\Coupling')

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
interface.trigger_setup_manual(scope, ch=3, level=.2, slope='RISE', rec_length=5000)
interface.vertical_scale_reset(scope, ch=1, ycenter=0, ydiv=0.02, yoffset=0)
interface.vertical_scale_reset(scope, ch=2, ycenter=0, ydiv=0.05, yoffset=0)
interface.vertical_scale_reset(scope, ch=3, ycenter=0, ydiv=0.2, yoffset=0)


''' Upload Excitation Signals '''
wfm_num = input('Enter Waveform Number: ')
# tf_fn = 'transferfunction' + wfm_num + '.csv'
tf_fn = 'hec_last.csv'
nb_fn = 'narrowband' + wfm_num + '.csv'
tft,tf_sig = ut.read_cst_csv(tfdir / tf_fn,1)
# tft = np.loadtxt(tfdir / tf_fn)
# nbt,nb_sig = ut.read_cst_csv(tfdir / nb_fn,1)

''' Transfer Function Shot'''
interface.awg_upload_wfm(awg, np.asarray(tf_sig), .5, FS, 'current_iter.seqx')
interface.capture_single_sequence(scope)
interface.awg_fire(awg)
time.sleep(1)

tf_time, tf_DUT_wfm,_ = interface.get_waveform_from_scope(scope, ch=1)
_, tf_TEM_wfm,_ = interface.get_waveform_from_scope(scope, ch=2)

filt_tfdut = sigs.wfm_filter(tf_time, np.squeeze(tf_DUT_wfm), .7e9, 2e9)
filt_tftem = sigs.wfm_filter(tf_time, np.squeeze(tf_TEM_wfm), .7e9, 2e9)
ut.signal_plot(tf_time,np.squeeze(filt_tfdut),0,3e9,unit='V', titlestr='TF DUT ')
ut.signal_plot(tf_time,np.squeeze(filt_tftem),0,3E9,unit='V', titlestr='TF TEM ')

# ''' Narrowband Shot'''
# interface.capture_single_sequence(scope)
# interface.awg_upload_wfm(awg, np.asarray(nb_sig), .5, FS, 'current_iter.seqx')
# interface.awg_fire(awg)
# time.sleep(1)

# nb_time, nb_DUT_wfm,_ = interface.get_waveform_from_scope(scope, ch=1)
# _, nb_TEM_wfm,_ = interface.get_waveform_from_scope(scope, ch=2)

# filt_nbdut = sigs.wfm_filter(nb_time, np.squeeze(nb_DUT_wfm), .7e9, 2e9)
# filt_nbtem = sigs.wfm_filter(nb_time, np.squeeze(nb_TEM_wfm), .7e9, 2e9)
# ut.signal_plot(nb_time,np.squeeze(filt_nbdut),0,3e9,unit='V', titlestr='NB DUT ')
# ut.signal_plot(nb_time,np.squeeze(filt_nbtem),0,3E9,unit='V', titlestr='NB TEM ')

''' Create File Names'''
dut_num = input('Enter DUT Number: ')
fn1 = 'DUT' + dut_num + '_' + wfm_num + '.csv'
# fn2 = 'DUT' + dut_num + '_NB' +  wfm_num + '.csv'

''' Save Data '''
csv_header = ','.join(('Time (s)','DUT (V)', 'TEM (V)'))
ut.array_to_csv(coupdir / fn1, csv_header, tf_time, filt_tfdut.real, filt_tftem.real)
# ut.array_to_csv(coupdir / fn2, csv_header, nb_time, filt_nbtem.real, filt_nbtem.real)

''' Clear Instruments'''
awg.close()
scope.close()
rm.close()
del awg, scope, rm