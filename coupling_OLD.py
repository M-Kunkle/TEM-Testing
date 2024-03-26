# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 12:41:51 2024

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

# connect with awg and oscope
rm = pyvisa.ResourceManager()
awg_address = interface.get_address(TEK_AWG70001B, rm)
oscope_address = interface.get_address(TEK_DPO71304SX, rm)
scope, rm = interface.connect_scope(oscope_address, rm)
awg = interface.awg_connect(awg_address)
interface.trigger_setup_manual(scope, ch=3, level=.2, slope='RISE', rec_length=5000)
interface.vertical_scale_reset(scope, ch=1, ycenter=0, ydiv=0.02, yoffset=0)
interface.vertical_scale_reset(scope, ch=2, ycenter=0, ydiv=0.05, yoffset=0)
interface.vertical_scale_reset(scope, ch=3, ycenter=0, ydiv=0.2, yoffset=0)

t,sig = ut.read_cst_csv('D:\\Mohamed Testing 3-21-2024\\Signals\\transferfunction4.csv',1)

interface.awg_upload_wfm(awg, np.asarray(sig[:-1]), .5, FS, 'current_iter.seqx')
interface.capture_single_sequence(scope)
interface.awg_fire(awg)

xx,zz,_ = interface.get_waveform_from_scope(scope, ch=1)
cc,vv,_ = interface.get_waveform_from_scope(scope, ch=2)

filtzz = sigs.wfm_filter(xx, np.squeeze(zz), .7e9, 2e9)
filtvv = sigs.wfm_filter(xx, np.squeeze(vv), .7e9, 2e9)
ut.signal_plot(xx,np.squeeze(filtzz),0,3e9,unit='V')
ut.signal_plot(cc,np.squeeze(filtvv),0,3E9,unit='V')

filename = 'DUT10_WC4_'

csv_file = f'D:\\Mohamed Testing 3-21-2024\\Results\\{filename}.csv'
csv_header = ','.join(('Time (s)','DUT (V)', 'TEM (V)'))
ut.array_to_csv(csv_file, csv_header, xx, filtzz.real, filtvv.real)

# t,sig = ut.read_cst_csv('D:\\Mohamed Testing 3-21-2024\\Signals\\narrowband4.csv',1)

# interface.capture_single_sequence(scope)
# interface.awg_upload_wfm(awg, np.asarray(sig[:]), .5, FS, 'current_iter.seqx')
# interface.awg_fire(awg)


# xx,zz,_ = interface.get_waveform_from_scope(scope, ch=1)
# cc,vv,_ = interface.get_waveform_from_scope(scope, ch=2)

# filtzz = sigs.wfm_filter(xx, np.squeeze(zz), .7e9, 2e9)
# filtvv = sigs.wfm_filter(xx, np.squeeze(vv), .7e9, 2e9)
# ut.signal_plot(xx,np.squeeze(filtzz),0,3e9,unit='V')
# ut.signal_plot(cc,np.squeeze(filtvv),0,3E9,unit='V')

# filename = 'DUT10_NB4_'

# csv_file = f'D:\\Mohamed Testing 3-21-2024\\Results\\{filename}.csv'
# csv_header = ','.join(('Time (s)','DUT (V)', 'TEM (V)'))
# ut.array_to_csv(csv_file, csv_header, xx, filtzz.real, filtvv.real)

awg.close()
scope.close()
del awg
del scope