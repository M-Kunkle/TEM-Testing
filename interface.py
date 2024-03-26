# -*- coding: utf-8 -*-
"""
Interface functions for the oscope and the awg

@author: Simeon Karnes, mkrxp
"""

import numpy as np
import os, sys, time
from qcodes.instrument_drivers.tektronix import TektronixAWG70001B
import pyvisa
import sys
import utils as ut

def get_address(device_id, rm):
    ''' Returns the USB address of device'''
    
    device_list = rm.list_resources();
    for i in device_list:
        if i.find(device_id) > -1:
            return i
        
    raise Exception("Device not found")
    
    return

'''
###############################################################################
# AWG #########################################################################
###############################################################################
'''

def awg_connect(awg_address, log=False):
    ''' Connect to the awg '''
    
    awg = TektronixAWG70001B(name='awg', address=awg_address, timeout=5)
    # Print out current parameters of the AWG
    if log:
        awg.ch1.print_readable_snapshot(update=True)
    awg.mode('AWG') # Set AWG mode to AWG
    
    return awg
    
def awg_clean_up(awg):
    '''
    This function is a "reset" to clean out all waveforms and sequences from
    the AWG for a new setup.
    '''
    # Change the system mode to "AWG"
    awg.mode('AWG')
    # Set channel 1 resolution. Bits option (8, 9, or 10)
    bits = 9
    awg.ch1.resolution(bits)
    # Set channel 1 amplitude (Still figuring this out really, it does some weird scaling)
    awg.ch1.awg_amplitude(0.5)
    # Clear the sequences list
    awg.clearSequenceList()
    # Clear the waveforms list out from the waveforms tab on the right of the screen
    awg.clearWaveformList()
    print("================================================AWG Clean up done!")
    return

def awg_upload_wfm(awg, signal, amp_vpp, SR, signal_name,
                   awg_location=r'C:\Users\tek_local_admin\Desktop\TempWaveforms/',
                   clean=False):
    ''' NOTE TO SELF: Probs turn the seqx file creation into its own function '''
    if (len(signal) % 2) != 0:
        signal = signal[:-1]
    
    signal = 0.5*amp_vpp* (signal / max(abs(signal)))
    
    
    if clean == True:
        awg_clean_up()
        
    if amp_vpp > 0.500:
        print('Error: Signal amplitude is too high for AWG to output. Quiting code.')
        sys.exit()
    elif amp_vpp < 0.250:
        print('Error: Signal amplitude is too low for AWG to output. Quiting code.')
        sys.exit()
        
    awg.sample_rate(SR) # Set AWG sample rate (dt for arbitrary signal = 1/SR[Hz])
    awg.ch1.awg_amplitude(amp_vpp) # Set AWG amplitude to 100 mVpp
    ch1_amp = awg.ch1.awg_amplitude() # Ask AWG it's amplitude
    
    m1_1 = np.ones((2000,))
    m1_0 = np.zeros((len(signal)-2000),)
    m1 = np.concatenate((m1_1, m1_0)) # Create marker columns of zeros
    m2 = np.zeros((len(signal),)) #
    m2 = np.zeros((len(signal),)) #
    
    wfm_ch1_n1 = np.array([signal, m1, m2]) # Append signal and markers to list
    
    ### Trigger and Sequence Parameters ###
    trig_waits = [1]  # 0: off, 1: trigA, 2: trigB, 3: EXT
    nreps = [1]  # 0 corresponds to infinite
    event_jumps = [0] # 0: off, 1: trigA, 2: trigB, 3: EXT
    event_jump_to = [0]  # irrelevant if event-jump is 0, else the sequence pos. to jump to
    go_to = [0]  # 0 means next
    wfms = [[wfm_ch1_n1]]
     
    # Create .seqx file
    seqx = awg.makeSEQXFile(trig_waits,
                            nreps,
                            event_jumps,
                            event_jump_to,
                            go_to,
                            wfms,
                            [ch1_amp],
                            signal_name)
    
    awg.sendSEQXFile(seqx, os.path.join(awg_location,signal_name)) # Send trigger and sequence parameters to the AWG
    awg.loadSEQXFile(signal_name,awg_location[2:]) # Load sequence to the "Sequences" tab
    awg.ch1.setSequenceTrack(signal_name, 1) # Assign channel 1 the sequence
    
    return 

def awg_fire(awg):
    awg.ch1.state(1) # Turn channel 1 output on ("safety off")
    awg.play() # Play channel 1, but it is waiting on trigger A
    awg.force_triggerA() # Trigger A. This is the single output shot
    #time.sleep(.01) # Wait one second for output.
    awg.stop()
    awg.ch1.state(0) # Turn channel 1 output off ("safety on")
    print('AWG Fired')
    
    return
    

def awg_single_shot_waveform(awg, signal, amp_vpp, SR, signal_name, awg_location=r'C:\Users\tek_local_admin\Desktop\TempWaveforms/', clean=False):
    '''
    Send the Tektronix AWG70001B Arbitrary waveform generator
    an arbitrary wave and fire a single output.
       
   Notes:
       ***THIS FUNCTION WILL CLEAR OUT ALL WAVEFORMS AND SEQUENCES BEFORE
       AND AFTER RUNNING.
       
       ***THIS FUNCTION WILL FIRE WHEN RAN***
       
       ***THIS FUNCTION WILL OVERWRITE SEQUENCES ALREADY ON THE AWG***
   
   Inputs:
       signal:
           A column vector of amplitude N points (no time column).
           N must be >= 4800 points
           signal shape should be (N,)
           This function will normalize the amplitude to 1 and then scale
           by amp_vpp.
           ***This has only been tested on sinusoids***
           
       amp_vpp:
           The peak-to-peak amplitude of the output signal.
           This must be entered in millivolts and has upp and lower limits
           0.250 V <= amp_vpp <= 0.500 V
           ***If your signal is asymetric about the vertical axis, amp_vpp 
           should be equal to 2*max(abs(signal)) to scale correctly. This
           is still a work in progress.***
           
       SR:
           The sample rate at which the AWG needs to produce time steps.
           Thinks of this as providing the dt of your time vector for the
           signal. Units of SR are in Hz.
           
           ex:
               SR=50E9 would be a 50 GHz sample rate

       signal_name:
           The name you wish to give your sequence file as a string.
              ex:
                  signal_name = "Example_Signal_Name.seqx"

       awg_location:
           The file directory you wish to store your sequence file.
              ex:
                  r"C:/Users/tek_local_admin/Desktop/Example_Folder/"
       clean:
           If True, this will clear all waveform and sequences already
           loaded on to the AWG. Default parameter state is False.
           
        Outputs:
       None
    '''
     
    awg.sample_rate(SR) # Set AWG sample rate (dt for arbitrary signal = 1/SR[Hz])
    awg.ch1.awg_amplitude(amp_vpp) # Set AWG amplitude to 100 mVpp
    ch1_amp = awg.ch1.awg_amplitude() # Ask AWG it's amplitude
    
    m1 = np.zeros((len(signal),)) # Create marker columns of zeros
    m2 = np.zeros((len(signal),)) #
    
    wfm_ch1_n1 = np.array([signal, m1, m2]) # Append signal and markers to list
    
    ### Trigger and Sequence Parameters ###
    trig_waits = [1]  # 0: off, 1: trigA, 2: trigB, 3: EXT
    nreps = [1]  # 0 corresponds to infinite
    event_jumps = [0] # 0: off, 1: trigA, 2: trigB, 3: EXT
    event_jump_to = [0]  # irrelevant if event-jump is 0, else the sequence pos. to jump to
    go_to = [0]  # 0 means next
    wfms = [[wfm_ch1_n1]]
     
    # Create .seqx file
    seqx = awg.makeSEQXFile(trig_waits,
                            nreps,
                            event_jumps,
                            event_jump_to,
                            go_to,
                            wfms,
                            [ch1_amp],
                            signal_name)
    
    awg.sendSEQXFile(seqx, os.path.join(awg_location,signal_name)) # Send trigger and sequence parameters to the AWG
    awg.loadSEQXFile(signal_name,awg_location[2:]) # Load sequence to the "Sequences" tab
    awg.ch1.setSequenceTrack(signal_name, 1) # Assign channel 1 the sequence
    awg.ch1.state(1) # Turn channel 1 output on ("safety off")
    awg.play() # Play channel 1, but it is waiting on trigger A
    awg.force_triggerA() # Trigger A. This is the single output shot
    #time.sleep(1) # Wait one second for output.
    awg.stop()
    awg.ch1.state(0) # Turn channel 1 output off ("safety on")
    
    return

'''
###############################################################################
# Oscilloscope ################################################################
###############################################################################
'''

def connect_scope(address, rm, time_out=2500):
    """Connect to the scope and check it's error status. This function will
       stop your code if it detects ANY error on the scope"""
    
       
    scope = rm.open_resource(address, timeout=time_out) # Timeout set for 30 seconds
    #print('Scope timeout: ' + str(time_out) + 'seconds')
    # scope.write('*RST')
    print('Scope name: ' + scope.query('*IDN?')) # Ask the scope to identify itself
    # print('Checking for errors...')
    time.sleep(0.25)
    scope.write('*CLS')
    # scope.write("*ESR?") # Ask the scope if it has any errors to report
    time.sleep(1) # Wait one seconds after trigger
    # error_check = scope.read_raw()
    # scope.write('EVENT?')
    # time.sleep(1)
    # event = int.from_bytes(scope.read_raw(), "little")
    # # print(event)
    
    
    # if int.from_bytes(error_check, "little") == 0:
    #     pass #print('No errors detected')
    # else:
        
    #     print('ERROR DETECTED!!!')
    #     print('Error code: ' + str(int.from_bytes(error_check, "little")))
    #     print('Closing connection and stopping code...')
    #     scope.close() # End communication with the scope
    #     sys.exit()       
    return scope, rm

def vertical_scale_reset(scope, ch=1, ycenter=0, ydiv=0.01, yoffset=0):
    """Sets up the vertical scale on the Tektronic DPO7000SX series O-scope"""
    
    scope.write('CH'+str(ch)+':SCALE ' +str(ydiv)) # Set the volts/div on the vertical axis of the scope
    scope.write('CH'+str(ch)+':POSITION ' +str(ycenter)) # sets the center position shown on the scope
    scope.write('CH'+str(ch)+':OFFSET ' +str(yoffset)) # removes any artificial offset on the scope
    
    return

def trigger_setup_manual(scope, ch=1, level=1E-2, slope='RISE', fs=50E9, rec_length=5000):
    '''Sets the trigger parameters to catch a waveform in manual mode with
       constant sample rates and a set number of data points.
       
       Notes:
       
       Inputs:
           ch:
               Enter a integer to command which channel you are setting as the
               trigger.
               
           level:
               Sets the trigger level in volts.
           
           slope:
               The trigger slope can be either RISE, FALL, or EITHER.
               
           fs:
               Sample rate of the scope in Hz.
               
           rec_length:
               Sets the number of data points to take per frame
                              
       Outputs:
           None
           '''
    
    scope.write('TRIG:MAIN:EDGE:SOU CH' + str(ch) + ';*OPC')
    scope.write('TRIG:MAIN:EDGE:SLOP ' + slope + ';*OPC')
    scope.write('TRIG:MAIN:LEVEL ' + str(level) + ';*OPC')
    scope.write('HORizontal:MODE MANUAL' + ';*OPC')
    scope.write('HORizontal:MODE:RECOrdlengt ' + str(rec_length) + ';*OPC')
    scope.write('HORizontal:MODE:SAMPLERate ' + str(fs) + ';*OPC')
    print('Trigger set to CH ' + str(ch) + ' at ' + str(level) + ' volts on a ' + slope + ' edge')
    
    return

def capture_single_sequence(scope):
    '''Sets up the scope to prepare a single sequence screen grab.
       
       Notes:
           *** Waits one second before returning in order to prevent the 
           waveform from running before the scope is ready.
    '''
    
    print('Stopping refresh')
    scope.write('ACQ:STATE STOP') #Stops the scope from running
    print('Setting to single sequence')
    scope.write('ACQuire:STOPAfter SEQuence')   # Sets the scope to catch one frame on trigger
    print('Trigger Ready')
    scope.write('ACQ:STATE RUN')    # The scope is now ready to catch 1 frame upon trigger
    time.sleep(0.1) # Wait one seconds between run and trigger
    
    return

def force_trigger(scope):
    '''Force trigger on the scope.
       
       Notes:
           Works like a typical force trigger button.
    '''
    scope.write('TRIG FORCE')    # Force the trigger to catch a single frame
    print('Trigger forced!')
    
    return

def get_waveform_from_scope(scope, rec_length=50000, ch=1):
    '''Pulls the waveform currently on the specified channel from the scope and
       downloads it to the host computer.
       
       Notes:
           *This function will only download one channel at a time.
           *This function will download the entire series regardless of zooming.
       
       Inputs:
           Ch:
               Specifies the channel to pull the waveform from. This will only
               work for a single channel at a time. Defaults to channel 1.
                              
       Outputs:
           signal:
               The voltage signal as a numpy array returned as a column vector.
               
           fs:
               The sample rate of the data points from the scope.
               *Note: fs = 1/dt and/or dt = 1/fs
               
           time_ns:
               Time vector in nano-seconds to make plotting easier.
           '''
    scope.write('DATa:SOUrce CH' + str(ch)) # Set record source
    scope.write('DATA:ENCdg:ASCII') # Set data encoding to Binary
    time.sleep(0.1)
    N = rec_length
    scope.write('DATA:START 1') # Set initial data point of record to pull
    time.sleep(.01)
    scope.write('DATA:STOP ' + str(int(N))) # Grab the entire record
    time.sleep(.01)
    print(scope.query('WFMOutpre?'))
    # bytes_waveform = scope.query_binary_values('CURVE?')
    time.sleep(0.1)
    scope.write('CURV?')
    bytes_waveform = scope.read_raw() # Get waveform from scope
    hex_waveform = bytes_waveform.hex() # Call the hex values of the waveform
    hex_waveform = bytes.fromhex(hex_waveform) # Call hex data from bytes data
    signal = np.frombuffer(hex_waveform, dtype=np.uint8) # Convert hex data to 0-255 unsigned integer of 8 bits
    signal = signal.reshape((len(signal),1)) # Reshape to a column vector
    signal = signal.astype(np.float16) # Convert to float to do math on
    signal = uint8_to_signed_float(signal) # Shift values to account for +/-
    signal = signal[7:int(len(bytes_waveform))-1] # The first 6 and last 1 data point are irrelevant
    time.sleep(.01)
    
    y_scale = float(scope.read_raw(scope.write('WFMOutpre:YMUlt?'))) # Get y-scale to convert to decimal
    y_offset = float(scope.read_raw(scope.write('CH1:OFFSET?'))) # Check for y-axis offset
    # y_zero = float(scope.read_raw(scope.write('WFMOutpre:YZERO?'))) 

    signal = (signal * y_scale) + y_offset # Convert from +/- 0-128 to a decimal number
    fs = float(scope.read_raw(scope.write("HORizontal:MODE:SAMPLERate?"))) # Get sample rate
    # time_ns = 1E9*np.arange(0, (len(signal[:,0]))*(1/fs), 1/fs) # Create artificial time column
    time_ns = np.linspace(0, len(signal)*(1/fs), len(signal))
    
    
    return time_ns, signal, fs


def uint8_to_signed_float(uint8_data):
    '''Applies a sign to an unsigned 8-bit integer value
       
       Notes:
           ***Must be passed in as a float type data
       
       Inputs:
           uint8_data:
               A column or row vector or list containing float data integers
               unsigned.
                              
       Outputs:
           uint8_data:
               Returns the same vector, but with signs and shifted.
           '''
    
    for i in range(0,len(uint8_data)):
        if uint8_data[i] > 128:
            uint8_data[i] = uint8_data[i] - 256 
        else:
            continue
        
    return uint8_data

'''
###############################################################################
# Testing ####################################################################
###############################################################################
'''

if __name__ == '__main__':
    import utils as ut
    
    tic,toc = ut.tictoc()
    tic()
    
    rm = pyvisa.ResourceManager()
    
    awg_address = get_address('B020541', rm)
    oscope_address = get_address('B322209', rm)
    
    scope = connect_scope(oscope_address, rm)
    awg = awg_connect(awg_address)
    
    signal, fs, t = get_waveform_from_scope(scope)
    awg_clean_up(awg)
    
    scope.close()
    del awg
    print(toc())
    