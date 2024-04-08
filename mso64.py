# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 11:08:45 2024

@author: mkrxp
"""

import numpy as np
import time
import pyvisa

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
# Oscilloscope ################################################################
###############################################################################
'''

def connect_scope(address, rm, time_out=2500):
    """Connect to the scope and check it's error status. This function will
       stop your code if it detects ANY error on the scope"""
       
    scope = rm.open_resource(address, timeout=time_out) # Timeout set for 30 seconds
    print('Scope name: ' + scope.query('*IDN?')) # Ask the scope to identify itself
    time.sleep(0.25)
    scope.write('*CLS')
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
    
    # scope.write('TRIG:MAIN:EDGE:SOU CH' + str(ch) + ';*OPC')
    # scope.write('TRIG:MAIN:EDGE:SLOP ' + slope + ';*OPC')
    scope.write('TRIG:A:EDGE:SOU CH' + str(ch))
    scope.write('TRIG:A:LEVEL:CH4 ')+ str(level)
    # scope.write('TRIG:MAIN:LEVEL ' + str(level) + ';*OPC')
    scope.write('HORizontal:MODE MANUAL' + ';*OPC')
    scope.write('HORizontal:MODE:RECOrdlengt ' + str(rec_length) + ';*OPC')
    scope.write('HORizontal:MODE:SAMPLERate ' + str(fs) + ';*OPC')
    print('Trigger set to CH')
    
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

    signal = (signal * y_scale) + y_offset # Convert from +/- 0-128 to a decimal number
    fs = float(scope.read_raw(scope.write("HORizontal:MODE:SAMPLERate?"))) # Get sample rate
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

    
    rm = pyvisa.ResourceManager()
    oscope_address, rm = get_address('B322209', rm)
    
    scope = connect_scope(oscope_address, rm)
    
    signal, fs, t = get_waveform_from_scope(scope)
    
    scope.close()
    del scope
    