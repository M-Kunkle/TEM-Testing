# -*- coding: utf-8 -*-
"""
Signal processing functions

@author: mkrxp
"""
import numpy as np
from numpy.fft import fft, fftfreq, ifft
from scipy.signal.windows import tukey, boxcar
from scipy.signal import gausspulse, chirp, butter, sosfilt
from math import floor, ceil
import utils as ut

def wfm_generate(wfm_type, start_freq, stop_freq, fs, pulse_width, sample_count):
    '''
    Generation of different waveforms for use as interrogation impulses or
    even for testing purposes. Does not include any zero padding.
    
    Parameters
    ----------
    wfm_type : 
        Selects signal waveform from: 1=narrowband, 2=sinc, 3=gauss, 4=chirp
    start_freq : 
        The starting frequency of the signal
    stop_freq : 
        The stopping frequency of the signal
    fs : 
        The frequency sampling rate, recall fs = 1 / (time between samples)
    pulse_width :
        The width of the signal in the time domain, should be FWHM

    Returns
    -------
    t :
        The time vector for the signal
    wfm :
        The time domain amplitude of the signal
        
    '''
    # calculate useful quantities and time vector
    bandwidth = stop_freq - start_freq
    center_freq = start_freq + (bandwidth / 2)
    dt = 1 / fs
    df = 1 / pulse_width
    t = np.arange(0, pulse_width, dt)
    
    # window used for smoothing wfm ends to remove high freq noise
    length = len(t)
    tukwin = tukey(length, 0.05)
    
    # 1 NARROWBAND
    if wfm_type == 1:
        square_pulse = boxcar(length)
        win_pulse = tukwin * square_pulse
        sig = np.cos(2 * np.pi * center_freq * t);
        narrowband = sig * win_pulse
        t = np.linspace(0, dt*len(narrowband) - dt, len(narrowband)) 
        wfm = narrowband / (max(np.abs(narrowband)))
        t,wfm = zeropad_to_num(t, wfm, sample_count)
        return t, wfm
    
    # 2 SINC IMPULSE
    elif wfm_type == 2:
        freq = np.arange(0, fs, df)
        n = pulse_width / dt

        pulse_f = [0] * len(freq)
        for idx, x in enumerate(freq):
            if x >= start_freq and x <= stop_freq:
                pulse_f[idx] = 1;
        # pulse_f = pulse_f + np.conj(np.flipud(pulse_f))
        pulse = np.real(ifft(pulse_f))
        wfm = np.concatenate((pulse[-int(floor(n/2)):], pulse[0:int(ceil(n/2))]));
        wfm = (wfm*tukwin) / max(np.abs(wfm*tukwin))
        wfm = (wfm) / max(np.abs(wfm))
        t,wfm = zeropad_to_num(t, ut.normalize(wfm), sample_count)
            
        return t, wfm
    
    # 3 GAUSSIAN PULSE
    elif wfm_type == 3:
        tt = t - (max(t)/2)
        bw = bandwidth / center_freq
        wfm = gausspulse(tt, center_freq, bw)
        t,wfm = zeropad_to_num(t, wfm, sample_count)
        return t, wfm
    
    # 4 CHIRP
    elif wfm_type == 4:
        tt = t - (max(t)/2)
        wfm = chirp(t, start_freq, t[-1], stop_freq)*tukwin
        t,wfm = zeropad_to_num(t, wfm, sample_count)
        return t, wfm
    
def zeropad_to_num(t, wfm, sample_count):
    if len(wfm) < sample_count:
        dt = t[1] - t[0]
        pad_length = sample_count-len(wfm)
        wfm = np.pad(wfm, pad_width=(0, pad_length), mode='constant')
        t = np.append(t, np.linspace(t[-1] + dt, pad_length*dt, pad_length))
    return t, wfm
    
    
def generate_spectrum(time, wfm, dB=1, norm=1):
    '''
    Creates a frequency domain representation of a time domain signal. Can
    be normalized and converted to dB if wanted.

    Parameters
    ----------
    time :
        Time vector of the signal
    wfm :
        Waveform of the signal
    dB :
        Converts result to dBm if true
    norm :
        Normalizes result if true

    Returns
    -------
    freq :
        Frequency vector of the spectrum
    spectrum :
        Waveform of the spectrum

    '''
    dt = time[1] - time[0]
    length = len(wfm)
    y = fft(np.real(wfm)) / length
    y = 2 * abs(y[1:ceil(length/2)])
    freq = fftfreq(length, dt)
    
    if dB == 1 and norm == 1:
        y = y / max(abs(y))
        y = 20 * np.log10(y)
    elif dB == 1 and norm == 0:
        y = ((y/np.sqrt(2))**2)/50
        y = 10*np.log10(y) + 30
    spectrum = y
        
    return freq[1:ceil(length/2)], spectrum
    
def butter_bandpass(lowcut, highcut, fs, order=5):
    ''' Helper function for butterworth (not really used) '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

def butter_bandpass_filter(time, wfm, lowcut, highcut, order=10):
    ''' Apply a butterworth bandpass filter, not in use at all in my code '''
    fs = 1 / (time[1] - time[0])
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered = sosfilt(sos, wfm)
    return filtered
    
def wfm_filter(time, wfm, low_freq, high_freq):
    '''
    Applies a bandpass filter to the signal. Time vector of filtered signal
    is identical to unfiltered, so only waveform is returned.

    Parameters
    ----------
    time :
        Time vector of unfiltered signal
    wfm :
        Waveform of unfiltered signal
    low_freq :
        Bottom cutoff frequency (-3dB)
    high_freq : TYPE
        Top cutoff frequency (-3dB)

    Returns
    -------
    filtered :
        New waveform of filtered signal

    '''
    fs = 1 / (time[1] - time[0])
    n = len(wfm)
    fft_freq = np.linspace(0, fs, n)
    
    if n % 2 == 0:
        fft_freq[int(n/2) + 1:] = fft_freq[int(n/2) + 1:] - fs
    else:
        fft_freq[int((n+1)/2) + 1:] = fft_freq[int((n+1)/2) + 1:] - fs
    
    h_freq = []
    low_idx = []
    
    for idx, x in enumerate(fft_freq):
        if x >= low_freq and x <= high_freq:
            h_freq.append(x)
            low_idx.append(idx)
            
    h_two_sided = [1] + ([0] * (len(fft_freq)-1))
    for i in low_idx:
        h_two_sided[i] = 1
        h_two_sided[len(h_two_sided)-i-1] = 1
    
    filtered = ifft(h_two_sided * fft(wfm))
    
    return filtered

def window(time, sig, t0=0, tf=10e-9, alpha=.25):
    ''' Apply a tukey window '''
    idx = ut.indices(time, lambda x: x >= t0 and x <= tf)
    tukwin = tukey(len(idx), alpha)
    windowed_signal = [0] * (len(time)-1)
    windowed_signal[idx[0]:idx[-1]] = [sig[x] for x in idx]*tukwin
    return windowed_signal

def reverse_signal(signal):
    ''' Better time complexity than reverse() '''
    size = len(signal)
    i = 0
    while(i<=size//2):
        signal[i],signal[size-i-1]=signal[size-i-1],signal[i]
        if((i!=i+1 and size-i-1 != size-i-2) and (i!=size-i-2 and size-i-1!=i+1)):
                signal[i+1],signal[size-i-2]=signal[size-i-2],signal[i+1]
        i += 2
    return signal

def add_pre_signal(t, wfm, start_freq, stop_freq, fs, pulse_width):
    wfm = wfm / max(abs(wfm))
    init_length = int(np.floor(len(t)*.05))
    _, s_pre = wfm_generate(2, start_freq, stop_freq, fs, pulse_width, sample_count=init_length)
    wfm[len(s_pre)-1:-1] = wfm[0:-len(s_pre)]
    wfm[0:len(s_pre)] = s_pre
    
    return wfm

def rdwfft(time_array, signal_amplitudes):
    # Compute the FFT of the signal
    spectrum = np.fft.fft(signal_amplitudes)
    # Calculate the corresponding frequency values
    dt = time_array[1] - time_array[0]
    frequency_array = np.fft.fftfreq(len(time_array), d=dt)
    return frequency_array, spectrum

def rdwifft(frequency_array, spectal_density, desired_time_array):
    num_samp = len(desired_time_array)
    init_shift = np.fft.ifft(spectal_density, n=num_samp, axis=0)
    new_signal = np.fft.ifftshift(init_shift)
    new_signal = new_signal.real
    return new_signal

def filter_spectrum(frequency_array, amplitude_array):
    freq_range = (frequency_array >= 1e9) & (frequency_array <= 2e9)
    filt_freq = frequency_array[freq_range]
    filt_amps = amplitude_array[freq_range]
    return filt_freq, filt_amps

if __name__ == '__main__':
    a,b = wfm_generate(2, 2e9, 3e9, 50e9, 20e-9, 4800)
    ut.signal_plot(a, b,1e9,8e9)
    #result = window(a,b,2e-9,10e-9)
    #result = reverse_signal(result)
    #ut.quick_plot(a, result)
    #ut.array_to_csv('test_waveform.csv', 'time, wfm', a,b)