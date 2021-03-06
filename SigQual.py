# -*- coding: utf-8 -*-
"""
Copyright (C) 2020, Mahdad Jafarzadeh

THIS IS A CLASS FOR ANALYZING THE QUALITY OF SIGNAL.

Using this class, one can compare the quality of signals simultaneously 
recorded different devices.

"""
# =============================== import libs =============================== #

import mne
import numpy as np
from scipy.signal import butter, lfilter, periodogram, spectrogram, welch, filtfilt, iirnotch
import matplotlib.pyplot as plt
from scipy import signal

# =============================== defining calss ============================ #
class SigQual:
    
    # ~~~~~~~~~~~~~~~~~~~~~ Data and info reading section ~~~~~~~~~~~~~~~~~~~ #
    #%% Read text
    def read_txt(self,main_path, file_name, dtype = 'str',delimiter='\n'):
        from numpy import loadtxt
        
        # Check if the input has the ".txt"
        if (main_path + file_name)[-4:] =='.txt':   
            output_file = loadtxt(main_path + file_name         , dtype = dtype, delimiter = delimiter)
       
        # Otherwise
        else: 
            output_file = loadtxt(main_path + file_name + ".txt", dtype = dtype, delimiter = delimiter)
        
        return output_file
    #%% Read Excel file
    def read_excel(self, main_path, filename, fileformat = '.xlsx'):
        import pandas as pd
        Output = pd.read_excel(main_path + filename + fileformat)

        return Output
    #%% Read EDF
    def read_edf_file(self, path_folder, filename, preload = True):
        
        if (path_folder + filename)[-4:]=='.edf':
            raw_data = mne.io.read_raw_edf(path_folder + filename         , preload = preload)
            
        else:
            raw_data = mne.io.read_raw_edf(path_folder + filename + ".edf", preload = preload)
        
        return raw_data
    
    #%% Extract EDF info
    def edf_info(self, data):
        
        # Extract all info
        Info = data.info
        
        # extract fs
        fs = int(Info['sfreq'])
        
        # extract available channels
        availableChannels = Info['ch_names']
        
        return Info, fs, availableChannels
    
    # ~~~~~~~~~~~~~~~~~~~~ Filter and pre-process section ~~~~~~~~~~~~~~~~~~~ #
    #%% Notch-filter    
    def NotchFilter(self, data, Fs, f0, Q):
        w0 = f0/(Fs/2)
        b, a = iirnotch(w0, Q)
        y = filtfilt(b, a, data)
        return y
    
    #%% Low-pass butterworth
    def butter_lowpass_filter(self, data, cutoff, fs, order=2):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y    
    
    #%% Band-pass Filtering section
    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order = 2):
        nyq = 0.5 * fs
        low = lowcut /nyq
        high = highcut/nyq
        b, a = butter(order, [low, high], btype='band')
        #print(b,a)
        y = filtfilt(b, a, data)
        return y
    
    #%% high-pass Filtering section
    def butter_highpass_filter(self, data, highcut, fs, order):
        nyq = 0.5 * fs
        high = highcut/nyq
        b, a = butter(order, high, btype='highpass')
        y = filtfilt(b, a, data)
        return y
    #%% mne object filter
    
    def mne_obj_filter(self, data, l_freq, h_freq):
        
        filtered_sig = data.filter(l_freq=l_freq, h_freq=h_freq)
        
        return filtered_sig
        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~ Resampling section ~~~~~~~~~~~~~~~~~~~~~~~~~ #
    #%% Resampling higher freq to lower
    def resample_data(self, data1, data2, fs1, fs2):
        if fs1 != fs2:
            
            if fs1 < fs2:
                data2 = data2.resample(int(fs1), npad="auto")
                
            else:
                data1 = data1.resample(int(fs2), npad="auto")
                
        # Define resampled fs
        fs_res = np.min([fs1, fs2])
        
        return fs_res, data1, data2
    
    #%% Synchronization section
    def sync_data(self, fs_res, LRLR_start_iBand, LRLR_end_iBand, LRLR_start_somno, LRLR_end_somno,\
                  data_R_resampled_filtered, data_L_resampled_filtered, \
                  EEG_somno_resampled_filtered, AvailableChannels, subj_night, save_name, \
                  RequiredChannels = ['F4'], Ref_channel = ['A1'], save_fig = False, dpi = 1000,\
                  save_dir = "F:\Zmax_Data\Results\SignalQualityAnalysis",
                  report_pearson_corr_during_sync  = False,\
                  report_spearman_corr_during_sync = False,\
                  plot_cross_corr_lag = True, scale_somno = 1,\
                  amplitude_range = [-1000e-6, 1000e-6]):
        
        """ Please note: the event detection should be preliminary given to algorithm 
        by visual inspection.
        
        events can be blinks, eye movements, etc.
        """
        # ===================== start of LRLR for sync ========================= #
    
        # Headband
        LRLR_start_iBand = LRLR_start_iBand   #sec
        LRLR_end_iBand   = LRLR_end_iBand     #sec
        
        # Somno
        LRLR_start_somno = LRLR_start_somno #sec
        LRLR_end_somno   = LRLR_end_somno   #sec
        
        # Define a period around sync point ro perform alignment
        iBand_plotting_secs = [LRLR_start_iBand,LRLR_end_iBand]
        somno_plotting_secs = [LRLR_start_somno, LRLR_end_somno]
        
        # Finding corresponding samples of sync period
        iBand_plotting_samples  = np.arange(iBand_plotting_secs[0] *fs_res, iBand_plotting_secs[1] * fs_res)
        somno_plotting_samples = np.arange(somno_plotting_secs[0] *fs_res, somno_plotting_secs[1] * fs_res)
        
        # Convert (probable) floats into int
        somno_plotting_samples = somno_plotting_samples.astype(np.int32)
        iBand_plotting_samples  = iBand_plotting_samples.astype(np.int32)
        
        # R EEG (Zmax) --> sync period
        zmax_data_R = np.ravel(data_R_resampled_filtered)
        
        # L EEG (Zmax) --> sync period
        zmax_data_L = np.ravel(data_L_resampled_filtered)
        
        # Define channel of interest
        RequiredChannels  = RequiredChannels # main electrodes
        
        # init index of reeuired channel(s)   
        Idx               = []
        
        
        # Initializing index lists
        Idx = []
        Idx_Ref_channel = []
           
        # Find index of required channels     
        for indx, c in enumerate(RequiredChannels):
            if c in AvailableChannels:
                Idx.append(AvailableChannels.index(c))
                
        # Find index of refernces (e.g. Mastoids) 
                
        if Ref_channel:
            for indx, c in enumerate(Ref_channel):
                if c in AvailableChannels:
                    Idx_Ref_channel.append(AvailableChannels.index(c))
                
        # pick Somno channel
        if Ref_channel:
            Somno_reqChannel = EEG_somno_resampled_filtered[Idx,:] - EEG_somno_resampled_filtered[Idx_Ref_channel,:]
        else:
            Somno_reqChannel = EEG_somno_resampled_filtered[Idx,:] 
        
        # np.ravel somno signal(s)
        Somno_reqChannel = np.ravel(Somno_reqChannel)
        
        # plt R EEG (zmax) and required channel of Somno BEFORE sync
        plt.figure()
        figure = plt.gcf()  # get current figure
        plt.xlabel('Samples',size = 15)
        plt.ylabel('Amp',size = 15)
        figure.set_size_inches(32, 18)
        
        sig_iBand    = zmax_data_R[iBand_plotting_samples]
        sig_somno    = Somno_reqChannel[somno_plotting_samples]
        
        # Compute correlation
        corr = signal.correlate(sig_iBand, sig_somno)
        
        # find lag
        lag = np.argmax(np.abs(corr)) - len(zmax_data_R[iBand_plotting_samples]) + 1
        
        # Plot before lag correction
        plt.plot(np.arange(0, len(iBand_plotting_samples)), sig_iBand,label = 'Headband EEG', color = 'black')
        #plt.plot(np.arange(0, len(somno_plotting_samples)), sig_somno, label = 'Somno F3-F4', color = 'gray', linestyle = ':')
        plt.title('Syncing Somno and Headband data (Sync period only) - '+str(subj_night), size = 15)
        
        # Plot after lag correction
        #plt.plot(np.arange(0+lag, len(somno_plotting_samples)+lag), sig_somno, label = 'Somno F4 - synced',color = 'red')
        plt.plot(np.arange(0, len(somno_plotting_samples)), scale_somno*Somno_reqChannel[somno_plotting_samples-lag], label = 'Synced Somno '+str(RequiredChannels)[2:-2]+'-'+str(Ref_channel)[2:-2],color = 'red')
        #plt.plot(np.arange(0-lag, len(zmax_plotting_samples)-lag), sig_zmax, label = 'zmax - synced',color = 'cyan')
        
        plt.legend(prop={"size":20})
        
        # Show ylim
        if amplitude_range != None:
            plt.ylim((amplitude_range[0], amplitude_range[1]))
        # Save figure
        if save_fig == True:
            self.save_figure(directory=save_dir, saving_name= save_name,
                         dpi=dpi, saving_format = '.png',
                         full_screen = False)
            
        # Retrieving synced signals
        sig1 = Somno_reqChannel[somno_plotting_samples-lag]
        sig2 = sig_iBand
        
        # Report Pearson correlations during sync period
        if report_pearson_corr_during_sync == True:
            
            self.pearson_corr(sig1, sig2)
        
        # Report spearman correlations during sync period
        if report_spearman_corr_during_sync == True:

            self.spearman_corr(sig1, sig2)
        
        # Plot the cross-corr by which the lag was found
        if plot_cross_corr_lag == True:
            
            fig, ax = plt.subplots(1,1, figsize=(26, 14))
    
            ax.plot(np.arange(-len(zmax_data_R[iBand_plotting_samples])+1,len(zmax_data_R[iBand_plotting_samples])), corr, color = 'blue')
            plt.title('Cross-correlation to find lag between Zmax & Somno during eye movements', size=15)
            
            # Marking max correlation value to find lag
            ymax = np.max(np.abs(corr)) 
            
            # If negative peak, put the arrow below it
            if np.max(np.abs(corr)) != np.max(corr) :
                ymax = -ymax
                
            xpos = lag
            xmax = lag
            
            # Creating arrow to point to max
            ax.annotate('max correlation', xy=(xmax, ymax), xytext=(xmax, ymax+ymax/10),
                        arrowprops=dict(facecolor='red', shrink=0.05),
                        )
            
            # title, etc
            plt.title('Cross-correlation during event emergence', size = 20)
            plt.xlabel('Lag (samples)', size = 15)
            plt.ylabel('Amplitude', size = 15)
            plt.show()
            
        return lag, corr, Somno_reqChannel, zmax_data_R, sig1, sig2
            
    #%% Pearson correlation
            
    def pearson_corr(self, sig1, sig2, abs_value = True, print_results = True):
        
        from scipy.stats import pearsonr
        
        try:
            pearson_corr,pval = pearsonr(sig1, sig2)
            
        except TypeError:
            pearson_corr,pval = pearsonr(np.ravel(sig1), np.ravel(sig2))

        
        # calculate absolute corr if needed:
        if abs_value == True:
            pearson_corr = np.abs(pearson_corr)
        
        if print_results == True:
            print(f'Pearson corr during sync period between signal1 and signal2\
              is {pearson_corr}, p-value: {pval}')
        
        return pearson_corr,pval
    
    #%% Spearman correlation
    def spearman_corr(self, sig1, sig2, abs_value = True, print_results = True):
        
        from scipy.stats import spearmanr
        
        try: 
            spearman_corr,pval = spearmanr(sig1, sig2)
            
        except spearman_corr:
            spearman_corr,pval  = spearmanr(np.ravel(sig1), np.ravel(sig2))
        
        # calculate absolute corr if needed:
        if abs_value == True:
            spearman_corr = np.abs(spearman_corr)
            
        if print_results == True:
            print(f'Spearman corr during sync period between signal1 and signal2\
              is {spearman_corr}, p-value: {pval}')
        
        return spearman_corr,pval
     
    #%% Create COMPLETE signals after synchronization
    
    def create_full_sig_after_sync(self, LRLR_start_somno, LRLR_start_zmax, fs_res,
                                 lag, full_sig_somno_before_sync,
                                 full_sig_zmax_before_sync, plot_full_sig = False,\
                                 standardize_data = True):
        
        # rough lag 
        rough_lag = (LRLR_start_somno - LRLR_start_zmax) * fs_res
        
        # Total lag = rough lag +- lag during sync
        total_lag = int(rough_lag - lag)
        
        # truncate the lag period from somno BEGINNING
        truncated_beginning_somno = full_sig_somno_before_sync[total_lag:]
        
        # Truncate the end of LONGER signal
        len_s = len(truncated_beginning_somno)
        len_z = len(full_sig_zmax_before_sync)
        
        # if somno data is larger
        if len_s > len_z:
            somno_final = truncated_beginning_somno[:len_z]
            zmax_final  = full_sig_zmax_before_sync
        else: 
            zmax_final  = full_sig_zmax_before_sync[:len_s]
            somno_final = truncated_beginning_somno
        
        # Standardize 
        if standardize_data == True:
            try:
                zmax_final  = self.Standardadize_data_fit_transform(zmax_final)
                somno_final = self.Standardadize_data_fit_transform(somno_final)
                
            except ValueError:
                zmax_final  = self.Standardadize_data_fit_transform(np.expand_dims(zmax_final, axis = 1))
                somno_final = self.Standardadize_data_fit_transform(np.expand_dims(somno_final, axis = 1))
            
        # Calculate final length
        common_length = np.min([len_s, len_z])  
        
        # Plot truncated sigs
        if plot_full_sig == True:
            plt.figure()
            plt.plot(np.arange(0, common_length) / fs_res / 60, zmax_final, color = 'blue', label = 'iBand+ Fp1-Fp2 EEG')
            plt.plot(np.arange(0, common_length) / fs_res / 60, somno_final, \
                     color = 'red', label = 'Somno F3-F4')
            plt.title('Complete iBand and Somno data after initial sync', size = 20)
            plt.xlabel('Time (mins)', size = 15)
            plt.ylabel('Amplitude (v)', size = 15)
            plt.legend(prop={"size":20}, loc = "upper right") 
        
        return zmax_final, somno_final, total_lag
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Plot section ~~~~~~~~~~~~~~~~~~~~~~~~~ #
    
    #%% plot EDF
    def plot_edf(self, data, higpass = .1, lowpass = 30, duration = 30, n_channels =1):
        
        data.plot(duration = duration, highpass = higpass , lowpass = lowpass, n_channels = n_channels)
        
    #%% Save plot
    def save_figure(self, directory, saving_name, dpi, saving_format = '.png',
                full_screen = False):
        if full_screen == True:
            manager = plt.get_current_fig_manager()
            manager.window.showMaximized()
        plt.savefig(directory+saving_name+saving_format,dpi = dpi)  
        
    #%% plot spectrogram
    def spectrogram_creation(self, sig1, sig2, subj_night, fs, save_name, save_fig = False, dpi = 1000,\
                             save_dir = "F:\Zmax_Data\Results\SignalQualityAnalysis"):
        
        from lspopt import spectrogram_lspopt
        import numpy as np
        import matplotlib.pyplot as plt
    
        #==== plot 1st sig =======  
        try: 
            f, t, Sxx = spectrogram_lspopt(x=sig1, fs=fs, c_parameter=20.0, nperseg=int(30*fs), \
                                           scaling='density')
        except ValueError:
            f, t, Sxx = spectrogram_lspopt(x=np.ravel(sig1), fs=fs, c_parameter=20.0, nperseg=int(30*fs), \
                                           scaling='density')
            
        Sxx = 10 * np.log10(Sxx) #power to db
            
        # Limit Sxx to the largest freq of interest:
        f_sig1 = f[0:750]
        Sxx_sig1 = Sxx[0:750, :]
        fig, axs = plt.subplots(2,1, figsize=(26, 14))
        plt.axes(axs[0])
        
        plt.pcolormesh(t, f_sig1, Sxx_sig1)
        plt.ylabel('Frequency [Hz]', size=15)
        #plt.xlabel('Time [sec]', size=15)
        plt.title('Somnoscreeen data (F3-F4) - Multi-taper Spectrogram ('+str(subj_night)+")" , size=20)
        plt.colorbar()
        # ==== plot 2nd sig ==== #
        plt.axes(axs[1])
        
        try:
            f, t, Sxx = spectrogram_lspopt(x=sig2, fs=fs, c_parameter=20.0, nperseg=int(30*fs), \
                                           scaling='density')
        except ValueError:
            f, t, Sxx = spectrogram_lspopt(x=np.ravel(sig2), fs=fs, c_parameter=20.0, nperseg=int(30*fs), \
                                       scaling='density')
        Sxx = 10 * np.log10(Sxx) #power to db
            
        # Limit Sxx to the largest freq of interest:
        f_sig2 = f[0:750]
        Sxx_sig2 = Sxx[0:750, :]
        plt.pcolormesh(t, f_sig2, Sxx_sig2)
        plt.ylabel('Frequency [Hz]', size=15)
        plt.xlabel('Time [sec]', size=15)
        plt.title('iBand+ data (Fp1-Fp2) - Multi-taper Spectrogram ('+str(subj_night)+")", size=20)
    
        plt.colorbar()
        #==== 1st Way =======
        
        #=== Maximize ====
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(32, 18)
        plt.show()
        
        #Save figure
        if save_fig == True:
            self.save_figure(directory=save_dir, saving_name= save_name,
                         dpi=dpi, saving_format = '.png',
                         full_screen = False)
            
        #=== Maximize ====
        return f_sig1, f_sig2, Sxx_sig1, Sxx_sig2
    
    #%% Computing Coherence of signals
    def plot_coherence(self, sig1, sig2, Fs , NFFT = 256):
        plt.figure()
        try: 
            coh, f = plt.cohere(sig1, sig2, Fs = Fs, NFFT = NFFT)
        except ValueError:
            coh, f = plt.cohere(np.ravel(sig1), np.ravel(sig2), Fs = Fs, NFFT = NFFT)
        plt.xlim([0, 30])
        
    #%% Plot PSD
    def psd(self, sig1, sig2, fs,subj_night, NFFT = 2**11, plot_psd = True, log_power = True):
        
        from matplotlib.mlab import psd
        
        # Compute power spectrums
        try:
            psd_s1, f_psd_s1 = psd(x = sig1, Fs = fs, NFFT = NFFT, scale_by_freq= True)      
            psd_s2, f_psd_s2 = psd(x = sig2, Fs = fs, NFFT = NFFT, scale_by_freq= True)  
            
        except ValueError:
            psd_s1, f_psd_s1 = psd(x = np.ravel(sig1), Fs = fs, NFFT = NFFT, scale_by_freq= True)      
            psd_s2, f_psd_s2 = psd(x = np.ravel(sig2), Fs = fs, NFFT = NFFT, scale_by_freq= True)           
     
        # Compute log of power (optional)
            
        if log_power == True:
            
            psd_s1 = 10*np.log10(psd_s1)
            psd_s2 = 10*np.log10(psd_s2)
            
        # ================== plot dashed lines of freq bins ========================= #
        if plot_psd == True:
            
            # Open a new fig
            fig, ax = plt.subplots(1,1, figsize=(20, 10))
            
            # Global setting for axes values size
            plt.rc('xtick',labelsize=16)
            plt.rc('ytick',labelsize=16)
            
            # Plot signals 
            plt.plot(f_psd_s1, psd_s1, label = 'Zmax EEG R', color = 'blue')
            plt.plot(f_psd_s2, psd_s2, label = 'Somno F4:A1', color = 'red')

            #Delta
            plt.axvline(.5, linestyle = '--', color = 'black')
            plt.axvline(4, linestyle = '--', color = 'black')
            
            #Theta
            plt.axvline(8, linestyle = '--', color = 'black')
            
            # Alpha
            plt.axvline(12, linestyle = '--', color = 'black')
            
            # Title and labels
            plt.title('Power spectral density throughout the night - '+str(subj_night), size = 20)
            plt.xlabel('Frequency (Hz)', size = 20)
            plt.ylabel('Power spectral density (dB/ Hz)', size = 20)
            
            # Legend 
            plt.legend(['iBand+ Fp1-Fp2', 'Somno F3-F4'], prop = {'size':20})
            
            # Deactivate grid
            plt.grid(False)
            
            # Adding labels
            plt.text(1.5, 2, 'Delta',size =18)
            plt.text(5, 2, 'Theta',size =18)
            plt.text(9, 2, 'Alpha',size =18)
            plt.text(13, 2, 'Beta',size =18)
            
            # Limiting x-axis to 0-30 Hz
            plt.xlim([0, 30])
        
        return psd_s1, f_psd_s1, psd_s2, f_psd_s2

    #%% save_dic
    def save_dictionary(self, path, fname, dic):
        import pickle        
        with open(path+fname+'.pickle',"wb") as f:
            pickle.dump(dic, f)
            
    #%% Load pickle files to access features and labels     
    def load_dictionary(self, path, fname):
        import pickle
        with open(path + fname + '.pickle', "rb") as f: 
            dic = pickle.load(f)
            
        return dic   
    #%% Window by window cross correlation
    def win_by_win_corr(self, sig1, sig2, fs, subj_night, win_size = 30, plot_synced_winodws = False,\
                        plot_correlation = True, report_correlation = True,\
                        estimate_polynomial = True, poly_order = 3):
        
        """ This function gets the signals which have been already synced based 
        on an event.(e.g. using self.sync_data function)
        
        To compensate the drift in measurement devices, this function looks into data
        window by window, align them using cross-corr and compute measures such
        as pearson or spearman correaltion. 
        
        Sig1 is the reference and windowing will be based on this. Then sig2 
        will be shifted to find the best correlation per window.
        
        The elements of the output dictionarry comprises:
        """
        
        # init lists/dicts to save outputs
        list_lags            = []
        list_pearson_corr    = []
        list_pearson_pval    = []
        list_spearman_corr   = []
        list_spearman_pval   = []
        signal1_dic_windowed = dict()
        signal2_dic_windowed = dict()
        signal1_dic_windowed_before_sync = dict()
        signal2_dic_windowed_before_sync = dict()
        Outcome_dic_windowed = dict()
        f_coherence          = dict()
        overall_Cxy          = np.empty((0, 129))
        
        # e.g. Sig1 : Zmax , Sig2 : Somno
        win_size    = win_size #secs
        len_epoch   = fs * win_size
           
        # Define the loop of window size 
        for i in np.arange(0, np.floor(np.shape(sig1)[0] / (fs * win_size))):
            i = int(i)
            
            # define the sample range of the current window
            lower_boundary_samples  = i * win_size * fs
            higher_boundary_samples = (i+1) * win_size * fs
            
            # merging current samples
            plotting_samples = np.arange(lower_boundary_samples, higher_boundary_samples)
            
            # designing current windows of both signals
            curr_sig1 = sig1[plotting_samples]
            curr_sig2 = sig2[plotting_samples]
            
            # Compute correlation
            corr = signal.correlate(curr_sig1, curr_sig2)
            
            # find lag
            lag  = np.argmax(np.abs(corr)) - len(curr_sig1) + 1
            
            # shift "curr_sig2" with "lag" to sync with "curr_sig1"
            curr_sig2_synced = sig2[plotting_samples - lag]
            
            # Compute Pearson corr of the current win
            pear_corr, pear_pval = self.pearson_corr(curr_sig1, curr_sig2_synced, abs_value = True , print_results = False)
            
            # Compute Spearman corr of the current win
            spea_corr, spea_pval = self.spearman_corr(curr_sig1, curr_sig2_synced, abs_value = True, print_results = False)
            
# =============================================================================
#             # Compute coherence
#             for i in np.arange(0,6):
#                 i = int(i*5)
#                 # Lower boundary is i and higher boundary is (i+1)
#                 curr_sig1_5sec = curr_sig1[i * fs: (i+5)*fs]
#                 curr_sig2_5sec = curr_sig2_synced[i * fs: (i+5)*fs]
#                 # Dompute coherence
#                 f, Cxy      = signal.coherence(curr_sig1_5sec, curr_sig2_5sec, fs = fs)
#                 overall_Cxy = np.row_stack((overall_Cxy, Cxy))
# =============================================================================
            
            # Concatenate the values of pearson and spearman corr per window
            list_pearson_corr.append(pear_corr)
            list_pearson_pval.append(pear_pval)
            list_spearman_corr.append(spea_corr)
            list_spearman_pval.append(spea_pval)
            
            # concatenate lags per win
            list_lags.append(lag)
            
            # Convert lags: smaples --> secs
            list_lags_sec = [x / fs for x in list_lags]
            
            # Also keep the synced signals for any further analysis
            signal1_dic_windowed['window'+str(i)] = curr_sig1
            signal2_dic_windowed['window'+str(i)] = curr_sig2_synced
            
            # Also store non-synced signals
            signal1_dic_windowed_before_sync['window'+str(i)] = curr_sig1
            signal2_dic_windowed_before_sync['window'+str(i)] = curr_sig2
            
            if plot_synced_winodws==True:
                # Plot before lag correction
                plt.plot(plotting_samples, curr_sig1,label = 'Signal1 (reference)', color = 'black')
                plt.plot(plotting_samples, curr_sig2, label = 'Signal2 ', color = 'gray', linestyle = ':')
                plt.title('Syncing signal1 and signal2 data (Sync period only)', size = 15)
            
                # Plot after lag correction
                plt.plot(plotting_samples, curr_sig2_synced, label = 'Signal2 - synced',color = 'red')
                
                plt.legend(prop={"size":20})
        
        # Fit a polynomial to correelation values
        if estimate_polynomial ==True:
            
            # find coefficients for polynomial
            coefficients = np.polyfit(np.arange(len(list_pearson_corr)), list_pearson_corr, poly_order)
            poly = np.poly1d(coefficients)
            estimated_y = poly(np.arange(len(list_pearson_corr)))
            
        # Plot cross-corr and p-values
        if plot_correlation == True:
            fig1 = plt.figure()
            
            # Plot plearson corr
            plt.subplot(2, 1, 1)
            plt.stem(np.arange(len(list_pearson_corr)), list_pearson_corr,label = 'Pearson corr',\
                     markerfmt = 'black')
            plt.title('Pearson corr per epoch - ' + str(subj_night))
            plt.xlabel('Epoch #')
            plt.ylabel('Pearson corr')
            plt.xlim([0,len(list_pearson_corr)])
            
            # Plot the fitted polynomial, if requested
            if estimate_polynomial ==True:
                plt.plot(np.arange(len(list_pearson_corr)), estimated_y, color = 'red',\
                         linewidth = 3, label = "Fitted polynomial - order: "+str(poly_order))
            
            # Plot a threshold line on a specific corr value, e.g. 30%
            thresh = [0.3]
            plt.plot(np.arange(len(list_pearson_corr)),thresh * len(list_pearson_corr),'c:',\
                     linewidth = 3,label = "threshold for Pearson R "+str(thresh))
            plt.legend()    
            
            # Plot p-values
            plt.subplot(2, 1, 2)
            plt.stem(np.arange(len(list_pearson_pval)), list_pearson_pval)
            plt.title('P-values of pearson corr per epoch')
            plt.ylim((0, .05))
            plt.xlabel('Epoch #')    
            plt.ylabel('p-value')
            plt.rcParams.update({'font.size': 15})
            plt.xlim([0,len(list_pearson_corr)])
            
        # Report pearson_corr and corresponding p-values
        if report_correlation == True:
            print(f'Mean of pearson corr among all windows: {np.mean(list_pearson_corr)} +- {np.std(list_pearson_corr)}')
                
        # pack all outcomes to return 
        Outcome_dic_windowed['Pearson_corr']     = list_pearson_corr
        Outcome_dic_windowed['Pearson_pval']     = list_pearson_pval
        Outcome_dic_windowed['Spearman_corr']    = list_spearman_corr
        Outcome_dic_windowed['Spearman_pval']    = list_spearman_pval
        Outcome_dic_windowed['lags_sample']      = list_lags
        Outcome_dic_windowed['lags_sec']         = list_lags_sec
        Outcome_dic_windowed['signal1_windowed'] = signal1_dic_windowed
        Outcome_dic_windowed['signal2_windowed'] = signal2_dic_windowed
        Outcome_dic_windowed['Coherence']        = overall_Cxy
        Outcome_dic_windowed['signal1_full']     = sig1
        Outcome_dic_windowed['signal2_full']     = sig2
        Outcome_dic_windowed['signal1_windowed_before_sync'] = signal1_dic_windowed_before_sync
        Outcome_dic_windowed['signal2_windowed_before_sync'] = signal2_dic_windowed_before_sync
        return Outcome_dic_windowed
    
    #%% Compute coherence per synced window    
    def coherence_per_sleep_stage(self, subjective_corr_dic, subj_hyps,subj_night):
        
        # Init
        coherence_per_stage            = dict()
        Coherence_subjective_per_stage = dict()
        
        # Iterate over subjects
        for i,subj in enumerate(subj_night):
            
            # Init dicts (Wake, N1, N2, SWS, REM)
            coherecne_W   = np.empty((0, 129))
            coherecne_N1  = np.empty((0, 129))
            coherecne_N2  = np.empty((0, 129))
            coherecne_SWS = np.empty((0, 129))
            coherecne_REM = np.empty((0, 129))
            coherence_per_stage_tmp        = dict()
            
            # Pick the array of Coherence of current subject
            sig1_values = subjective_corr_dic[subj]['signal1_windowed']
            sig2_values = subjective_corr_dic[subj]['signal2_windowed']
            
            # Pick the address of current hyp
            tmp_hyp_adr = subj_hyps[i]
            
            # read current hypnogram
            curr_hyp = self.read_txt(main_path = tmp_hyp_adr, file_name = "",\
                                  dtype = None,delimiter=None) 
            
            # First ensure that hyp and Coh arrays have the same length
            #self.Ensure_data_label_length(curr_subj_coherence, curr_hyp)
            
            # Separate windows per sleep stage
            Wake_idx = [ii for ii,j in enumerate(curr_hyp[:,0]) if (j == 0) ]
            N1_idx   = [ii for ii,j in enumerate(curr_hyp[:,0]) if (j == 1) ]
            N2_idx   = [ii for ii,j in enumerate(curr_hyp[:,0]) if (j == 2) ]
            N3_idx   = [ii for ii,j in enumerate(curr_hyp[:,0]) if (j == 3) ]
            REM_idx  = [ii for ii,j in enumerate(curr_hyp[:,0]) if (j == 5) ]
            
            # Coherence of 5 sec windows per sleep stage
            
            # Wake
            coherence_w_tmp = self.short_window_coherence_per_sleep_stage(Wake_idx,\
                                               sig1_values, sig2_values,\
                                               subjective_corr_dic, fs = 256)
                
            # N1
            coherence_N1_tmp = self.short_window_coherence_per_sleep_stage(N1_idx,\
                                               sig1_values, sig2_values,\
                                               subjective_corr_dic, fs = 256)


            # N2
            coherence_N2_tmp = self.short_window_coherence_per_sleep_stage(N2_idx,\
                                               sig1_values, sig2_values,\
                                               subjective_corr_dic, fs = 256)


            # SWS
            coherence_SWS_tmp = self.short_window_coherence_per_sleep_stage(N3_idx,\
                                               sig1_values, sig2_values,\
                                               subjective_corr_dic, fs = 256)


            # REM
            coherence_REM_tmp = self.short_window_coherence_per_sleep_stage(REM_idx,\
                                               sig1_values, sig2_values,\
                                               subjective_corr_dic, fs = 256)

        	# Concatenate stages per subject to find overall subjects
            coherecne_W   = np.row_stack((coherecne_W, coherence_w_tmp))
            coherecne_N1  = np.row_stack((coherecne_N1, coherence_N1_tmp))
            coherecne_N2  = np.row_stack((coherecne_N2, coherence_N2_tmp))
            coherecne_SWS = np.row_stack((coherecne_SWS, coherence_SWS_tmp))
            coherecne_REM = np.row_stack((coherecne_REM, coherence_REM_tmp))
            
            # subjective per-stage arrays in a dic
            coherence_per_stage_tmp["Coherence_W"]   = coherence_w_tmp
            coherence_per_stage_tmp["Coherence_N1"]  = coherence_N1_tmp
            coherence_per_stage_tmp["Coherence_N2"]  = coherence_N2_tmp
            coherence_per_stage_tmp["Coherence_SWS"] = coherence_SWS_tmp
            coherence_per_stage_tmp["Coherence_REM"] = coherence_REM_tmp
            
            # Keep all stages in a subjective dic
            Coherence_subjective_per_stage[subj] = coherence_per_stage_tmp
            
            
        # Keep per-stage arrays in a dic
# =============================================================================
#         coherence_per_stage["Coherence_W"]   = coherecne_W
#         coherence_per_stage["Coherence_N1"]  = coherecne_N1
#         coherence_per_stage["Coherence_N2"]  = coherecne_N2
#         coherence_per_stage["Coherence_SWS"] = coherecne_SWS
#         coherence_per_stage["Coherence_REM"] = coherecne_REM
# =============================================================================
        
        return Coherence_subjective_per_stage
    #%% short window coherence computation
    def short_window_coherence_per_sleep_stage(self, stage_idx,\
                                               sig1_values, sig2_values,\
                                               subjective_corr_dic, fs=256):
        
        # init 
        overall_Cxy          = np.empty((0, 129))
        
        # Take the corresponding windows of each sleep stage
        for k in stage_idx:
            
            curr_win_no = 'window' + str(k)
            sig1_curr_epoch = sig1_values[curr_win_no]
            sig2_curr_epoch = sig2_values[curr_win_no]
        
            # Go 5 second by 5 second within the current epoch (30s)
            for jj in np.arange(0,6):
                
                # 5 second time-window
                jj = int(jj*5)
                
                # Lower boundary is i and higher boundary is (i+1)
                curr_sig1_5sec = sig1_curr_epoch[jj * fs: (jj+5)*fs]
                curr_sig2_5sec = sig2_curr_epoch[jj * fs: (jj+5)*fs]
                
                # Dompute coherence
# =============================================================================
#                 try:
#                     f, Cxy      = signal.coherence(curr_sig1_5sec, curr_sig2_5sec, fs = fs)
#                 except ValueError:
# =============================================================================
                f, Cxy      = signal.coherence(np.ravel(curr_sig1_5sec), np.ravel(curr_sig2_5sec), fs = fs)
                    
                overall_Cxy = np.row_stack((overall_Cxy, Cxy))
                
        return overall_Cxy
            
    #%% Get overall measures
    def get_overall_measure(self, subjective_corr_dic, subj_night, measure = "Pearson_corr"):
        
        # Init an array to append all the measure over subjects
        measure_all_subjs = []
        
        # Iterate over subjects to retrieve the required measure
        for subj in subj_night:
            
            # find a subject
            tmp_subj    = subjective_corr_dic[subj]
            
            # Take corresponding measure
            tmp_measure = tmp_subj[measure]
            
            # Append therequried measure to the array
            measure_all_subjs.append(tmp_measure)
            
            del tmp_subj, tmp_measure 
            
        return measure_all_subjs
    
    #%% Z-score the dataset -> fit_to_one_transform_to_other
    
    def Standardadize_data_fit_to_one_transform_to_other(self, data1, data2):
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        data1_normalized = sc.fit_transform(data1)
        data2_normalized = sc.transform(data2)
        
        return data1_normalized, data2_normalized
    
    #%% Z-score the dataset
    
    def Standardadize_data_fit_transform(self, data):
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        data_normalized = sc.fit_transform(data)
        
        return data_normalized 
    
    #%% Plot boxplot of data
    def plot_boxplot(self, data, Xlabels, Title, showmeans= True):
        
        fig, ax = plt.subplots()
        
        # positioniong labels
        fig.canvas.draw()
        
        # Define labels 
        labels = [item.get_text() for item in ax.get_xticklabels()]
        
        # Check if the user wants to plot overall as well
        
        labels      = np.append(Xlabels, "Overall")
        
        overall_val = []
        for i in np.arange(len(data)):
            overall_val = np.concatenate((overall_val, data[i]))
            
        data.append(overall_val.tolist())
            
        # Set labels
        ax.set_xticklabels(labels, rotation=45, size = 15)
        
        # Boxplot
        red_square = dict(markerfacecolor='r', marker='s')
        
        ax.boxplot(data, showmeans = showmeans, meanprops=red_square)
        
        # title and onther info
        plt.title("Boxplot of subjective epoch-wise "+ Title + 
                 " between Zmax EEG R and Somno F4:A1", size= 14)
    
    #%% Quanifying a metric
    def quanitifying_metric(self, subj_night, data_metric, metric_title):
        
        # init array for overall result
        overall_val = []
        
        for i,subj in enumerate(subj_night):
            
            # take the current row of the metric array
            tmp_metric_val = data_metric[i]
            
            # Current mean and std 
            curr_mean = np.mean(tmp_metric_val) * 100
            curr_std  = np.std(tmp_metric_val)  * 100
            
            # Concatenate to compute overall val
            overall_val    = np.concatenate((overall_val, tmp_metric_val))
            
            # Print current mean and std of subject
            print(f'The mean and std of {metric_title} for {subj} is: {"{:.2f}".format(curr_mean)} +- {"{:.2f}".format(curr_std)} ')
            
            del curr_mean, curr_std, tmp_metric_val
        
        # print overall outcome
        overall_mean = np.mean(overall_val) * 100
        overall_std  = np.std(overall_val)  * 100
        print(f'The overall mean and std of {metric_title} is: {"{:.2f}".format(overall_mean)} +- {"{:.2f}".format(overall_std)} ')   
        
    #%% ensure train data and labels have the same length
    def Ensure_data_label_length(self, X, y):
        len_x, len_y = np.shape(X)[0], np.shape(y)[0]
        if len_x == len_y:
            print("Length of data and hypnogram are identical! Perfect!")
        else:
            raise ValueError("Lengths of data epochs and hypnogram labels are different!!!")
    #%% Permutation test on on PSD
    def split_psd_whole_night(self, subj_night, subjective_corr_dic,subj_hyps,\
                             fs, NFFT = 2**11):
                
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Init ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # Dict of all vals
        psd_WholeNight_dict      = dict()
        
        # Full range (0-30 Hz)
        all_subjs_psd_sig1_full  = np.empty((0,1))
        all_subjs_psd_sig2_full  = np.empty((0,1))
        
        # Delta
        all_subjs_psd_sig1_delta = np.empty((0,1))
        all_subjs_psd_sig2_delta = np.empty((0,1))
        
        # Theta
        all_subjs_psd_sig1_theta = np.empty((0,1))
        all_subjs_psd_sig2_theta = np.empty((0,1))
        
        # Alpha
        all_subjs_psd_sig1_alpha = np.empty((0,1))
        all_subjs_psd_sig2_alpha = np.empty((0,1))
        
        # Beta
        all_subjs_psd_sig1_beta = np.empty((0,1))
        all_subjs_psd_sig2_beta = np.empty((0,1))
        
        freq_ix            = dict()
        
        # Defining EEG bands:
        eeg_bands = {'Delta'     : (0.5, 4),
                     'Theta'     : (4  , 8),
                     'Alpha'     : (8  , 12),
                     'Beta'      : (12 , 30),
                     '0_30Hz'    : (0  , 30)}

        # Iterate over subjects
        for i,subj in enumerate(subj_night):
            
            # Receive sig1 and sig2 
            sig1_tmp = subjective_corr_dic[subj]['signal1_full']
            sig2_tmp = subjective_corr_dic[subj]['signal2_full']
            
            psd_s1, f_psd_s1, psd_s2, f_psd_s2 = self.psd(sig1_tmp, sig2_tmp, fs=fs,\
                                                          NFFT = NFFT, plot_psd = False, log_power = True)
            
            # Find corresponding idx of required samples
            for band in eeg_bands:
                freq_ix[band] = np.where((f_psd_s1 >= eeg_bands[band][0]) &   
                               (f_psd_s1 <= eeg_bands[band][1]))[0]  
            
            # ================== Concatenate psd of signals  ================ #
            
            try:
                
                # [0-30 Hz] bandwidth
                all_subjs_psd_sig1_full = np.row_stack((all_subjs_psd_sig1_full, psd_s1[freq_ix['0_30Hz']]))
                all_subjs_psd_sig2_full = np.row_stack((all_subjs_psd_sig2_full, psd_s2[freq_ix['0_30Hz']]))
                
                # Delta
                all_subjs_psd_sig1_delta = np.row_stack((all_subjs_psd_sig1_delta, psd_s1[freq_ix['Delta']]))
                all_subjs_psd_sig2_delta = np.row_stack((all_subjs_psd_sig2_delta, psd_s2[freq_ix['Delta']]))
                
                # Theta
                all_subjs_psd_sig1_theta = np.row_stack((all_subjs_psd_sig1_theta, psd_s1[freq_ix['Theta']]))
                all_subjs_psd_sig2_theta = np.row_stack((all_subjs_psd_sig2_theta, psd_s2[freq_ix['Theta']]))
                
                # Alpha
                all_subjs_psd_sig1_alpha = np.row_stack((all_subjs_psd_sig1_alpha, psd_s1[freq_ix['Alpha']]))
                all_subjs_psd_sig2_alpha = np.row_stack((all_subjs_psd_sig2_alpha, psd_s2[freq_ix['Alpha']]))
                
                # Beta
                all_subjs_psd_sig1_beta = np.row_stack((all_subjs_psd_sig1_beta, psd_s1[freq_ix['Beta']]))
                all_subjs_psd_sig2_beta = np.row_stack((all_subjs_psd_sig2_beta, psd_s2[freq_ix['Beta']]))
                
            except ValueError:
                
                # [0-30 Hz] bandwidth
                all_subjs_psd_sig1_full = np.row_stack((all_subjs_psd_sig1_full, np.expand_dims(psd_s1[freq_ix['0_30Hz']], axis = 1)))
                all_subjs_psd_sig2_full = np.row_stack((all_subjs_psd_sig2_full, np.expand_dims(psd_s2[freq_ix['0_30Hz']], axis = 1)))
                
                # Delta
                all_subjs_psd_sig1_delta = np.row_stack((all_subjs_psd_sig1_delta, np.expand_dims(psd_s1[freq_ix['Delta']], axis = 1)))
                all_subjs_psd_sig2_delta = np.row_stack((all_subjs_psd_sig2_delta, np.expand_dims(psd_s2[freq_ix['Delta']], axis = 1)))
                
                # Theta
                all_subjs_psd_sig1_theta = np.row_stack((all_subjs_psd_sig1_theta, np.expand_dims(psd_s1[freq_ix['Theta']], axis = 1)))
                all_subjs_psd_sig2_theta = np.row_stack((all_subjs_psd_sig2_theta, np.expand_dims(psd_s2[freq_ix['Theta']], axis = 1)))
                
                # Alpha
                all_subjs_psd_sig1_alpha = np.row_stack((all_subjs_psd_sig1_alpha, np.expand_dims(psd_s1[freq_ix['Alpha']], axis = 1)))
                all_subjs_psd_sig2_alpha = np.row_stack((all_subjs_psd_sig2_alpha, np.expand_dims(psd_s2[freq_ix['Alpha']], axis = 1)))
                
                # Beta
                all_subjs_psd_sig1_beta = np.row_stack((all_subjs_psd_sig1_beta, np.expand_dims(psd_s1[freq_ix['Beta']], axis = 1)))
                all_subjs_psd_sig2_beta = np.row_stack((all_subjs_psd_sig2_beta, np.expand_dims(psd_s2[freq_ix['Beta']], axis = 1)))
                
            del psd_s1, psd_s2, f_psd_s1, f_psd_s2
        
        # Store the psd of freq_bins in a dict
        psd_WholeNight_dict['0_30Hz_sig1'] = all_subjs_psd_sig1_full
        psd_WholeNight_dict['0_30Hz_sig2'] = all_subjs_psd_sig2_full
        
        psd_WholeNight_dict['Delta_sig1']  = all_subjs_psd_sig1_delta
        psd_WholeNight_dict['Delta_sig2']  = all_subjs_psd_sig2_delta
        
        psd_WholeNight_dict['Theta_sig1']  = all_subjs_psd_sig1_theta
        psd_WholeNight_dict['Theta_sig2']  = all_subjs_psd_sig2_theta
        
        psd_WholeNight_dict['Alpha_sig1']  = all_subjs_psd_sig1_alpha
        psd_WholeNight_dict['Alpha_sig2']  = all_subjs_psd_sig2_alpha
        
        psd_WholeNight_dict['Beta_sig1']  = all_subjs_psd_sig1_beta
        psd_WholeNight_dict['Beta_sig2']  = all_subjs_psd_sig2_beta
        
        return psd_WholeNight_dict
    
    #%% Random permutation of PSDs
    def permutation_test_psd(self, psd_WholeNight_dict, n_perm = 1000, print_reference_means = True):
        
        # Import 
        import random 
        
        # Init 
        diff_delta = []
        diff_theta = []
        diff_alpha = []
        diff_beta  = []
        diff_full  = []
        p_val      = dict()
        # ============== Define ground truth per freq bin =================== #
        
        # 0-30 Hz
        ref_full  = np.abs(np.mean(psd_WholeNight_dict['0_30Hz_sig2']) - np.mean(psd_WholeNight_dict['0_30Hz_sig1']))
        
        # Delta
        ref_delta = np.abs(np.mean(psd_WholeNight_dict['Delta_sig2']) - np.mean(psd_WholeNight_dict['Delta_sig1']))
        
        # Theta 
        ref_theta = np.abs(np.mean(psd_WholeNight_dict['Theta_sig2']) - np.mean(psd_WholeNight_dict['Theta_sig1']))
        
        # Alpha
        ref_alpha = np.abs(np.mean(psd_WholeNight_dict['Alpha_sig2']) - np.mean(psd_WholeNight_dict['Alpha_sig1']))
        
        # Beta
        ref_beta  = np.abs(np.mean(psd_WholeNight_dict['Beta_sig2']) - np.mean(psd_WholeNight_dict['Beta_sig1']))
        
        # Print reference means
        
        if print_reference_means == True:
            print(f'The reference mean for Delta, Theta, Alpha, Beta, and [0-30 Hz]')
            print(f'are: {"{:.2f}".format(ref_delta)}, {"{:.2f}".format(ref_theta)}, {"{:.2f}".format(ref_alpha)}, {"{:.2f}".format(ref_beta)}, {"{:.2f}".format(ref_full)}')
        
        # ~~~~~~~~~~~~~~~~~~ Pooling data per freq bin ~~~~~~~~~~~~~~~~~~~~~~ #
            
        # Delta
        pooled_delta = np.row_stack((psd_WholeNight_dict['Delta_sig1'], psd_WholeNight_dict['Delta_sig2']))
        pooled_delta_tmp = pooled_delta
        
        # Theta
        pooled_theta = np.row_stack((psd_WholeNight_dict['Theta_sig1'], psd_WholeNight_dict['Theta_sig2']))
        pooled_theta_tmp = pooled_theta
        
        # Alpha
        pooled_alpha = np.row_stack((psd_WholeNight_dict['Alpha_sig1'], psd_WholeNight_dict['Alpha_sig2']))
        pooled_alpha_tmp = pooled_alpha
        
        # Beta
        pooled_beta  = np.row_stack((psd_WholeNight_dict['Beta_sig1'], psd_WholeNight_dict['Beta_sig2']))
        pooled_beta_tmp = pooled_beta
        
        # [0-30 Hz]
        pooled_full  = np.row_stack((psd_WholeNight_dict['0_30Hz_sig1'], psd_WholeNight_dict['0_30Hz_sig2']))
        pooled_full_tmp = pooled_full
        
        # ========================== Shuffle groups ========================= #
        for j in np.arange(n_perm):
            
            # Delta
            random.shuffle(pooled_delta_tmp)
            
            # Theta
            random.shuffle(pooled_theta_tmp)
            
            # Alpha
            random.shuffle(pooled_alpha_tmp)
            
            # Beta
            random.shuffle(pooled_beta_tmp)
            
            # full ([0 - 30] Hz)
            random.shuffle(pooled_full_tmp)
            
            # == Compute permuted absolute difference of two distributions == #
           
            # Delta
            diff_delta.append(np.abs(np.average(pooled_delta_tmp[0:int(len(pooled_delta_tmp)/2)]) - np.average(pooled_delta_tmp[int(len(pooled_delta_tmp)/2):])))
            
            # Theta
            diff_theta.append(np.abs(np.average(pooled_theta_tmp[0:int(len(pooled_theta_tmp)/2)]) - np.average(pooled_theta_tmp[int(len(pooled_theta_tmp)/2):])))
            
            # Alpha
            diff_alpha.append(np.abs(np.average(pooled_alpha_tmp[0:int(len(pooled_alpha_tmp)/2)]) - np.average(pooled_alpha_tmp[int(len(pooled_alpha_tmp)/2):])))
            
            # Beta
            diff_beta.append(np.abs(np.average(pooled_beta_tmp[0:int(len(pooled_beta_tmp)/2)]) - np.average(pooled_beta_tmp[int(len(pooled_beta_tmp)/2):])))
            
            # full ([0 - 30] Hz)
            diff_full.append(np.abs(np.average(pooled_full_tmp[0:int(len(pooled_full_tmp)/2)]) - np.average(pooled_full_tmp[int(len(pooled_full_tmp)/2):])))
            
        # =========================== Compuite p-val ======================== #
            
        p_val_delta = len(np.where(diff_delta >= ref_delta)[0]) / n_perm
        
        p_val_theta = len(np.where(diff_theta >= ref_theta)[0]) / n_perm
        
        p_val_alpha = len(np.where(diff_alpha >= ref_alpha)[0]) / n_perm
         
        p_val_beta  = len(np.where(diff_beta >= ref_beta)[0]) / n_perm
          
        p_val_full  = len(np.where(diff_full >= ref_full)[0]) / n_perm
        
        # Put all p-vals in a dict
        p_val['delta'] = p_val_delta
        p_val['theta'] = p_val_theta
        p_val['alpha'] = p_val_alpha
        p_val['beta']  = p_val_beta
        p_val['full']  = p_val_full
         
        return p_val_delta, p_val

            
            
    #%% Plot coherence per sleep stage
    def plot_coherence_per_sleep_stage(self, subj_night, Coherence_subjective_per_stage,\
                                       plot_all_subj_all_stage = True,\
                                       plot_mean_per_stage = True,\
                                       freq_range= "all",
                                       print_resutls = True):
        
        # Init arrays to plot
        coh_W_all   = np.empty((0,129))
        coh_N1_all  = np.empty((0,129))
        coh_N2_all  = np.empty((0,129))
        coh_SWS_all = np.empty((0,129))
        coh_REM_all = np.empty((0,129))
        
        # Assign a shorter name
        coh = Coherence_subjective_per_stage
        
        for i,subj in enumerate(subj_night):
            
            # Take the current subject coherence
            curr_coh = coh[subj]
            
            # Concatenate coherence per sleep stage
            coh_W_all   = np.row_stack((coh_W_all, curr_coh["Coherence_W"]))
            coh_N1_all  = np.row_stack((coh_N1_all, curr_coh["Coherence_N1"]))
            coh_N2_all  = np.row_stack((coh_N2_all, curr_coh["Coherence_N2"]))
            coh_SWS_all = np.row_stack((coh_SWS_all, curr_coh["Coherence_SWS"]))
            coh_REM_all = np.row_stack((coh_REM_all, curr_coh["Coherence_REM"]))
            
            del curr_coh
        
        # Compute overall coherence (independent of sleep stage)
        coh_full = np.row_stack((coh_W_all,coh_N1_all))
        coh_full = np.row_stack((coh_full,coh_N2_all))
        coh_full = np.row_stack((coh_full, coh_SWS_all))
        coh_full = np.row_stack((coh_full, coh_REM_all))
        
        # Compute mean per stage
        mean_w  = np.mean(coh_W_all, axis = 0)
        std_w   = np.std(coh_W_all, axis = 0)
        
        mean_n1 = np.mean(coh_N1_all, axis = 0)
        std_n1  = np.std(coh_N1_all, axis = 0)
        
        mean_n2 = np.mean(coh_N2_all, axis = 0)
        std_n2  = np.std(coh_N2_all, axis = 0)
        
        mean_sws = np.mean(coh_SWS_all, axis = 0)
        std_sws  = np.std(coh_SWS_all, axis = 0)
        
        mean_rem = np.mean(coh_REM_all, axis = 0)
        std_rem  = np.std(coh_REM_all, axis = 0)
        
        mean_full = np.mean(coh_full, axis = 0)
        std_full  = np.std(coh_full, axis = 0) 
            
        # Define frequency range
        f = np.arange(0,129)
        
        # Which freq range to show?
        if freq_range == "Delta":
            Lim = [0, 4]
            
        elif freq_range == "Theta":
            Lim = [4, 8]
            
        elif freq_range == "Alpha":
            Lim = [8, 12]
            
        elif freq_range == "Beta":
            Lim = [12, 30]
            
        elif freq_range == "Sleep":
            Lim = [0, 30]
        else:
            Lim = [0, 128]
        
        # ====================== Print per-stage results ==================== #
        # Range
        range_  = np.arange(Lim[0], Lim[1]+1)
        
        # Means
        meanW    = np.mean(mean_w[range_])*100
        meanN1   = np.mean(mean_n1[range_])*100
        meanN2   = np.mean(mean_n2[range_])*100
        meanSWS  = np.mean(mean_sws[range_])*100
        meanREM  = np.mean(mean_rem[range_])*100
        meanFull = np.mean(mean_full[range_])*100
        
        # Std
        stdW    = np.std(mean_w[range_])*100
        stdN1   = np.std(mean_n1[range_])*100
        stdN2   = np.std(mean_n2[range_])*100
        stdSWS  = np.std(mean_sws[range_])*100
        stdREM  = np.std(mean_rem[range_])*100
        stdFull = np.std(mean_full[range_])*100
        
        if print_resutls == True:
            print(f'Frequency range is chosen as: {freq_range} ({Lim})' )
            print(f'Coherence over all subjects - Wake: {"{:.2f}".format(meanW)} +- {"{:.2f}".format(stdW)}')
            print(f'Coherence over all subjects - N1: {"{:.2f}".format(meanN1)} +- {"{:.2f}".format(stdN1)}')
            print(f'Coherence over all subjects - N2: {"{:.2f}".format(meanN2)} +- {"{:.2f}".format(stdN2)}')
            print(f'Coherence over all subjects - SWS: {"{:.2f}".format(meanSWS)} +- {"{:.2f}".format(stdSWS)}')
            print(f'Coherence over all subjects - REM: {"{:.2f}".format(meanREM)} +- {"{:.2f}".format(stdREM)}')
            print(f'Coherence over all subjects - Overall: {"{:.2f}".format(meanFull)} +- {"{:.2f}".format(stdFull)}')
        # =================== Plot Coherence during awake =================== #
        
        if plot_all_subj_all_stage == True:
            
            fig, ax = plt.subplots(5,1, figsize=(20, 10))
            
            # Wake
            plt.axes(ax[0])
            plt.plot(f, np.transpose(coh_W_all))
            plt.xlim(Lim)
            plt.title("Coherence - all subjects - Wake", size =14)
            plt.ylabel("Coherence",size = 15)
            
            # N1
            plt.axes(ax[1])
            plt.plot(f, np.transpose(coh_N1_all))
            plt.xlim(Lim)
            plt.title("Coherence - all subjects - N1",size =14)
            plt.ylabel("Coherence",size = 15)
            
            # N2
            plt.axes(ax[2])
            plt.plot(f, np.transpose(coh_N2_all))
            plt.xlim(Lim)
            plt.title("Coherence - all subjects - N2", size =14)
            plt.ylabel("Coherence",size = 15)
            
            # SWS
            plt.axes(ax[3])
            plt.plot(f, np.transpose(coh_SWS_all))
            plt.xlim(Lim)
            plt.title("Coherence - all subjects - SWS",size =14)
            plt.ylabel("Coherence",size = 15)
            
            # REM
            plt.axes(ax[4])
            plt.plot(f, np.transpose(coh_REM_all))
            plt.xlim(Lim)
            plt.title("Coherence - all subjects - REM",size =14)
            plt.xlabel("Frequency (Hz)", size = 15)
            plt.ylabel("Coherence",size = 15)
            
        # ======================= plot mean per stage ======================= #
        # Plot all epochs at once (per stage)
        if plot_mean_per_stage == True:
            
            fig, ax = plt.subplots(4,1, figsize=(20, 10))
            # N1
            plt.axes(ax[0])
            plt.plot(f, mean_n1,  color = 'blue',    label = 'N1' , linewidth = 3)
            plt.fill_between(f, mean_n1 - std_n1, mean_n1 + std_n1, alpha=0.2, color = 'blue')
            plt.xlim(Lim)
            plt.title('Averaged coherence of all subjects N1 - Freq bin: ' + freq_range,size = 14)
            plt.ylabel("Coherence",size = 15)
            # N2
            plt.axes(ax[1])
            plt.plot(f, mean_n2,  color = 'red',     label = 'N2', linewidth = 3)
            plt.fill_between(f, mean_n2 - std_n2, mean_n2 + std_n2, alpha=0.2, color = 'red')
            plt.xlim(Lim)
            plt.title('Averaged coherence of all subjects N2 - Freq bin: ' + freq_range,size = 14)
            plt.ylabel("Coherence",size = 15)
            # SWS
            plt.axes(ax[2])
            plt.plot(f, mean_sws, color = 'green',   label = 'SWS', linewidth = 3)
            plt.fill_between(f, mean_sws - std_sws, mean_sws + std_sws, alpha=0.2,color = 'green')
            plt.xlim(Lim)
            plt.title('Averaged coherence of all subjects SWS - Freq bin: ' + freq_range,size = 14)
            plt.ylabel("Coherence",size = 15)
            # REM
            plt.axes(ax[3])
            plt.plot(f, mean_rem, color = 'cyan', label = 'REM', linewidth = 3)
            plt.fill_between(f, mean_rem - std_rem, mean_rem + std_rem, alpha=0.2,color = 'cyan')
            plt.title('Averaged coherence of all subjects REM - Freq bin: ' + freq_range,size = 14)
            plt.xlabel("Frequency (Hz)", size = 15)
            plt.ylabel("Coherence",size = 15)
            plt.xlim(Lim)
            ########## Plot all together
# =============================================================================
#             plt.axes(ax[4])
#             # W
#             plt.plot(f, mean_w,   color = 'black',   label = 'Wake', linewidth = 3)
#             plt.fill_between(f, mean_w - std_w, mean_w + std_w, alpha=0.2,color = 'black')
#             # N1
#             plt.plot(f, mean_n1,  color = 'blue',    label = 'N1' , linewidth = 3)
#             plt.fill_between(f, mean_n1 - std_n1, mean_n1 + std_n1, alpha=0.2, color = 'blue')
#             # N2
#             plt.plot(f, mean_n2,  color = 'red',     label = 'N2', linewidth = 3)
#             plt.fill_between(f, mean_n2 - std_n2, mean_n2 + std_n2, alpha=0.2, color = 'red')
#             # SWS
#             plt.plot(f, mean_sws, color = 'green',   label = 'SWS', linewidth = 3)
#             plt.fill_between(f, mean_sws - std_sws, mean_sws + std_sws, alpha=0.2,color = 'green')
#             # REM
#             plt.plot(f, mean_rem, color = 'cyan', label = 'REM', linewidth = 3)
#             plt.fill_between(f, mean_rem - std_rem, mean_rem + std_rem, alpha=0.2,color = 'cyan')
#             plt.xlim([0,30])
#             plt.legend(prop={"size":20})
#             plt.ylim([0,1])
#             
# =============================================================================

#%% Analyze drift of lags after sync period
    def analyze_dift_after_sync(self, subjective_dic, subj_night, save_fig = False,\
                                save_dir = "C:/"):
        
        for i in np.arange(len(subj_night)):
            
            # Select a subject
            Subj = subjective_dic[subj_night[i]]
            Lags = Subj['lags_sec']    
            
            # Show all lags
            plt.figure()
            plt.plot(np.arange(len(Lags)), Lags, linewidth = 2, label = "lag")
            
            # Calculate regression 
            m, b = np.polyfit(np.arange(len(Lags)), Lags, 1)
            
            # Plot regression
            plt.plot(np.arange(len(Lags)), m*np.arange(len(Lags)) + b, linewidth = 5, \
                     color = 'red', label = 'Regeression')
            
            # add title etc
            plt.title('Analysis of drift after initial alignment - ' + str(subj_night[i]), size = 20)
            plt.xlabel("Epochs", size=  18)
            plt.ylabel("second", size = 18)
            plt.legend(prop={"size":20})
            
            # Global setting for axes values size
            plt.rc('xtick',labelsize=16)
            plt.rc('ytick',labelsize=16)
            
            #=== Maximize ====
            figure = plt.gcf()  # get current figure
            figure.set_size_inches(32, 18)
            plt.show()
            
            #Save figure
            if save_fig == True:
                self.save_figure(directory=save_dir, saving_name= "analyze_dift_after_sync"+\
                                 str(subj_night[i]),
                                 dpi=300, saving_format = '.png',
                                 full_screen = False)
                    
#%% Periodic sync
    def periodic_sync_to_compensate_drift(self, Output_dic, iBand_final, \
                                          somno_final, sync_periodicity = 100,
                                          fs = 256, win_size = 30):
        
        """ Uisng this function one can align the data of a headband and a 
        golden standard after every specific number of epochs (sync_periodicity).
        This function doesn't touch golden standard signal after initial alignemnet
        (which has to be already done). Instead, it syncs the investigational headband
        data. """
        
        #Receive all the lags
        Lags_samp = Output_dic['lags_sample']
        Lags_samp_old = Lags_samp
        # create temp signals to
        sig_headband_tmp = np.expand_dims(iBand_final, axis = 1)
        
        
        # Create loop for periodic sync
        for i in np.arange(sync_periodicity, len(Lags_samp)-1, sync_periodicity):
            
            # initiate neighb_lag
            neighb_lag = []
                        
            # First iteration: Get the beginning "n" epochs
            if i == sync_periodicity:
                
                periodically_synced_iBand = sig_headband_tmp[0:i * fs * win_size]                
            
            # compute the mean of lag in the neighbouring epochs [-2, -1,  0,  1,  2]
            
            for j in np.arange(-2,3):
                
                # Define a threshold to exclude lags computed from noisy epochs
                if Lags_samp[i+j] > -(8 * fs) and  Lags_samp[i+j] < -250:
                    
                    neighb_lag.append(Lags_samp[i+j])
                    
            # Mean periodic lag
            mean_neighb_lag = int(np.floor(np.mean(neighb_lag)))
            
            # compensate the lag for all Lags_samp
            Lags_samp = [x - mean_neighb_lag for x in Lags_samp]
            
            # Keep prior portion up to sync point from headband signal
            periodically_synced_iBand = np.row_stack((periodically_synced_iBand,\
                                        sig_headband_tmp[i * fs * win_size + mean_neighb_lag:(i+sync_periodicity) * fs * win_size + mean_neighb_lag]))
        
    #%% Sync signal based on a window-by-window correlation
                
    def win_by_win_drift_compensation(self, sig1, sig2, fs, win_size = 30, plot_synced_winodws = True,\
                        plot_correlation = True, report_correlation = True,\
                        drift_comp_thresh = 5):
            
        # init lists/dicts to save outputs
        list_lags            = []
        list_pearson_corr    = []
        list_pearson_pval    = []
        list_spearman_corr   = []
        list_spearman_pval   = []
        signal1_dic_windowed = dict()
        signal2_dic_windowed = dict()
        signal1_dic_windowed_before_sync = dict()
        signal2_dic_windowed_before_sync = dict()
        Outcome_dic_windowed = dict()
        f_coherence          = dict()
        overall_Cxy          = np.empty((0, 129))
        
        # define epoch size
        len_epoch   = fs * win_size
        
        # Change dimensionality of sigs
        sig1        = np.expand_dims(sig1, axis = 1) 
        sig2        = np.expand_dims(sig2, axis = 1) 
        
        # Init lags
        lag     = 0
        lag_tmp = 0
        
        # Define the loop of window size 
        for i in np.arange(0,200):#for i in np.arange(0, np.floor(np.shape(sig1)[0] / (fs * win_size)) - 1):
            i = int(i)
            
            # define the sample range of the current window
            lower_boundary_samples  = i * win_size * fs
            higher_boundary_samples = (i+1) * win_size * fs
            next_epoch_end_samples  = (i+2) * win_size * fs
            
            # merging current samples
            plotting_samples = np.arange(lower_boundary_samples, higher_boundary_samples)
            
            # Next epoch range
            next_epoch_samps = np.arange(higher_boundary_samples, next_epoch_end_samples)
            
            # designing current windows of signal 1
            
            if lag >= 0:
        
                curr_sig1 = sig1[plotting_samples - lag]
                
            else:
                curr_sig1 = sig1[plotting_samples + lag]
            
            # designing current windows of signal 2
            curr_sig2 = sig2[plotting_samples]
            
            # 1st Itration? keep iBand sig as is
            if i == 0:
                periodically_synced_sig1 = curr_sig1
                
            # Compute correlation
            corr = signal.correlate(curr_sig1, curr_sig2)
            
            # find lag (either positive or negative)
            min_max_corr = np.abs([np.min(corr), np.max(corr)])
            
            # Is the lag neg or pos?
            corr_sign = np.max(min_max_corr)
            
            # ============= Shift sig 1 FORWARD if the corr < 0 ============= #
            if corr_sign == min_max_corr[0]: # if negative corr
                
                lag_tmp  = np.argmin(corr) - len(curr_sig1) + 1
                
                # Define threshold of acceptable lag 
                if np.abs(lag_tmp) < drift_comp_thresh * fs:
                    
                    print(f'lag withing the threshold ({lag_tmp/fs} secs)')
                    
                    # shift iBand signal "OF NEXT EPOCH" to sync with ground truth
                    next_epoch_sig1_synced = sig1[next_epoch_samps + lag]
                    
                    # Maybe due to noise, corr exceeds thresh --> keep it without change
                else:
                    print(f'lag bigger than threshold ({lag_tmp/fs} secs)')
                    next_epoch_sig1_synced = sig1[next_epoch_samps]
                
                
            # =========== Shift sig 1 BACKWARD if the corr > 0 ============== #
            else:   # if positive corr
                
                lag_tmp  = np.argmax(corr) - len(curr_sig1) + 1
                
                # Define threshold of acceptable lag 
                if np.abs(lag_tmp) < drift_comp_thresh * fs:
                    
                    print(f'lag withing the threshold ({lag_tmp/fs} secs)')
                    
                    # shift iBand signal "OF NEXT EPOCH" to sync with ground truth
                    next_epoch_sig1_synced = sig1[next_epoch_samps - lag]
                    
                    # Maybe due to noise, corr exceeds thresh --> keep it without change
                else:
                    print(f'lag bigger than threshold ({lag_tmp/fs} secs)')
                    next_epoch_sig1_synced = sig1[next_epoch_samps]
             
            # Store the overall trend of lag over epochs
            lag =lag + lag_tmp
# =============================================================================
#             # find lag
#             #lag_tmp  = np.argmax(np.abs(corr)) - len(curr_sig1) + 1
#             lag_tmp  = np.argmax(corr) - len(curr_sig1) + 1
#             lag = lag + lag_tmp
#             
#             # Define threshold of acceptable lag 
#             if np.abs(lag_tmp) < drift_comp_thresh * fs:
#                 print(f'lag withing the threshold ({lag_tmp/fs} secs)')
#                 # shift iBand signal "OF NEXT EPOCH" to sync with ground truth
#                 next_epoch_sig1_synced = sig1[next_epoch_samps - lag]
#                 
#                 # Maybe due to noise, corr exceed thresh --> keep it without change
#             else:
#                 print(f'lag bigger than threshold ({lag_tmp/fs} secs)')
#                 next_epoch_sig1_synced = sig1[next_epoch_samps]
# =============================================================================
            
            # Concatenate the next to the current epoch
            periodically_synced_sig1 = np.row_stack((periodically_synced_sig1,\
                                                     next_epoch_sig1_synced))
            
            # Make next epoch sig 2 to compute correlaiton
            next_epoch_sig2 = sig2[next_epoch_samps]   
                
            # Compute Pearson corr of the current win
            pear_corr, pear_pval = self.pearson_corr(next_epoch_sig1_synced, next_epoch_sig2, abs_value = True , print_results = False)
            
            # Compute Spearman corr of the current win
            spea_corr, spea_pval = self.spearman_corr(next_epoch_sig1_synced, next_epoch_sig2, abs_value = True, print_results = False)
            
            # Concatenate the values of pearson and spearman corr per window
            list_pearson_corr.append(pear_corr)
            list_pearson_pval.append(pear_pval)
            list_spearman_corr.append(spea_corr)
            list_spearman_pval.append(spea_pval)
            
            # concatenate lags per win
            list_lags.append(lag)
            
            # Convert lags: smaples --> secs
            list_lags_sec = [x / fs for x in list_lags]
            
            # Also keep the synced signals for any further analysis
            signal1_dic_windowed['window'+str(i)] = curr_sig1
            signal2_dic_windowed['window'+str(i)] = curr_sig2
            
            # Also store non-synced signals
            signal1_dic_windowed_before_sync['window'+str(i)] = curr_sig1
            signal2_dic_windowed_before_sync['window'+str(i)] = curr_sig2
            
        # Plot cross-corr and p-values
        if plot_correlation == True:
            fig1 = plt.figure()
            
            # Plot plearson corr
            plt.subplot(2, 1, 1)
            plt.plot(np.arange(len(list_pearson_corr)), list_pearson_corr)
            plt.title('Pearson corr per epoch')
            
            # Plot p-values
            plt.subplot(2, 1, 2)
            plt.plot(np.arange(len(list_pearson_pval)), list_pearson_pval)
            plt.title('P-values of pearson corr per epoch')
            plt.ylim((0, .05))
                
        # Report pearson_corr and corresponding p-values
        if report_correlation == True:
            print(f'Mean of pearson corr among all windows: {np.mean(list_pearson_corr)} +- {np.std(list_pearson_corr)}')
                
        # pack all outcomes to return 
        Outcome_dic_windowed['Pearson_corr']     = list_pearson_corr
        Outcome_dic_windowed['Pearson_pval']     = list_pearson_pval
        Outcome_dic_windowed['Spearman_corr']    = list_spearman_corr
        Outcome_dic_windowed['Spearman_pval']    = list_spearman_pval
        Outcome_dic_windowed['lags_sample']      = list_lags
        Outcome_dic_windowed['lags_sec']         = list_lags_sec
        Outcome_dic_windowed['signal1_windowed'] = signal1_dic_windowed
        Outcome_dic_windowed['signal2_windowed'] = signal2_dic_windowed
        Outcome_dic_windowed['Coherence']        = overall_Cxy
        Outcome_dic_windowed['signal1_full']     = sig1
        Outcome_dic_windowed['signal2_full']     = sig2
        Outcome_dic_windowed['signal1_windowed_before_sync'] = signal1_dic_windowed_before_sync
        Outcome_dic_windowed['signal2_windowed_before_sync'] = signal2_dic_windowed_before_sync
        return Outcome_dic_windowed, periodically_synced_sig1
    
    
    #%% Create COMPLETE signals after synchronization
    
    def derive_final_sigs_after_init_and_periodic_alignment(self, LRLR_start_somno, LRLR_start_zmax, fs_res,
                                 lag, full_sig_somno_before_sync, full_sig_zmax_before_sync,\
                                 subj_night, regression_slope,\
                                 plot_full_sig = False,\
                                 epoch_periodic_alignment = 50,\
                                 standardize_data = True, amplitude_range = None):
        
        LRLR_start_zmax = int(LRLR_start_zmax)
        
        # rough lag 
        rough_lag = (LRLR_start_somno - LRLR_start_zmax) * fs_res
        
        # Total lag = rough lag +- lag during sync
        total_lag_init = int(rough_lag - lag)
        
        # truncate the lag period from somno BEGINNING
        truncated_beginning_somno = full_sig_somno_before_sync[total_lag_init:]
        
        # Truncate the end of LONGER signal
        len_s = len(truncated_beginning_somno)
        len_z = len(full_sig_zmax_before_sync)
        
        # find the rest of signal duration after initial sync
        iBand_length_after_init_sync = np.floor((len(full_sig_zmax_before_sync)/fs_res - LRLR_start_zmax))
        n_epochs_after_sync = int(np.floor(iBand_length_after_init_sync / 30))
        
        # Compute estimated lag over every n epochs for periodic alignmnet
        t_regression = int(np.floor((epoch_periodic_alignment * regression_slope) * fs_res))
        
        # Final_sig
        Final_sig_headband = full_sig_zmax_before_sync[:LRLR_start_zmax * fs_res]
        
        # Go over every n epochs for periodic alignmnet
        for j in np.arange(epoch_periodic_alignment,n_epochs_after_sync, epoch_periodic_alignment):
            
            # Separate the signal to before and after periodic event
            sig_tmp_before_event = full_sig_zmax_before_sync[(LRLR_start_zmax + 30*(j-epoch_periodic_alignment)) * fs_res : (LRLR_start_zmax + 30*j) * fs_res]
            sig_tmp_after_event  = full_sig_zmax_before_sync[(LRLR_start_zmax + 30*j) * fs_res:]                                                                                                                                                                                                          
                
            # Append sig_tmp_before_event to final
            Final_sig_headband = np.append(Final_sig_headband, sig_tmp_before_event) 
                                             
            # Copy the ending "t_regression" amount of before periodic event sig
            copy_tmp = sig_tmp_before_event[-t_regression:]
            
            # Append the copied section to the "sig_tmp_before_event"
            Final_sig_headband= np.append(Final_sig_headband, copy_tmp)
        
        # Adding the remaining tail of signals, after the last periodic alignment
        Final_sig_headband  =  np.append(Final_sig_headband, sig_tmp_after_event)
        Final_sig_reference =  truncated_beginning_somno[:len(Final_sig_headband)]
        
        # Standardize 
        if standardize_data == True:
            try:
                Final_sig_headband  = self.Standardadize_data_fit_transform(Final_sig_headband)
                Final_sig_reference = self.Standardadize_data_fit_transform(Final_sig_reference)
                
            except ValueError:
                Final_sig_headband  = self.Standardadize_data_fit_transform(np.expand_dims(Final_sig_headband, axis = 1))
                Final_sig_reference = self.Standardadize_data_fit_transform(np.expand_dims(Final_sig_reference, axis = 1))
            
        # Calculate final length
        common_length = np.min([len_s, len_z])  
        
        # Plot truncated sigs
        if plot_full_sig == True:
            plt.figure()
            plt.plot(np.arange(0, len(Final_sig_headband)) / fs_res / 60, Final_sig_headband, label = 'iBand+ Fp1-Fp2 EEG')
            plt.plot(np.arange(0, len(Final_sig_reference)) / fs_res / 60, Final_sig_reference, \
                     color = 'red', label = 'Somno F3-F4')
            plt.title('Complete iBand and Somno data after initial sync - '+str(subj_night), size = 20)
            plt.xlabel('Time (mins)', size = 15)
            plt.ylabel('Amplitude (v)', size = 15)
            plt.legend(prop={"size":20}, loc = "upper right") 
            
            # Show ylim
            if amplitude_range != None:
                plt.ylim((amplitude_range[0], amplitude_range[1]))
        
        return Final_sig_headband, Final_sig_reference, total_lag_init
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Plot section ~~~~~~~~~~~~~~~~~~~~~~~~~ #