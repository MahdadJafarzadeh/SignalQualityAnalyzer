# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 14:13:18 2020

CopyRight: Mahdad Jafarzadeh 

SigQual: The package to analyze the quality of signals!

"""
#%% Import libs
#####===================== Importiung libraries =========================#####
import mne
import numpy as np
from scipy.integrate import simps
from   numpy import loadtxt
import h5py
import time
import os 
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd
from scipy import signal
from scipy.signal import butter, lfilter, periodogram, spectrogram, welch, filtfilt, iirnotch
from scipy.stats import pearsonr, spearmanr
import matplotlib.mlab as mlab
import pandas as pd
from SigQual import SigQual
from matplotlib.mlab import psd
%matplotlib qt

#%% Initiate an object from SigQual class
Object = SigQual()

#%% Read in data (Somno + iBand)

#####=========================== Reading data ============================#####

# Main path
main_path       = "C:/Users/mahda/OneDrive/Documents/iBand_SigQual/"
 
# Read location of Somno data
subj_ids_somno  = Object.read_txt(main_path = main_path, file_name =  "SigQual_Somno_data_loc",\
                                  dtype = 'str',delimiter='\n') 
    
# Read iBand data
subj_ids_iBand   =  Object.read_txt(main_path = main_path, file_name =  "SigQual_iBand_data_loc",\
                                  dtype = 'str',delimiter='\n') 


# Read subject_night id
subj_night      = Object.read_txt(main_path = main_path, file_name =  "Subject_Night",\
                                  dtype = 'str',delimiter='\n') 

# read event markers path to sync data
sync_markers_main_path = main_path
event_markers = Object.read_excel(main_path = sync_markers_main_path, filename = "Sync_periods")

# =============================================================================
# # Read Hypnograms
# subj_hyps     = Object.read_txt(main_path = main_path, file_name =  "Subject_Hyps",\
#                                   dtype = 'str',delimiter='\n') 
# =============================================================================
#%% initializing dictionaries to save output
Sxx_somno_dic       = dict()
Sxx_iBand_dic        = dict()
f_spect_somno_dic   = dict()
f_spect_iBand_dic    = dict()
psd_somno_dic       = dict()
psd_iBand_dic        = dict()
f_psd_somno_dic     = dict()
f_psd_iBand_dic      = dict()
subjective_dic      = dict()
total_lags_full_sig = dict()

#%% Main loop of analysis
#####======================== Iterating through subjs=====================#####

for idx, c_subj in enumerate(subj_ids_somno):
    idx= 1
    # define the current iBand data
    curr_iBand  = subj_ids_iBand[idx]
    
    # define current somno data
    curr_somno = subj_ids_somno[idx]
    
    # Reading EEG fp1 - Fp2 (iBand)
    data_iBand     = Object.read_edf_file(path_folder=curr_iBand, filename="", preload = True)
    
    # Read somno data    
    EEG_somno  =Object.read_edf_file(path_folder=curr_somno, filename="", preload = True)  
    
    # Reading info header (Somno)
    Info_s, fs_somno, AvailableChannels_s = Object.edf_info(EEG_somno)
    
    # Reading info header (curr_iBand)
    Info_iBand, fs_iBand, AvailableChannels_iBand = Object.edf_info(data_iBand)
    
    # ======================= Data representation =========================== #
    
    # iBand
    Object.plot_edf(data = data_iBand, higpass = .1, lowpass = 30, duration = 30, n_channels =1)
    
    #Somno
    Object.plot_edf(data = EEG_somno, higpass = .1, lowpass = 30, duration = 30, n_channels =10)
    
    # ======================= Filter data before resample =================== #
    
    data_iBand    = Object.mne_obj_filter(data = data_iBand, l_freq = .1, h_freq=30)
    EEG_somno     = Object.mne_obj_filter(data = EEG_somno, l_freq = .1, h_freq=30)
    
    # ======================= Resampling to lower freq ====================== #
    # Check if fs1 != fs2 
    fs_res, data_iBand, EEG_somno = Object.resample_data(data_iBand, EEG_somno, fs_iBand, fs_somno)

    # ========================== Get data arrays ============================ #
    
    data_iBand_resampled_filtered = data_iBand.get_data()
    EEG_somno_resampled_filtered = EEG_somno.get_data()
    
    # ====================== Synchronization of data ======================== #
    
    # required inputs to sync
    LRLR_start_iBand = event_markers['LRLR_start_iBand'][idx] #sec
    LRLR_end_iBand   = event_markers['LRLR_end_iBand'][idx] #sec
    LRLR_start_somno = event_markers['LRLR_start_somno'][idx] #sec
    LRLR_end_somno   = event_markers['LRLR_end_somno'][idx] #sec
    
    # sync
    lag, corr, Somno_reqChannel, iBand_data, sig1, sig2 = Object.sync_data(fs_res, LRLR_start_iBand, LRLR_end_iBand,\
                                                                          LRLR_start_somno, LRLR_end_somno,\
                                                                          data_iBand_resampled_filtered, data_iBand_resampled_filtered, \
                                                                          EEG_somno_resampled_filtered, AvailableChannels_s,subj_night[idx],\
                                                                          save_name = subj_night[idx], \
                                                                          RequiredChannels = ['F3'], Ref_channel = ['F4'], save_fig = False, dpi = 1000,\
                                                                          save_dir = "D:/Zmax_Data/Results/SignalQualityAnalysis/",
                                                                          report_pearson_corr_during_sync  = True,\
                                                                          report_spearman_corr_during_sync = True,\
                                                                          plot_cross_corr_lag = False, scale_somno=1,\
                                                                          amplitude_range = None)
      
    # ======================= Plot full sig after sync ====================== #
        
    full_sig_somno_before_sync  = Somno_reqChannel
    full_sig_iBand_before_sync  = iBand_data
    
    # ======== Apply periodic alignemnt and derive the final signals ======== #
    Final_sig_iBand, Final_sig_somno, total_lag_init = Object.derive_final_sigs_after_init_and_periodic_alignment(
                                 LRLR_start_somno, LRLR_start_iBand, fs_res,
                                 lag, full_sig_somno_before_sync, full_sig_iBand_before_sync,\
                                 subj_night[idx], regression_slope = 0.0627,\
                                 plot_full_sig = True,\
                                 epoch_periodic_alignment = 50,\
                                 standardize_data = True, amplitude_range = None)
        
    # =================== Compute correlations win by win =================== #
    
    Output_dic = Object.win_by_win_corr(sig1 = Final_sig_iBand, sig2 = Final_sig_somno,\
                                    fs = fs_res,subj_night = subj_night[idx],\
                                    win_size = 30, plot_synced_winodws = False,\
                                    plot_correlation = True, \
                                    estimate_polynomial = True, poly_order = 9)
        
    # keep outcomes subjectively
    subjective_dic[subj_night[idx]] = Output_dic
    
    # =================== Plot comparative spectrograms ===================== #
        
    f_spect_s, f_spect_z, Sxx_s, Sxx_z = Object.spectrogram_creation(Final_sig_somno, Final_sig_iBand,\
                                         subj_night[idx], fs_res,\
                                         save_name="spect_"+subj_night[idx], save_fig = False, dpi = 1000,\
                                         save_dir = "D:\Zmax_Data\Results\SignalQualityAnalysis")
        
    # ============================== Plot psd =============================== #
    psd_s1, f_psd_s1, psd_s2, f_psd_s2 = Object.psd(sig1 = Final_sig_iBand,\
                                sig2 = Final_sig_somno, fs = fs_res, subj_night = subj_night[idx],\
                                NFFT = 2**11,\
                                plot_psd = True, log_power = True)
        