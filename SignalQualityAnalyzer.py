# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 14:13:18 2020

CopyRight: Mahdad Jafarzadeh 

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
import tensorflow as tf
from scipy import signal
from scipy.signal import butter, lfilter, periodogram, spectrogram, welch, filtfilt, iirnotch
from scipy.stats import pearsonr, spearmanr
import matplotlib.mlab as mlab
import pandas as pd
from SigQual import SigQual

#%% Initiate an object from SigQual class
Object = SigQual()

#%% Read in data (Somno + Zmax)

#####=========================== Reading data ============================#####

# Main path
main_path   = "F:/Zmax_Data/features/"

# Read location of Somno data
subj_ids_somno  = Object.read_txt(main_path = main_path, file_name =  "SigQual_Somno_data_loc",\
                                  dtype = 'str',delimiter='\n') 
    
# Read Zmax data
subj_ids_zmax   =  Object.read_txt(main_path = main_path, file_name =  "SigQual_Zmax_data_loc",\
                                  dtype = 'str',delimiter='\n') 


# Read subject_night id
subj_night   = Object.read_txt(main_path = main_path, file_name =  "Subject_Night",\
                                  dtype = 'str',delimiter='\n') 

# read event markers path to sync data
sync_markers_main_path = "F:/Zmax_Data/features/"
event_markers = Object.read_excel(main_path = sync_markers_main_path, filename = "Sync_periods")


#%% initializing dictionaries to save output
Sxx_somno_dic       = dict()
Sxx_zmax_dic        = dict()
f_spect_somno_dic   = dict()
f_spect_zmax_dic    = dict()
psd_somno_dic       = dict()
psd_zmax_dic        = dict()
f_psd_somno_dic     = dict()
f_psd_zmax_dic      = dict()
subjective_corr_dic = dict()

#%% Main loop of analysis
#####======================== Iterating through subjs=====================#####
#subj_ids_somno = ["F:/Zmax_Data/Somnoscreen_Data/P_21/P_21 night2_UB.15.01.2019/P_21_night2_UB_Recording_(2)_(1).edf"]
for idx, c_subj in enumerate(subj_ids_somno):
    
    # define the current zmax data
    curr_zmax  = subj_ids_zmax[idx]
    
    # define current somno data
    curr_somno = c_subj
    
    # Reading EEG left and right (Zmax)
    data_L     = Object.read_edf_file(path_folder=curr_zmax, filename="EEG L", preload = True)
    data_R     = Object.read_edf_file(path_folder=curr_zmax, filename="EEG R", preload = True)  
    
    # Read somno data    
    EEG_somno  =Object.read_edf_file(path_folder=curr_somno, filename="", preload = True)  
    
    # Reading info header (Somno)
    Info_s, fs_somno, AvailableChannels_s = Object.edf_info(EEG_somno)
    
    # Reading info header (Zmax)
    Info_z, fs_zmax, AvailableChannels_z = Object.edf_info(data_R)
    
    # ======================= Data representation =========================== #
    
# =============================================================================
#     Object.plot_edf(data = data_R, higpass = .1, lowpass = 30, duration = 30, n_channels =1)
#     Object.plot_edf(data = data_L, higpass = .1, lowpass = 30, duration = 30, n_channels =1)
#     Object.plot_edf(data = EEG_somno, higpass = .1, lowpass = 30, duration = 30, n_channels =4)
# =============================================================================
    
    # ======================= Filter data before resample =================== #
    #Data_R_filt = Object.mne_obj_filter(data = data_R, sfreq = fs_zmax, l_freq = .1, h_freq=30, picks = AvailableChannels_z)
    
    # ======================= Resampling to lower freq ====================== #
     
    fs_res, data_R, EEG_somno = Object.resample_data(data_R, EEG_somno, fs_zmax, fs_somno)
    _ , data_L, _      = Object.resample_data(data_L, EEG_somno, fs_zmax, fs_somno)

    # ========================== Get data arrays ============================ #
    data_L_get = data_L.get_data()
    data_R_get = data_R.get_data()
    data_somno_get = EEG_somno.get_data()
    
    # ====================== Filtering resampled data ======================= #
    
    data_L_resampled_filtered    = Object.butter_bandpass_filter(data_L_get, lowcut=.1, highcut=30, fs=fs_res, order = 2)
    data_R_resampled_filtered    = Object.butter_bandpass_filter(data_R_get, lowcut=.1, highcut=30, fs=fs_res, order = 2)
    EEG_somno_resampled_filtered = Object.butter_bandpass_filter(data_somno_get, lowcut=.1, highcut=30, fs=fs_res, order = 2)

    # ====================== Synchronization of data ======================== #
    
    # required inputs to sync
    LRLR_start_zmax = event_markers['LRLR_start_zmax'][idx] #sec
    LRLR_end_zmax   = event_markers['LRLR_end_zmax'][idx] #sec
    LRLR_start_somno = event_markers['LRLR_start_somno'][idx] #sec
    LRLR_end_somno   = event_markers['LRLR_end_somno'][idx] #sec
    
    # sync
    lag, corr, Somno_reqChannel, zmax_data_R = Object.sync_data(fs_res, LRLR_start_zmax, LRLR_end_zmax, LRLR_start_somno, LRLR_end_somno,\
                  data_R_resampled_filtered, data_L_resampled_filtered, \
                  EEG_somno_resampled_filtered, AvailableChannels_s, save_name = subj_night[idx], \
                  RequiredChannels = ['F4:A1'], save_fig = False, dpi = 1000,\
                  save_dir = "F:/Zmax_Data/Results/SignalQualityAnalysis/",
                  report_pearson_corr_during_sync  = True,\
                  report_spearman_corr_during_sync = True,\
                  plot_cross_corr_lag = False)
        
    # ======================= Plot full sig after sync ====================== #
        
    full_sig_somno_before_sync = Somno_reqChannel
    full_sig_zmax_before_sync  = zmax_data_R
    
    # Get final sigs and plot them
    zmax_final, somno_final = Object.create_full_sig_after_sync(LRLR_start_somno, LRLR_start_zmax, fs_res,
                                 lag, full_sig_somno_before_sync,
                                 full_sig_zmax_before_sync, plot_full_sig = False)
    
    # =================== Compute correlations win by win =================== #
    Output_dic = Object.win_by_win_corr(sig1 = zmax_final, sig2 = somno_final,\
                                    fs = fs_res, win_size = 30, plot_synced_winodws = False)
    # save it subjectively
    subjective_corr_dic[subj_night[idx]] = Output_dic
    
    # =================== Plot spectrgoram of somno vs Zmax ================= #
    f_spect_s, f_spect_z, Sxx_s, Sxx_z = Object.spectrogram_creation(somno_final, zmax_final, fs_res,\
                                         save_name="spect_"+subj_night[idx], save_fig = False, dpi = 1000,\
                                         save_dir = "F:\Zmax_Data\Results\SignalQualityAnalysis")
    
    # ========================= Plot coherence ============================== #
    coh, f = Object.plot_coherence(somno_final, zmax_final, Fs = fs_res, NFFT = 256)
    
    # ============================== Plot psd =============================== #
    psd_s1, f_psd_s1, psd_s2, f_psd_s2 = Object.plot_psd(sig1 = zmax_final,\
                                sig2 = somno_final, fs = fs_res, NFFT = 2**11)

#%% Save outcomes        
# ============================= save results ================================ #
Object.save_dictionary( "F:/Zmax_Data/Results/SignalQualityAnalysis/",\
                       "subjective_corr_results_140720", subjective_corr_dic)

#%% Load corr_outcomes
# =========================== Load windowed corrs =========================== #
subjective_corr_dic = Object.load_dictionary( "F:/Zmax_Data/features/",\
                       "subjective_corr_results_140720")
    
#%% Get overall corr
# ======================== Retrieve overall metrics ========================= #
Overall_pearson_corr  = Object.get_overall_measure(subjective_corr_dic,\
                                                  subj_night, measure = "Pearson_corr")
    
Overall_spearman_corr = Object.get_overall_measure(subjective_corr_dic,\
                                                  subj_night, measure = "Spearman_corr")
    
#%% Boxplot     
Object.plot_boxplot(Overall_pearson_corr, Xlabels = subj_night, showmeans= True,\
                    Title = "Pearson correlation")

Object.plot_boxplot(Overall_spearman_corr, Xlabels = subj_night, showmeans= True,\
                    Title = "Spearman correlation")
    
#%% Quantifying some metrics
Object.quanitifying_metric(subj_night, Overall_spearman_corr, metric_title = "spearman corr")
