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

#%% Read in data (Somno + Zmax)

#####=========================== Reading data ============================#####

# Main path
main_path       = "D:/Zmax_Data/features/"
 
# Read location of Somno data
subj_ids_somno  = Object.read_txt(main_path = main_path, file_name =  "SigQual_Somno_data_loc2",\
                                  dtype = 'str',delimiter='\n') 
    
# Read Zmax data
subj_ids_zmax   =  Object.read_txt(main_path = main_path, file_name =  "SigQual_Zmax_data_loc2",\
                                  dtype = 'str',delimiter='\n') 


# Read subject_night id
subj_night      = Object.read_txt(main_path = main_path, file_name =  "Subject_Night",\
                                  dtype = 'str',delimiter='\n') 

# read event markers path to sync data
sync_markers_main_path = "D:/Zmax_Data/features/"
event_markers = Object.read_excel(main_path = sync_markers_main_path, filename = "Sync_periods")

# Read Hypnograms
subj_hyps     = Object.read_txt(main_path = main_path, file_name =  "Subject_Hyps",\
                                  dtype = 'str',delimiter='\n') 
#%% initializing dictionaries to save output
Sxx_somno_dic       = dict()
Sxx_zmax_dic        = dict()
f_spect_somno_dic   = dict()
f_spect_zmax_dic    = dict()
psd_somno_dic       = dict()
psd_zmax_dic        = dict()
f_psd_somno_dic     = dict()
f_psd_zmax_dic      = dict()
subjective_dic      = dict()
total_lags_full_sig = dict()
#%% Main loop of analysis
#####======================== Iterating through subjs=====================#####

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
    
    data_R    = Object.mne_obj_filter(data = data_R, l_freq = .1, h_freq=30)
    data_L    = Object.mne_obj_filter(data = data_L, l_freq = .1, h_freq=30)
    EEG_somno = Object.mne_obj_filter(data = EEG_somno, l_freq = .1, h_freq=30)
    
    # ======================= Resampling to lower freq ====================== #
     
    fs_res, data_R, EEG_somno = Object.resample_data(data_R, EEG_somno, fs_zmax, fs_somno)
    _ , data_L, _      = Object.resample_data(data_L, EEG_somno, fs_zmax, fs_somno)

    # ========================== Get data arrays ============================ #
    
    data_L_resampled_filtered = data_L.get_data()
    data_R_resampled_filtered = data_R.get_data()
    EEG_somno_resampled_filtered = EEG_somno.get_data()
    
    # ====================== Synchronization of data ======================== #
    
    # required inputs to sync
    LRLR_start_zmax = event_markers['LRLR_start_zmax'][idx] #sec
    LRLR_end_zmax   = event_markers['LRLR_end_zmax'][idx] #sec
    LRLR_start_somno = event_markers['LRLR_start_somno'][idx] #sec
    LRLR_end_somno   = event_markers['LRLR_end_somno'][idx] #sec
    
    # sync
    lag, corr, Somno_reqChannel, zmax_data_R, sig1, sig2 = Object.sync_data(fs_res, LRLR_start_zmax, LRLR_end_zmax, LRLR_start_somno, LRLR_end_somno,\
                  data_R_resampled_filtered, data_L_resampled_filtered, \
                  EEG_somno_resampled_filtered, AvailableChannels_s, save_name = subj_night[idx], \
                  RequiredChannels = ['F4:A1'], save_fig = False, dpi = 1000,\
                  save_dir = "D:/Zmax_Data/Results/SignalQualityAnalysis/",
                  report_pearson_corr_during_sync  = True,\
                  report_spearman_corr_during_sync = True,\
                  plot_cross_corr_lag = False)
        
    # ======================= Plot full sig after sync ====================== #
        
    full_sig_somno_before_sync = Somno_reqChannel
    full_sig_zmax_before_sync  = zmax_data_R
    
    # Get final sigs and plot them
    zmax_final, somno_final, total_lag = Object.create_full_sig_after_sync(LRLR_start_somno, LRLR_start_zmax, fs_res,
                                 lag, full_sig_somno_before_sync,
                                 full_sig_zmax_before_sync, plot_full_sig = False,\
                                 standardize_data = True)
    total_lags_full_sig[subj_night[idx]] = total_lag
    # =================== Compute correlations win by win =================== #
    Output_dic = Object.win_by_win_corr(sig1 = zmax_final, sig2 = somno_final,\
                                    fs = fs_res, win_size = 30, plot_synced_winodws = False)
    # keep it subjectively
    subjective_dic[subj_night[idx]] = Output_dic
    
    # =================== Plot spectrgoram of somno vs Zmax ================= #
    f_spect_s, f_spect_z, Sxx_s, Sxx_z = Object.spectrogram_creation(somno_final, zmax_final, fs_res,\
                                         save_name="spect_"+subj_night[idx], save_fig = False, dpi = 1000,\
                                         save_dir = "D:\Zmax_Data\Results\SignalQualityAnalysis")
    
    # ========================= Plot coherence ============================== #
    coh, f = Object.plot_coherence(somno_final, zmax_final, Fs = fs_res, NFFT = 256)
    
    # ============================== Plot psd =============================== #
    psd_s1, f_psd_s1, psd_s2, f_psd_s2 = Object.psd(sig1 = zmax_final,\
                                sig2 = somno_final, fs = fs_res, NFFT = 2**11,\
                                plot_psd = False, log_power = True)
        
    plt.plot(f_psd_s2, 10*np.log10(psd_s2), color ='cyan')
        
#%% Save outcomes        
# ============================= save results ================================ #
Object.save_dictionary( "D:/Zmax_Data/features/",\
                       "subjective_results_Normalized_5min_win_by_win_corr_031020", subjective_dic)

#%% Load corr_outcomes
# =========================== Load windowed corrs =========================== #
subjective_dic = Object.load_dictionary( "D:/Zmax_Data/features/",\
                       "subjective_results_Normalized_5min_win_by_win_corr_031020")

#%% Analyze drift aftersync --> does it cause epoch shift between devices?

Object.analyze_dift_after_sync(subjective_dic, subj_night, save_fig = True,\
                                save_dir = "M:/Music/")
#%% Get overall corr
# ======================== Retrieve overall metrics ========================= #
Overall_pearson_corr  = Object.get_overall_measure(subjective_dic,\
                                                  subj_night, measure = "Pearson_corr")
    
Overall_spearman_corr = Object.get_overall_measure(subjective_dic,\
                                                  subj_night, measure = "Spearman_corr")
    
#%% Boxplot     
Object.plot_boxplot(Overall_pearson_corr, Xlabels = subj_night, showmeans= True,\
                    Title = "Pearson correlation")

Object.plot_boxplot(Overall_spearman_corr, Xlabels = subj_night, showmeans= True,\
                    Title = "Spearman correlation")
    
#%% Quantifying some metrics
Object.quanitifying_metric(subj_night, Overall_pearson_corr, metric_title = "pearson corr")

#%% split the coherences per sleep stage
Coherence_subjective_per_stage = Object.coherence_per_sleep_stage(subjective_dic, subj_hyps,subj_night)

#%% Plot coherence
Object.plot_coherence_per_sleep_stage(subj_night, Coherence_subjective_per_stage,\
                                      plot_all_subj_all_stage = False,\
                                      plot_mean_per_stage = True,\
                                      freq_range= "Theta",print_resutls = True)

#%% Divide the whole night PSDs based on freq bins
psd_WholeNight_dict = Object.split_psd_whole_night(subj_night, subjective_dic,subj_hyps,\
                             fs = 256, NFFT = 2**11)
    
#%% Apply permutation test
p_val_delta, p_val = Object.permutation_test_psd(psd_WholeNight_dict,  n_perm = 1000,\
                                                 print_reference_means = True)