# -*- coding: utf-8 -*-
"""
Immitance calculation subscript of IMTB

@author: YLI, AUF @ Energinet

Update history:
    21 Mar 2025 - v1.0 - first public version
    
"""

# =============================================================================
# Global settings
# =============================================================================

# Rounding negative scientific exponent. Helping with numeric problems in the script
ROUNDING_TO_EXP_FREQ = 6 # For frequency rounding
ROUNDING_TO_EXP_TIME = 6 # For time steps rounding

ENABLE_DEDUCTION_DFT_NONINJ = False # for enabling deduction of background DFT analysis
# Note: please use "False" by not deducting background DFT analysis, as a bug was found with SISO and dq scan. 
NEW_MIMO_CALC_METHOD = False # for using the new MIMO impedance calculation method
# Note: please use "False", as larger inaccuracies were identified using the new method
# new impedance calculation method: abc -> DFT -> alpha-beta complex or dq
# old IMTB impedance calculation method: abc -> alpha-beta -> DFT -> alpha-beta complex or dq

# =============================================================================
# Modules Imports
# =============================================================================

import sys, os
import shutil
# for data handling
import pandas as pd 
# for saving file to csv
import csv

import mhi.psout

import numpy as np
import numpy.linalg as lnlg 

import IMTB_pscad_sim as pscad_sim
from tqdm import tqdm

from plotly.subplots import make_subplots
import plotly.graph_objects as go

# for impedance calculation
import IM_analysis_lib_v1 as IM
# import math
import cmath

# =============================================================================
# Internal Functions
# =============================================================================

def get_settings(simPath):
    
    def is_float(element: any) -> bool:
        #If you expect None to be passed:
        if element is None: 
            return False
        try:
            float(element)
            return True
        except ValueError:
            return False
        
    def is_bool(element: any) -> bool:
        if element is None:
            return False
        elif element == "True" or element == "False":
            return True
        else:
            return False
        
    def transform_to_bool(element: any) -> bool:
        if element == "True":
            return True
        else:
            return False

    # get settings
    setting_file = os.path.join(simPath, 'settings.csv')
    with open(setting_file, newline="") as fp:
        csv_reader = csv.DictReader(fp)        
        settings = []     
        for row in csv_reader:
            # Append the row as a dictionary to the list
            settings.append(row)
    settings = settings[0]
    for key in settings:
        value = settings[key]
        if key == "Timestamp":
            settings[key] = str(value)
        elif value.isdigit():
            settings[key] = int(value)
        elif is_float(value):
            settings[key] = float(value)
        elif is_bool(value):
            settings[key] = transform_to_bool(value)
    return settings


def read_raw(settings, filepath):
    """ Helping function to get all signal data from .psout files for IMTB """
    IMTB_name = settings["IMTB name"]
    if "IMTB canvas" in settings.keys():
        IMTB_canvas = settings["IMTB canvas"]
    else:
        IMTB_canvas = "Main"
        
    # helping function for getting 3 phase signals
    def _get_traces(signal_call):
        traces = []
        for call in signal_call.calls():
            trace = run.trace(call)
            traces.append(trace.data.tolist())
        
        return traces
    
    def _get_time(signal_call):
        for call in signal_call.calls():
            trace = run.trace(call)
            domain = trace.domain
            time = domain.data.tolist() 
        return time
        
    with mhi.psout.File(filepath) as file:
        i_dut_call = file.call("Root/"+IMTB_canvas+"/"+IMTB_name+"/i_dut/0")
        v_dut_call = file.call("Root/"+IMTB_canvas+"/"+IMTB_name+"/v_dut/0")
        if settings["Calculate NET"]:
            i_net_call = file.call("Root/"+IMTB_canvas+"/"+IMTB_name+"/i_net/0")
            v_net_call = file.call("Root/"+IMTB_canvas+"/"+IMTB_name+"/v_net/0")
        
                
        run = file.run(0)
        
        data = list()
        for subtrace in _get_traces(i_dut_call): data.append(subtrace)
        for subtrace in _get_traces(v_dut_call): data.append(subtrace)
        if settings["Calculate NET"]:
            for subtrace in _get_traces(i_net_call): data.append(subtrace)
            for subtrace in _get_traces(v_net_call): data.append(subtrace)
        
    idx = 0
    for dataset in data:
        data[idx] = dataset[settings["idx_start"]:settings["idx_end"]+1]
        idx+=1

    return data
    
def get_time_idxs(settings, filepath):
    """ Helping function to get idxs for time data from .psout files for IMTB"""
    IMTB_name = settings["IMTB name"]
    if "IMTB canvas" in settings.keys():
        IMTB_canvas = settings["IMTB canvas"]
    else:
        IMTB_canvas = "Main"

    def _get_time(signal_call):
        for call in signal_call.calls():
            trace = run.trace(call)
            domain = trace.domain
            time = domain.data.tolist() 
        return time
    
    # get the time string    
    with mhi.psout.File(filepath) as file:
        
        call = file.call("Root/"+IMTB_canvas+"/"+IMTB_name+"/i_dut/0")
                
        run = file.run(0)
        
        data = _get_time(call)
        
    # get variables for easier use in function
    if settings["Snapshot function"]:
        snap_time = settings["Snapshot time"]        
    else:
        snap_time = 0
    
    settl_time = settings["Settling time"]
    inj_time = settings["Injection time"]
    timestep = settings["Plot timestep"]*1e-6
        
    # round the time array
    time = [round(x,ROUNDING_TO_EXP_TIME) for x in data]
    
    time_start = round(snap_time + settl_time + timestep, ROUNDING_TO_EXP_TIME)
    time_end = round(snap_time + settl_time + inj_time, ROUNDING_TO_EXP_TIME)
    duration = round(time_end-time_start, ROUNDING_TO_EXP_TIME)
    n_samples = int(round(duration/timestep,0))+1
    
    if time_start in time:
        idx_start = time.index(time_start)
    else:
        print("Problem to find starting index in IMTB_calculate. Check if PSCAD settings were set up correct. Trying anyway")
        idx_start = 0
        while time[idx_start]<time_start:
            idx_start+=1
     
    settings["idx_start"] = idx_start
    settings["idx_end"] = idx_start + n_samples-1
    settings["time_start"] = time_start
    return

def get_dftdata(settings, filepath, f_inj=None, calc_NET=False):
    """ Helping function to get the data format for immitance calc """
    data = read_raw(settings, filepath)
    f0 = settings["Fundamental freq"]
    
    dataout = list()
    for idx in range(6):
        if type(f_inj) == type(None):
            if calc_NET:
                f_dft, datathis = IM.get_FFT(data[idx+6], settings["fs"])
            else:
                f_dft, datathis = IM.get_FFT(data[idx], settings["fs"])
        else:
            f_dft = np.append(f_inj, f_inj-2*f0, f0)
            datathis = np.zeros(np.shape(f_dft), dtype="complex128")
            idx_f = 0
            for f in tqdm(f_dft):
                if calc_NET:
                    datathis[idx_f] = (IM.DFT_1f(data[idx+6], settings["fs"], f))
                else:
                    datathis[idx_f] = (IM.DFT_1f(data[idx], settings["fs"], f))
                idx_f += 1
        dataout.append(datathis)
    if type(f_dft) == type(np.array(0.0)):
        f_dft = np.ndarray.tolist(f_dft)
    return dataout, f_dft

def get_dftdata_ab(settings, filepath, f_inj=None,calc_NET=False):
    """ Helping function to get the data format for immitance calc """
    data = read_raw(settings, filepath)
    data_ab = []
    if calc_NET:
        ia = _get_alpha(data[6], data[7], data[8])
        ib = _get_beta(data[6], data[7], data[8])
        va = _get_alpha(data[9], data[10], data[11])
        vb = _get_beta(data[9], data[10], data[11])
    else:
        ia = _get_alpha(data[0], data[1], data[2])
        ib = _get_beta(data[0], data[1], data[2])
        va = _get_alpha(data[3], data[4], data[5])
        vb = _get_beta(data[3], data[4], data[5])
        
    data_ab.append(ia)
    data_ab.append(ib)
    data_ab.append(va)
    data_ab.append(vb)
    
    f0 = settings["Fundamental freq"]
    
    dataout = list()
    for idx in range(4):
        if type(f_inj) == type(None):
            f_dft, datathis = IM.get_FFT(data_ab[idx], settings["fs"])
                
        else:
            f_dft = np.append(f_inj, f_inj-2*f0, f0)
            datathis = np.zeros(np.shape(f_dft), dtype="complex128")
            idx_f = 0
            for f in tqdm(f_dft):
                datathis[idx_f] = (IM.DFT_1f(data[idx], settings["fs"], f))
                    
                idx_f += 1
        dataout.append(datathis)
    if type(f_dft) == type(np.array(0.0)):
        f_dft = np.ndarray.tolist(f_dft)
    return dataout, f_dft

def get_dftdata_1f(settings, filepath, f, calc_NET=False):
    """ Helping function to get the data format for immitance calc """
    data = read_raw(settings, filepath)
    
    dataout = list()
    for idx in range(6):
        if calc_NET:
            datathis = IM.DFT_1f(data[idx+6], settings["fs"], f) # this can calculate for negative frequency
        else:
            datathis = IM.DFT_1f(data[idx], settings["fs"], f) # this can calculate for negative frequency
        dataout.append(datathis)
    return dataout
def get_dftdata_ab_1f(settings, filepath, f, calc_NET=False):
    """ Helping function to get the data format for immitance calc """
    data = read_raw(settings, filepath)
    data_ab = []
    if calc_NET:
        ia = _get_alpha(data[6], data[7], data[8])
        ib = _get_beta(data[6], data[7], data[8])
        va = _get_alpha(data[9], data[10], data[11])
        vb = _get_beta(data[9], data[10], data[11])
    else:
        ia = _get_alpha(data[0], data[1], data[2])
        ib = _get_beta(data[0], data[1], data[2])
        va = _get_alpha(data[3], data[4], data[5])
        vb = _get_beta(data[3], data[4], data[5])

    data_ab.append(ia)
    data_ab.append(ib)
    data_ab.append(va)
    data_ab.append(vb)
    
    dataout = list()
    for idx in range(4):
        datathis = IM.DFT_1f(data_ab[idx], settings["fs"], f) # this can calculate for negative frequency
            
        dataout.append(datathis)
    return dataout
# global help variables and functions
a = np.exp(1j*2/3*np.pi)
a2 = np.exp(-1j*2/3*np.pi)

def _get_pos(A, B, C):
    return (A + a*B + a2*C)/3

def _get_neg(A, B, C):
    return (A + a2*B + a*C)/3

def _get_zero(A, B, C):
    return (A + B + C)/3
    
def _get_alpha(A, B, C):
    if len(A) == 1:
        return 2*(A - 0.5*B - 0.5*C)/3
    else:
        out=[]
        for idx in range(len(A)):
            out.append(2*(A[idx] - 0.5*B[idx] - 0.5*C[idx])/3)
        return out

def _get_beta(A, B, C):
    if len(A) == 1:
        return (np.sqrt(3)*B - np.sqrt(3)*C)/3
    else:
        out=[]
        for idx in range(len(A)):
            out.append((np.sqrt(3)*B[idx] - np.sqrt(3)*C[idx])/3)
        return out

def _get_pos_fra_ab(a,b):
    return a+1j*b

def _get_neg_fra_ab(a,b):
    return a-1j*b


def calc_Yf_SISO(inj_comp, data_f, data_noinj, idx_f):
    """ Calculate admittance for one frequency """
    
    # get voltages and current for later calculation
    if inj_comp == "pos":
        # Positive sequence component
                
        if NEW_MIMO_CALC_METHOD:
            I_noinj = _get_pos(data_noinj[0][idx_f],
                               data_noinj[1][idx_f],
                               data_noinj[2][idx_f],)
            V_noinj = _get_pos(data_noinj[3][idx_f],
                               data_noinj[4][idx_f],
                               data_noinj[5][idx_f],)
            
            I_f = _get_pos(data_f[0],
                           data_f[1],
                           data_f[2],)
            V_f = _get_pos(data_f[3],
                           data_f[4],
                           data_f[5],)
        else:
            I_noinj = _get_pos_fra_ab(data_noinj[0][idx_f],
                               data_noinj[1][idx_f])
            V_noinj = _get_pos_fra_ab(data_noinj[2][idx_f],
                               data_noinj[3][idx_f])
            
            I_f = _get_pos_fra_ab(data_f[0],
                           data_f[1])
            V_f = _get_pos_fra_ab(data_f[2],
                           data_f[3])
            
        
        
    elif inj_comp == "neg":
        # Negative sequence component
        if NEW_MIMO_CALC_METHOD:
            I_noinj = _get_neg(data_noinj[0][idx_f],
                               data_noinj[1][idx_f],
                               data_noinj[2][idx_f],)
            V_noinj = _get_neg(data_noinj[3][idx_f],
                               data_noinj[4][idx_f],
                               data_noinj[5][idx_f],)
            
            I_f = _get_neg(data_f[0],
                           data_f[1],
                           data_f[2],)
            V_f = _get_neg(data_f[3],
                           data_f[4],
                           data_f[5],)
        else:
            I_noinj = _get_neg_fra_ab(data_noinj[0][idx_f],
                               data_noinj[1][idx_f])
            V_noinj = _get_neg_fra_ab(data_noinj[2][idx_f],
                               data_noinj[3][idx_f])
            
            I_f = _get_neg_fra_ab(data_f[0],
                           data_f[1])
            V_f = _get_neg_fra_ab(data_f[2],
                           data_f[3])
        
    
    elif inj_comp == "zero":
        # Zero component
        
        I_noinj = _get_zero(data_noinj[0][idx_f],
                           data_noinj[1][idx_f],
                           data_noinj[2][idx_f],)
        V_noinj = _get_zero(data_noinj[3][idx_f],
                           data_noinj[4][idx_f],
                           data_noinj[5][idx_f],)
        
        I_f = _get_zero(data_f[0],
                       data_f[1],
                       data_f[2],)
        V_f = _get_zero(data_f[3],
                       data_f[4],
                       data_f[5],)
        
    elif inj_comp == "a":
        # A phase only
        
        I_noinj = data_noinj[0][idx_f]
        V_noinj = data_noinj[3][idx_f]
                  
        I_f = data_f[0]
        V_f = data_f[3]
    
    elif inj_comp == "b":
        # B phase only
        
        I_noinj = data_noinj[1][idx_f]
        V_noinj = data_noinj[4][idx_f]
                  
        I_f = data_f[1]
        V_f = data_f[4]
        
    elif inj_comp == "c":
        # A phase only
        
        I_noinj = data_noinj[2][idx_f]
        V_noinj = data_noinj[5][idx_f]
                  
        I_f = data_f[2]
        V_f = data_f[5]
    
    elif inj_comp == "alpha":
        # Alpha component from Clarke
        
        I_noinj = _get_alpha(data_noinj[0][idx_f],
                             data_noinj[1][idx_f],
                             data_noinj[2][idx_f],)
        V_noinj = _get_alpha(data_noinj[3][idx_f],
                             data_noinj[4][idx_f],
                             data_noinj[5][idx_f],)
        
        I_f = _get_alpha(data_f[0],
                         data_f[1],
                         data_f[2],)
        V_f = _get_alpha(data_f[3],
                         data_f[4],
                         data_f[5],)
    
    elif inj_comp == "beta":
        # Beta component from Clarke
        
        I_noinj = _get_beta(data_noinj[0][idx_f],
                            data_noinj[1][idx_f],
                            data_noinj[2][idx_f],)
        V_noinj = _get_beta(data_noinj[3][idx_f],
                            data_noinj[4][idx_f],
                            data_noinj[5][idx_f],)
        
        I_f = _get_beta(data_f[0],
                        data_f[1],
                        data_f[2],)
        V_f = _get_beta(data_f[3],
                        data_f[4],
                        data_f[5],)
        
    elif inj_comp == "ab":
        # DC calculation of positive pole (between A and B)
        
        I_noinj = data_noinj[0][idx_f]
        V_noinj = data_noinj[3][idx_f]-data_noinj[4][idx_f]
        
        I_f = data_f[0]
        V_f = data_f[3]-data_f[4]
        
    elif inj_comp == "cb":
        # DC calculation of negative pole (between C and B)
        
        I_noinj = data_noinj[2][idx_f]
        V_noinj = data_noinj[5][idx_f]-data_noinj[4][idx_f]
        
        I_f = data_f[2]
        V_f = data_f[5]-data_f[4]
    
    else:
        print("ERROR! Calculation not implemented yet!") 
        return None
    
    # Calculate addmittance and return
    if ENABLE_DEDUCTION_DFT_NONINJ:
        Y = (I_f - I_noinj) / (V_f - V_noinj)
    else:
        Y = I_f / V_f 
    
    return Y

def calc_Yf_DC_MIMO(inj_comp, data_f, data_noinj, idx_f):
    """ Getting voltages and currents for admittance calculation for DC MIMO """
    
    # get voltages and current for later calculation
    
    I_noinj_a = data_noinj[0][idx_f]
    I_noinj_c = data_noinj[2][idx_f]
    V_noinj_ab = data_noinj[3][idx_f]-data_noinj[4][idx_f]
    V_noinj_cb = data_noinj[5][idx_f]-data_noinj[4][idx_f]
    
    I_f_a = data_f[0]
    I_f_c = data_f[2]
    V_f_ab = data_f[3]-data_f[4]
    V_f_cb = data_f[5]-data_f[4]
    
    
    # Get voltages and currents for later addmittance calculation
    if ENABLE_DEDUCTION_DFT_NONINJ:
        Ia = I_f_a - I_noinj_a
        Ic = I_f_c - I_noinj_c
        Vab = V_f_ab - V_noinj_ab
        Vcb = V_f_cb - V_noinj_cb
    else:
        Ia = I_f_a
        Ic = I_f_c
        Vab = V_f_ab
        Vcb = V_f_cb
        
    return Ia, Ic, Vab, Vcb

def calc_Yf_ab_complex(inj_comp, data_f1, data_f2, data_noinj, f_dft, f, f0):
    """ Calculate admittance for one frequency """
    f1 = f
    f2 = f-2*f0
    
    if abs(f1) in f_dft:
        if f1>=0:
            idx_f = f_dft.index(f1)
            
            if NEW_MIMO_CALC_METHOD:
                I_noinj = _get_pos(data_noinj[0][idx_f],
                                   data_noinj[1][idx_f],
                                   data_noinj[2][idx_f],)
                V_noinj = _get_pos(data_noinj[3][idx_f],
                                   data_noinj[4][idx_f],
                                   data_noinj[5][idx_f],)
                
                I_f = _get_pos(data_f1[0],
                               data_f1[1],
                               data_f1[2],)
                V_f = _get_pos(data_f1[3],
                               data_f1[4],
                               data_f1[5],)
            else:
                I_noinj = _get_pos_fra_ab(data_noinj[0][idx_f],
                                   data_noinj[1][idx_f],)
                V_noinj = _get_pos_fra_ab(data_noinj[2][idx_f],
                                   data_noinj[3][idx_f],)
                
                I_f = _get_pos_fra_ab(data_f1[0],
                               data_f1[1])
                V_f = _get_pos_fra_ab(data_f1[2],
                               data_f1[3],)
            
            if ENABLE_DEDUCTION_DFT_NONINJ:
                I_f1 = I_f - I_noinj
                V_f1 = V_f - V_noinj
            else:
                I_f1 = I_f 
                V_f1 = V_f
            
             
        else:
            idx_f = f_dft.index(-f1)
            
            if NEW_MIMO_CALC_METHOD:
                I_noinj = _get_pos(data_noinj[0][idx_f],
                                   data_noinj[1][idx_f],
                                   data_noinj[2][idx_f],)
                V_noinj = _get_pos(data_noinj[3][idx_f],
                                   data_noinj[4][idx_f],
                                   data_noinj[5][idx_f],)
                
                I_f = _get_pos(data_f1[0],
                               data_f1[1],
                               data_f1[2],)
                V_f = _get_pos(data_f1[3],
                               data_f1[4],
                               data_f1[5],)
            else:
                I_noinj = _get_pos_fra_ab(data_noinj[0][idx_f],
                                   data_noinj[1][idx_f],)
                V_noinj = _get_pos_fra_ab(data_noinj[2][idx_f],
                                   data_noinj[3][idx_f],)
                
                I_f = _get_pos_fra_ab(data_f1[0],
                               data_f1[1])
                V_f = _get_pos_fra_ab(data_f1[2],
                               data_f1[3],)
            
            if ENABLE_DEDUCTION_DFT_NONINJ:
                I_f1 = I_f - np.conj(I_noinj)
                V_f1 = V_f - np.conj(V_noinj)
            else:
                I_f1 = I_f 
                V_f1 = V_f
            
             
    else:
        print("f1 not in f_dft. Check injection time window for MIMO scan.")
        return None
        
    if f0 in f_dft:
        idx_f = f_dft.index(f0)
        
        if NEW_MIMO_CALC_METHOD:
            V_f0 = _get_pos(data_noinj[3][idx_f],
                               data_noinj[4][idx_f],
                               data_noinj[5][idx_f],)
        else:
            V_f0 = _get_pos_fra_ab(data_noinj[2][idx_f],
                               data_noinj[3][idx_f])
                
        
        phi_f0 = np.angle(V_f0)
    
    if abs(f2) in f_dft:
        if f2>=0:
            idx_f = f_dft.index(f2)
            
            if NEW_MIMO_CALC_METHOD:
                I_noinj = _get_pos(data_noinj[0][idx_f],
                                   data_noinj[1][idx_f],
                                   data_noinj[2][idx_f],)
                V_noinj = _get_pos(data_noinj[3][idx_f],
                                   data_noinj[4][idx_f],
                                   data_noinj[5][idx_f],)
                
                I_f = _get_pos(data_f1[0],
                               data_f1[1],
                               data_f1[2],)
                V_f = _get_pos(data_f1[3],
                               data_f1[4],
                               data_f1[5],)
            else:
                I_noinj = _get_pos_fra_ab(data_noinj[0][idx_f],
                                   data_noinj[1][idx_f],)
                V_noinj = _get_pos_fra_ab(data_noinj[2][idx_f],
                                   data_noinj[3][idx_f],)
                
                I_f = _get_pos_fra_ab(data_f1[0],
                               data_f1[1])
                V_f = _get_pos_fra_ab(data_f1[2],
                               data_f1[3],)
                
            if ENABLE_DEDUCTION_DFT_NONINJ:
                I_f2 = (I_f - I_noinj) * cmath.exp(2j*phi_f0)
                V_f2 = (V_f - V_noinj) * cmath.exp(2j*phi_f0)
            else:
                I_f2 = I_f * cmath.exp(2j*phi_f0)
                V_f2 = V_f * cmath.exp(2j*phi_f0)
            
            
        else:
            idx_f = f_dft.index(-f2)
            
            if NEW_MIMO_CALC_METHOD:
                I_noinj = _get_pos(data_noinj[0][idx_f],
                                   data_noinj[1][idx_f],
                                   data_noinj[2][idx_f],)
                V_noinj = _get_pos(data_noinj[3][idx_f],
                                   data_noinj[4][idx_f],
                                   data_noinj[5][idx_f],)
                
                I_f = _get_pos(data_f1[0],
                               data_f1[1],
                               data_f1[2],)
                V_f = _get_pos(data_f1[3],
                               data_f1[4],
                               data_f1[5],)
            else:
                I_noinj = _get_pos_fra_ab(data_noinj[0][idx_f],
                                   data_noinj[1][idx_f],)
                V_noinj = _get_pos_fra_ab(data_noinj[2][idx_f],
                                   data_noinj[3][idx_f],)
                
                I_f = _get_pos_fra_ab(data_f1[0],
                               data_f1[1])
                V_f = _get_pos_fra_ab(data_f1[2],
                               data_f1[3],)
                
            if ENABLE_DEDUCTION_DFT_NONINJ:
                I_f2 = (I_f - np.conj(I_noinj)) * cmath.exp(2j*phi_f0)
                V_f2 = (V_f - np.conj(V_noinj)) * cmath.exp(2j*phi_f0)
            else:
                I_f2 = I_f * cmath.exp(2j*phi_f0)
                V_f2 = V_f * cmath.exp(2j*phi_f0)
            
    else:
        print("f2 not in f_dft. Check injection time window for MIMO scan.")
        return None
        

    return I_f1,V_f1,I_f2,V_f2
    
def calc_Yf_dq_complex(inj_comp, data_f1, data_f2, data_noinj, f_dft, f, f0):
    """ Calculate admittance for one frequency """
    f1 = f # f0+fdq = 51 Hz
    f2 = f-2*f0 # fdq-f0= -49 Hz
    # fdq 1 Hz
    if f0 in f_dft:
        idx_f = f_dft.index(f0)
        if NEW_MIMO_CALC_METHOD:
            V_f0 = _get_pos(data_noinj[3][idx_f],
                               data_noinj[4][idx_f],
                               data_noinj[5][idx_f],)
        else:
            V_f0 = _get_pos_fra_ab(data_noinj[2][idx_f],
                               data_noinj[3][idx_f])
                

        phi_f0 = np.angle(V_f0)
    
    if abs(f1) in f_dft:
        if f1>=0:
            idx_f = f_dft.index(f1)
            if NEW_MIMO_CALC_METHOD:
                I_noinj = _get_pos(data_noinj[0][idx_f],
                                   data_noinj[1][idx_f],
                                   data_noinj[2][idx_f],)
                V_noinj = _get_pos(data_noinj[3][idx_f],
                                   data_noinj[4][idx_f],
                                   data_noinj[5][idx_f],)
                
                I_f = _get_pos(data_f1[0],
                               data_f1[1],
                               data_f1[2],)
                V_f = _get_pos(data_f1[3],
                               data_f1[4],
                               data_f1[5],)
            else:
                I_noinj = _get_pos_fra_ab(data_noinj[0][idx_f],
                                   data_noinj[1][idx_f],)
                V_noinj = _get_pos_fra_ab(data_noinj[2][idx_f],
                                   data_noinj[3][idx_f],)
                
                I_f = _get_pos_fra_ab(data_f1[0],
                               data_f1[1])
                V_f = _get_pos_fra_ab(data_f1[2],
                               data_f1[3],)
            
            
            if ENABLE_DEDUCTION_DFT_NONINJ:
                I1_fdq = (I_f - I_noinj) * cmath.exp(-1j*phi_f0) # 51 Hz pos -> 1 Hz dqp
                V1_fdq = (V_f - V_noinj) * cmath.exp(-1j*phi_f0)
            else:
                I1_fdq = I_f * cmath.exp(-1j*phi_f0)
                V1_fdq = V_f * cmath.exp(-1j*phi_f0)
            
             
        else:
            idx_f = f_dft.index(-f1)
            
            if NEW_MIMO_CALC_METHOD:
                I_noinj = _get_pos(data_noinj[0][idx_f],
                                   data_noinj[1][idx_f],
                                   data_noinj[2][idx_f],)
                V_noinj = _get_pos(data_noinj[3][idx_f],
                                   data_noinj[4][idx_f],
                                   data_noinj[5][idx_f],)
                
                I_f = _get_pos(data_f1[0],
                               data_f1[1],
                               data_f1[2],)
                V_f = _get_pos(data_f1[3],
                               data_f1[4],
                               data_f1[5],)
            else:
                I_noinj = _get_pos_fra_ab(data_noinj[0][idx_f],
                                   data_noinj[1][idx_f],)
                V_noinj = _get_pos_fra_ab(data_noinj[2][idx_f],
                                   data_noinj[3][idx_f],)
                
                I_f = _get_pos_fra_ab(data_f1[0],
                               data_f1[1])
                V_f = _get_pos_fra_ab(data_f1[2],
                               data_f1[3],)
            
            if ENABLE_DEDUCTION_DFT_NONINJ:
                I1_fdq = (I_f - np.conj(I_noinj)) * cmath.exp(-1j*phi_f0)
                V1_fdq = (V_f - np.conj(V_noinj)) * cmath.exp(-1j*phi_f0)
            else:
                I1_fdq = I_f * cmath.exp(-1j*phi_f0)
                V1_fdq = V_f * cmath.exp(-1j*phi_f0)
            
             
    else:
        print("f1 not in f_dft. Check injection time window for MIMO scan.")
        return None
        
    
    
    if abs(f2) in f_dft: # -49 Hz
        if f2>=0:
            idx_f = f_dft.index(f2)
            
            if NEW_MIMO_CALC_METHOD:
                I_noinj = _get_pos(data_noinj[0][idx_f],
                                   data_noinj[1][idx_f],
                                   data_noinj[2][idx_f],)
                V_noinj = _get_pos(data_noinj[3][idx_f],
                                   data_noinj[4][idx_f],
                                   data_noinj[5][idx_f],)
                
                I_f = _get_pos(data_f1[0],
                               data_f1[1],
                               data_f1[2],)
                V_f = _get_pos(data_f1[3],
                               data_f1[4],
                               data_f1[5],)
            else:
                I_noinj = _get_pos_fra_ab(data_noinj[0][idx_f],
                                   data_noinj[1][idx_f],)
                V_noinj = _get_pos_fra_ab(data_noinj[2][idx_f],
                                   data_noinj[3][idx_f],)
                
                I_f = _get_pos_fra_ab(data_f1[0],
                               data_f1[1])
                V_f = _get_pos_fra_ab(data_f1[2],
                               data_f1[3],)
            
            if ENABLE_DEDUCTION_DFT_NONINJ:
                I2_fdq = (I_f - I_noinj) * cmath.exp(1j*phi_f0)
                V2_fdq = (V_f - V_noinj) * cmath.exp(1j*phi_f0)
            else:
                I2_fdq = I_f * cmath.exp(1j*phi_f0)
                V2_fdq = V_f * cmath.exp(1j*phi_f0)
            
            
        else: # -49 Hz, neg   Id-jIq (1Hz) = exp(theta) * (Ia-jIb) (-49Hz)
            idx_f = f_dft.index(-f2) # get 49 Hz
            
            if NEW_MIMO_CALC_METHOD:
                I_noinj = _get_pos(data_noinj[0][idx_f],
                                   data_noinj[1][idx_f],
                                   data_noinj[2][idx_f],)
                V_noinj = _get_pos(data_noinj[3][idx_f],
                                   data_noinj[4][idx_f],
                                   data_noinj[5][idx_f],)
                
                I_f = _get_pos(data_f1[0],
                               data_f1[1],
                               data_f1[2],)
                V_f = _get_pos(data_f1[3],
                               data_f1[4],
                               data_f1[5],)
            else:
                I_noinj = _get_pos_fra_ab(data_noinj[0][idx_f],
                                   data_noinj[1][idx_f],)
                V_noinj = _get_pos_fra_ab(data_noinj[2][idx_f],
                                   data_noinj[3][idx_f],)
                
                I_f = _get_pos_fra_ab(data_f1[0],
                               data_f1[1])
                V_f = _get_pos_fra_ab(data_f1[2],
                               data_f1[3],)
            
            if ENABLE_DEDUCTION_DFT_NONINJ:
                I2_fdq = (I_f - np.conj(I_noinj)) * cmath.exp(1j*phi_f0)
                V2_fdq = (V_f - np.conj(V_noinj)) * cmath.exp(1j*phi_f0)
            else:
                I2_fdq = I_f * cmath.exp(1j*phi_f0)
                V2_fdq = V_f * cmath.exp(1j*phi_f0)
            
    else:
        print("f2 not in f_dft. Check injection time window for MIMO scan.")
        return None
        

    return I1_fdq,V1_fdq,I2_fdq,V2_fdq

def save_to_files(filename,str_list,data_list):
    data = {}
    for k in range(len(str_list)):
        data[str_list[k]] = data_list[k]
    
    df = pd.DataFrame.from_dict(data)
    df.to_csv(filename+".csv",index = False)
    # df.to_excel(filename+".xlsx",index = False)
    return data

def plot_noinj(settings, f, data, name):
    """ Plotting harmonic data for no injection """
    
    # Creating plotly figure object
    fig = make_subplots(rows=2,cols=1, shared_xaxes=True,)
    
        
    for idx in range(6):
        # Creating a line and adding to fig
        trace_obj = go.Scatter(x=f, y=np.abs(data[idx]))
        if idx <= 2: # for currents
            fig.append_trace(trace_obj, row=2, col=1)
        else: # for voltages
            fig.append_trace(trace_obj, row=1, col=1)
        
    
    title = settings["Timestamp"] + " - " + settings["simname"] + " - Harmonics profile without injection"
    
    # Add main title and axis labels
    fig.update_layout(title_text=title)
    fig.update_layout(template="ggplot2")
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
    fig.update_yaxes(title_text="Voltage mag. (kV)", row=1, col=1)
    fig.update_yaxes(title_text="Current mag. (kA)", row=2, col=1)
        
    
    # saving fig to html
    fig.write_html((settings["Working folder"] + 
                   "\\IMTB_data\\" + 
                   settings["Timestamp"] +"_"+ settings["simname"] + 
                   "\\"+settings["simname"]+"_"+name+"harmonics.html"))
    print("Harmonic profile without injection plot saved")
    return

# get Fourier response at f using y_window data
def get_Yf(f,fs,y_window):
    # get FFT result at f    
    if len(y_window) !=0:
        # Using FFT
        # f_fft, Y_fft = IM.get_FFT(y_window, fs)
        # Y_f = IM.get_FFT_at_f_v2(f_fft, Y_fft, f)
    
        # Using DFT at single frequency
        Y_f = IM.DFT_1f(y_window, fs, f) # new version
        
        
        return Y_f

# get Fourier responses for complex values in alpha-beta frame for postive-sequence impedance calculation
def get_Yab1_complex_value(f,fs,yalpha,ybeta):
    y_window = yalpha
    # response at f
    Yalpha_f = get_Yf(f, fs, y_window)

    y_window = ybeta
    # response at f
    Ybeta_f = get_Yf(f, fs, y_window)
        
    Yab1 = Yalpha_f + 1j*Ybeta_f   
    return Yab1
        
# get Fourier responses for complex values in alpha-beta frame for negative-sequence impedance calculation
def get_Yab2_complex_value(f,fs,yalpha,ybeta):
    y_window = yalpha
    # response at f
    Yalpha_f = get_Yf(f, fs, y_window)

    y_window = ybeta
    # response at f
    Ybeta_f = get_Yf(f, fs, y_window)
        
    Yab2 = Yalpha_f - 1j*Ybeta_f   
    return Yab2

# get Fourier responses for complex vectors in alpha-beta frame for MIMO impedance calculation
def get_Yab_complex_vector(f,f0,phi_f0,fs,yalpha,ybeta):
    
    y_window = yalpha 
    # response at f
    Yalpha_f = get_Yf(f, fs, y_window)
    # response at f-2f0
    Yalpha_f2 = get_Yf((f-2*f0), fs, y_window)
    
    y_window = ybeta
    # response at f
    Ybeta_f = get_Yf(f, fs, y_window)
    # response at f-2f0
    Ybeta_f2 = get_Yf((f-2*f0), fs, y_window)
    
    
    Yab1 = Yalpha_f + 1j*Ybeta_f
    Yab2 = (Yalpha_f2 - 1j*Ybeta_f2)*cmath.exp(2j*phi_f0)
        
    return Yab1, Yab2

# get Fourier responses for dq vectors in dq frame for MIMO impedance calculation
# calc_Vq: True - it is a voltage phasor, Vq to be adjusted as 0
#          False - it is a current phasor, no change of initial phase
def get_Ydq_real_vector(f,f0,phi_f0,fs,yalpha,ybeta,calc_Vq = False):
    
    ya_window = yalpha 
    yb_window = ybeta
    
    yd_window = []
    yq_window = []   
    
    
    for idx in range(len(ya_window)):

        t_idx = 1/fs*idx 
        ya = ya_window[idx]
        yb = yb_window[idx]
        
        phi = 2*np.pi*f0*t_idx + phi_f0
        
        yd = np.cos(phi) * ya + np.sin(phi) * yb
        yq = - np.sin(phi) * ya + np.cos(phi) * yb
        
        yd_window.append(yd)
        yq_window.append(yq)
        
    if calc_Vq: # for voltage phasor
        Vd = np.mean(yd_window)
        Vq = np.mean(yq_window)
        # print("Vq = %.6f" %Vq)
        # print("Vd = %.6f" %Vd)
        angle =float(np.angle(Vd+1j*Vq))
        # print(angle)
        
        phi_f0_cmp = phi_f0 + angle
        
        yd_window = []
        yq_window = []   
        
        # Do a dq transformation compensation to get Vq=0
        for idx in range(len(ya_window)):

            t_idx = 1/fs*idx 
            ya = ya_window[idx]
            yb = yb_window[idx]
            
            phi = 2*np.pi*f0*t_idx + phi_f0_cmp
            
            yd = np.cos(phi) * ya + np.sin(phi) * yb
            yq = - np.sin(phi) * ya + np.cos(phi) * yb
            
            yd_window.append(yd)
            yq_window.append(yq)
            

    else:
        phi_f0_cmp = phi_f0
        
    
    # response at f
    Yd_f = get_Yf(f, fs, yd_window)
    Yq_f = get_Yf(f, fs, yq_window)
    
        
    return Yd_f, Yq_f, phi_f0_cmp
    
def IMTB_AC_Zab_MIMO(data_pos, data_neg, f, phi_f0, settings, calc_NET=False):
        
    fs = settings["fs"]
    f0 = float(settings["Fundamental freq"])
    
        
    fs = settings["fs"]
    f0 = float(settings["Fundamental freq"])
    
    if calc_NET:
        print('NET side impedance calculation:')
        ialpha1 = _get_alpha(data_pos[6], data_pos[7], data_pos[8])
        ibeta1 = _get_beta(data_pos[6], data_pos[7], data_pos[8])
        valpha1 = _get_alpha(data_pos[9], data_pos[10], data_pos[11])
        vbeta1 = _get_beta(data_pos[9], data_pos[10], data_pos[11])
        
        ialpha2 = _get_alpha(data_neg[6], data_neg[7], data_neg[8])
        ibeta2 = _get_beta(data_neg[6], data_neg[7], data_neg[8])
        valpha2 = _get_alpha(data_neg[9], data_neg[10], data_neg[11])
        vbeta2 = _get_beta(data_neg[9], data_neg[10], data_neg[11])
    else:
        print('DUT side impedance calculation:')
        ialpha1 = _get_alpha(data_pos[0], data_pos[1], data_pos[2])
        ibeta1 = _get_beta(data_pos[0], data_pos[1], data_pos[2])
        valpha1 = _get_alpha(data_pos[3], data_pos[4], data_pos[5])
        vbeta1 = _get_beta(data_pos[3], data_pos[4], data_pos[5])
        
        ialpha2 = _get_alpha(data_neg[0], data_neg[1], data_neg[2])
        ibeta2 = _get_beta(data_neg[0], data_neg[1], data_neg[2])
        valpha2 = _get_alpha(data_neg[3], data_neg[4], data_neg[5])
        vbeta2 = _get_beta(data_neg[3], data_neg[4], data_neg[5])

    
    # V at pos scan - 1st inj
    Vab1_inj1,Vab2_inj1 = get_Yab_complex_vector(f,f0,phi_f0,fs,valpha1,vbeta1)

    # I at pos scan - 1st inj
    Iab1_inj1,Iab2_inj1 = get_Yab_complex_vector(f,f0,phi_f0,fs,ialpha1,ibeta1)    

    
    # V at neg scan - 2nd inj
    Vab1_inj2,Vab2_inj2 = get_Yab_complex_vector(f,f0,phi_f0,fs,valpha2,vbeta2)

    # I at neg scan - 2nd inj
    Iab1_inj2,Iab2_inj2 = get_Yab_complex_vector(f,f0,phi_f0,fs,ialpha2,ibeta2)            
    
    Vmat = np.array([[Vab1_inj1,Vab1_inj2],[Vab2_inj1,Vab2_inj2]]) 
    Imat = np.array([[Iab1_inj1,Iab1_inj2],[Iab2_inj1,Iab2_inj2]]) 
    # print(Vmat)
    # print(Imat)
    
    Zab = np.matmul(Vmat,lnlg.inv(Imat))
    # print(Zab)
    
    return Zab

def IMTB_AC_Zdq_MIMO(data_pos, data_neg, f, phi_f0, settings, calc_NET=False):

    fs = settings["fs"]
    f0 = float(settings["Fundamental freq"])
    # time_start = settings["time_start"]
    # print(f0)
    if calc_NET:
        print('NET side impedance calculation:')
        ialpha1 = _get_alpha(data_pos[6], data_pos[7], data_pos[8])
        ibeta1 = _get_beta(data_pos[6], data_pos[7], data_pos[8])
        valpha1 = _get_alpha(data_pos[9], data_pos[10], data_pos[11])
        vbeta1 = _get_beta(data_pos[9], data_pos[10], data_pos[11])
        
        ialpha2 = _get_alpha(data_neg[6], data_neg[7], data_neg[8])
        ibeta2 = _get_beta(data_neg[6], data_neg[7], data_neg[8])
        valpha2 = _get_alpha(data_neg[9], data_neg[10], data_neg[11])
        vbeta2 = _get_beta(data_neg[9], data_neg[10], data_neg[11])
    else:
        print('DUT side impedance calculation:')
        ialpha1 = _get_alpha(data_pos[0], data_pos[1], data_pos[2])
        ibeta1 = _get_beta(data_pos[0], data_pos[1], data_pos[2])
        valpha1 = _get_alpha(data_pos[3], data_pos[4], data_pos[5])
        vbeta1 = _get_beta(data_pos[3], data_pos[4], data_pos[5])
        
        ialpha2 = _get_alpha(data_neg[0], data_neg[1], data_neg[2])
        ibeta2 = _get_beta(data_neg[0], data_neg[1], data_neg[2])
        valpha2 = _get_alpha(data_neg[3], data_neg[4], data_neg[5])
        vbeta2 = _get_beta(data_neg[3], data_neg[4], data_neg[5])
    
    # V at pos scan - 1st inj
    Vd_inj1,Vq_inj1,phi_f0_cmp = get_Ydq_real_vector(f-f0, f0, phi_f0, fs, valpha1, vbeta1,calc_Vq = True)
    
    # I at pos scan - 1st inj
    Id_inj1,Iq_inj1,phi_f0_cmp = get_Ydq_real_vector(f-f0, f0, phi_f0_cmp, fs, ialpha1, ibeta1,calc_Vq = False)
    
    
    # V at pos scan - 2nd inj
    Vd_inj2,Vq_inj2,phi_f0_cmp = get_Ydq_real_vector(f-f0, f0, phi_f0, fs, valpha2, vbeta2,calc_Vq = True)

    # I at pos scan - 2nd inj
    Id_inj2,Iq_inj2,phi_f0_cmp = get_Ydq_real_vector(f-f0, f0, phi_f0_cmp, fs, ialpha2, ibeta2,calc_Vq = False)
    
       
    Vmat = np.array([[Vd_inj1,Vd_inj2],[Vq_inj1,Vq_inj2]]) 
    Imat = np.array([[Id_inj1,Id_inj2],[Iq_inj1,Iq_inj2]]) 
    
    Zdq = np.matmul(Vmat,lnlg.inv(Imat))
    
    
    return Zdq




def Single_IM_calc(rawfolder, sc_name, f_inj, settings, DQ_calc):
    
    # Selection if SISO or MIMO procedure
    if settings["Immitance type"]=="SISO":
        # Define helping variables
        inj_comp = settings["Injection components"]
        rawfile_prefix = rawfolder+"\\"+settings["simname"]+"_"+sc_name+inj_comp+"_"
        
        # Get indexes for time series signals
        print("Finding time idexes for DFT")
        get_time_idxs(settings, rawfile_prefix+"01.psout")
        
        # get all data for no inj
        print("Getting DFT of no injection file")
        if NEW_MIMO_CALC_METHOD:
            data_noinj, f_dft = get_dftdata(settings, rawfile_prefix+"01.psout")
            if settings["Calculate NET"]:
                data_noinj_net, f_dft_net = get_dftdata(settings, rawfile_prefix+"01.psout",calc_NET=True)

        else:
            if settings["Terminal type"]=="AC":
                data_noinj, f_dft = get_dftdata_ab(settings, rawfile_prefix+"01.psout")
                if settings["Calculate NET"]:
                    data_noinj_net, f_dft_net = get_dftdata_ab(settings, rawfile_prefix+"01.psout",calc_NET=True)
                    
                    
            elif settings["Terminal type"]=="DC":
                data_noinj, f_dft = get_dftdata(settings, rawfile_prefix+"01.psout")
                if settings["Calculate NET"]:
                    data_noinj_net, f_dft_net = get_dftdata(settings, rawfile_prefix+"01.psout",calc_NET=True)

                
        # Plotting no injection data 
        if settings["Plot harmonics"]:
            data_harm, f_harm = get_dftdata(settings, rawfile_prefix+"01.psout")
            plot_noinj(settings, f_harm, data_harm, sc_name)
        
        # Here stuff for the power flow check!
        #!!!
        if NEW_MIMO_CALC_METHOD or settings["Terminal type"]=="DC": 
            # initialize impedance
            Y = np.zeros((len(f_inj)-1),dtype="complex")
            
            print("Calculating DUT immitance data for: "+ inj_comp)
            idx = 1
            for f in tqdm(f_inj[1:]):
                                
                # set up all filenames and indexes
                rawfile_this = rawfile_prefix+f"{idx+1:02d}.psout"
                if f in f_dft:
                    idx_f = f_dft.index(f)
                    # get DFT for ABC components
                    data_f = get_dftdata_1f(settings, rawfile_this, f)
                    
                    Y[idx-1] = calc_Yf_SISO(inj_comp, data_f, data_noinj, idx_f)
                    
                    # Here is room to check if calculated correctly
                    #!!!
                else:
                    print(f"Injection frequency can not be calculated or rounding error at {f} Hz!")  
                idx+=1
            print("Done.")
            
            print("Recalculating to impedance data and saving")
            # recalculate for impedance and other types
            Z = 1/Y       
            Z_TF = IM.TF(f_inj[1:],Z)
            
            if settings["Calculate NET"]:
                Y = np.zeros((len(f_inj)-1),dtype="complex")
                
                print("Calculating NET immitance data for: "+ inj_comp)
                idx = 1
                for f in tqdm(f_inj[1:]):
                                    
                    # set up all filenames and indexes
                    rawfile_this = rawfile_prefix+f"{idx+1:02d}.psout"
                    if f in f_dft:
                        idx_f = f_dft.index(f)
                        # get DFT for ABC components
                        data_f = get_dftdata_1f(settings, rawfile_this, f, calc_NET=True)
                        
                        Y[idx-1] = calc_Yf_SISO(inj_comp, data_f, data_noinj_net, idx_f)
                        
                        # Here is room to check if calculated correctly
                        #!!!
                    else:
                        print(f"Injection frequency can not be calculated or rounding error at {f} Hz!")  
                    idx+=1
                print("Done.")
                
                print("Recalculating to impedance data and saving")
                # recalculate for impedance and other types
                Z = 1/Y       
                Znet_TF = IM.TF(f_inj[1:],Z)
        else:
            
            # initialize impedance
            Y = np.zeros((len(f_inj)-1),dtype="complex")
            
            print("Calculating DUT immitance data for: "+ inj_comp)
            idx = 1
            for f in tqdm(f_inj[1:]):
                                
                # set up all filenames and indexes
                rawfile_this = rawfile_prefix+f"{idx+1:02d}.psout"
                if f in f_dft:
                    idx_f = f_dft.index(f)
                    # get DFT for alpha-beta components
                    data_f = get_dftdata_ab_1f(settings, rawfile_this, f)
                    
                    Y[idx-1] = calc_Yf_SISO(inj_comp, data_f, data_noinj, idx_f)
                    
                    # Here is room to check if calculated correctly
                    #!!!
                else:
                    print(f"Injection frequency can not be calculated or rounding error at {f} Hz!")  
                idx+=1
            print("Done.")
            
            print("Recalculating to impedance data and saving")
            # recalculate for impedance and other types
            Z = 1/Y       
            Z_TF = IM.TF(f_inj[1:],Z)
            
            if settings["Calculate NET"]:
                # initialize impedance
                Y = np.zeros((len(f_inj)-1),dtype="complex")
                
                print("Calculating NET immitance data for: "+ inj_comp)
                idx = 1
                for f in tqdm(f_inj[1:]):
                                    
                    # set up all filenames and indexes
                    rawfile_this = rawfile_prefix+f"{idx+1:02d}.psout"
                    if f in f_dft:
                        idx_f = f_dft.index(f)
                        # get DFT for alpha-beta components
                        data_f = get_dftdata_ab_1f(settings, rawfile_this, f, calc_NET=True)
                        
                        Y[idx-1] = calc_Yf_SISO(inj_comp, data_f, data_noinj_net, idx_f)
                        
                        # Here is room to check if calculated correctly
                        #!!!
                    else:
                        print(f"Injection frequency can not be calculated or rounding error at {f} Hz!")  
                    idx+=1
                print("Done.")
                
                print("Recalculating to impedance data and saving")
                # recalculate for impedance and other types
                Z = 1/Y       
                Znet_TF = IM.TF(f_inj[1:],Z)
            
        print('SISO impedance calculation is done. Saving results...')

        # save to IM CSV file
        IM_csvname = "IM_" + inj_comp +"_" + settings["simname"]+ "_" + sc_name + inj_comp + ".csv"
        IM_csv_file = (settings["Working folder"] + "\\IMTB_data\\" + 
                    settings["Timestamp"] + "_" + settings["simname"] + "\\" +
                    IM_csvname)
        if settings["Calculate NET"]:
            IM_str = ['f','Zdut','Znet']
            IM_list = [Z_TF.f,Z_TF.values,Znet_TF.values]
        else:
            IM_str = ['f','Zdut']
            IM_list = [Z_TF.f,Z_TF.values]
        

        IM_data = IM.IMTB_AC_immittace_toCSV(IM_csv_file, IM_str, IM_list)
        
        print('SISO impedance data has been saved in:')
        print(IM_csv_file)
        
        return IM_data
    

    # for MIMO impedance        
    elif settings["Immitance type"]=="MIMO":
        if settings["Injection components"] == 'posneg':
            injtype_all = ["pos", "neg"]
        
            f0 = settings["Fundamental freq"]
                
            data_noinj = {}
            f_dft = {}
            if settings["Calculate NET"]:
                data_noinj_net = {}
                f_dft_net = {}
          
            for inj_comp in injtype_all:
                rawfile_prefix = rawfolder+"\\"+settings["simname"]+"_"+sc_name+inj_comp+"_"
                
                # Get indexes for time series signals
                print("Finding time idexes for DFT")
                get_time_idxs(settings, rawfile_prefix+"01.psout")
                
                # get all data for no inj
                print("Getting DFT of no injection file")
                # data_noinj[inj_comp], f_dft[inj_comp] = get_dftdata(settings, rawfile_prefix+"01.psout")
                if NEW_MIMO_CALC_METHOD: # get DFT from abc 
                    data_noinj[inj_comp], f_dft[inj_comp] = get_dftdata(settings, rawfile_prefix+"01.psout")
                    if settings["Calculate NET"]:
                        data_noinj_net[inj_comp], f_dft_net[inj_comp] = get_dftdata(settings, rawfile_prefix+"01.psout",calc_NET=True)

                else: # get DFT from alpha beta
                    data_noinj[inj_comp], f_dft[inj_comp] = get_dftdata_ab(settings, rawfile_prefix+"01.psout")
                    if settings["Calculate NET"]:
                        data_noinj_net[inj_comp], f_dft_net[inj_comp] = get_dftdata_ab(settings, rawfile_prefix+"01.psout", calc_NET=True)
                        
                # Plotting no injection data 
                if settings["Plot harmonics"]:
                    data_harm, f_harm = get_dftdata(settings, rawfile_prefix+"01.psout")
                    plot_noinj(settings, f_harm, data_harm, sc_name)
                
            print("Calculating MIMO immitance data:")
            # Here stuff for the power flow check!
            #!!!            
            
            # calculate steady states from DUT
            if NEW_MIMO_CALC_METHOD:
                if f0 in f_dft["pos"]:
                    idx_f = f_dft["pos"].index(f0)
                    
                    V_f0 = _get_pos(data_noinj["pos"][3][idx_f],
                                       data_noinj["pos"][4][idx_f],
                                       data_noinj["pos"][5][idx_f],)
                    Vss = np.abs(V_f0)
                    phi_f0 = np.angle(V_f0)
                    
                    I_f0 = _get_pos(data_noinj["pos"][0][idx_f],
                                       data_noinj["pos"][1][idx_f],
                                       data_noinj["pos"][2][idx_f],)
                    
                    if settings["Calculate NET"]:
                        idx_f = f_dft["pos"].index(f0)
                        
                        V_f0_net = _get_pos(data_noinj_net["pos"][3][idx_f],
                                           data_noinj_net["pos"][4][idx_f],
                                           data_noinj_net["pos"][5][idx_f],)
                        Vss_net = np.abs(V_f0_net)
                        phi_f0_net = np.angle(V_f0_net)
                        
                        I_f0_net = _get_pos(data_noinj_net["pos"][0][idx_f],
                                           data_noinj_net["pos"][1][idx_f],
                                           data_noinj_net["pos"][2][idx_f],)
                    

            else:
                if f0 in f_dft["pos"]:
                    idx_f = f_dft["pos"].index(f0)
                    
                    V_f0 = _get_pos_fra_ab(data_noinj["pos"][2][idx_f],
                                        data_noinj["pos"][3][idx_f],)
                    Vss = np.abs(V_f0)
                    phi_f0 = np.angle(V_f0)
                    
                    I_f0 = _get_pos_fra_ab(data_noinj["pos"][0][idx_f],
                                        data_noinj["pos"][1][idx_f])
                    
                    if settings["Calculate NET"]:
                        idx_f = f_dft_net["pos"].index(f0)
                        
                        V_f0_net = _get_pos_fra_ab(data_noinj_net["pos"][2][idx_f],
                                            data_noinj_net["pos"][3][idx_f],)
                        Vss_net = np.abs(V_f0_net)
                        phi_f0_net = np.angle(V_f0_net)
                        
                        I_f0_net = _get_pos_fra_ab(data_noinj_net["pos"][0][idx_f],
                                            data_noinj_net["pos"][1][idx_f])
                    
            S_f0 = V_f0 * np.conjugate(I_f0)
            Pss = np.real(S_f0)
            Qss = np.imag(S_f0)
            if settings["Calculate NET"]:
                S_f0_net = V_f0_net * np.conjugate(I_f0_net)
                Pss_net = np.real(S_f0_net)
                Qss_net = np.imag(S_f0_net)
            
            
            # calculate MIMO models
            print("Responses at the following frequency points will be calculated.")
            print(f_inj[1:])
            
            Zab_dut = []
            Zdq_dut = []
            if settings["Calculate NET"]:
                Zab_net = []
                Zdq_net = []
            idx = 1
            fab_list = []
            fdq_list = []
            for f in tqdm(f_inj[1:]):  
                fab_list.append(f)
                fdq = f - f0
                fdq_list.append(fdq)
                if NEW_MIMO_CALC_METHOD:
                    # new impedance calculation method: abc -> DFT -> alpha-beta complex or dq
                    I_f1 = {}
                    V_f1 = {}
                    I_f2 = {}
                    V_f2 = {}
                    I1_fdq = {}
                    V1_fdq = {}
                    I2_fdq = {}
                    V2_fdq = {}
                
                    for inj_comp in injtype_all:
                        rawfile_prefix = rawfolder+"\\"+settings["simname"]+"_"+sc_name+inj_comp+"_"
                        # # set up all filenames and indexes
                        rawfile_this = rawfile_prefix+f"{idx+1:02d}.psout"
                        
                        
                        data_f1 = get_dftdata_1f(settings, rawfile_this, f) 
                        data_f2 = get_dftdata_1f(settings, rawfile_this, f-2*f0) 
                        
                        I_f1[inj_comp],V_f1[inj_comp],I_f2[inj_comp],V_f2[inj_comp] = calc_Yf_ab_complex(inj_comp, data_f1, data_f2, data_noinj[inj_comp], f_dft[inj_comp], f, f0)
                        I1_fdq[inj_comp],V1_fdq[inj_comp],I2_fdq[inj_comp],V2_fdq[inj_comp] = calc_Yf_dq_complex(inj_comp, data_f1, data_f2, data_noinj[inj_comp], f_dft[inj_comp], f, f0)

                    
                    Vmat = np.array([[V_f1["pos"],V_f1["neg"]],[V_f2["pos"],V_f2["neg"]]]) 
                    Imat = np.array([[I_f1["pos"],I_f1["neg"]],[I_f2["pos"],I_f2["neg"]]]) 
                    
                    Vmatdq = np.array([[V1_fdq["pos"],V1_fdq["neg"]],[V2_fdq["pos"],V2_fdq["neg"]]]) 
                    Imatdq = np.array([[I1_fdq["pos"],I1_fdq["neg"]],[I2_fdq["pos"],I2_fdq["neg"]]]) 
                    
                    Zab_dut.append(np.matmul(Vmat,lnlg.inv(Imat)))
                    Zdq_dut.append(np.matmul(Vmatdq,lnlg.inv(Imatdq)))
                    
                    if settings["Calculate NET"]:
                        # new impedance calculation method: abc -> DFT -> alpha-beta complex or dq
                        I_f1 = {}
                        V_f1 = {}
                        I_f2 = {}
                        V_f2 = {}
                        I1_fdq = {}
                        V1_fdq = {}
                        I2_fdq = {}
                        V2_fdq = {}
                    
                        for inj_comp in injtype_all:
                            rawfile_prefix = rawfolder+"\\"+settings["simname"]+"_"+sc_name+inj_comp+"_"
                            # # set up all filenames and indexes
                            rawfile_this = rawfile_prefix+f"{idx+1:02d}.psout"
                            
                            data_f1 = get_dftdata_1f(settings, rawfile_this, f, calc_NET=True) 
                            data_f2 = get_dftdata_1f(settings, rawfile_this, f-2*f0, calc_NET=True) 
                            
                            I_f1[inj_comp],V_f1[inj_comp],I_f2[inj_comp],V_f2[inj_comp] = calc_Yf_ab_complex(inj_comp, data_f1, data_f2, data_noinj[inj_comp], f_dft[inj_comp], f, f0)
                            I1_fdq[inj_comp],V1_fdq[inj_comp],I2_fdq[inj_comp],V2_fdq[inj_comp] = calc_Yf_dq_complex(inj_comp, data_f1, data_f2, data_noinj[inj_comp], f_dft[inj_comp], f, f0)

                        
                        Vmat = np.array([[V_f1["pos"],V_f1["neg"]],[V_f2["pos"],V_f2["neg"]]]) 
                        Imat = np.array([[I_f1["pos"],I_f1["neg"]],[I_f2["pos"],I_f2["neg"]]]) 
                        
                        Vmatdq = np.array([[V1_fdq["pos"],V1_fdq["neg"]],[V2_fdq["pos"],V2_fdq["neg"]]]) 
                        Imatdq = np.array([[I1_fdq["pos"],I1_fdq["neg"]],[I2_fdq["pos"],I2_fdq["neg"]]]) 
                        
                        Zab_net.append(np.matmul(Vmat,lnlg.inv(Imat)))
                        Zdq_net.append(np.matmul(Vmatdq,lnlg.inv(Imatdq)))
                else:
                    # old IMTB impedance calculation method: abc -> alpha-beta -> DFT -> alpha-beta complex or dq
                    
                    rawfile_prefix = rawfolder+"\\"+settings["simname"]+"_"+sc_name+"pos"+"_"
                    
                    # set up all filenames and indexes
                    rawfile_this = rawfile_prefix+f"{idx+1:02d}.psout"
                    data_pos = read_raw(settings, rawfile_this)
                    rawfile_prefix = rawfolder+"\\"+settings["simname"]+"_"+sc_name+"neg"+"_"
                    
                    # set up all filenames and indexes
                    rawfile_this = rawfile_prefix+f"{idx+1:02d}.psout"
                    data_neg = read_raw(settings, rawfile_this)
                                       
                    Zab = IMTB_AC_Zab_MIMO(data_pos, data_neg, f, phi_f0, settings)
                    Zab_dut.append(Zab)
                    Zdq = IMTB_AC_Zdq_MIMO(data_pos, data_neg, f, phi_f0, settings)
                    Zdq_dut.append(Zdq)
                    
                    if settings["Calculate NET"]:
                        Zab = IMTB_AC_Zab_MIMO(data_pos, data_neg, f, phi_f0_net, settings, calc_NET=True)
                        Zab_net.append(Zab)
                        Zdq = IMTB_AC_Zdq_MIMO(data_pos, data_neg, f, phi_f0_net, settings, calc_NET=True)
                        Zdq_net.append(Zdq)
                
                idx = idx + 1
            
        elif settings["Injection components"] == 'dq':
            injtype_all = ["d", "q"]
           
            f0 = settings["Fundamental freq"]
                
            data_noinj = {}
            f_dft = {}
            if settings["Calculate NET"]:
                data_noinj_net = {}
                f_dft_net = {}
                
          
            for inj_comp in injtype_all:
                rawfile_prefix = rawfolder+"\\"+settings["simname"]+"_"+sc_name+inj_comp+"_"
                
                # Get indexes for time series signals
                print("Finding time idexes for DFT")
                get_time_idxs(settings, rawfile_prefix+"01.psout")
                
                # get all data for no inj
                print("Getting DFT of no injection file")
                if NEW_MIMO_CALC_METHOD: # get DFT from abc 
                    data_noinj[inj_comp], f_dft[inj_comp] = get_dftdata(settings, rawfile_prefix+"01.psout")
                    if settings["Calculate NET"]:
                        data_noinj_net[inj_comp], f_dft_net[inj_comp] = get_dftdata(settings, rawfile_prefix+"01.psout",calc_NET=True)

                else: # get DFT from alpha beta
                    data_noinj[inj_comp], f_dft[inj_comp] = get_dftdata_ab(settings, rawfile_prefix+"01.psout")
                    if settings["Calculate NET"]:
                        data_noinj_net[inj_comp], f_dft_net[inj_comp] = get_dftdata_ab(settings, rawfile_prefix+"01.psout",calc_NET=True)

                # Plotting no injection data 
                if settings["Plot harmonics"]:
                    data_harm, f_harm = get_dftdata(settings, rawfile_prefix+"01.psout")
                    plot_noinj(settings, f_harm, data_harm, sc_name)
            # Here stuff for the power flow check!
            #!!!
           
            print("Calculating MIMO immitance data:")
            # calculate steady states
            if NEW_MIMO_CALC_METHOD:
                if f0 in f_dft["d"]:
                    idx_f = f_dft["d"].index(f0)
                    V_f0 = _get_pos(data_noinj["d"][3][idx_f],
                                        data_noinj["d"][4][idx_f],
                                        data_noinj["d"][5][idx_f],)
                    Vss = np.abs(V_f0)
                    phi_f0 = np.angle(V_f0)
                    
                    I_f0 = _get_pos(data_noinj["d"][0][idx_f],
                                        data_noinj["d"][1][idx_f],
                                        data_noinj["d"][2][idx_f],)
                    
                    if settings["Calculate NET"]:
                        idx_f = f_dft_net["d"].index(f0)
                        V_f0_net = _get_pos(data_noinj_net["d"][3][idx_f],
                                            data_noinj_net["d"][4][idx_f],
                                            data_noinj_net["d"][5][idx_f],)
                        Vss_net = np.abs(V_f0_net)
                        phi_f0_net = np.angle(V_f0_net)
                        
                        I_f0_net = _get_pos(data_noinj_net["d"][0][idx_f],
                                            data_noinj_net["d"][1][idx_f],
                                            data_noinj_net["d"][2][idx_f],)
                        
                    
            else:
                if f0 in f_dft["d"]:
                    idx_f = f_dft["d"].index(f0)
                    
                    V_f0 = _get_pos_fra_ab(data_noinj["d"][2][idx_f],
                                        data_noinj["d"][3][idx_f],)
                    Vss = np.abs(V_f0)
                    phi_f0 = np.angle(V_f0)
                    
                    I_f0 = _get_pos_fra_ab(data_noinj["d"][0][idx_f],
                                        data_noinj["d"][1][idx_f])
                    
                    if settings["Calculate NET"]:
                        idx_f = f_dft_net["d"].index(f0)
                        
                        V_f0_net = _get_pos_fra_ab(data_noinj_net["d"][2][idx_f],
                                            data_noinj_net["d"][3][idx_f],)
                        Vss_net = np.abs(V_f0_net)
                        phi_f0_net = np.angle(V_f0_net)
                        
                        I_f0_net = _get_pos_fra_ab(data_noinj_net["d"][0][idx_f],
                                            data_noinj_net["d"][1][idx_f])
                    
                    
            S_f0 = V_f0 * np.conjugate(I_f0)
            Pss = np.real(S_f0)
            Qss = np.imag(S_f0)
            if settings["Calculate NET"]:
                S_f0_net = V_f0_net * np.conjugate(I_f0_net)
                Pss_net = np.real(S_f0_net)
                Qss_net = np.imag(S_f0_net)
                
            # calculate MIMO models
            print("Responses at the following frequency points (dq frame) will be calculated.")
            print(f_inj[1:])
            
            Zab_dut = []
            Zdq_dut = []
            if settings["Calculate NET"]:
                Zab_net = []
                Zdq_net = []
            idx = 1
            fab_list = []
            fdq_list = []
            for fdq in tqdm(f_inj[1:]):    
                f = fdq + f0
                fab_list.append(f)
                fdq_list.append(fdq)
                if NEW_MIMO_CALC_METHOD:
                    # new impedance calculation method: abc -> DFT -> alpha-beta complex or dq
                    I_f1 = {}
                    V_f1 = {}
                    I_f2 = {}
                    V_f2 = {}
                    I1_fdq = {}
                    V1_fdq = {}
                    I2_fdq = {}
                    V2_fdq = {}
                
                    for inj_comp in injtype_all:
                        rawfile_prefix = rawfolder+"\\"+settings["simname"]+"_"+sc_name+inj_comp+"_"
                        # # set up all filenames and indexes
                        rawfile_this = rawfile_prefix+f"{idx+1:02d}.psout"
                        
                        
                        data_f1 = get_dftdata_1f(settings, rawfile_this, f) 
                        data_f2 = get_dftdata_1f(settings, rawfile_this, f-2*f0) 
                        
                        I_f1[inj_comp],V_f1[inj_comp],I_f2[inj_comp],V_f2[inj_comp] = calc_Yf_ab_complex(inj_comp, data_f1, data_f2, data_noinj[inj_comp], f_dft[inj_comp], f, f0)
                        I1_fdq[inj_comp],V1_fdq[inj_comp],I2_fdq[inj_comp],V2_fdq[inj_comp] = calc_Yf_dq_complex(inj_comp, data_f1, data_f2, data_noinj[inj_comp], f_dft[inj_comp], f, f0)

                    Vmat = np.array([[V_f1["d"],V_f1["q"]],[V_f2["d"],V_f2["q"]]]) 
                    Imat = np.array([[I_f1["d"],I_f1["q"]],[I_f2["d"],I_f2["q"]]]) 
                    
                    Vmatdq = np.array([[V1_fdq["d"],V1_fdq["q"]],[V2_fdq["d"],V2_fdq["q"]]]) 
                    Imatdq = np.array([[I1_fdq["d"],I1_fdq["q"]],[I2_fdq["d"],I2_fdq["q"]]]) 
                    
                    Zab_dut.append(np.matmul(Vmat,lnlg.inv(Imat)))
                    Zdq_dut.append(np.matmul(Vmatdq,lnlg.inv(Imatdq)))
                    
                    if settings["Calculate NET"]:
                        # new impedance calculation method: abc -> DFT -> alpha-beta complex or dq
                        I_f1 = {}
                        V_f1 = {}
                        I_f2 = {}
                        V_f2 = {}
                        I1_fdq = {}
                        V1_fdq = {}
                        I2_fdq = {}
                        V2_fdq = {}
                    
                        for inj_comp in injtype_all:
                            rawfile_prefix = rawfolder+"\\"+settings["simname"]+"_"+sc_name+inj_comp+"_"
                            # # set up all filenames and indexes
                            rawfile_this = rawfile_prefix+f"{idx+1:02d}.psout"
                            
                            
                            data_f1 = get_dftdata_1f(settings, rawfile_this, f, calc_NET=True) 
                            data_f2 = get_dftdata_1f(settings, rawfile_this, f-2*f0, calc_NET=True) 
                            
                            I_f1[inj_comp],V_f1[inj_comp],I_f2[inj_comp],V_f2[inj_comp] = calc_Yf_ab_complex(inj_comp, data_f1, data_f2, data_noinj[inj_comp], f_dft[inj_comp], f, f0)
                            I1_fdq[inj_comp],V1_fdq[inj_comp],I2_fdq[inj_comp],V2_fdq[inj_comp] = calc_Yf_dq_complex(inj_comp, data_f1, data_f2, data_noinj[inj_comp], f_dft[inj_comp], f, f0)

                        Vmat = np.array([[V_f1["d"],V_f1["q"]],[V_f2["d"],V_f2["q"]]]) 
                        Imat = np.array([[I_f1["d"],I_f1["q"]],[I_f2["d"],I_f2["q"]]]) 
                        
                        Vmatdq = np.array([[V1_fdq["d"],V1_fdq["q"]],[V2_fdq["d"],V2_fdq["q"]]]) 
                        Imatdq = np.array([[I1_fdq["d"],I1_fdq["q"]],[I2_fdq["d"],I2_fdq["q"]]]) 
                        
                        Zab_net.append(np.matmul(Vmat,lnlg.inv(Imat)))
                        Zdq_net.append(np.matmul(Vmatdq,lnlg.inv(Imatdq)))
                    
                else:
                    # old IMTB impedance calculation method: abc -> alpha-beta -> DFT -> alpha-beta complex or dq
                    
                    rawfile_prefix = rawfolder+"\\"+settings["simname"]+"_"+sc_name+"d"+"_"
                    # set up all filenames and indexes
                    rawfile_this = rawfile_prefix+f"{idx+1:02d}.psout"
                    data_pos = read_raw(settings, rawfile_this)
                    
                    rawfile_prefix = rawfolder+"\\"+settings["simname"]+"_"+sc_name+"q"+"_"
                    # set up all filenames and indexes
                    rawfile_this = rawfile_prefix+f"{idx+1:02d}.psout"
                    data_neg = read_raw(settings, rawfile_this)
                                       
                    Zab = IMTB_AC_Zab_MIMO(data_pos, data_neg, f, phi_f0, settings)
                    Zab_dut.append(Zab)
                    Zdq = IMTB_AC_Zdq_MIMO(data_pos, data_neg, f, phi_f0, settings)
                    Zdq_dut.append(Zdq)
                    
                    if settings["Calculate NET"]:
                        Zab = IMTB_AC_Zab_MIMO(data_pos, data_neg, f, phi_f0_net, settings, calc_NET=True)
                        Zab_net.append(Zab)
                        Zdq = IMTB_AC_Zdq_MIMO(data_pos, data_neg, f, phi_f0_net, settings, calc_NET=True)
                        Zdq_net.append(Zdq)
                        
                    
                idx = idx + 1
        
        elif settings["Injection components"] == 'ab-cb':
            # MIMO calculation for DC terminal type
            injtype_all = ["ab", "cb"]
           
            f0 = settings["Fundamental freq"]
                
            data_noinj = {}
            f_dft = {}
            if settings["Calculate NET"]:
                data_noinj_net = {}
                f_dft_net = {}
          
            for inj_comp in injtype_all:
                rawfile_prefix = rawfolder+"\\"+settings["simname"]+"_"+sc_name+inj_comp+"_"
                
                # Get indexes for time series signals
                print("Finding time idexes for DFT")
                get_time_idxs(settings, rawfile_prefix+"01.psout")
                
                # get all data for no inj
                print("Getting DFT of no injection file")
                data_noinj[inj_comp], f_dft[inj_comp] = get_dftdata(settings, rawfile_prefix+"01.psout")
                if settings["Calculate NET"]:
                    data_noinj_net[inj_comp], f_dft_net[inj_comp] = get_dftdata(settings, rawfile_prefix+"01.psout",calc_NET=True)

                        
                # Plotting no injection data 
                if settings["Plot harmonics"]:
                    data_harm, f_harm = get_dftdata(settings, rawfile_prefix+"01.psout")
                    plot_noinj(settings, f_harm, data_harm, sc_name)
          
                       
            print("Calculating MIMO immitance data:")
            
            
            # calculate MIMO models
            print("Responses at the following frequency points will be calculated.")
            print(f_inj[1:])
            
            Zdc_dut = []
            if settings["Calculate NET"]:
                Zdc_net = []
            idx = 1
            fdc_list = []
            for f in tqdm(f_inj[1:]):    
                fdc_list.append(f)
                # new impedance calculation method: abc -> DFT -> DC terminals
                Ia = {}
                Ic = {}
                Vab = {}
                Vcb = {}
            
                for inj_comp in injtype_all:
                    rawfile_prefix = rawfolder+"\\"+settings["simname"]+"_"+sc_name+inj_comp+"_"
                    # # set up all filenames and indexes
                    rawfile_this = rawfile_prefix+f"{idx+1:02d}.psout"
                    
                    data_f = get_dftdata_1f(settings, rawfile_this, f) 
                    
                    Ia[inj_comp], Ic[inj_comp], Vab[inj_comp], Vcb[inj_comp] = calc_Yf_DC_MIMO(inj_comp, data_f, data_noinj[inj_comp], idx)
                
                Vmat = np.array([[Vab["ab"],Vab["cb"]],[Vcb["ab"],Vcb["cb"]]]) 
                Imat = np.array([[Ia["ab"],Ia["cb"]],[Ic["ab"],Ic["cb"]]]) 
                
                Zdc_dut.append(np.matmul(Vmat,lnlg.inv(Imat)))
                
                if settings["Calculate NET"]:
                    # new impedance calculation method: abc -> DFT -> DC terminals
                    Ia = {}
                    Ic = {}
                    Vab = {}
                    Vcb = {}
                
                    for inj_comp in injtype_all:
                        rawfile_prefix = rawfolder+"\\"+settings["simname"]+"_"+sc_name+inj_comp+"_"
                        # # set up all filenames and indexes
                        rawfile_this = rawfile_prefix+f"{idx+1:02d}.psout"
                        
                        data_f = get_dftdata_1f(settings, rawfile_this, f, calc_NET=True) 
                        
                        Ia[inj_comp], Ic[inj_comp], Vab[inj_comp], Vcb[inj_comp] = calc_Yf_DC_MIMO(inj_comp, data_f, data_noinj[inj_comp], idx)
                    
                    Vmat = np.array([[Vab["ab"],Vab["cb"]],[Vcb["ab"],Vcb["cb"]]]) 
                    Imat = np.array([[Ia["ab"],Ia["cb"]],[Ic["ab"],Ic["cb"]]]) 
                    
                    Zdc_net.append(np.matmul(Vmat,lnlg.inv(Imat)))
                
                
                idx = idx + 1
                
        # convert calculated impedance to TF object
        if settings["Terminal type"]   == "AC":
            # get Zab, Zdq and Kn in TF forms
            Zab_dut_TF = IM.TF(fab_list,Zab_dut)
            Zdq_dut_TF = IM.TF(fdq_list,Zdq_dut)
            Kn_dut_TF = IM.Zdq2P(Zdq_dut_TF, Pss, Qss, Vss)
            if settings["Calculate NET"]:
                Zab_net_TF = IM.TF(fab_list,Zab_net)
                Zdq_net_TF = IM.TF(fdq_list,Zdq_net)
                Kn_net_TF = IM.Zdq2P(Zdq_net_TF, Pss_net, Qss_net, Vss_net)

            print('MIMO impedances and tranfer matrix calculation is done. Saving results...')
            
            # data preparation for saving Zab
            Z11 = []
            Z12 = []
            Z21 = []
            Z22 = []
            for Zk in Zab_dut_TF.values:
                Z11.append(Zk[0][0])
                Z12.append(Zk[0][1])
                Z21.append(Zk[1][0])
                Z22.append(Zk[1][1])
            
            if settings["Calculate NET"]: 
                # data preparation for saving Zab
                Z11_net = []
                Z12_net = []
                Z21_net = []
                Z22_net = []
                for Zk in Zab_net_TF.values:
                    Z11_net.append(Zk[0][0])
                    Z12_net.append(Zk[0][1])
                    Z21_net.append(Zk[1][0])
                    Z22_net.append(Zk[1][1])
      
            
            # save to IM CSV file
            IM_csvname = "IM_Zab_MIMO_" + settings["simname"]+ "_" + sc_name + settings["Injection components"]  + ".csv"
            IM_csv_file = (settings["Working folder"] + "\\IMTB_data\\" + 
                        settings["Timestamp"] + "_" + settings["simname"] + "\\" +
                        IM_csvname)
            
            if settings["Calculate NET"]: 
                IM_str = ['f','Zdut_ab_11','Zdut_ab_12','Zdut_ab_21','Zdut_ab_22','Znet_ab_11','Znet_ab_12','Znet_ab_21','Znet_ab_22']
                IM_list = [Zab_dut_TF.f,Z11,Z12,Z21,Z22,Z11_net,Z12_net,Z21_net,Z22_net]
            else:
                IM_str = ['f','Zdut_ab_11','Zdut_ab_12','Zdut_ab_21','Zdut_ab_22']
                IM_list = [Zab_dut_TF.f,Z11,Z12,Z21,Z22]
            
            IM_data_ab = IM.IMTB_AC_immittace_toCSV(IM_csv_file, IM_str, IM_list)
            print('Alpha-beta impedance data has been saved in:')
            print(IM_csv_file)
            # print(IM_csv_file.replace(".csv", ".xlsx"))
            
            # data preparation for saving Zdq
            Z11 = []
            Z12 = []
            Z21 = []
            Z22 = []
            for Zk in Zdq_dut_TF.values:
                Z11.append(Zk[0][0])
                Z12.append(Zk[0][1])
                Z21.append(Zk[1][0])
                Z22.append(Zk[1][1])
            
            if settings["Calculate NET"]: 
                # data preparation for saving Zdq
                Z11_net = []
                Z12_net = []
                Z21_net = []
                Z22_net = []
                for Zk in Zdq_net_TF.values:
                    # print(Zk)
                    Z11_net.append(Zk[0][0])
                    Z12_net.append(Zk[0][1])
                    Z21_net.append(Zk[1][0])
                    Z22_net.append(Zk[1][1])
      
           
            
            # save to IM CSV file
            IM_csvname = "IM_Zdq_MIMO_" + settings["simname"]+ "_" + sc_name + settings["Injection components"]  + ".csv"
            IM_csv_file = (settings["Working folder"] + "\\IMTB_data\\" + 
                        settings["Timestamp"] + "_" + settings["simname"] + "\\" +
                        IM_csvname)
            
            if settings["Calculate NET"]: 
                IM_str = ['f','Zdut_dq_11','Zdut_dq_12','Zdut_dq_21','Zdut_dq_22','Znet_dq_11','Znet_dq_12','Znet_dq_21','Znet_dq_22']
                IM_list = [Zdq_dut_TF.f,Z11,Z12,Z21,Z22,Z11_net,Z12_net,Z21_net,Z22_net]
            else:
                IM_str = ['f','Zdut_dq_11','Zdut_dq_12','Zdut_dq_21','Zdut_dq_22']
                IM_list = [Zdq_dut_TF.f,Z11,Z12,Z21,Z22]
            
            
    
            IM_data_dq = IM.IMTB_AC_immittace_toCSV(IM_csv_file, IM_str, IM_list)
            print('DQ impedance data has been saved in:')
            print(IM_csv_file)
            # print(IM_csv_file.replace(".csv", ".xlsx"))    
            
            # # data preparation for saving Kn
            Z11 = []
            Z12 = []
            Z21 = []
            Z22 = []
            for Zk in Kn_dut_TF.values:
                Z11.append(Zk[0][0])
                Z12.append(Zk[0][1])
                Z21.append(Zk[1][0])
                Z22.append(Zk[1][1])
            
            if settings["Calculate NET"]: 
                # data preparation for saving Kn
                Z11_net = []
                Z12_net = []
                Z21_net = []
                Z22_net = []
                for Zk in Kn_net_TF.values:
                    Z11_net.append(Zk[0][0])
                    Z12_net.append(Zk[0][1])
                    Z21_net.append(Zk[1][0])
                    Z22_net.append(Zk[1][1])
                 
            
            # save to IM CSV file
            IM_csvname = "IM_Kn_MIMO_" + settings["simname"]+ "_" + sc_name + settings["Injection components"]  + ".csv"
            IM_csv_file = (settings["Working folder"] + "\\IMTB_data\\" + 
                        settings["Timestamp"] + "_" + settings["simname"] + "\\" +
                        IM_csvname)
            
            if settings["Calculate NET"]: 
                IM_str = ['f','Kdut_11','Kdut_12','Kdut_21','Kdut_22','Knet_11','Knet_12','Knet_21','Knet_22']
                IM_list = [Kn_dut_TF.f,Z11,Z12,Z21,Z22,Z11_net,Z12_net,Z21_net,Z22_net]
            else:
                IM_str = ['f','Kdut_11','Kdut_12','Kdut_21','Kdut_22']
                IM_list = [Kn_dut_TF.f,Z11,Z12,Z21,Z22]
                
            
    
            IM_data_Kn = IM.IMTB_AC_immittace_toCSV(IM_csv_file, IM_str, IM_list)
            print('Jacobian matrix data has been saved in:')
            print(IM_csv_file)
            # print(IM_csv_file.replace(".csv", ".xlsx")) 
       
            return IM_data_ab, IM_data_dq, IM_data_Kn
        else:
            # FOR DC MIMO CALCULATION
            # get Zdc in TF forms
            Zdc_dut_TF = IM.TF(fdc_list,Zdc_dut)
            if settings["Calculate NET"]:
                Zdc_net_TF = IM.TF(fdc_list,Zdc_net)
            
            
            print('MIMO impedances and tranfer matrix calculation is done. Saving results...')
    
            
            # data preparation for saving Zab
            Z11 = []
            Z12 = []
            Z21 = []
            Z22 = []
            for Zk in Zdc_dut_TF.values:
                Z11.append(Zk[0][0])
                Z12.append(Zk[0][1])
                Z21.append(Zk[1][0])
                Z22.append(Zk[1][1])
            
            if settings["Calculate NET"]:
                Z11_net = []
                Z12_net = []
                Z21_net = []
                Z22_net = []
                for Zk in Zdc_net_TF.values:
                    Z11_net.append(Zk[0][0])
                    Z12_net.append(Zk[0][1])
                    Z21_net.append(Zk[1][0])
                    Z22_net.append(Zk[1][1])
      
            
            # save to IM CSV file
            IM_csvname = "IM_Zdc_MIMO_" + settings["simname"]+ "_" + sc_name + settings["Injection components"]  + ".csv"
            IM_csv_file = (settings["Working folder"] + "\\IMTB_data\\" + 
                        settings["Timestamp"] + "_" + settings["simname"] + "\\" +
                        IM_csvname)
            if settings["Calculate NET"]:
                IM_str = ['f','Zdut_dc_11','Zdut_dc_12','Zdut_dc_21','Zdut_dc_22','Znet_dc_11','Znet_dc_12','Znet_dc_21','Znet_dc_22']
                IM_list = [Zdc_dut_TF.f,Z11,Z12,Z21,Z22,Z11_net,Z12_net,Z21_net,Z22_net]
            else:
                IM_str = ['f','Zdut_dc_11','Zdut_dc_12','Zdut_dc_21','Zdut_dc_22']
                IM_list = [Zdc_dut_TF.f,Z11,Z12,Z21,Z22]
            
    
            IM_data_dc = IM.IMTB_AC_immittace_toCSV(IM_csv_file, IM_str, IM_list)
            print('DC MIMO impedance data has been saved in:')
            print(IM_csv_file)
            # print(IM_csv_file.replace(".csv", ".xlsx"))
            
            
            return IM_data_dc
        
         

# =============================================================================
# Run function definition
# =============================================================================

def run(settings):
    """ Calculate and save the immitances from PSCAD simulation """  

    # Getting all injection frequencies from pscad simulation script
    f_inj = pscad_sim.setupfreqs(settings)
    settings["fs"] = round(1/settings["Plot timestep"]*1e6,ROUNDING_TO_EXP_FREQ)
    
    # folder definitions
    rawfolder = (settings["Working folder"] + "\\IMTB_data\\" + 
                settings["Timestamp"] + "_" + settings["simname"] + "\\raw")
    
    # looping over number of scenarios
    for idx_sc in range(settings["Nr scenarios"]):
        if settings["Multiple scenarios"]:
            sc_name = f"sc{idx_sc+1:02d}_"
            print("Calculating for scenario nr "+str(idx_sc+1))
        else:
            sc_name = ""
            
        Single_IM_calc(rawfolder, sc_name, f_inj, settings, DQ_calc=0)
      
    return settings
    

# =============================================================================
# Main function for testing only
# =============================================================================

if __name__ == "__main__":
    # path for the simulation folder
    simPath = r'INSERT PATH HERE'
    settings = get_settings(simPath)
    
    run(settings)
    