# -*- coding: utf-8 -*-
"""
Subscript of IMTB to run the simulations

@author: AUF, YLI @ Energinet

Update history:
    21 Mar 2025 - v1.0 - first public version
    
"""
# =============================================================================
# Import modules
# =============================================================================

import sys, os
import shutil
# import math
import time
import csv
import pandas as pd
import numpy as np
import mhi
import mhi.pscad.utilities.file 


# =============================================================================
# Global settings of this sub-script
# =============================================================================

# Rounding negative scientific exponent. Helping with numeric problems in the script
ROUNDING_TO_EXP_FREQ = 4 # For frequency rounding
ROUNDING_TO_EXP_TIME = 6 # For time steps rounding

SNAPSHOT_PLOTSTEP = 1000 # Plot step for snapshot creation (us)

TRUELOG_NEG_FREQ_START = 1 # If truelog is choosen and startfreq is negative this will be the start freq for positive side (float)

# Frequency sweep settings
POINTS_BETWEEN_LOG = 10 # Number of injection points to be simulated between log steps (int)
# Sub-synchronous preset values (lin)
SUBSYNCH_FREQSTART = 0.5 # Start frequency (Hz)
SUBSYNCH_FREQSTOP = 5 # Stop frequency (Hz)
SUBSYNCH_FREQSTEP = 0.5 # Frequency step (Hz)
SUBSYNCH_PLOTSTEP = 1000.0 # Plotstep for the frequency range (us)
# Near-synchronous preset values (lin)
NEARSYNCH_FREQDELTA_LOW = 20 # Start frequency as in f0-X (Hz)
NEARSYNCH_FREQDELTA_HIGH = 20 # Stop frequency as in f0+X (Hz)
NEARSYNCH_FREQSTEP = 1 # Frequency step (Hz)
NEARSYNCH_PLOTSTEP = 1000.0 # Plotstep for the frequency range (us)
# Super-synchronous preset values (log)
SUPERSYNCH_FREQSTART_MULT = 2 # Start frequency as multiple of f0 (-)
SUPERSYNCH_FREQSTOP_MULT = 50 # Stop frequency as multiple of f0 (-)
SUPERSYNCH_FREQSTEP = 0 # Frequency step (Hz) (not used in log)
SUPERSYNCH_PLOTSTEP = 160.0 # Plotstep for the frequency range (us)
# Combined preset values (lin+lin or lin+log or lin+truelog)
COMBINED_FREQSTART = 10
COMBINED_FREQSTEP = 10
COMBINED_FREQMID = 100
COMBINED_FREQSTEP2 = 20 # for linear
COMBINED_POINTS_BETWEEN_LOG = 10 # for log
COMBINED_NUM_POINTS = 50 # for truelog
COMBINED_FREQSTOP = 2500
# Complete preset values (log or truelog)
COMPLETE_FREQBAND1 = 10
COMPLETE_FREQSTEP1 = 1
COMPLETE_FREQBAND2 = 100
COMPLETE_FREQSTEP2 = 2
COMPLETE_POINTS_BETWEEN_LOG = 10 # for log
COMPLETE_NUM_POINTS = 50 # for truelog
# COMPLETE_FREQSTOP = 2500

# Removes scanning at harmonics, due to errors based on harmonic injection
REMOVE_HARMONIC_INJ = False

# =============================================================================
# Internal functions
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

def get_single_type_finj(settings, freqstart,freqstop,freqstep,f0,resolution_type='linear'):
    if resolution_type == 'linear':
        f_inj1 = np.arange(freqstart,freqstop+freqstep,freqstep)
        
        f_inj1 = list(np.round(f_inj1,ROUNDING_TO_EXP_FREQ))
        
        # Remove fundamental or harmonics if included
        try:
            if REMOVE_HARMONIC_INJ:
                for fk in f_inj1:
                    if fk % f0 == 0:
                        f_inj1.remove(fk)
            else:
                if settings["Injection components"] == "dq":
                    f_inj1.remove(0)
                elif settings["Injection components"] == "d":
                    f_inj1.remove(0)
                elif settings["Injection components"] == "q":
                    f_inj1.remove(0)
                else:
                    for fk in f_inj1:
                        if fk % f0 == 0 and fk < 200:
                            f_inj1.remove(fk)
        except:
            pass
        
    elif resolution_type == 'log':
        f_inj1 = []
        
        # Getting exponent of starting frequency
        f_exp = np.floor(np.log10(float(freqstart))).astype(int)
        
        # Adjusting the first frequency to nearest sci int
        f = np.round(float(freqstart),f_exp)
        
        N_between_log = int(freqstep)
        if N_between_log == 0:
            N_between_log = POINTS_BETWEEN_LOG
            
        if f % f0 == 0 and REMOVE_HARMONIC_INJ:
            f += 1/N_between_log*10**f_exp/2
            
        # Looping through the sweep series
        while f<=float(freqstop):
            # appending to overall list if not f0
            if REMOVE_HARMONIC_INJ:
                if f % f0 != 0:
                    f_inj1.append(f)
            else:
                if f != f0 or f != 2*f0 or f != 3*f0:
                    f_inj1.append(f)
                            
            # adjusting the frequency for next step
            # adding one sci int
            f += 1/N_between_log*10**f_exp                
            
            # checking if f_exp need to be adjusted
            if np.floor(np.log10(f))>f_exp:
                f_exp += 1
                
            # rounding f to get round values and get last frequency
            f = np.round(f,ROUNDING_TO_EXP_FREQ)
            
    elif resolution_type == 'truelog':
        start_neg = freqstart<0
        if start_neg:
            f_start_exp = np.log10(TRUELOG_NEG_FREQ_START)
            f_start_abs = abs(freqstart)
        else:
            f_start_exp = np.log10(freqstart)
            
        f_end_exp = np.log10(freqstop)
        N_freqs = freqstep
        
        
        f_inj1 = np.logspace(f_start_exp,f_end_exp,num=int(N_freqs))
        
        f_inj1 = list(np.round(f_inj1,0))
        
        # Remove fundamental or harmonics if included
        try:
            if REMOVE_HARMONIC_INJ:
                for fk in f_inj1:
                    if fk % f0 == 0:
                        f_inj1.remove(fk)
            else:
                if settings["Injection components"] == "dq":
                    f_inj1.remove(0)
                elif settings["Injection components"] == "d":
                    f_inj1.remove(0)
                elif settings["Injection components"] == "q":
                    f_inj1.remove(0)
                else:
                    for fk in f_inj1:
                        if fk % f0 == 0 and fk < 200:
                            f_inj1.remove(fk)
        except:
            pass
        
        # Adding negative side if start freq negative
        if start_neg:
            idx = 0
            while f_inj1[idx]<f_start_abs:
                f_inj1.append(-1*f_inj1[idx])
                idx+=1
            
        f_inj = list(set(f_inj1))
        f_inj.sort()
        
        f_inj1 = f_inj
    else:
        print('Resolution type is not defined.')
    
    return f_inj1
        
def get_plot_timestep(freqmax):
    # Look up table for plotstep choices
    if freqmax < 500:
        plot_Tstep = 1000.0
    elif freqmax < 625:
        plot_Tstep = 800.0
    elif freqmax < 1000:
        plot_Tstep = 500.0
    elif freqmax < 1250:
        plot_Tstep = 400.0
    elif freqmax < 2000:
        plot_Tstep = 250.0
    elif freqmax < 2500:
        plot_Tstep = 200.0
    elif freqmax < 3125:
        plot_Tstep = 160.0
    elif freqmax < 4000:
        plot_Tstep = 125.0
    elif freqmax < 5000:
        plot_Tstep = 100.0
    elif freqmax < 6250:
        plot_Tstep = 80.0
    elif freqmax < 10000:
        plot_Tstep = 50.0
    else:
        plot_Tstep = 50.0
        print("Frequency stop too high. Please change to values lower than 10kHz")
    return plot_Tstep

def setupfreqs(settings):
    """ Function to create frequency series """
    # Helping functions
    f0 = float(settings["Fundamental freq"])

    # switch between different freq range settings
    if settings["Freq range"]=="sub-synchronous":
        settings_subsyn = settings
        settings_subsyn["Freq range"]="custom" # overriding the user settings
        settings_subsyn["Freq type"]="linear"
        settings_subsyn["Freq start"]=SUBSYNCH_FREQSTART
        settings_subsyn["Freq stop"]=SUBSYNCH_FREQSTOP
        settings_subsyn["Freq step"]=SUBSYNCH_FREQSTEP
        
        # Changing settings plot timestep and returning lists
        settings["Plot timestep"] = SUBSYNCH_PLOTSTEP
        return setupfreqs(settings_subsyn)
    elif settings["Freq range"]=="near-synchronous":
        settings_nearsyn = settings
        settings_nearsyn["Freq range"]="custom" # overriding the user settings
        settings_nearsyn["Freq type"]="linear"
        settings_nearsyn["Freq start"]=f0-NEARSYNCH_FREQDELTA_LOW
        settings_nearsyn["Freq stop"]=f0+NEARSYNCH_FREQDELTA_HIGH
        settings_nearsyn["Freq step"]=NEARSYNCH_FREQSTEP
        
        # Changing settings plot timestep and returning lists
        settings["Plot timestep"] = NEARSYNCH_PLOTSTEP
        return setupfreqs(settings_nearsyn)
    elif settings["Freq range"]=="super-synchronous":
        settings_supersyn = settings
        settings_supersyn["Freq range"]="custom" # overriding the user settings
        settings_supersyn["Freq type"]="log"
        settings_supersyn["Freq start"]=SUPERSYNCH_FREQSTART_MULT*f0
        settings_supersyn["Freq stop"]=SUPERSYNCH_FREQSTOP_MULT*f0
        settings_supersyn["Freq step"]=SUPERSYNCH_FREQSTEP # is not used in log mode
        
        # Changing settings plot timestep and returning lists
        settings["Plot timestep"] = SUPERSYNCH_PLOTSTEP
        return setupfreqs(settings_supersyn)
    elif settings["Freq range"] == "complete(log)":
        if settings["Terminal type"]=="DC" or settings["Injection components"] == "dq" or settings["Injection components"] == "d" or settings["Injection components"] == "q":
            # for freq range defined in DC/dq domain
            f_inj1 = get_single_type_finj(settings, COMPLETE_FREQSTEP1, 
                                          COMPLETE_FREQBAND1, 
                                          COMPLETE_FREQSTEP1, 
                                          f0, resolution_type='linear')
            f_inj2 = get_single_type_finj(settings, COMPLETE_FREQBAND1, 
                                          COMPLETE_FREQBAND2, 
                                          COMPLETE_FREQSTEP2, 
                                          f0, resolution_type='linear')
            f_inj3 = get_single_type_finj(settings, COMPLETE_FREQBAND2, 
                                          settings["Freq stop"], 
                                          COMPLETE_POINTS_BETWEEN_LOG, 
                                          f0, resolution_type='log')
            
            if f_inj2[0] == f_inj1[-1]:
                f_inj2 = f_inj2[1:]
            
            if f_inj3[0] == f_inj2[-1]:
                f_inj3 = f_inj3[1:]
            
            f_inj2 = np.append(f_inj2,f_inj3)
            
            f_inj = np.append(np.array([0,]),np.append(f_inj1,f_inj2))
            
        else: 
            # for freq range defined in phase domain
            if f0 - COMPLETE_FREQBAND2 < 0:
                freq_start = COMPLETE_FREQSTEP2
            else:
                freq_start = f0 - COMPLETE_FREQBAND2
            f_inj1 = get_single_type_finj(settings, freq_start, 
                                          f0 - COMPLETE_FREQBAND1, 
                                          COMPLETE_FREQSTEP2, 
                                          f0, resolution_type='linear')
            f_inj2 = get_single_type_finj(settings, f0 - COMPLETE_FREQBAND1, 
                                          f0 + COMPLETE_FREQBAND1, 
                                          COMPLETE_FREQSTEP1, 
                                          f0, resolution_type='linear')
            f_inj3 = get_single_type_finj(settings, f0 + COMPLETE_FREQBAND1, 
                                          f0 + COMPLETE_FREQBAND2, 
                                          COMPLETE_FREQSTEP2, 
                                          f0, resolution_type='linear')
            f_inj4 = get_single_type_finj(settings, f0 + COMPLETE_FREQBAND2, 
                                          settings["Freq stop"], 
                                          COMPLETE_POINTS_BETWEEN_LOG, 
                                          f0, resolution_type='log')
            
            if f_inj2[0] == f_inj1[-1]:
                f_inj2 = f_inj2[1:]
            
            if f_inj3[0] == f_inj2[-1]:
                f_inj3 = f_inj3[1:]
                
            if f_inj4[0] == f_inj3[-1]:
                f_inj4 = f_inj4[1:]
                
            f_inj3 = np.append(f_inj3,f_inj4)
            f_inj2 = np.append(f_inj2,f_inj3)
            
            f_inj = np.append(np.array([0,]),np.append(f_inj1,f_inj2))
            
        # Changing settings plot timestep and returning lists
        settings["Plot timestep"] = get_plot_timestep(settings["Freq stop"])
        
        return f_inj
    
    elif settings["Freq range"]=="complete(truelog)":
        if settings["Terminal type"]=="DC" or settings["Injection components"] == "dq" or settings["Injection components"] == "d" or settings["Injection components"] == "q":
            # for freq range defined in DC/dq domain
            f_inj1 = get_single_type_finj(settings, COMPLETE_FREQSTEP1, 
                                          COMPLETE_FREQBAND1, 
                                          COMPLETE_FREQSTEP1, 
                                          f0, resolution_type='linear')
            f_inj2 = get_single_type_finj(settings, COMPLETE_FREQBAND1, 
                                          COMPLETE_FREQBAND2, 
                                          COMPLETE_FREQSTEP2, 
                                          f0, resolution_type='linear')
            f_inj3 = get_single_type_finj(settings, COMPLETE_FREQBAND2, 
                                          settings["Freq stop"],
                                          COMPLETE_NUM_POINTS, 
                                          f0, resolution_type='truelog')
            
            if f_inj2[0] == f_inj1[-1]:
                f_inj2 = f_inj2[1:]
            
            if f_inj3[0] == f_inj2[-1]:
                f_inj3 = f_inj3[1:]
            
            f_inj2 = np.append(f_inj2,f_inj3)
            
            f_inj = np.append(np.array([0,]),np.append(f_inj1,f_inj2))
            
        else: 
            # for freq range defined in phase domain
            if f0 - COMPLETE_FREQBAND2 < 0:
                freq_start = COMPLETE_FREQSTEP2
            else:
                freq_start = f0 - COMPLETE_FREQBAND2
            f_inj1 = get_single_type_finj(settings, freq_start, 
                                          f0 - COMPLETE_FREQBAND1, 
                                          COMPLETE_FREQSTEP2, 
                                          f0, resolution_type='linear')
            f_inj2 = get_single_type_finj(settings, f0 - COMPLETE_FREQBAND1, 
                                          f0 + COMPLETE_FREQBAND1, 
                                          COMPLETE_FREQSTEP1, 
                                          f0, resolution_type='linear')
            f_inj3 = get_single_type_finj(settings, f0 + COMPLETE_FREQBAND1, 
                                          f0 + COMPLETE_FREQBAND2, 
                                          COMPLETE_FREQSTEP2, 
                                          f0, resolution_type='linear')
            f_inj4 = get_single_type_finj(settings, f0 + COMPLETE_FREQBAND2, 
                                          settings["Freq stop"],
                                          COMPLETE_NUM_POINTS, 
                                          f0, resolution_type='truelog')
            
            if f_inj2[0] == f_inj1[-1]:
                f_inj2 = f_inj2[1:]
            
            if f_inj3[0] == f_inj2[-1]:
                f_inj3 = f_inj3[1:]
                
            if f_inj4[0] == f_inj3[-1]:
                f_inj4 = f_inj4[1:]
                
            f_inj3 = np.append(f_inj3,f_inj4)
            f_inj2 = np.append(f_inj2,f_inj3)
            
            f_inj = np.append(np.array([0,]),np.append(f_inj1,f_inj2))
            
        # Changing settings plot timestep and returning lists
        settings["Plot timestep"] = get_plot_timestep(settings["Freq stop"])
        
        return f_inj
    elif settings["Freq range"]=="linear+linear":
        
        f_inj1 = get_single_type_finj(settings, COMBINED_FREQSTART, 
                                      COMBINED_FREQMID, 
                                      COMBINED_FREQSTEP, 
                                      f0, resolution_type='linear')
        
        f_inj2 = get_single_type_finj(settings, COMBINED_FREQMID, 
                                      COMBINED_FREQSTOP, 
                                      COMBINED_FREQSTEP2, 
                                      f0, resolution_type='linear')
        if f_inj2[0] == f_inj1[-1]:
            f_inj2 = f_inj2[1:]
            
        f_inj = np.append(np.array([0,]),np.append(f_inj1,f_inj2))
        
        # Changing settings plot timestep and returning lists
        settings["Plot timestep"] = get_plot_timestep(COMBINED_FREQSTOP)
        
        return f_inj
    elif settings["Freq range"]=="linear+log":
        
        f_inj1 = get_single_type_finj(settings, COMBINED_FREQSTART, 
                                      COMBINED_FREQMID, 
                                      COMBINED_FREQSTEP, 
                                      f0, resolution_type='linear')
        
        f_inj2 = get_single_type_finj(settings, COMBINED_FREQMID, 
                                      COMBINED_FREQSTOP, 
                                      COMBINED_POINTS_BETWEEN_LOG, 
                                      f0, resolution_type='log')
        if f_inj2[0] == f_inj1[-1]:
            f_inj2 = f_inj2[1:]
            
        f_inj = np.append(np.array([0,]),np.append(f_inj1,f_inj2))
        
        # Changing settings plot timestep and returning lists
        settings["Plot timestep"] = get_plot_timestep(COMBINED_FREQSTOP)
        
        return f_inj
    
    elif settings["Freq range"]=="linear+truelog":
        
        f_inj1 = get_single_type_finj(settings, COMBINED_FREQSTART, 
                                      COMBINED_FREQMID, 
                                      COMBINED_FREQSTEP, 
                                      f0, resolution_type='linear')
        
        f_inj2 = get_single_type_finj(settings, COMBINED_FREQMID, 
                                      COMBINED_FREQSTOP, 
                                      COMBINED_NUM_POINTS, 
                                      f0, resolution_type='truelog')
        if f_inj2[0] == f_inj1[-1]:
            f_inj2 = f_inj2[1:]
            
        f_inj = np.append(np.array([0,]),np.append(f_inj1,f_inj2))
        
        settings["Plot timestep"] = get_plot_timestep(COMBINED_FREQSTOP)
        
        return f_inj
        
    # Otherwise the custom time setup is appolied
    elif settings["Freq range"]=="custom":
        # Setting up the first frequency    
        # Linear by using the frequencies input in settings
        if settings["Freq type"]=="linear":
            
            f_inj1 = get_single_type_finj(settings, settings["Freq start"], 
                                          settings["Freq stop"], 
                                          settings["Freq step"], 
                                          f0, resolution_type='linear')
            
            # add 0 for no injection at the beginning 
            f_inj = np.append(np.array([0,]),f_inj1)
            
        # Logarithmic from start and stop. Between only "scientific" integer steps
        elif settings["Freq type"]=="log":
            f_inj1 = get_single_type_finj(settings, settings["Freq start"], 
                                          settings["Freq stop"], 
                                          settings["Freq step"], 
                                          f0, resolution_type='log')
            # add 0 for no injection at the beginning 
            f_inj = np.append(np.array([0,]),f_inj1)
        
        # Logarithmic scale using logspace but rounding to nearest INT for easier scan
        elif settings["Freq type"]=="truelog":
            f_inj1 = get_single_type_finj(settings, settings["Freq start"], 
                                          settings["Freq stop"], 
                                          settings["Freq step"], 
                                          f0, resolution_type='truelog')
            
            # add 0 for no injection at the beginning 
            f_inj = np.append(np.array([0,]),f_inj1)
            
        # Logging error in case of wrong input
        else:
            print("Error in frequency settings, while setting up times")
        
    
        return f_inj

def createtxt(settings,f_inj):
    """ Creates injection .txt file for IMTB frequency scan """
    # List for number of ranks
    ranks = list(range(1,len(f_inj)+1))
    # Save the files with times
    out = np.array([ranks,f_inj,])
    out = np.transpose(out)
    np.savetxt("IMTB_f.txt", out, delimiter=",",
               newline='\n', comments="! ", fmt="%.4f")
    shutil.move("IMTB_f.txt", os.path.join(settings["Working folder"], "IMTB_f.txt"))
    print("Injection frequencies .txt-file created")
    return None



def get_comp_canvaspath(component,project):
    # allocating output variable
    canvas_name = component.parent.name
    
    canvas_path = canvas_name
    
    while canvas_name!= "Main":
        def_name = project.name+":"+canvas_name
        comp = project.find_all(def_name)[0]
        canvas_name = comp.parent.name
        
        canvas_path = canvas_name + "/" + canvas_path
    
    return canvas_path


def start_pscad(settings):
    """ Starts PSCAD in right version and right way for further steps """
    # Start right PSCAD version
    print("Starting PSCAD v" + settings["PSCAD version"])
    print("Fortran compiler: " + settings["Fortran version"])
    # 
    if settings["License type"]=="certificate":
        load_up = True
    else:
        load_up = True
    pscad = mhi.pscad.launch(version=settings["PSCAD version"],load_user_profile=load_up, silence=True, splash=False)
    
    return pscad

def pscad_licensing(settings, pscad):
    """ dealing with pscad licensing settings """
    
    if settings["License type"] == "certificate":
    
        # Set PSCAD to certificate license type
        # pscad.settings(cl_use_advanced="true")
        
        # setting PSCAD to give up the license after usage for householding
        try:
            pscad.settings({"cl_exit_behaviour":0})
        except:
            print("Remeber to close license after use")
            
        # Release certificate if already exists
        pscad.release_certificate()
        
        # help function to get volley nr of license certificate
        def get_cert_volley(cert):
            return cert.feature("EMTDC Instances").value()
        
        # Grab the license with the most appropriate number of parallel processes found and use the certificate to license PSCAD
        if(pscad.logged_in() == True):
            certs = pscad.get_available_certificates()
            if len(certs) > 0:
                best_cert = None
                # finding a license with open instances
                for cert in list(certs.values()):
                    if cert.available() > 0:
                        # choose certificate with most parallel licenses
                        if best_cert:
                            # exchange best cert if better fit to max volley settings
                            if get_cert_volley(cert) < get_cert_volley(best_cert):
                                if get_cert_volley(cert) >= settings["Max volley"]:
                                    # replace with current certificate to not use higher capability licenses
                                    best_cert = cert
                            else:
                                if get_cert_volley(best_cert) < settings["Max volley"]:
                                    # replace because higher number of parallel licenses and user need not met before
                                    best_cert = cert
                        else:
                            best_cert = cert
                if best_cert:
                    print("Acquiring Certificate Now! : %s", str(best_cert))
                    pscad.get_certificate(best_cert)
                    # replace max volley setting if certificate cannot comply with it
                    if settings["Max volley"] > get_cert_volley(best_cert):
                        settings["Max volley"] = get_cert_volley(best_cert)
                    print("PSCAD should have a license now")
                    
                if pscad.licensed() == False:
                    print("All PSCAD Licenses are in use right now!")
            else:
                print("No certificate licenses available on server")
                print("Starting PSCAD in unlicensed mode")
        else:
            print("You must log in (top right on PSCAD) and then restart script")
    
    else:
        
        # Set PSCAD to lock-based license type
        # pscad.settings(cl_use_advanced="false")
        
        # try to activate professional license
        try:
            pscad.activate_pro_license()
        except:
            # try to activate educational license
            try:
                pscad.activate_edu_license()
            except:
                pass
        
        if pscad.licensed() == False:
            print("There is a problem to activate lock-based license. Check PSCAD license settings.")
        
    return None

# =============================================================================
# Run function (Main)
# =============================================================================

def run(settings):
    
    print("Simulation name: " + settings["simname"])
    
    # =============================================================================
    # Start PSCAD
    # =============================================================================

    pscad = start_pscad(settings)    
    
    # =============================================================================
    # PSCAD Licence management
    # =============================================================================
    
    if settings["License type"]=="certificate":
        pscad_licensing(settings, pscad)
    
    # =============================================================================
    # PSCAD compiler settings
    # =============================================================================
    
    # Setting some PSCAD settings
    pscad_options = {"fortran_version":settings["Fortran version"], "start_page_startup":False, "cl_use_advanced":True}
    pscad.settings(pscad_options)
    
    if pscad:
        # Open PSCAD workspace
        print("Opening Workspace: " + settings["Workspace name"] + " in " + settings["Working folder"])
        pscad.load(settings["Working folder"] + "\\" + settings["Workspace name"])
        time.sleep(5) # Wait for 5 sec in case PSCAD needs more time to load files
        
        # =============================================================================
        # Get IMTB instance
        # =============================================================================
        
        # Set simulation set
        simset = pscad.simulation_set(settings["SimSet name"])
                
        # Get all projects in active simulation set
        projectname_all = simset.list_tasks()
        
        # Raise error if no Project is assigned
        if len(projectname_all) == 0:
            print("ERROR! No PSCAD project is assigned to simulation set: " + settings["SimSet name"])
        
        # adapt max volley to number of tasks in the simulation set
        settings["Max volley"] = int(settings["Max volley"]/len(projectname_all))
        # check if enough licenses
        if settings["Max volley"]==0:
            print("Assigned PSCAD license does not allow for "+ str(len(projectname_all))+" parallel simulations!")
            
        print("Maximal volley set to: " + str(settings["Max volley"]))
        
        print("Searching for instance of IMTB in Simulation Set: " + settings["SimSet name"])
        
        IMTB_all = []
        
        # Iterate through the projects and search for IMTB and get parameters in a list of dict
        for projectname_this in projectname_all:
            # Set current project
            project = pscad.project(projectname_this)
            # Find all IMTBs in this project
            IMTB_all = IMTB_all + (project.find_all("IMTB:IMTB"))
            
            # check if it is not disabled and no name (works from PSCAD v502 only)
            try:
                idx = 0
                for IMTB_this in IMTB_all:
                    
                    
                    if IMTB_this.enabled == False:
                        IMTB_all.pop(idx)
                    else:
                        # check if there is no name -> give representative naming
                        IMTB_para_this = IMTB_this.parameters()
                        if IMTB_para_this['Name'] == "":
                            IMTB_this.parameters(Name=projectname_this+"_"+str(idx))
                        
                        # expand the idx
                        idx += 1
            except:
                pass         
        
                
        if len(IMTB_all) == 0:
            print("ERROR! No IMTB toolboxes found!")
        elif len(IMTB_all) > 1:
            print("ERROR! Number of active IMTB instances in the simulation set higher that 1!")
        else:
            # Find all important PSCAD components
            IMTB = IMTB_all[0]
            settings["IMTB name"] = IMTB.parameters()["Name"]
            settings["IMTB canvas"] = get_comp_canvaspath(IMTB,project)
            # Injection signal creator
            INJcreator = IMTB.canvas().find("IMTB:create_inj")
            # Setting series or shunt injection into PSCAD hidden parameter
            if settings["Injection type"]=="shunt":
                IMTB.parameters(shunt_inj=1)
            else:
                IMTB.parameters(shunt_inj=0)
            # AC or DC terminal setting in PSCAD IMTB component
            if settings["Terminal type"]=="AC":
                IMTB.parameters(ac_src=1)
            else:
                IMTB.parameters(ac_src=0)
        
        # =============================================================================
        # Setup different scenarios simulations
        # =============================================================================
        
        if settings["Multiple scenarios"]:
            print("Setting up multiple scenarios")
            # read Scenarios excel file, get values and transpose
            df = pd.read_excel(settings["Scenarios filepath"])
            constnames = list(df.columns)[1:]
            scenarios_paras = df.values.transpose()
            settings["Nr scenarios"] = len(scenarios_paras[0])
            
            # find constants in PSCAD models
            idx_const = 0
            scenarios_consts = []
            for constname in constnames:
                # find all possible constants components in all projects
                scenarios_consts.append([])
                for projectname_this in projectname_all:
                    # Set current project
                    project = pscad.project(projectname_this)
                    #find all and add to list
                    scenarios_consts[idx_const] += project.find_all("master:const", Name=constname)
                    
                # check if unique and exists, otherwise overwrite with component
                if len(scenarios_consts[idx_const]) == 0:
                    print("ERROR! Multiple scenario constant \'"+ constname+"\' can not be found!")
                elif len(scenarios_consts[idx_const]) > 1:
                    print("ERROR! Multiple scenario constant \'"+ constname+"\' exists in multiple places/projects!")
                else:
                    scenarios_consts[idx_const] = scenarios_consts[idx_const][0]
                idx_const +=1
                
        # No multiple scenarios choosen
        else:
            print("No multiple scenarios choosen.")
            # create a single row scenarios matrix
            sc_name = ""
            settings["Nr scenarios"] = 1
        
        # =============================================================================
        # Start single simulation set of the toolbox
        # =============================================================================
        
        # Create injection frequency sequence and save as TXT
        f_inj = setupfreqs(settings)
        createtxt(settings, f_inj)
        
        # Disabling all plots ("pgb"s) in PSCAD projects (optimizing simulation time and storage space)
        project_all = pscad.projects()
        for project_this in project_all:
            # Finding all "pgb"s in this project
            project = pscad.project(project_this["name"])
            pgb_all = project.find_all("master:pgb")
            
            if project_this["type"] == "Library":
                if project_this["name"] != "master":
                    if project_this["name"] != "IMTB":
                        # Disable all in other libraries
                        for pgb_this in pgb_all:
                            pgb_this.disable()
                    else:
                        # What to do in IMTB library
                        for pgb_this in pgb_all:
                            pgb_para = pgb_this.parameters()
                            pgbname = pgb_para["Name"]
                            if any([pgbname == "v_dut", pgbname == "i_dut"]):
                                pgb_this.enable()
                            else:
                                pgb_this.disable()
                            # enable the plotting of NET side waveforms
                            if settings["Calculate NET"]:
                                if any([pgbname == "v_net", pgbname == "i_net"]):
                                    pgb_this.enable()
            else:
                # Disable all in all of the cases
                for pgb_this in pgb_all:
                    pgb_this.disable()
        
        
        for idx_sc in range(settings["Nr scenarios"]):
            # Handling scenarios values
            if settings["Multiple scenarios"]:
                print(f"Setting up scenario nr {idx_sc+1:02d}")
                info_string = ""
                for idx in range(len(constnames)):
                    info_string = (info_string +
                                      constnames[idx]+": "+str(scenarios_paras[idx+1][idx_sc])+"; ")
                print(info_string)
                sc_name = f"sc{idx_sc+1:02d}_"
                
                # Set constants for thism ultiple scenario
                idx = 0
                for const in scenarios_consts:
                    const.parameters(Value=scenarios_paras[idx+1][idx_sc])
                    idx += 1                
            
            
            if settings["Snapshot function"]:
                if settings["Start from snapshot"]==True and int(settings["Nr scenarios"])==1: # only for single scenario case
                    print("Will start from snapshot: "+ settings["Snapshot name"])
                    
                else:                     
                    print("Setting up the snapshot")
                    
                    # Assign name for later reference
                    settings["Snapshot name"] = "snp_"+settings["Timestamp"]+".snp"
                    # Setting up the parameters of all projects according to settings for snapshot
                    project_para = {
                                    "description":settings["simname"] + ":snapshot",
                                    "time_duration":settings["Snapshot time"],
                                    "time_step":settings["Solution timestep"],
                                    "sample_step":SNAPSHOT_PLOTSTEP,#settings["Plot timestep"], # override because not saved data
                                    "StartType":"0",
                                    "PlotType":"0",
                                    "SnapType":"1",
                                    "snapshot_filename":settings["Snapshot name"],
                                    "SnapTime":settings["Snapshot time"]
                                    }
                    
                    # Setting the simulation parameters of all projects to same
                    for projectname_this in projectname_all:
                        # Set current project
                        project = pscad.project(projectname_this)
                        # Set all parameters to current project
                        project.parameters(**project_para)
                                            
                    # setting no injection to IMTB toolbox for snapshot
                    IMTB.parameters(f_r = settings["Fundamental freq"])
                    INJcreator.parameters(inj_type=0, amplitude =0, f0 = settings["Fundamental freq"])
                    
                    # Setting the volley settings of all projects in simulation set to 1 and no tracing
                    for task_this in simset.tasks():
                        task_this.parameters(affinity_type = "0",
                                             ammunition = "1",
                                             volley = "1")
                    
                    
                    # Simulating snapshot
                    print("Simulating the snapshot for {} seconds".format(settings["Snapshot time"]))
                    pscad.run_simulation_sets(settings["SimSet name"])
                        
                    print("Snapshot ready")
            
            # setting up simulation parameters for injection
            print("Setting up for frequency scan")
            # Setting up the parameters of all projects according to settings for snapshot
            settings["Output filename"] = "psout_"+sc_name+settings["Timestamp"]+".psout"
            
            if settings["Snapshot function"]:
                project_para = {
                                "description":settings["simname"],
                                "time_duration":settings["Settling time"]+settings["Injection time"],
                                "time_step":settings["Solution timestep"],
                                "sample_step":settings["Plot timestep"],
                                "StartType":"1",
                                "startup_filename":settings["Snapshot name"],
                                "PlotType":"0",
                                "output_filename":settings["Output filename"],
                                "SnapType":"0",
                                "remove_time_offset":False
                                }
            else:
                project_para = {
                                "description":settings["simname"],
                                "time_duration":settings["Settling time"]+settings["Injection time"],
                                "time_step":settings["Solution timestep"],
                                "sample_step":settings["Plot timestep"],
                                "StartType":"0",
                                "PlotType":"0",
                                "output_filename":settings["Output filename"],
                                "SnapType":"0",
                                }

            
            
            # Setting the simulation parameters of all projects to same
            for projectname_this in projectname_all:
                # Set current project
                project = pscad.project(projectname_this)
                # Set all parameters to current project
                project.parameters(**project_para)
            
            # activating output file only on IMTB project
            project = pscad.project(IMTB.project_name)
            project.parameters(PlotType = "2")
            
            
            # Setting up a list of injection component types
            if settings["Injection components"] == "pos":
                injtype_all = ["pos"]
            elif settings["Injection components"] == "posneg0":
                injtype_all = ["pos", "neg", "zero"]
            elif settings["Injection components"] == "abc":
                injtype_all = ["a", "b", "c"]
            elif settings["Injection components"] == "alphabeta0":
                injtype_all = ["alpha", "beta", "zero"]
            elif settings["Injection components"] == "dq0":
                injtype_all = ["d", "q", "zero"]
            elif settings["Injection components"] == "posneg":
                injtype_all = ["pos", "neg"]
            elif settings["Injection components"] == "pos0":
                injtype_all = ["pos", "zero"]
            elif settings["Injection components"] == "dq":
                injtype_all = ["d", "q"]
            elif settings["Injection components"] == "neg":
                injtype_all = ["neg"]
            elif settings["Injection components"] == "zero":
                injtype_all = ["zero"]
            elif settings["Injection components"] == "d":
                injtype_all = ["d"]
            elif settings["Injection components"] == "q":
                injtype_all = ["q"]
            elif settings["Injection components"] == "alpha":
                injtype_all = ["alpha"]
            elif settings["Injection components"] == "beta":
                injtype_all = ["beta"]
            elif settings["Injection components"] == "a":
                injtype_all = ["a"]
            elif settings["Injection components"] == "b":
                injtype_all = ["b"]
            elif settings["Injection components"] == "c":
                injtype_all = ["c"]
            elif settings["Injection components"] == "ab":
                injtype_all = ["ab"]
            elif settings["Injection components"] == "cb":
                injtype_all = ["cb"]
            elif settings["Injection components"] == "ab-cb":
                injtype_all = ["ab","cb"]
            else:
                print("Injection component type not known!")
                
            # loop for different injection components
            for injtype_this in injtype_all:
                # Set injection type for IMTB
                print("Setting injection for: " + injtype_this)
                
                # getting the index of injection type for PSCAD injection creator
                if injtype_this == "a":
                    injidx = 1
                elif injtype_this == "b":
                    injidx = 2
                elif injtype_this == "c":
                    injidx = 3
                elif injtype_this == "pos":
                    injidx = 4
                elif injtype_this == "neg":
                    if settings["Injection components"] == "posneg" and settings["Immitance type"] == "MIMO":
                        injidx = 13
                    else:
                        injidx = 5
                elif injtype_this == "zero":
                    injidx = 6
                elif injtype_this == "alpha":
                    injidx = 7
                elif injtype_this == "beta":
                    injidx = 8
                elif injtype_this == "d":
                    injidx = 9
                elif injtype_this == "q":
                    injidx = 10
                elif injtype_this == "ab":
                    injidx = 11
                elif injtype_this == "cb":
                    injidx = 12
                else:
                    injidx = 0
                    
                INJcreator.parameters(inj_type=injidx,
                                      amplitude = settings["Injection amplitude"],)
                
                # Setting the project descriptions for logging
                for projectname_this in projectname_all:
                    # Set current project
                    project = pscad.project(projectname_this)
                    # Set all parameters to current project
                    project.parameters(description=settings["simname"] + ":" + injtype_this)
                
                # Setting up a parallel simulation
                for task_this in simset.tasks():
                    task_this.parameters(affinity_type = "0",
                                         ammunition = len(f_inj),
                                         volley = settings["Max volley"])
                
                # Simulating for this injection
                print("Start simulation")
                pscad.run_simulation_sets(settings["SimSet name"])
                time.sleep(10) # Waiting to let PSCAD finish creating .psout files just in case
                print("Simulation done")
                
                print("Saving results")
                
                destination_folder = (settings["Working folder"] + "\\IMTB_data\\" + 
                            settings["Timestamp"] + "_" + settings["simname"] + "\\raw")
                os.makedirs(destination_folder, exist_ok=True)
                
                # looping through each sim and shifting file while renaming a bit
                for idx in range(1,len(f_inj)+1):
                    # setting index string according to how PSCAD saves files
                    idxstr = f"{idx:02d}"
                    
                    # moving .psout file for post processing folder
                    source_path = (settings["Working folder"] + "\\" + 
                                IMTB.project_name + settings["Folder extension"] + "\\" +
                                settings["Output filename"][0:-6] + "_" + 
                                idxstr + ".psout")
                    
                    destination_path = (destination_folder + 
                                        "\\" + settings["simname"] + "_" + 
                                        sc_name +
                                        injtype_this + 
                                        "_" + idxstr +".psout")
                    
                    if os.path.isfile(source_path):
                        shutil.move(source_path,destination_path)
                    else:
                        source_path = (settings["Working folder"] + "\\" + 
                                    IMTB.project_name + settings["Folder extension"] + "\\" +
                                    settings["Output filename"][0:-6] + ".psout")
                        if os.path.isfile(source_path) and len(f_inj) % settings["Max volley"] == 1 and idx == len(f_inj):
                            shutil.move(source_path,destination_path)
                        else:
                            print("Result .psout file not found!")
                            
                        
        
            # Scenario done
        
        # Saving settings for later use
        destination = settings["Working folder"] + "\\IMTB_data\\" + settings["Timestamp"] +"_"+ settings["simname"] + "\\"
        with open(destination + "settings.csv", "w", newline="") as fp:
            # creating header object
            writer = csv.DictWriter(fp, fieldnames=settings.keys())
            
            # writing the data from settings
            writer.writeheader()
            writer.writerow(settings)
       
        # everything is simulated, close PSCAD
        print("Simulation done, closing PSCAD")
        if settings["License type"]=="certificate":
            pscad.release_certificate()
        pscad.quit()
    else:
        print("Error with PSCAD")

# =============================================================================
# Main function just for testing !!
# =============================================================================

if __name__ == "__main__":
        
    simPath = r'INSERT PATH HERE'
    settings = get_settings(simPath)
    
    run(settings)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    