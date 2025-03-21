# -*- coding: utf-8 -*-
"""
Subscript of IMTB for impedance plotting

@author: YLI, AUF @ Energinet

Update history:
    21 Mar 2025 - v1.0 - first public version
    
"""
# =============================================================================
# Global settings
# =============================================================================

# Multi-scenario plot setting
Single_plot = True # True means plotting all scenario data only in one plot
Multi_plots = True # True means plotting individual scenatio data in individual plots

# True means to add non-passive region on IM plots
Neg_Z_region = True

# Plot functions
log_axis = False 
dB_unit = True
unwrap_on = False

# =============================================================================
# Modules Imports
# =============================================================================

import numpy as np
import os.path
import csv

import IM_analysis_lib_v1 as IM


from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px


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

def create_colormap(nfig=2):
    n_colors = nfig
    # color_idx = np.linspace(0, 1, nfig)
    colors = px.colors.sample_colorscale(
        "turbo", [n/(n_colors - 1) for n in range(n_colors)])
    # colormap = plt.cm.gist_ncar
    return colors


def get_IM_data(simPath, sc_name, settings):
    all_items = os.listdir(simPath)
    Z = [] 
    IM_file_name = []
    
    if settings["Immitance type"]=="SISO":
        
        inj_comp = settings["Injection components"]
        IM_csvnames = [item for item in all_items if "IM_" + inj_comp in item and sc_name in item and '.csv' in item]
        IM_csvname = IM_csvnames[0]

        IM_csv_file = simPath + "\\" + IM_csvname
        Z.append(IM.IM_read_CSV(IM_csv_file))
        print("Get SISO impedance data from:")
        print(IM_csv_file)
        IM_file_name.append(IM_csvname.rsplit('.')[0])
        
        IM_folder = simPath

    elif settings["Immitance type"]=="MIMO":
        if settings["Injection components"] == 'posneg':
            # Zab
            IM_csvnames = [
                item for item in all_items if 'IM_Zab_MIMO_' in item and sc_name in item and '.csv' in item]
            IM_csvname = IM_csvnames[0]
            IM_csv_file = simPath + "\\" + IM_csvname
            Z.append(IM.IM_read_CSV(IM_csv_file))
            print("Get alpha-beta impedance data from:")
            print(IM_csv_file)
            IM_file_name.append(IM_csvname.rsplit('.')[0])
            # Zdq
            IM_csvnames = [
                item for item in all_items if 'IM_Zdq_MIMO_' in item and sc_name in item and '.csv' in item]
            IM_csvname = IM_csvnames[0]
            IM_csv_file = simPath + "\\" + IM_csvname
            Z.append(IM.IM_read_CSV(IM_csv_file))
            print("Get DQ impedance data from:")
            print(IM_csv_file)
            IM_file_name.append(IM_csvname.rsplit('.')[0])
            
            # Kn
            IM_csvnames = [
                item for item in all_items if 'IM_Kn_MIMO_' in item and sc_name in item and '.csv' in item]
            IM_csvname = IM_csvnames[0]
            IM_csv_file = simPath + "\\" + IM_csvname
            Z.append(IM.IM_read_CSV(IM_csv_file))
            print("Get Jacobian matrix data from:")
            print(IM_csv_file)
            IM_file_name.append(IM_csvname.rsplit('.')[0])
            
            
            IM_folder = simPath
        elif settings["Injection components"] == 'dq':
            # Zab
            IM_csvnames = [
                item for item in all_items if 'IM_Zab_MIMO_' in item and sc_name in item and '.csv' in item]
            IM_csvname = IM_csvnames[0]
            IM_csv_file = simPath + "\\" + IM_csvname
            Z.append(IM.IM_read_CSV(IM_csv_file))
            print("Get alpha-beta impedance data from:")
            print(IM_csv_file)
            IM_file_name.append(IM_csvname.rsplit('.')[0])
            # Zdq
            IM_csvnames = [
                item for item in all_items if 'IM_Zdq_MIMO_' in item and sc_name in item and '.csv' in item]
            IM_csvname = IM_csvnames[0]
            IM_csv_file = simPath + "\\" + IM_csvname
            Z.append(IM.IM_read_CSV(IM_csv_file))
            print("Get DQ impedance data from:")
            print(IM_csv_file)
            IM_file_name.append(IM_csvname.rsplit('.')[0])
            
            # Kn
            IM_csvnames = [
                item for item in all_items if 'IM_Kn_MIMO_' in item and sc_name in item and '.csv' in item]
            IM_csvname = IM_csvnames[0]
            IM_csv_file = simPath + "\\" + IM_csvname
            Z.append(IM.IM_read_CSV(IM_csv_file))
            print("Get Jacobian matrix data from:")
            print(IM_csv_file)
            IM_file_name.append(IM_csvname.rsplit('.')[0])
            
            
            IM_folder = simPath
            
        elif settings["Injection components"] == 'ab-cb':
            # Zdc
            IM_csvnames = [
                item for item in all_items if 'IM_Zdc_MIMO_' in item and sc_name in item and '.csv' in item]
            IM_csvname = IM_csvnames[0]
            IM_csv_file = simPath + "\\" + IM_csvname
            Z.append(IM.IM_read_CSV(IM_csv_file))
            print("Get DC MIMO impedance data from:")
            print(IM_csv_file)
            IM_file_name.append(IM_csvname.rsplit('.')[0])
            
            IM_folder = simPath
        

    return Z, IM_folder, IM_file_name

def get_ZMIMO_element_list(x,y,Z):
    nf = len(Z.f)
    Zelement = []
    for idx in range(nf):
        Zelement.append(Z.values[idx][x,y])

    return Zelement

def get_Bode_plot_html(Z, figurePath, pName, DUT_plot=True, NET_plot=True, log_axis=True, dB_unit=True,unwrap_on=True):

    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02)

    colors = create_colormap(nfig=2)
    
    if unwrap_on:
        f_low = Z['Zdut'].f[0]
        f_high = Z['Zdut'].f[len(Z['Zdut'].f)-1]
        add_filled_area_on_phase_plot(fig,2,1,f_low,f_high,phase_band_no=2)
    else:
        f_low = Z['Zdut'].f[0]
        f_high = Z['Zdut'].f[len(Z['Zdut'].f)-1]
        add_filled_area_on_phase_plot(fig,2,1,f_low,f_high,phase_band_no=1)
    
    if DUT_plot:
        if dB_unit:
            fig.append_trace(go.Scatter(x=Z['Zdut'].f, y=IM.Mat_mag((Z['Zdut'].values)), name='Zdut_mag', line_color=colors[0], mode='lines'), row=1, col=1, ) 
            
        else:               
            fig.append_trace(go.Scatter(x=Z['Zdut'].f, y=np.abs(Z['Zdut'].values), 
                                        name='Zdut_mag', line_color=colors[0], mode='lines'), row=1, col=1, )
        
        if unwrap_on:
            pha = np.unwrap(IM.Mat_pha(Z['Zdut'].values),period=360)
        else:
            pha = IM.Mat_pha(Z['Zdut'].values)
        
        fig.append_trace(go.Scatter(x=Z['Zdut'].f, y=pha, name='Zdut_pha', line_color=colors[0], mode='lines'), row=2, col=1, )

    if NET_plot and "Znet" in Z.keys():
        if dB_unit:
            fig.append_trace(go.Scatter(x=Z['Znet'].f, y=IM.Mat_mag((Z['Znet'].values)), name='Znet_mag', line_color=colors[1], mode='lines'), row=1, col=1, )
        else:
            fig.append_trace(go.Scatter(x=Z['Znet'].f, y=np.abs(Z['Znet'].values), 
                                        name='Znet_mag', line_color=colors[1], mode='lines'), row=1, col=1, )
        
        if unwrap_on:
            pha = np.unwrap(IM.Mat_pha(Z['Znet'].values),period=360)
        else:
            pha = IM.Mat_pha(Z['Znet'].values)
        fig.append_trace(go.Scatter(x=Z['Znet'].f, y=pha, name='Znet_pha', line_color=colors[1], mode='lines'), row=2, col=1, )

    if log_axis:
        fig.update_xaxes(type="log", row=1, col=1)
        fig.update_xaxes(type="log", row=2, col=1)

    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
    
    if dB_unit:
        fig.update_yaxes(title_text="Amplitude (dB)", row=1, col=1) 
        
    else:
        fig.update_yaxes(title_text="Amplitude (ohm)", row=1, col=1)
        fig.update_yaxes(type="log", row=1, col=1)
        
    fig.update_yaxes(title_text="Phase (deg)", row=2, col=1)
    fig.update_layout(title='Impedance Bode plot')
    
    fig.write_html(figurePath + "\\" + pName + ".html")
    print('Impedance figure saved as:')
    print(figurePath + "\\" + pName + ".html")
    

def add_Bode_plot_html(fig, color, legend_name, Z, log_axis=True, dB_unit=True, unwrap_on=True, NET_plot=False):
    if dB_unit:
        fig.append_trace(go.Scatter(x=Z['Zdut'].f, y=IM.Mat_mag(
            (Z['Zdut'].values)), name=legend_name+'Zdut_mag', line_color=color, mode='lines'), row=1, col=1, ) 
        
    else:               
        fig.append_trace(go.Scatter(x=Z['Zdut'].f, y=np.abs(Z['Zdut'].values), 
                                    name=legend_name+'Zdut_mag', line_color=color, mode='lines'), row=1, col=1, )
    if unwrap_on:
        pha = np.unwrap(IM.Mat_pha(Z['Zdut'].values),period=360)
    else:
        pha = IM.Mat_pha(Z['Zdut'].values)
    fig.append_trace(go.Scatter(x=Z['Zdut'].f, y=pha, name=legend_name+'Zdut_pha', line_color=color, mode='lines'), row=2, col=1, )

        
    if NET_plot and "Znet" in Z.keys():
        if dB_unit:
            fig.append_trace(go.Scatter(x=Z['Znet'].f, y=IM.Mat_mag(
                (Z['Znet'].values)), name=legend_name+'Znet_mag', line_color=color, mode='lines'), row=1, col=1, ) 
            
        else:               
            fig.append_trace(go.Scatter(x=Z['Znet'].f, y=np.abs(Z['Znet'].values), 
                                        name=legend_name+'Znet_mag', line_color=color, mode='lines'), row=1, col=1, )
        if unwrap_on:
            pha = np.unwrap(IM.Mat_pha(Z['Znet'].values),period=360)
        else:
            pha = IM.Mat_pha(Z['Znet'].values)
        fig.append_trace(go.Scatter(x=Z['Znet'].f, y=pha, name=legend_name+'Znet_pha', line_color=color, mode='lines'), row=2, col=1, )
        


    if log_axis:
        fig.update_xaxes(type="log", row=1, col=1)
        fig.update_xaxes(type="log", row=2, col=1)

    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
    
    if dB_unit:
        fig.update_yaxes(title_text="Amplitude (dB)", row=1, col=1) 
        
    else:
        fig.update_yaxes(title_text="Amplitude (ohm)", row=1, col=1)
        fig.update_yaxes(type="log", row=1, col=1)
        
    fig.update_yaxes(title_text="Phase (deg)", row=2, col=1)
    fig.update_layout(title='Impedance Bode plot')
    return fig
    

def get_Z_MIMO_Bode_plot_html(Z, figurePath, pName, DUT_plot=True, NET_plot=True, log_axis=True, dB_unit=True, unwrap_on=True):

    fig = make_subplots(rows=4, cols=2,
                        shared_xaxes=True,
                        vertical_spacing=0.02)

    colors = create_colormap(nfig=2)

    if DUT_plot:
        # mag plots
        if dB_unit:
            fig.append_trace(go.Scatter(x=Z['Zdut'].f, y=IM.Mat_mag(
                (get_ZMIMO_element_list(0, 0, Z['Zdut']))), name='Zdut_mag', line_color=colors[0], mode='lines'), row=1, col=1, )
            fig.append_trace(go.Scatter(x=Z['Zdut'].f, y=IM.Mat_mag(
                (get_ZMIMO_element_list(0, 1, Z['Zdut']))), name='Zdut_mag', line_color=colors[0], mode='lines'), row=1, col=2, )
            fig.append_trace(go.Scatter(x=Z['Zdut'].f, y=IM.Mat_mag(
                (get_ZMIMO_element_list(1, 0, Z['Zdut']))), name='Zdut_mag', line_color=colors[0], mode='lines'), row=3, col=1, )
            fig.append_trace(go.Scatter(x=Z['Zdut'].f, y=IM.Mat_mag(
                (get_ZMIMO_element_list(1, 1, Z['Zdut']))), name='Zdut_mag', line_color=colors[0], mode='lines'), row=3, col=2, )
            
        else:
            
            fig.append_trace(go.Scatter(x=Z['Zdut'].f, y=np.abs(get_ZMIMO_element_list(0, 0, Z['Zdut'])), 
                                        name='Zdut_mag', line_color=colors[0], mode='lines'), row=1, col=1, )
            fig.append_trace(go.Scatter(x=Z['Zdut'].f, y=np.abs(get_ZMIMO_element_list(0, 1, Z['Zdut'])), 
                                        name='Zdut_mag', line_color=colors[0], mode='lines'), row=1, col=2, )
            fig.append_trace(go.Scatter(x=Z['Zdut'].f, y=np.abs(get_ZMIMO_element_list(1, 0, Z['Zdut'])), 
                                        name='Zdut_mag', line_color=colors[0], mode='lines'), row=3, col=1, )
            fig.append_trace(go.Scatter(x=Z['Zdut'].f, y=np.abs(get_ZMIMO_element_list(1, 1, Z['Zdut'])), 
                                        name='Zdut_mag', line_color=colors[0], mode='lines'), row=3, col=2, )
        
        # phase plots
        if unwrap_on:
            pha11 = np.unwrap(IM.Mat_pha(get_ZMIMO_element_list(0, 0, Z['Zdut'])),period=360)
            pha12 = np.unwrap(IM.Mat_pha(get_ZMIMO_element_list(0, 1, Z['Zdut'])),period=360)
            pha21 = np.unwrap(IM.Mat_pha(get_ZMIMO_element_list(1, 0, Z['Zdut'])),period=360)
            pha22 = np.unwrap(IM.Mat_pha(get_ZMIMO_element_list(1, 1, Z['Zdut'])),period=360)
        else:
            pha11 = IM.Mat_pha(get_ZMIMO_element_list(0, 0, Z['Zdut']))
            pha12 = IM.Mat_pha(get_ZMIMO_element_list(0, 1, Z['Zdut']))
            pha21 = IM.Mat_pha(get_ZMIMO_element_list(1, 0, Z['Zdut']))
            pha22 = IM.Mat_pha(get_ZMIMO_element_list(1, 1, Z['Zdut']))
            
        fig.append_trace(go.Scatter(x=Z['Zdut'].f, y=pha11, name='Zdut_pha', line_color=colors[0], mode='lines'), row=2, col=1, )
        fig.append_trace(go.Scatter(x=Z['Zdut'].f, y=pha12, name='Zdut_pha', line_color=colors[0], mode='lines'), row=2, col=2, )
        fig.append_trace(go.Scatter(x=Z['Zdut'].f, y=pha21, name='Zdut_pha', line_color=colors[0], mode='lines'), row=4, col=1, )
        fig.append_trace(go.Scatter(x=Z['Zdut'].f, y=pha22, name='Zdut_pha', line_color=colors[0], mode='lines'), row=4, col=2, )
        
    if NET_plot and "Znet" in Z.keys():
        # mag plots
        if dB_unit:
            fig.append_trace(go.Scatter(x=Z['Znet'].f, y=IM.Mat_mag(
                (get_ZMIMO_element_list(0, 0, Z['Znet']))), name='Znet_mag', line_color=colors[1], mode='lines'), row=1, col=1, )
            fig.append_trace(go.Scatter(x=Z['Znet'].f, y=IM.Mat_mag(
                (get_ZMIMO_element_list(0, 1, Z['Znet']))), name='Znet_mag', line_color=colors[1], mode='lines'), row=1, col=2, )
            fig.append_trace(go.Scatter(x=Z['Znet'].f, y=IM.Mat_mag(
                (get_ZMIMO_element_list(1, 0, Z['Znet']))), name='Znet_mag', line_color=colors[1], mode='lines'), row=3, col=1, )
            fig.append_trace(go.Scatter(x=Z['Znet'].f, y=IM.Mat_mag(
                (get_ZMIMO_element_list(1, 1, Z['Znet']))), name='Znet_mag', line_color=colors[1], mode='lines'), row=3, col=2, )
        else:
            
            fig.append_trace(go.Scatter(x=Z['Znet'].f, y=np.abs(get_ZMIMO_element_list(0, 0, Z['Znet'])), 
                                        name='Znet_mag', line_color=colors[1], mode='lines'), row=1, col=1, )
            fig.append_trace(go.Scatter(x=Z['Znet'].f, y=np.abs(get_ZMIMO_element_list(0, 1, Z['Znet'])), 
                                        name='Znet_mag', line_color=colors[1], mode='lines'), row=1, col=2, )
            fig.append_trace(go.Scatter(x=Z['Znet'].f, y=np.abs(get_ZMIMO_element_list(1, 0, Z['Znet'])), 
                                        name='Znet_mag', line_color=colors[1], mode='lines'), row=3, col=1, )
            fig.append_trace(go.Scatter(x=Z['Znet'].f, y=np.abs(get_ZMIMO_element_list(1, 1, Z['Znet'])), 
                                        name='Znet_mag', line_color=colors[1], mode='lines'), row=3, col=2, )
            
        # phase plots
        if unwrap_on:
            pha11 = np.unwrap(IM.Mat_pha(get_ZMIMO_element_list(0, 0, Z['Znet'])),period=360)
            pha12 = np.unwrap(IM.Mat_pha(get_ZMIMO_element_list(0, 1, Z['Znet'])),period=360)
            pha21 = np.unwrap(IM.Mat_pha(get_ZMIMO_element_list(1, 0, Z['Znet'])),period=360)
            pha22 = np.unwrap(IM.Mat_pha(get_ZMIMO_element_list(1, 1, Z['Znet'])),period=360)
        else:
            pha11 = IM.Mat_pha(get_ZMIMO_element_list(0, 0, Z['Znet']))
            pha12 = IM.Mat_pha(get_ZMIMO_element_list(0, 1, Z['Znet']))
            pha21 = IM.Mat_pha(get_ZMIMO_element_list(1, 0, Z['Znet']))
            pha22 = IM.Mat_pha(get_ZMIMO_element_list(1, 1, Z['Znet']))
        fig.append_trace(go.Scatter(x=Z['Znet'].f, y=pha11, name='Znet_pha', line_color=colors[1], mode='lines'), row=2, col=1, )
        fig.append_trace(go.Scatter(x=Z['Znet'].f, y=pha12, name='Znet_pha', line_color=colors[1], mode='lines'), row=2, col=2, )
        fig.append_trace(go.Scatter(x=Z['Znet'].f, y=pha21, name='Znet_pha', line_color=colors[1], mode='lines'), row=4, col=1, )
        fig.append_trace(go.Scatter(x=Z['Znet'].f, y=pha22, name='Znet_pha', line_color=colors[1], mode='lines'), row=4, col=2, )

    if log_axis:
        fig.update_xaxes(type="log", row=1, col=1)
        fig.update_xaxes(type="log", row=2, col=1)
        fig.update_xaxes(type="log", row=1, col=2)
        fig.update_xaxes(type="log", row=2, col=2)
        fig.update_xaxes(type="log", row=3, col=1)
        fig.update_xaxes(type="log", row=4, col=1)
        fig.update_xaxes(type="log", row=3, col=2)
        fig.update_xaxes(type="log", row=4, col=2)
        


    fig.update_xaxes(title_text="Frequency (Hz)", row=4, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=4, col=2)
    if dB_unit:
        fig.update_yaxes(title_text="Amplitude (dB)", row=1, col=1)
        fig.update_yaxes(title_text="Amplitude (dB)", row=3, col=1)
    else:        
        fig.update_yaxes(title_text="Amplitude (ohm)", row=1, col=1)        
        fig.update_yaxes(title_text="Amplitude (ohm)", row=3, col=1)
        fig.update_yaxes(type="log", row=1, col=1)
        fig.update_yaxes(type="log", row=1, col=2)
        fig.update_yaxes(type="log", row=3, col=1)
        fig.update_yaxes(type="log", row=3, col=2)
    
    fig.update_yaxes(title_text="Phase (deg)", row=2, col=1)
    fig.update_yaxes(title_text="Phase (deg)", row=4, col=1)
    fig.update_layout(title='MIMO impedance Bode plot')
    f_low = Z['Zdut'].f[0]
    f_high = Z['Zdut'].f[len(Z['Zdut'].f)-1]
    if f_low<0:
        adjust_freq_range_MIMO(fig, f_low=0, f_high=f_high)
    else:
        adjust_freq_range_MIMO(fig, f_low=f_low, f_high=f_high)
    
    fig.write_html(figurePath + "\\" + pName + ".html")
    print('Impedance figure saved as:')
    print(figurePath + "\\" + pName + ".html")
    
def add_Z_MIMO_Bode_plot_html(fig, color, legend_name, Z, log_axis=True, dB_unit=True, unwrap_on=True, NET_plot=False):
    # mag plots
    if dB_unit:
        fig.append_trace(go.Scatter(x=Z['Zdut'].f, y=IM.Mat_mag(
            (get_ZMIMO_element_list(0, 0, Z['Zdut']))), name=legend_name+'Zdut11_mag', line_color=color, mode='lines'), row=1, col=1, )
        fig.append_trace(go.Scatter(x=Z['Zdut'].f, y=IM.Mat_mag(
            (get_ZMIMO_element_list(0, 1, Z['Zdut']))), name=legend_name+'Zdut12_mag', line_color=color, mode='lines'), row=1, col=2, )
        fig.append_trace(go.Scatter(x=Z['Zdut'].f, y=IM.Mat_mag(
            (get_ZMIMO_element_list(1, 0, Z['Zdut']))), name=legend_name+'Zdut21_mag', line_color=color, mode='lines'), row=3, col=1, )
        fig.append_trace(go.Scatter(x=Z['Zdut'].f, y=IM.Mat_mag(
            (get_ZMIMO_element_list(1, 1, Z['Zdut']))), name=legend_name+'Zdut22_mag', line_color=color, mode='lines'), row=3, col=2, )
        
    else:
        
        fig.append_trace(go.Scatter(x=Z['Zdut'].f, y=np.abs(get_ZMIMO_element_list(0, 0, Z['Zdut'])), 
                                    name=legend_name+'Zdut11_mag', line_color=color, mode='lines'), row=1, col=1, )
        fig.append_trace(go.Scatter(x=Z['Zdut'].f, y=np.abs(get_ZMIMO_element_list(0, 1, Z['Zdut'])), 
                                    name=legend_name+'Zdut12_mag', line_color=color, mode='lines'), row=1, col=2, )
        fig.append_trace(go.Scatter(x=Z['Zdut'].f, y=np.abs(get_ZMIMO_element_list(1, 0, Z['Zdut'])), 
                                    name=legend_name+'Zdut21_mag', line_color=color, mode='lines'), row=3, col=1, )
        fig.append_trace(go.Scatter(x=Z['Zdut'].f, y=np.abs(get_ZMIMO_element_list(1, 1, Z['Zdut'])), 
                                    name=legend_name+'Zdut22_mag', line_color=color, mode='lines'), row=3, col=2, )
    
    # phase plots
    if unwrap_on:
        pha11 = np.unwrap(IM.Mat_pha(get_ZMIMO_element_list(0, 0, Z['Zdut'])),period=360)
        pha12 = np.unwrap(IM.Mat_pha(get_ZMIMO_element_list(0, 1, Z['Zdut'])),period=360)
        pha21 = np.unwrap(IM.Mat_pha(get_ZMIMO_element_list(1, 0, Z['Zdut'])),period=360)
        pha22 = np.unwrap(IM.Mat_pha(get_ZMIMO_element_list(1, 1, Z['Zdut'])),period=360)
    else:
        pha11 = IM.Mat_pha(get_ZMIMO_element_list(0, 0, Z['Zdut']))
        pha12 = IM.Mat_pha(get_ZMIMO_element_list(0, 1, Z['Zdut']))
        pha21 = IM.Mat_pha(get_ZMIMO_element_list(1, 0, Z['Zdut']))
        pha22 = IM.Mat_pha(get_ZMIMO_element_list(1, 1, Z['Zdut']))
    fig.append_trace(go.Scatter(x=Z['Zdut'].f, y=pha11, name=legend_name+'Zdut11_pha', line_color=color, mode='lines'), row=2, col=1, )

    fig.append_trace(go.Scatter(x=Z['Zdut'].f, y=pha12, name=legend_name+'Zdut12_pha', line_color=color, mode='lines'), row=2, col=2, )

    fig.append_trace(go.Scatter(x=Z['Zdut'].f, y=pha21, name=legend_name+'Zdut21_pha', line_color=color, mode='lines'), row=4, col=1, )

    fig.append_trace(go.Scatter(x=Z['Zdut'].f, y=pha22, name=legend_name+'Zdut22_pha', line_color=color, mode='lines'), row=4, col=2, )
    
    if NET_plot and "Znet" in Z.keys():
        if dB_unit:
            fig.append_trace(go.Scatter(x=Z['Znet'].f, y=IM.Mat_mag(
                (get_ZMIMO_element_list(0, 0, Z['Znet']))), name=legend_name+'Znet11_mag', line_color=color, mode='lines'), row=1, col=1, )
            fig.append_trace(go.Scatter(x=Z['Znet'].f, y=IM.Mat_mag(
                (get_ZMIMO_element_list(0, 1, Z['Znet']))), name=legend_name+'Znet12_mag', line_color=color, mode='lines'), row=1, col=2, )
            fig.append_trace(go.Scatter(x=Z['Znet'].f, y=IM.Mat_mag(
                (get_ZMIMO_element_list(1, 0, Z['Znet']))), name=legend_name+'Znet21_mag', line_color=color, mode='lines'), row=3, col=1, )
            fig.append_trace(go.Scatter(x=Z['Znet'].f, y=IM.Mat_mag(
                (get_ZMIMO_element_list(1, 1, Z['Znet']))), name=legend_name+'Znet22_mag', line_color=color, mode='lines'), row=3, col=2, )
            
        else:
            
            fig.append_trace(go.Scatter(x=Z['Znet'].f, y=np.abs(get_ZMIMO_element_list(0, 0, Z['Znet'])), 
                                        name=legend_name+'Znet11_mag', line_color=color, mode='lines'), row=1, col=1, )
            fig.append_trace(go.Scatter(x=Z['Znet'].f, y=np.abs(get_ZMIMO_element_list(0, 1, Z['Znet'])), 
                                        name=legend_name+'Znet12_mag', line_color=color, mode='lines'), row=1, col=2, )
            fig.append_trace(go.Scatter(x=Z['Znet'].f, y=np.abs(get_ZMIMO_element_list(1, 0, Z['Znet'])), 
                                        name=legend_name+'Znet21_mag', line_color=color, mode='lines'), row=3, col=1, )
            fig.append_trace(go.Scatter(x=Z['Znet'].f, y=np.abs(get_ZMIMO_element_list(1, 1, Z['Znet'])), 
                                        name=legend_name+'Znet22_mag', line_color=color, mode='lines'), row=3, col=2, )
        
        # phase plots
        if unwrap_on:
            pha11 = np.unwrap(IM.Mat_pha(get_ZMIMO_element_list(0, 0, Z['Znet'])),period=360)
            pha12 = np.unwrap(IM.Mat_pha(get_ZMIMO_element_list(0, 1, Z['Znet'])),period=360)
            pha21 = np.unwrap(IM.Mat_pha(get_ZMIMO_element_list(1, 0, Z['Znet'])),period=360)
            pha22 = np.unwrap(IM.Mat_pha(get_ZMIMO_element_list(1, 1, Z['Znet'])),period=360)
        else:
            pha11 = IM.Mat_pha(get_ZMIMO_element_list(0, 0, Z['Znet']))
            pha12 = IM.Mat_pha(get_ZMIMO_element_list(0, 1, Z['Znet']))
            pha21 = IM.Mat_pha(get_ZMIMO_element_list(1, 0, Z['Znet']))
            pha22 = IM.Mat_pha(get_ZMIMO_element_list(1, 1, Z['Znet']))
        fig.append_trace(go.Scatter(x=Z['Znet'].f, y=pha11, name=legend_name+'Znet11_pha', line_color=color, mode='lines'), row=2, col=1, )

        fig.append_trace(go.Scatter(x=Z['Znet'].f, y=pha12, name=legend_name+'Znet12_pha', line_color=color, mode='lines'), row=2, col=2, )

        fig.append_trace(go.Scatter(x=Z['Znet'].f, y=pha21, name=legend_name+'Znet21_pha', line_color=color, mode='lines'), row=4, col=1, )

        fig.append_trace(go.Scatter(x=Z['Znet'].f, y=pha22, name=legend_name+'Znet22_pha', line_color=color, mode='lines'), row=4, col=2, )
        
    
    if log_axis:
        fig.update_xaxes(type="log", row=1, col=1)
        fig.update_xaxes(type="log", row=2, col=1)
        fig.update_xaxes(type="log", row=1, col=2)
        fig.update_xaxes(type="log", row=2, col=2)
        fig.update_xaxes(type="log", row=3, col=1)
        fig.update_xaxes(type="log", row=4, col=1)
        fig.update_xaxes(type="log", row=3, col=2)
        fig.update_xaxes(type="log", row=4, col=2)
        

    fig.update_xaxes(title_text="Frequency (Hz)", row=4, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=4, col=2)
    if dB_unit:
        fig.update_yaxes(title_text="Amplitude (dB)", row=1, col=1)
        fig.update_yaxes(title_text="Amplitude (dB)", row=3, col=1)
    else:        
        fig.update_yaxes(title_text="Amplitude (ohm)", row=1, col=1)        
        fig.update_yaxes(title_text="Amplitude (ohm)", row=3, col=1)
        fig.update_yaxes(type="log", row=1, col=1)
        fig.update_yaxes(type="log", row=1, col=2)
        fig.update_yaxes(type="log", row=3, col=1)
        fig.update_yaxes(type="log", row=3, col=2)
    
    fig.update_yaxes(title_text="Phase (deg)", row=2, col=1)
    fig.update_yaxes(title_text="Phase (deg)", row=4, col=1)
    fig.update_layout(title='MIMO impedance Bode plot')
    return fig
    

def get_K_MIMO_Bode_plot_html(Z, figurePath, pName, DUT_plot=True, NET_plot=True, log_axis=True, dB_unit=True, unwrap_on=True):

    fig = make_subplots(rows=4, cols=2,
                        shared_xaxes=True,
                        vertical_spacing=0.02)

    colors = create_colormap(nfig=2)

    if DUT_plot:
        if dB_unit:
            fig.append_trace(go.Scatter(x=Z['Kdut'].f, y=IM.Mat_mag(
                (get_ZMIMO_element_list(0, 0, Z['Kdut']))), name='Kdut_mag', line_color=colors[0], mode='lines'), row=1, col=1, )
            fig.append_trace(go.Scatter(x=Z['Kdut'].f, y=IM.Mat_mag(
                (get_ZMIMO_element_list(0, 1, Z['Kdut']))), name='Kdut_mag', line_color=colors[0], mode='lines'), row=1, col=2, )
            fig.append_trace(go.Scatter(x=Z['Kdut'].f, y=IM.Mat_mag(
                (get_ZMIMO_element_list(1, 0, Z['Kdut']))), name='Kdut_mag', line_color=colors[0], mode='lines'), row=3, col=1, )
            fig.append_trace(go.Scatter(x=Z['Kdut'].f, y=IM.Mat_mag(
                (get_ZMIMO_element_list(1, 1, Z['Kdut']))), name='Kdut_mag', line_color=colors[0], mode='lines'), row=3, col=2, )
        else:
            fig.append_trace(go.Scatter(x=Z['Kdut'].f, y=np.abs(get_ZMIMO_element_list(0, 0, Z['Kdut'])), name='Kdut_mag', line_color=colors[0], mode='lines'), row=1, col=1, )
            fig.append_trace(go.Scatter(x=Z['Kdut'].f, y=np.abs(get_ZMIMO_element_list(0, 1, Z['Kdut'])), name='Kdut_mag', line_color=colors[0], mode='lines'), row=1, col=2, )
            fig.append_trace(go.Scatter(x=Z['Kdut'].f, y=np.abs(get_ZMIMO_element_list(1, 0, Z['Kdut'])), name='Kdut_mag', line_color=colors[0], mode='lines'), row=3, col=1, )
            fig.append_trace(go.Scatter(x=Z['Kdut'].f, y=np.abs(get_ZMIMO_element_list(1, 1, Z['Kdut'])), name='Kdut_mag', line_color=colors[0], mode='lines'), row=3, col=2, )    
            
        
        if unwrap_on:
            pha11 = np.unwrap(IM.Mat_pha(get_ZMIMO_element_list(0, 0, Z['Kdut'])),period=360)
            pha12 = np.unwrap(IM.Mat_pha(get_ZMIMO_element_list(0, 1, Z['Kdut'])),period=360)
            pha21 = np.unwrap(IM.Mat_pha(get_ZMIMO_element_list(1, 0, Z['Kdut'])),period=360)
            pha22 = np.unwrap(IM.Mat_pha(get_ZMIMO_element_list(1, 1, Z['Kdut'])),period=360)
        else:
            pha11 = IM.Mat_pha(get_ZMIMO_element_list(0, 0, Z['Kdut']))
            pha12 = IM.Mat_pha(get_ZMIMO_element_list(0, 1, Z['Kdut']))
            pha21 = IM.Mat_pha(get_ZMIMO_element_list(1, 0, Z['Kdut']))
            pha22 = IM.Mat_pha(get_ZMIMO_element_list(1, 1, Z['Kdut']))
        
        fig.append_trace(go.Scatter(x=Z['Kdut'].f, y=pha11, name='Kdut_pha', line_color=colors[0], mode='lines'), row=2, col=1, )
        
        fig.append_trace(go.Scatter(x=Z['Kdut'].f, y=pha12, name='Kdut_pha', line_color=colors[0], mode='lines'), row=2, col=2, )
       
        fig.append_trace(go.Scatter(x=Z['Kdut'].f, y=pha21, name='Kdut_pha', line_color=colors[0], mode='lines'), row=4, col=1, )
        
        fig.append_trace(go.Scatter(x=Z['Kdut'].f, y=pha22, name='Kdut_pha', line_color=colors[0], mode='lines'), row=4, col=2, )
        
    if NET_plot and "Knet" in Z.keys():
        if dB_unit:
            fig.append_trace(go.Scatter(x=Z['Knet'].f, y=IM.Mat_mag(
                (get_ZMIMO_element_list(0, 0, Z['Knet']))), name='Knet_mag', line_color=colors[1], mode='lines'), row=1, col=1, )
            fig.append_trace(go.Scatter(x=Z['Knet'].f, y=IM.Mat_mag(
                (get_ZMIMO_element_list(0, 1, Z['Knet']))), name='Knet_mag', line_color=colors[1], mode='lines'), row=1, col=2, )
            fig.append_trace(go.Scatter(x=Z['Knet'].f, y=IM.Mat_mag(
                (get_ZMIMO_element_list(1, 0, Z['Knet']))), name='Knet_mag', line_color=colors[1], mode='lines'), row=3, col=1, )
            fig.append_trace(go.Scatter(x=Z['Knet'].f, y=IM.Mat_mag(
                (get_ZMIMO_element_list(1, 1, Z['Knet']))), name='Knet_mag', line_color=colors[1], mode='lines'), row=3, col=2, )
        else:
            fig.append_trace(go.Scatter(x=Z['Knet'].f, y=np.abs(get_ZMIMO_element_list(0, 0, Z['Knet'])), name='Knet_mag', line_color=colors[1], mode='lines'), row=1, col=1, )
            fig.append_trace(go.Scatter(x=Z['Knet'].f, y=np.abs(get_ZMIMO_element_list(0, 1, Z['Knet'])), name='Knet_mag', line_color=colors[1], mode='lines'), row=1, col=2, )
            fig.append_trace(go.Scatter(x=Z['Knet'].f, y=np.abs(get_ZMIMO_element_list(1, 0, Z['Knet'])), name='Knet_mag', line_color=colors[1], mode='lines'), row=3, col=1, )
            fig.append_trace(go.Scatter(x=Z['Knet'].f, y=np.abs(get_ZMIMO_element_list(1, 1, Z['Knet'])), name='Knet_mag', line_color=colors[1], mode='lines'), row=3, col=2, )
                
            
        if unwrap_on:
            pha11 = np.unwrap(IM.Mat_pha(get_ZMIMO_element_list(0, 0, Z['Knet'])),period=360)
            pha12 = np.unwrap(IM.Mat_pha(get_ZMIMO_element_list(0, 1, Z['Knet'])),period=360)
            pha21 = np.unwrap(IM.Mat_pha(get_ZMIMO_element_list(1, 0, Z['Knet'])),period=360)
            pha22 = np.unwrap(IM.Mat_pha(get_ZMIMO_element_list(1, 1, Z['Knet'])),period=360)
        else:
            pha11 = IM.Mat_pha(get_ZMIMO_element_list(0, 0, Z['Knet']))
            pha12 = IM.Mat_pha(get_ZMIMO_element_list(0, 1, Z['Knet']))
            pha21 = IM.Mat_pha(get_ZMIMO_element_list(1, 0, Z['Knet']))
            pha22 = IM.Mat_pha(get_ZMIMO_element_list(1, 1, Z['Knet']))
        fig.append_trace(go.Scatter(x=Z['Knet'].f, y=pha11, name='Knet_pha', line_color=colors[1], mode='lines'), row=2, col=1, )
        fig.append_trace(go.Scatter(x=Z['Knet'].f, y=pha12, name='Knet_pha', line_color=colors[1], mode='lines'), row=2, col=2, )
        fig.append_trace(go.Scatter(x=Z['Knet'].f, y=pha21, name='Knet_pha', line_color=colors[1], mode='lines'), row=4, col=1, )
        fig.append_trace(go.Scatter(x=Z['Knet'].f, y=pha22, name='Knet_pha', line_color=colors[1], mode='lines'), row=4, col=2, )

    if log_axis:
        fig.update_xaxes(type="log", row=1, col=1)
        fig.update_xaxes(type="log", row=2, col=1)
        fig.update_xaxes(type="log", row=1, col=2)
        fig.update_xaxes(type="log", row=2, col=2)
        fig.update_xaxes(type="log", row=3, col=1)
        fig.update_xaxes(type="log", row=4, col=1)
        fig.update_xaxes(type="log", row=3, col=2)
        fig.update_xaxes(type="log", row=4, col=2)

    fig.update_xaxes(title_text="Frequency (Hz)", row=4, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=4, col=2)
    if dB_unit:
        fig.update_yaxes(title_text="Amplitude (dB)", row=1, col=1)
        fig.update_yaxes(title_text="Amplitude (dB)", row=3, col=1)
    else:
        fig.update_yaxes(title_text="Amplitude (MW/rad)", row=1, col=1)        
        fig.update_yaxes(title_text="Amplitude (MW/kV)", row=1, col=2)        
        fig.update_yaxes(title_text="Amplitude (MVAR/rad)", row=3, col=1)
        fig.update_yaxes(title_text="Amplitude (MVAR/kV)", row=3, col=2)
        fig.update_yaxes(title_text="Phase (deg)", row=2, col=2)
        fig.update_yaxes(title_text="Phase (deg)", row=4, col=2)
        fig.update_yaxes(type="log", row=1, col=1)
        fig.update_yaxes(type="log", row=1, col=2)
        fig.update_yaxes(type="log", row=3, col=1)
        fig.update_yaxes(type="log", row=3, col=2)
    
    fig.update_yaxes(title_text="Phase (deg)", row=2, col=1)
    fig.update_yaxes(title_text="Phase (deg)", row=4, col=1)
    
    fig.update_layout(title='Jacobian matrix Bode plot')
    f_low = Z['Kdut'].f[0]
    f_high = Z['Kdut'].f[len(Z['Kdut'].f)-1]
    if f_low<0:
        adjust_freq_range_MIMO(fig, f_low=0, f_high=f_high)
    else:
        adjust_freq_range_MIMO(fig, f_low=f_low, f_high=f_high)
    fig.write_html(figurePath + "\\" + pName + ".html")
    print('Jacobian matrix figure saved as:')
    print(figurePath + "\\" + pName + ".html")

def add_K_MIMO_Bode_plot_html(fig, color, legend_name, Z, log_axis=True, dB_unit=True, unwrap_on=True, NET_plot=False):

    if dB_unit:
        fig.append_trace(go.Scatter(x=Z['Kdut'].f, y=IM.Mat_mag(
            (get_ZMIMO_element_list(0, 0, Z['Kdut']))), name=legend_name+'Kdut11_mag', line_color=color, mode='lines'), row=1, col=1, )
        fig.append_trace(go.Scatter(x=Z['Kdut'].f, y=IM.Mat_mag(
            (get_ZMIMO_element_list(0, 1, Z['Kdut']))), name=legend_name+'Kdut12_mag', line_color=color, mode='lines'), row=1, col=2, )
        fig.append_trace(go.Scatter(x=Z['Kdut'].f, y=IM.Mat_mag(
            (get_ZMIMO_element_list(1, 0, Z['Kdut']))), name=legend_name+'Kdut21_mag', line_color=color, mode='lines'), row=3, col=1, )
        fig.append_trace(go.Scatter(x=Z['Kdut'].f, y=IM.Mat_mag(
            (get_ZMIMO_element_list(1, 1, Z['Kdut']))), name=legend_name+'Kdut22_mag', line_color=color, mode='lines'), row=3, col=2, )
    else:
        fig.append_trace(go.Scatter(x=Z['Kdut'].f, y=np.abs(get_ZMIMO_element_list(0, 0, Z['Kdut']))/180*np.pi, name=legend_name+'Kdut11_mag', line_color=color, mode='lines'), row=1, col=1, )
        fig.append_trace(go.Scatter(x=Z['Kdut'].f, y=np.abs(get_ZMIMO_element_list(0, 1, Z['Kdut'])), name=legend_name+'Kdut12_mag', line_color=color, mode='lines'), row=1, col=2, )
        fig.append_trace(go.Scatter(x=Z['Kdut'].f, y=np.abs(get_ZMIMO_element_list(1, 0, Z['Kdut']))/180*np.pi, name=legend_name+'Kdut21_mag', line_color=color, mode='lines'), row=3, col=1, )
        fig.append_trace(go.Scatter(x=Z['Kdut'].f, y=np.abs(get_ZMIMO_element_list(1, 1, Z['Kdut'])), name=legend_name+'Kdut22_mag', line_color=color, mode='lines'), row=3, col=2, )    
        
    if unwrap_on:
        pha11 = np.unwrap(IM.Mat_pha(get_ZMIMO_element_list(0, 0, Z['Kdut'])),period=360)
        pha12 = np.unwrap(IM.Mat_pha(get_ZMIMO_element_list(0, 1, Z['Kdut'])),period=360)
        pha21 = np.unwrap(IM.Mat_pha(get_ZMIMO_element_list(1, 0, Z['Kdut'])),period=360)
        pha22 = np.unwrap(IM.Mat_pha(get_ZMIMO_element_list(1, 1, Z['Kdut'])),period=360)
    else:
        pha11 = IM.Mat_pha(get_ZMIMO_element_list(0, 0, Z['Kdut']))
        pha12 = IM.Mat_pha(get_ZMIMO_element_list(0, 1, Z['Kdut']))
        pha21 = IM.Mat_pha(get_ZMIMO_element_list(1, 0, Z['Kdut']))
        pha22 = IM.Mat_pha(get_ZMIMO_element_list(1, 1, Z['Kdut']))
    fig.append_trace(go.Scatter(x=Z['Kdut'].f, y=pha11, name=legend_name+'Kdut11_pha', line_color=color, mode='lines'), row=2, col=1, )
    fig.append_trace(go.Scatter(x=Z['Kdut'].f, y=pha12, name=legend_name+'Kdut12_pha', line_color=color, mode='lines'), row=2, col=2, )
    fig.append_trace(go.Scatter(x=Z['Kdut'].f, y=pha21, name=legend_name+'Kdut21_pha', line_color=color, mode='lines'), row=4, col=1, )
    fig.append_trace(go.Scatter(x=Z['Kdut'].f, y=pha22, name=legend_name+'Kdut22_pha', line_color=color, mode='lines'), row=4, col=2, )
                
    if NET_plot and "Knet" in Z.keys():
        if dB_unit:
            fig.append_trace(go.Scatter(x=Z['Knet'].f, y=IM.Mat_mag(
                (get_ZMIMO_element_list(0, 0, Z['Knet']))), name=legend_name+'Knet11_mag', line_color=color, mode='lines'), row=1, col=1, )
            fig.append_trace(go.Scatter(x=Z['Knet'].f, y=IM.Mat_mag(
                (get_ZMIMO_element_list(0, 1, Z['Knet']))), name=legend_name+'Knet12_mag', line_color=color, mode='lines'), row=1, col=2, )
            fig.append_trace(go.Scatter(x=Z['Knet'].f, y=IM.Mat_mag(
                (get_ZMIMO_element_list(1, 0, Z['Knet']))), name=legend_name+'Knet21_mag', line_color=color, mode='lines'), row=3, col=1, )
            fig.append_trace(go.Scatter(x=Z['Knet'].f, y=IM.Mat_mag(
                (get_ZMIMO_element_list(1, 1, Z['Knet']))), name=legend_name+'Knet22_mag', line_color=color, mode='lines'), row=3, col=2, )
        else:
            fig.append_trace(go.Scatter(x=Z['Knet'].f, y=np.abs(get_ZMIMO_element_list(0, 0, Z['Knet']))/180*np.pi, name=legend_name+'Knet11_mag', line_color=color, mode='lines'), row=1, col=1, )
            fig.append_trace(go.Scatter(x=Z['Knet'].f, y=np.abs(get_ZMIMO_element_list(0, 1, Z['Knet'])), name=legend_name+'Knet12_mag', line_color=color, mode='lines'), row=1, col=2, )
            fig.append_trace(go.Scatter(x=Z['Knet'].f, y=np.abs(get_ZMIMO_element_list(1, 0, Z['Knet']))/180*np.pi, name=legend_name+'Knet21_mag', line_color=color, mode='lines'), row=3, col=1, )
            fig.append_trace(go.Scatter(x=Z['Knet'].f, y=np.abs(get_ZMIMO_element_list(1, 1, Z['Knet'])), name=legend_name+'Knet22_mag', line_color=color, mode='lines'), row=3, col=2, )    
            
        if unwrap_on:
            pha11 = np.unwrap(IM.Mat_pha(get_ZMIMO_element_list(0, 0, Z['Knet'])),period=360)
            pha12 = np.unwrap(IM.Mat_pha(get_ZMIMO_element_list(0, 1, Z['Knet'])),period=360)
            pha21 = np.unwrap(IM.Mat_pha(get_ZMIMO_element_list(1, 0, Z['Knet'])),period=360)
            pha22 = np.unwrap(IM.Mat_pha(get_ZMIMO_element_list(1, 1, Z['Knet'])),period=360)
        else:
            pha11 = IM.Mat_pha(get_ZMIMO_element_list(0, 0, Z['Knet']))
            pha12 = IM.Mat_pha(get_ZMIMO_element_list(0, 1, Z['Knet']))
            pha21 = IM.Mat_pha(get_ZMIMO_element_list(1, 0, Z['Knet']))
            pha22 = IM.Mat_pha(get_ZMIMO_element_list(1, 1, Z['Knet']))
        fig.append_trace(go.Scatter(x=Z['Knet'].f, y=pha11, name=legend_name+'Knet11_pha', line_color=color, mode='lines'), row=2, col=1, )
        fig.append_trace(go.Scatter(x=Z['Knet'].f, y=pha12, name=legend_name+'Knet12_pha', line_color=color, mode='lines'), row=2, col=2, )
        fig.append_trace(go.Scatter(x=Z['Knet'].f, y=pha21, name=legend_name+'Knet21_pha', line_color=color, mode='lines'), row=4, col=1, )
        fig.append_trace(go.Scatter(x=Z['Knet'].f, y=pha22, name=legend_name+'Knet22_pha', line_color=color, mode='lines'), row=4, col=2, )

    if log_axis:
        fig.update_xaxes(type="log", row=1, col=1)
        fig.update_xaxes(type="log", row=2, col=1)
        fig.update_xaxes(type="log", row=1, col=2)
        fig.update_xaxes(type="log", row=2, col=2)
        fig.update_xaxes(type="log", row=3, col=1)
        fig.update_xaxes(type="log", row=4, col=1)
        fig.update_xaxes(type="log", row=3, col=2)
        fig.update_xaxes(type="log", row=4, col=2)

    fig.update_xaxes(title_text="Frequency (Hz)", row=4, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=4, col=2)
    if dB_unit:
        fig.update_yaxes(title_text="Amplitude (dB)", row=1, col=1)
        fig.update_yaxes(title_text="Amplitude (dB)", row=3, col=1)
    else:
        fig.update_yaxes(title_text="Amplitude (MW/deg)", row=1, col=1)        
        fig.update_yaxes(title_text="Amplitude (MW/kV)", row=1, col=2)        
        fig.update_yaxes(title_text="Amplitude (MVAR/deg)", row=3, col=1)
        fig.update_yaxes(title_text="Amplitude (MVAR/kV)", row=3, col=2)
        fig.update_yaxes(title_text="Phase (deg)", row=2, col=2)
        fig.update_yaxes(title_text="Phase (deg)", row=4, col=2)
        fig.update_yaxes(type="log", row=1, col=1)
        fig.update_yaxes(type="log", row=1, col=2)
        fig.update_yaxes(type="log", row=3, col=1)
        fig.update_yaxes(type="log", row=3, col=2)
    
    fig.update_yaxes(title_text="Phase (deg)", row=2, col=1)
    fig.update_yaxes(title_text="Phase (deg)", row=4, col=1)
    
    fig.update_layout(title='Jacobian matrix Bode plot')
    
def add_filled_area_on_phase_plot(fig,row,col,f_low,f_high,phase_band_no=1):
    for k in range(phase_band_no):
        fig.append_trace(go.Scatter(x=[f_low, f_high], y=[90+k*360,90+k*360],
                                 fill=None, mode='lines',line_color='#F9BFBF',showlegend = False), row=row,col=col)
        fig.append_trace(go.Scatter(x=[f_low, f_high], y=[270+k*360,270+k*360],
                                 fill='tonexty', # fill area between trace0 and trace1
                                 mode='lines', line_color='#F9BFBF',showlegend = False), row=row,col=col)
        
        fig.append_trace(go.Scatter(x=[f_low, f_high], y=[-270-k*360,-270-k*360],
                                 fill=None, mode='lines',line_color='#F9BFBF',showlegend = False), row=row,col=col)
        fig.append_trace(go.Scatter(x=[f_low, f_high], y=[-90-k*360,-90-k*360],
                                 fill='tonexty', # fill area between trace0 and trace1
                                 mode='lines', line_color='#F9BFBF',showlegend = False), row=row,col=col)
    return fig

def Single_SC_plot(settings,simPath,sc_name, DUT_plot=True, NET_plot=True, log_axis=True, dB_unit=True, unwrap_on=True):
    if settings['Immitance type'] == 'SISO':
        print('Ploting SISO models')

        # Get impedance data and folder
        print('Reading data...')
        print('-------------------------------')
        Z, IM_folder, IM_file_name = get_IM_data(simPath,sc_name,settings)
        print('-------------------------------')

        # Get Bode plot and saved in html
        print('Plotting data...')
        print('-------------------------------')
        if settings["Freq type"] == 'log':
            get_Bode_plot_html(Z[0], figurePath=IM_folder, pName=IM_file_name[0],
                           DUT_plot=DUT_plot, NET_plot=NET_plot, log_axis=True, dB_unit=dB_unit, unwrap_on=unwrap_on)
        else:
            get_Bode_plot_html(Z[0], figurePath=IM_folder, pName=IM_file_name[0],
                           DUT_plot=DUT_plot, NET_plot=NET_plot, log_axis=log_axis, dB_unit=dB_unit, unwrap_on=unwrap_on)
        
        print('-------------------------------')

    elif settings['Immitance type'] == 'MIMO':
        print('Ploting MIMO models')
        
        # Get impedance data and folder
        print('Reading data...')
        print('-------------------------------')
        Z, IM_folder, IM_file_name = get_IM_data(simPath,sc_name,settings)
        print('-------------------------------')

        # Get Bode plot and saved in html
        print('Plotting data...')
        print('-------------------------------')
        get_Z_MIMO_Bode_plot_html(Z[0], figurePath=IM_folder, pName=IM_file_name[0],
                           DUT_plot=DUT_plot, NET_plot=NET_plot, log_axis=False, dB_unit=dB_unit, unwrap_on=unwrap_on)
        if settings["Terminal type"] == "AC":
            get_Z_MIMO_Bode_plot_html(Z[1], figurePath=IM_folder, pName=IM_file_name[1],
                                DUT_plot=DUT_plot, NET_plot=NET_plot, log_axis=False, dB_unit=dB_unit, unwrap_on=unwrap_on)
            get_K_MIMO_Bode_plot_html(Z[2], figurePath=IM_folder, pName=IM_file_name[2],
                                DUT_plot=DUT_plot, NET_plot=NET_plot, log_axis=False, dB_unit=dB_unit, unwrap_on=unwrap_on)
        print('-------------------------------')


def Multi_SC_plot(settings, simPath, DUT_plot=True, NET_plot=True, log_axis=True, dB_unit=True, unwrap_on=True):
    
    if settings['Immitance type'] == 'SISO':
        print('Ploting SISO models')
        
        fig = make_subplots(rows=2, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.02)
        
        
        n_sc=int(settings["Nr scenarios"])

        colors = create_colormap(nfig=n_sc)
        
        for idx_sc in range(int(settings["Nr scenarios"])):
        
            if settings["Multiple scenarios"]:
                sc_name = f"sc{idx_sc+1:02d}_"
                print("Plotting for scenario nr "+str(idx_sc+1))
            else:
                sc_name = ""
                
            # Get impedance data and folder

            print('-------------------------------')
            Z, IM_folder, IM_file_name = get_IM_data(simPath,sc_name,settings)
            print('-------------------------------')
            
            if idx_sc==0 and Neg_Z_region:
                
                f_low = Z[0]['Zdut'].f[0]
                f_high = Z[0]['Zdut'].f[len(Z[0]['Zdut'].f)-1]
                if unwrap_on:
                    add_filled_area_on_phase_plot(fig,2,1,f_low,f_high,phase_band_no=2)
                else:
                    add_filled_area_on_phase_plot(fig,2,1,f_low,f_high,phase_band_no=1)

            # Get Bode plot and saved in html

            if settings["Freq type"] == 'log':
                add_Bode_plot_html(fig, colors[idx_sc], sc_name, Z[0], 
                                log_axis=True, dB_unit=dB_unit, unwrap_on=unwrap_on, NET_plot=NET_plot)
            else:
                add_Bode_plot_html(fig, colors[idx_sc], sc_name, Z[0], 
                                log_axis=True, dB_unit=dB_unit, unwrap_on=unwrap_on, NET_plot=NET_plot)
            

            print('Saving plot...')
            print('-------------------------------')
            figurePath=IM_folder
            pName=IM_file_name[0].split('sc')[0]+"allSC"
            
            fig.write_html(figurePath + "\\" + pName + ".html")
            print('Impedance figure saved as:')
            print(figurePath + "\\" + pName + ".html")
            print('-------------------------------')


    elif settings['Immitance type'] == 'MIMO':
        print('Ploting MIMO models')
        n_sc=int(settings["Nr scenarios"])

        colors = create_colormap(nfig=n_sc)
        
        fig1 = make_subplots(rows=4, cols=2,
                            shared_xaxes=True,
                            vertical_spacing=0.02)
        fig2 = make_subplots(rows=4, cols=2,
                            shared_xaxes=True,
                            vertical_spacing=0.02)
        fig3 = make_subplots(rows=4, cols=2,
                            shared_xaxes=True,
                            vertical_spacing=0.02)
        
        
        
        for idx_sc in range(int(settings["Nr scenarios"])):
        
            if settings["Multiple scenarios"]:
                sc_name = f"sc{idx_sc+1:02d}_"
                print("Plotting for scenario nr "+str(idx_sc+1))
            else:
                sc_name = ""
            
            # Get impedance data and folder
            print('-------------------------------')
            Z, IM_folder, IM_file_name = get_IM_data(simPath,sc_name,settings)
            print('-------------------------------')
            
            if idx_sc==0 and Neg_Z_region:
                f_low = Z[0]['Zdut'].f[0]
                f_high = Z[0]['Zdut'].f[len(Z[0]['Zdut'].f)-1]
                if unwrap_on:
                    add_filled_area_on_phase_plot(fig1,2,1,f_low,f_high,phase_band_no=2)
                    add_filled_area_on_phase_plot(fig1,2,2,f_low,f_high,phase_band_no=2)
                    add_filled_area_on_phase_plot(fig1,4,1,f_low,f_high,phase_band_no=2)
                    add_filled_area_on_phase_plot(fig1,4,2,f_low,f_high,phase_band_no=2)
                else:
                    add_filled_area_on_phase_plot(fig1,2,1,f_low,f_high,phase_band_no=1)
                    add_filled_area_on_phase_plot(fig1,2,2,f_low,f_high,phase_band_no=1)
                    add_filled_area_on_phase_plot(fig1,4,1,f_low,f_high,phase_band_no=1)
                    add_filled_area_on_phase_plot(fig1,4,2,f_low,f_high,phase_band_no=1)
                
                
                f_low = Z[1]['Zdut'].f[0]
                f_high = Z[1]['Zdut'].f[len(Z[1]['Zdut'].f)-1]
                if unwrap_on:
                    add_filled_area_on_phase_plot(fig2,2,1,f_low,f_high,phase_band_no=2)
                    add_filled_area_on_phase_plot(fig2,2,2,f_low,f_high,phase_band_no=2)
                    add_filled_area_on_phase_plot(fig2,4,1,f_low,f_high,phase_band_no=2)
                    add_filled_area_on_phase_plot(fig2,4,2,f_low,f_high,phase_band_no=2)
                else:
                    add_filled_area_on_phase_plot(fig2,2,1,f_low,f_high,phase_band_no=1)
                    add_filled_area_on_phase_plot(fig2,2,2,f_low,f_high,phase_band_no=1)
                    add_filled_area_on_phase_plot(fig2,4,1,f_low,f_high,phase_band_no=1)
                    add_filled_area_on_phase_plot(fig2,4,2,f_low,f_high,phase_band_no=1)
                
            
           
            add_Z_MIMO_Bode_plot_html(fig1, colors[idx_sc], sc_name, Z[0], 
                                log_axis=False, dB_unit=dB_unit, unwrap_on=unwrap_on, NET_plot=NET_plot)
            f_low = Z[0]['Zdut'].f[0]
            f_high = Z[0]['Zdut'].f[len(Z[0]['Zdut'].f)-1]
            adjust_freq_range_MIMO(fig1, f_low=f_low, f_high=f_high)
            
            f_low = Z[1]['Zdut'].f[0]
            f_high = Z[0]['Zdut'].f[len(Z[0]['Zdut'].f)-1]
            add_Z_MIMO_Bode_plot_html(fig2, colors[idx_sc], sc_name, Z[1], 
                                log_axis=False, dB_unit=dB_unit, unwrap_on=unwrap_on, NET_plot=NET_plot)
            if f_low < 0:
                adjust_freq_range_MIMO(fig2, f_low=0, f_high=f_high)
            else:
                adjust_freq_range_MIMO(fig2, f_low=f_low, f_high=f_high)
            
            add_K_MIMO_Bode_plot_html(fig3, colors[idx_sc], sc_name, Z[2], 
                                 log_axis=False, dB_unit=dB_unit, unwrap_on=unwrap_on, NET_plot=NET_plot)
            if f_low < 0:
                adjust_freq_range_MIMO(fig3, f_low=0, f_high=f_high)
            else:
                adjust_freq_range_MIMO(fig3, f_low=f_low, f_high=f_high)
                
            print('Saving plot...')
            print('-------------------------------')
            print(dB_unit)

            figurePath=IM_folder
            pName=IM_file_name[0].split('sc')[0]+"allSC"
            fig1.write_html(figurePath + "\\" + pName + ".html")
            print('Impedance figure saved as:')
            print(figurePath + "\\" + pName + ".html")
            
            pName=IM_file_name[1].split('sc')[0]+"allSC"
            fig2.write_html(figurePath + "\\" + pName + ".html")
            print('Impedance figure saved as:')
            print(figurePath + "\\" + pName + ".html")
            
            pName=IM_file_name[2].split('sc')[0]+"allSC"
            fig3.write_html(figurePath + "\\" + pName + ".html")
            print('Jacobian matrix figure saved as:')
            print(figurePath + "\\" + pName + ".html")
            print('-------------------------------')
        
def adjust_freq_range_SISO(fig,f_low,f_high,logaxis=True):
    if log_axis==False:
        fig.update_layout(
            xaxis=dict(range=[f_low,f_high]),
            xaxis2=dict(range=[f_low,f_high]),
            )
    else:
        fig.update_layout(
            xaxis=dict(type='log',range=[np.log10(f_low),np.log10(f_high)]),
            xaxis2=dict(type='log',range=[np.log10(f_low),np.log10(f_high)]),
            )

def adjust_freq_range_MIMO(fig,f_low,f_high,logaxis=False):
    if log_axis==False:
        fig.update_layout(
            xaxis=dict(range=[f_low,f_high]),
            xaxis2=dict(range=[f_low,f_high]),
            xaxis3=dict(range=[f_low,f_high]),
            xaxis4=dict(range=[f_low,f_high]),
            xaxis5=dict(range=[f_low,f_high]),
            xaxis6=dict(range=[f_low,f_high]),
            xaxis7=dict(range=[f_low,f_high]),
            xaxis8=dict(range=[f_low,f_high]),
            )
    else:
        fig.update_layout(
            xaxis=dict(type='log',range=[np.log10(f_low),np.log10(f_high)]),
            xaxis2=dict(type='log',range=[np.log10(f_low),np.log10(f_high)]),
            xaxis3=dict(type='log',range=[np.log10(f_low),np.log10(f_high)]),
            xaxis4=dict(type='log',range=[np.log10(f_low),np.log10(f_high)]),
            xaxis5=dict(type='log',range=[np.log10(f_low),np.log10(f_high)]),
            xaxis6=dict(type='log',range=[np.log10(f_low),np.log10(f_high)]),
            xaxis7=dict(type='log',range=[np.log10(f_low),np.log10(f_high)]),
            xaxis8=dict(type='log',range=[np.log10(f_low),np.log10(f_high)]),
            )
        
# =============================================================================
# Run function definition
# =============================================================================

def run(settings, DUT_plot=True, NET_plot=True, log_axis=True, dB_unit=True, unwrap_on=False):
    
    simPath = (settings["Working folder"] + 
                   "\\IMTB_data\\" + settings["Timestamp"] +
                   "_"+ settings["simname"] )
    
    if Single_plot:
        if int(settings["Nr scenarios"])>1:
            # plot multiple scenario data in one plot
            Multi_SC_plot(settings, simPath, DUT_plot=True, NET_plot=False, log_axis=True, dB_unit=dB_unit, unwrap_on=unwrap_on) 
        else:
            sc_name = ""
            # plot one scenatio data in one plot
            Single_SC_plot(settings, simPath, sc_name, dB_unit=dB_unit,NET_plot=True, unwrap_on=unwrap_on)   
    
    if Multi_plots and int(settings["Nr scenarios"])>1:
        # looping over number of scenarios
        for idx_sc in range(int(settings["Nr scenarios"])):
        
            if int(settings["Nr scenarios"])>1:
                sc_name = f"sc{idx_sc+1:02d}_"
                print("Plotting for scenario nr "+str(idx_sc+1))
            else:
                sc_name = ""
            # plot one scenario data in one plot
            Single_SC_plot(settings, simPath, sc_name, dB_unit=dB_unit,NET_plot=True, unwrap_on=unwrap_on)
            


# =============================================================================
# Main function for testing only
# =============================================================================

if __name__ == "__main__":
    # path for the simulation folder
    simPath = r'INSERT PATH HERE'
    settings = get_settings(simPath)

    run(settings, log_axis=log_axis, dB_unit=dB_unit, unwrap_on=unwrap_on)
