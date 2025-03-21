# -*- coding: utf-8 -*-
"""
Subscript of IMTB to execute a user GUI for settings creation

@author: AUF, YLI @ Energinet

Update history:
    21 Mar 2025 - v1.0 - first public version

"""

# =============================================================================
# Global settings of this sub-script
# =============================================================================

SIMNAME = "NAME" # Simulation naming string
SOLUTION_TIMESTEP = 10 # Default timestep (us)
FREQ0 = 50 # Default fundamental frequency (Hz)
SNAPSHOT_TIME = 5.0 # Default snapshot time (sec)

# Injection settings
INJ_AMPLITUDE = 0.01 # Default amplitude of injection (pu)
INJ_FREQSTART = 100.0 # Default injection start frequency (Hz)
INJ_FREQSTOP = 2000.0 # Default injection stop frequency (Hz)
INJ_FREQSTEP = 25.0 # Default injection step frequency (Hz)

# Settings for injection times for fixed and variable
SETTLING_TIME = 0.2 # Default settling time (sec)
INJECTION_TIME = 0.3 # Default injection time (sec)
SETTLING_PERIODS = 3 # Default settling periods (int)
INJECTION_PERIODS = 5 # Default injection periods (int)

# Theme default settings
COLOR1 = "#008A8B" # Default primary/background color
COLOR2 = "#09505D" # Default secondary color
ICON_FILENAME = 'energinet.ico'


# =============================================================================
# Modules Imports
# =============================================================================

import sys, os
# for timestamp
import time
# for GUI
import PySimpleGUI as sg
# for saving file to csv
import csv
# for looking at the XML tree of PSCAD workspace
import xml.etree.ElementTree as ET
# for checking versions
import mhi.pscad

# =============================================================================
# Internal Functions
# =============================================================================

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def list_files_in_folder(folder_path):
    """ Lists all file in a defined folder """
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print("The specified path is not a valid folder.")
        return
    
    file_names = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            file_names.append(item)
    
    return file_names

def gettimestamp():
    """ function to get the timestamp of the simulation start for naming purposes """ 
    loctime = time.localtime()
    timestamp = "{year:04d}{month:02d}{day:02d}{hour:02d}{minute:02d}{seconds:02d}".format(
        year=loctime.tm_year, month=loctime.tm_mon, day=loctime.tm_mday,
        hour=loctime.tm_hour, minute=loctime.tm_min, seconds=loctime.tm_sec)
    return timestamp

def getfortran_ext(fortran):
    """ Getting the extension depending on fortran name string, needs improovement """
    #get the fortran extension
    fortran_ext = 'N/A'
    # Checking for version number
    if '12.' in fortran or '13.' in fortran or '14.' in fortran:
        fortran_ext = '.if12'
    elif '15.' in fortran or '16.' in fortran or '17.' in fortran:
        if '64-bit' in fortran:
            fortran_ext = '.if15'
        else:
            fortran_ext = '.if15_x86'
    elif '18.' in fortran or '19.' in fortran:
        if '64-bit' in fortran:
            fortran_ext = '.if18'
        else:
            fortran_ext = '.if18_x86'
    
    return fortran_ext
    
def findallsimsets(wspath):
    """ Finding all Simulation Sets in PSCAD using XML tree of the Workspace file """
    # allocating output variable
    simsets = []
    
    #getting the root of the workspace tree
    tree = ET.parse(wspath)
    root = tree.getroot()
    
    # looping through simulations in WS and saving SimSet names
    for simulation in root.findall("./simulations/"):
        if simulation.attrib["classid"]=="SimulationSet":
            simsets.append(simulation.attrib["name"])
            
    return simsets

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
        if value.isdigit():
            settings[key] = int(value)
        elif is_float(value):
            settings[key] = float(value)
        elif is_bool(value):
            settings[key] = transform_to_bool(value)
    return settings

def get_settings_from_gui(values):
    settings = {
                "Timestamp":gettimestamp(),
                "simname":values["-SIMNAME-"],
                "PSCAD version":values["-VERSION-"],
                "License type":values["-LICTYPE-"],
                "Fortran version":values["-FORTRAN-"],
                "Folder extension":values["-FORTRANEXT-"],
                "Max volley":int(values["-MAXVOLLEY-"]),
                "Solution timestep":float(values["-SOLTIME-"]),
                "Plot timestep":float(values["-PLOTTIME-"]),
                "Working folder":os.path.dirname(values["-WSPATH-"]),
                "Workspace name":os.path.basename(values["-WSPATH-"]),
                "SimSet name":values["-SIMSET-"],
                "Terminal type":values["-TERMTYPE-"],
                "Injection type":values["-INJTYPE-"],
                "Immitance type":values["-IMTYPE-"],
                "Injection components":values["-INJCOMP-"],
                "Injection amplitude":float(values["-INJAMPL-"]),
                "Freq range":values["-FREQRANGE-"],
                "Freq type":values["-FREQTYPE-"],
                "Freq start":float(values["-FREQSTART-"]),
                "Freq stop":float(values["-FREQSTOP-"]),
                "Freq step":float(values["-FREQSTEP-"]),
                "Snapshot function":bool(values["-SNAPSHOTFUNCTION-"]),
                "Start from snapshot":int(values["-STARTFROMSNAPSHOT-"]),
                "Snapshot name":os.path.basename(values["-SNPPATH-"]),
                "Snapshot time":float(values["-SNAPTIME-"]),
                "Sweep time type":values["-SWEEPTIMETYPE-"],
                "Settling time":float(values["-SETLTIME-"]),
                "Injection time":float(values["-INJTIME-"]),
                "Fundamental freq":float(values["-FUNDFREQ-"]),
                "Start frequency-scan":False,
                "Multiple scenarios":bool(values["-MULTSC-"]),
                "Scenarios filepath":values["-SCPATH-"],
                "Start immittance calculation":True,
                "Plot impedances":bool(values["-PLOTIMP-"]),
                "Plot harmonics":bool(values["-PLOTHARM-"]),
                "Calculate NET":bool(values["-CALCNET-"]),
                "Clear RAW":bool(values["-CLEARRAW-"]),
                }
    return settings

# =============================================================================
# IMTB Settings GUI
# =============================================================================

def run(settings):
    
    # Checking PSCAD installations on this machine and sorting too old versions out
    versions = mhi.pscad.versions()
    pscad_vers = []
    for value in versions:
        if value[1]==True and value[0][0]!="4":
            pscad_vers.append(value[0])

    # Checking Fortran versions on this machine and sorting too old versions out
    fortrans = mhi.pscad.fortran_versions()
    fortran_vers = []
    for fortran in fortrans:
        if ("GFortran" in fortran) == False:
            fortran_vers.append(fortran)

    inj_types_AC_SISO = [
                "pos",
                "neg",
                "zero",
                "a",
                "b",
                "c",
                # "d",
                # "q",
                # "alpha",
                # "beta",
                ]
    
    inj_types_AC_MIMO = [
                "posneg",
                "dq",
                ]
    
    inj_types_DC_SISO = [
                "ab",
                "cb",
                ]
    
    inj_types_DC_MIMO = [
                "ab-cb",
                ]

    # Here the settings (also for the impedance measurement) of the calculation are set
    print("Starting the settings GUI")

    # theme settings
    sg.theme_background_color(COLOR1)
    sg.theme_button_color(COLOR2)
    sg.theme_element_background_color(COLOR1)
    sg.theme_text_element_background_color(COLOR1)

    overhead_text = sg.Text("Immittance Measurement ToolBox v" + settings["IMTB version"],
                            font=("Arial", 11, "bold"))
    # overhead_logo = sg.Image(resource_path("logo.png"))

    sim_column = [
        [
         sg.Text("Simulation settings", font=("Arial", 11, "bold")) ,
         ],
        [
         sg.Text("Workspace (.pswx)"),
         sg.In(enable_events=True, key="-WSPATH-"),
         sg.FileBrowse(file_types=(("PSCAD Workspace","*.pswx"),)),
         ],
        [
         sg.Text("PSCAD Version"),
         sg.Combo(pscad_vers, key="-VERSION-", default_value=pscad_vers[-1])
         ],
        [
         sg.Text("PSCAD License type"),
         sg.Combo(["certificate","lock-based",], key="-LICTYPE-", default_value="certificate"),
         sg.Button("(i)", key="-INFOLICTYPE-", button_color=COLOR1, border_width=0),
         ],
        [
         sg.Text("Compiler version"),
         sg.Combo(fortran_vers, key="-FORTRAN-", default_value=fortran_vers[-1], enable_events=True),
         sg.Button("(i)", key="-INFOFORTRAN-", button_color=COLOR1, border_width=0),
         sg.Text("", key="-WARNINGSOLVER-"),
         ],
        [
         sg.Text("Compiler extension"),
         sg.In(default_text=getfortran_ext(fortran_vers[-1]), size=(10,1), key="-FORTRANEXT-"),
         sg.Button("(i)", key="-INFOFORTRANEXT-", button_color=COLOR1, border_width=0),
         ],
        [
         sg.Text("Max. parallel simulations"),
         sg.In(default_text=15, size=(10,1), key="-MAXVOLLEY-"),
         sg.Button("(i)", key="-INFOMAXVOLLEY-", button_color=COLOR1, border_width=0),
         ],
        [
         sg.Text("Simulation name"),
         sg.In(default_text=SIMNAME, key="-SIMNAME-"),
         ],
        [
         sg.Text("Simulation Set"),
         sg.Combo([], size=(20,1), key="-SIMSET-"),
         ],
        [
         sg.Text("Fundamental frequency (Hz)"),
         sg.In(default_text=FREQ0, size=(10,1), key="-FUNDFREQ-"),
         ],
        [
         sg.Text("Solution timestep (us)"),
         sg.In(default_text=SOLUTION_TIMESTEP, size=(10,1), key="-SOLTIME-"),
         sg.Text("", key="-WARNINGSOLTIME-"),
         ],
        [
         sg.Text("Plot timestep (us)"),
         sg.In(default_text=200, size=(10,1), disabled=True, key="-PLOTTIME-"),
         sg.Checkbox("Auto", default=True, enable_events=True, key="-AUTOPLOTTIME-"),
         sg.Button("(i)", key="-INFOPLOTTIME-", button_color=COLOR1, border_width=0),
         ],
        [
         sg.Text("Snapshot time (sec)"),
         sg.In(default_text=SNAPSHOT_TIME, size=(10,1), key="-SNAPTIME-"),
         sg.Checkbox("Snapshot function", default=True, enable_events=True, key="-SNAPSHOTFUNCTION-"),
         sg.Checkbox("Start from snapshot", default=False, enable_events=True, key="-STARTFROMSNAPSHOT-"),
         sg.Button("(i)", key="-INFOSNAPSHOT-", button_color=COLOR1, border_width=0),
         ],
        [
         sg.Text("Snapshot (.snp)"),
         sg.In(enable_events=True, disabled=True, key="-SNPPATH-"),
         sg.FileBrowse(file_types=(("PSCAD Snapshot","*.snp"),), disabled=True, key="-SNPBROWSE-"),
         ],
        [
         sg.Checkbox("Multiple scenarios", default=False, enable_events=True, key="-MULTSC-"),
         sg.Button("(i)", key="-INFOMULTSC-", button_color=COLOR1, border_width=0),
         ],
        [
         sg.Text("Scenarios list (.xlsx)"),
         sg.In(enable_events=False, disabled=True, key="-SCPATH-"),
         sg.FileBrowse(file_types=(("Excel scenarios file","*.xlsx"),), disabled=True, key="-SCBROWSE-"),
         ],
        ]

    fs_column = [
        [
         sg.Text("Frequency Scan settings", font=("Arial", 11, "bold")),
         ],
        [
         sg.Text("Terminal type"),
         sg.Combo(["AC","DC"], default_value="AC", disabled=False, enable_events=True, key="-TERMTYPE-"),
         ],
        [
         sg.Text("Injection type"),
         sg.Combo(["series","shunt"], default_value="series", disabled=False, key="-INJTYPE-"),
         sg.Button("(i)", key="-INFOINJTYPE-", button_color=COLOR1, border_width=0),
         ],
        [
         sg.Text("Immitance type"),
         sg.Combo(["SISO","MIMO"], default_value="SISO", enable_events=True, disabled=False, key="-IMTYPE-"),
         sg.Button("(i)", key="-INFOIMTYPE-", button_color=COLOR1, border_width=0),
         ],
        
        [
         sg.Text("Injection components"),
         sg.Combo(inj_types_AC_SISO, default_value=inj_types_AC_SISO[0], key="-INJCOMP-"),
         ],
        [
         sg.Text("Injection amplitude (pu)"),
         sg.In(default_text=INJ_AMPLITUDE, size=(10,1), key="-INJAMPL-"),
         sg.Button("(i)", key="-INFOINJAMPL-", button_color=COLOR1, border_width=0),
         ],
        [
         sg.Text("Frequency range"),
         sg.Combo([#"sub-synchronous",
                   #"near-synchronous",
                   #"super-synchronous",
                   #"linear+linear",
                   #"linear+log",
                   #"linear+truelog",
                   "complete(log)",
                   "complete(truelog)",
                   "custom",], default_value="complete(log)", enable_events=True, key="-FREQRANGE-"),
         sg.Button("(i)", key="-INFOFREQRANGE-", button_color=COLOR1, border_width=0),
        ],
       [
        sg.Button("See injection frequency array", key="-FREQARRAY-"),
        ],
        [
         sg.Text("Frequency sweep type"),
         sg.Combo(["linear","log","truelog"], default_value="linear", disabled=True, enable_events=True, key="-FREQTYPE-"),
         sg.Button("(i)", key="-INFOFREQTYPE-", button_color=COLOR1, border_width=0),
         ],
        [
         sg.Text("Start frequency (Hz)"),
         sg.In(default_text=INJ_FREQSTART, size=(10,1), disabled=True, key="-FREQSTART-"),
         ],
        [
         sg.Text("Stop frequency (Hz)"),
         sg.In(default_text=INJ_FREQSTOP, size=(10,1), disabled=False, enable_events=True, key="-FREQSTOP-"),
         ],
        [
         sg.Text("Frequency step (Hz)", key="-FREQSTEPTEXT-"),
         sg.In(default_text=INJ_FREQSTEP, size=(10,1), disabled=True, key="-FREQSTEP-"),
         ],
        [
          sg.Text("Sweep time type"),
          sg.Combo(["fixed","variable",], default_value="fixed", enable_events=True, disabled=True, key="-SWEEPTIMETYPE-"),
          # Note: "variable" sweep time type is not availible in this version yet
          ],
        [
         sg.Text("Settling time (sec)", key="-SETLTIMETEXT-"),
         sg.In(default_text=SETTLING_TIME, size=(10,1), key="-SETLTIME-"),
         sg.Button("(i)", key="-INFOSETLTIME-", button_color=COLOR1, border_width=0),
         ],
        [
         sg.Text("Injection time (sec)", key="-INJTIMETEXT-"),
         sg.In(default_text=INJECTION_TIME, size=(10,1), key="-INJTIME-"),
         sg.Button("(i)", key="-INFOINJTIME-", button_color=COLOR1, border_width=0),
         ],
        [
         sg.Checkbox("Plot impedances", default=True, key="-PLOTIMP-"),
         sg.Checkbox("Plot harmonics", default=False, enable_events=False, key="-PLOTHARM-"),
         ],
        [
         sg.Checkbox("Calculate NET", default=False, key="-CALCNET-"),
         sg.Checkbox("Clear RAW data", default=True, key="-CLEARRAW-"),
         ],
        ]

    execute = [
        sg.Button("START", key="-START-"),
        sg.Button("CLOSE", key="-CLOSE-"),
        sg.Button("HELP", key="-HELP-"),
        ]


    gui_layout = [
        [
         # overhead_logo,
         overhead_text,
         ],
        # [
        #  sg.HSeparator(),
        #  ],
        [
         sg.Column(sim_column),
         sg.VSeparator(),
         sg.Column(fs_column),
         ],
        [
         execute,
         ],
        ]

    gui_window = sg.Window("IMTB v" + settings["IMTB version"],
                           gui_layout,
                           icon=resource_path(ICON_FILENAME),
                           no_titlebar=False)

    # Settings GUI event loop

    while True:
        event, values = gui_window.read()
        
        # Closing the tool
        if event == "-CLOSE-" or event == sg.WINDOW_CLOSED:
            settings["Start frequency-scan"] = False
            settings["Start immittance calculation"] = False
            settings["Plot impedances"] = False
            settings["Clear RAW"] = False
            print("Closing programm.")
            break
        
        # Starting simulation
        if event == "-START-":
            
            # Get the settings from gui input
            settings = get_settings_from_gui(values)
            
            if "complete" in values["-FREQRANGE-"] and float(values["-FREQSTOP-"])<150.0:
                sg.popup("For complete freqeuncy range, stop frequency shall be above 150!",
                         title="Settings error",
                         icon=resource_path(ICON_FILENAME))
            else:
                import IMTB_pscad_sim as pscad_sim
                f_inj = list(pscad_sim.setupfreqs(settings))
                test_injtime = [(float(values["-INJTIME-"])*f)%1!=0 for f in f_inj]
            
                # Check if everything is correctly setup
                # No workspace defined
                if values["-WSPATH-"]=="":
                    sg.popup("Workspace not defined!                               \x00",
                             title="Settings error",
                             icon=resource_path(ICON_FILENAME))
                elif values["-FORTRANEXT-"] == "N/A":
                    sg.popup("Check fortran compiler extension!                    \x00",
                             title="Settings error",
                             icon=resource_path(ICON_FILENAME))
                # Plotstep multiple of solstep
                elif float(values["-PLOTTIME-"])%float(values["-SOLTIME-"]) != 0:
                    sg.popup("Plot timestep should be a multiple of solution timestep!",
                             title="Settings error",
                             icon=resource_path(ICON_FILENAME))
                # Enough injection time to calculate all frequencies
                elif any(test_injtime):
                    wrong_inj_setting = []
                    idx = 0
                    for f in f_inj:
                        if test_injtime[idx]:
                            wrong_inj_setting.append(f)
                        idx+=1
                    sg.popup("Not enough or wrong injection time to calculate all frequencies!"+
                             os.linesep+
                              f"Wrong frequencies: {[f'{wrong_f}' for wrong_f in wrong_inj_setting]}",
                             title="Settings error",
                             icon=resource_path(ICON_FILENAME))
                # no settings error found
                else:
                    settings["Start frequency-scan"] = True
                    # Saving settings to csv and starting
                    destination = settings["Working folder"] + "\\IMTB_data\\" + settings["Timestamp"] +"_"+ settings["simname"] + "\\"
                    os.makedirs(destination)
                    with open(destination + "settings.csv", "w", newline="") as fp:
                        # creating header object
                        writer = csv.DictWriter(fp, fieldnames=settings.keys())
                        
                        # writing the data from settings
                        writer.writeheader()
                        writer.writerow(settings)
                    
                    # go out of the GUI loop
                    break
        
        # Opening PDF help file
        if event == "-HELP-":
            # os.startfile(resource_path("help.pdf"))
            try:
                os.system("start \"\" https://github.com/Energinet-SimTools/IMTB")
            except:
                sg.popup("See documentation under:\nhttps://github.com/Energinet-SimTools/IMTB",
                         title="Help",
                         icon=resource_path(ICON_FILENAME))
        
        # Give the freq array of the current settings
        if event == "-FREQARRAY-":
            settings = get_settings_from_gui(values)
            
            import IMTB_pscad_sim as pscad_sim
            try:
                f_inj = list(pscad_sim.setupfreqs(settings))
                f_inj.remove(0.0)
                
                if len(f_inj)<=300:
                    sg.popup("Array of injection frequencies for current settings:"+
                             os.linesep+
                              f"{[f for f in f_inj]}",
                             title="Frequency array popup",
                             icon=resource_path(ICON_FILENAME))
                else:
                    sg.popup("Frequency array is too long to display correctly ("+str(len(f_inj))+" elements). First 200 elements are:"+
                             os.linesep+
                             f"{[f for f in f_inj[0:199]]}",
                             title="Frequency array popup",
                             icon=resource_path(ICON_FILENAME))
            except:
                sg.popup("Wrong frequency settings. For complete frequency range the stop frequency should be above 150 Hz!",
                          title="Frequency array popup",
                          icon=resource_path(ICON_FILENAME))
            
                
        
        # Changing the FS type (freq step)
        if event == "-FREQTYPE-":
            if values["-FREQTYPE-"] == "log":
                gui_window["-FREQSTEP-"].update(disabled=False)
                gui_window["-FREQSTEP-"].update("10")
                gui_window["-FREQSTEPTEXT-"].update("Points between log")
            elif values["-FREQTYPE-"] == "truelog":
                gui_window["-FREQSTEP-"].update(disabled=False)
                gui_window["-FREQSTEP-"].update("100")
                gui_window["-FREQSTEPTEXT-"].update("Number of points")
            else:
                gui_window["-FREQSTEP-"].update(disabled=False)
                gui_window["-FREQSTEP-"].update("10")
                gui_window["-FREQSTEPTEXT-"].update("Frequency step (Hz)")
                
        # Finding Simulation sets in a choosen Workspace
        if event == "-WSPATH-":
            # Getting the Simset names from XML tree
            simset_list = findallsimsets(values["-WSPATH-"])
            # updating the dropdown list, clean and add new
            gui_window["-SIMSET-"].update(values=[])
            gui_window["-SIMSET-"].update(values=simset_list)
            
            if not simset_list:
                sg.popup("No simulation sets are defined in the PSCAD Workspace!",
                         title="Settings error",
                         icon=resource_path(ICON_FILENAME))
            else:
                gui_window["-SIMSET-"].update(simset_list[0])
                    
            # Try to load previous settings and put them into GUI
            try:
                workfolder = os.path.dirname(values["-WSPATH-"])
                # getting the newewst one
                for dirpath in os.listdir(workfolder + "/IMTB_data/"):
                    prevsim_name = dirpath
                
                # reading the setting csv from the file
                filename = workfolder + "/IMTB_data/" + prevsim_name# + "/settings.csv"
                # with open(filename, "r") as infile:
                    
                #     # creating a reader object
                #     reader = csv.DictReader(infile)
                    
                #     # iterating for rows
                #     for row in reader:
                #         prevsim_settings = row
                        
                prevsim_settings = get_settings(filename)
                
                # updating the gui window with previous sim settings values
                gui_window["-SIMNAME-"].update(prevsim_settings["simname"])
                gui_window["-VERSION-"].update(prevsim_settings["PSCAD version"])
                gui_window["-LICTYPE-"].update(prevsim_settings["License type"])
                gui_window["-FORTRAN-"].update(prevsim_settings["Fortran version"])
                gui_window["-FORTRANEXT-"].update(prevsim_settings["Folder extension"])
                gui_window["-MAXVOLLEY-"].update(int(prevsim_settings["Max volley"]))
                gui_window["-SOLTIME-"].update(prevsim_settings["Solution timestep"])
                gui_window["-PLOTTIME-"].update(prevsim_settings["Plot timestep"])
                gui_window["-SIMSET-"].update(prevsim_settings["SimSet name"])
                gui_window["-TERMTYPE-"].update(prevsim_settings["Terminal type"])
                gui_window["-INJTYPE-"].update(prevsim_settings["Injection type"])
                gui_window["-IMTYPE-"].update(prevsim_settings["Immitance type"])
                gui_window["-INJCOMP-"].update(prevsim_settings["Injection components"])
                gui_window["-INJAMPL-"].update(prevsim_settings["Injection amplitude"])
                gui_window["-FREQRANGE-"].update(prevsim_settings["Freq range"])
                gui_window["-FREQTYPE-"].update(prevsim_settings["Freq type"])
                gui_window["-FREQSTART-"].update(prevsim_settings["Freq start"])
                gui_window["-FREQSTOP-"].update(prevsim_settings["Freq stop"])    
                gui_window["-FREQSTEP-"].update(prevsim_settings["Freq step"])    
                gui_window["-SNAPTIME-"].update(round(float(prevsim_settings["Snapshot time"]),9))
                gui_window["-SWEEPTIMETYPE-"].update(prevsim_settings["Sweep time type"])
                gui_window["-SETLTIME-"].update(prevsim_settings["Settling time"])
                gui_window["-INJTIME-"].update(prevsim_settings["Injection time"])
                gui_window["-FUNDFREQ-"].update(prevsim_settings["Fundamental freq"])
                gui_window["-SNPPATH-"].update(prevsim_settings["Snapshot name"])
                gui_window["-SCPATH-"].update(prevsim_settings["Scenarios filepath"])
                gui_window["-PLOTIMP-"].update(prevsim_settings["Plot impedances"])
                gui_window["-PLOTHARM-"].update(prevsim_settings["Plot harmonics"])
                gui_window["-CALCNET-"].update(prevsim_settings["Calculate NET"])
                gui_window["-CLEARRAW-"].update(prevsim_settings["Clear RAW"])
                gui_window["-SNAPSHOTFUNCTION-"].update(prevsim_settings["Snapshot function"])
                
                #Adjusting the window to previous settings
                if prevsim_settings["Terminal type"] == "AC":
                    if prevsim_settings["Immitance type"] == "SISO":
                        gui_window["-INJCOMP-"].update(values=inj_types_AC_SISO, value=inj_types_AC_SISO[0])
                    else:
                        gui_window["-INJCOMP-"].update(values=inj_types_AC_MIMO, value=inj_types_AC_MIMO[0])
                else:
                    if prevsim_settings["Immitance type"] == "SISO":
                        gui_window["-INJCOMP-"].update(values=inj_types_DC_SISO, value=inj_types_DC_SISO[0])
                    else:
                        gui_window["-INJCOMP-"].update(values=inj_types_DC_MIMO, value=inj_types_DC_MIMO[0])
                
                
                if prevsim_settings["Freq type"] == "log":
                    gui_window["-FREQSTEP-"].update(disabled=False)
                    gui_window["-FREQSTEPTEXT-"].update("Points between log")
                elif prevsim_settings["Freq type"] == "truelog":
                    gui_window["-FREQSTEP-"].update(disabled=False)
                    gui_window["-FREQSTEPTEXT-"].update("Number of points")
                else:
                    gui_window["-FREQSTEP-"].update(disabled=False)
                    gui_window["-FREQSTEPTEXT-"].update("Frequency step (Hz)")
                
                if prevsim_settings["Sweep time type"]=="fixed":
                    gui_window["-SETLTIMETEXT-"].update("Settling time (sec)")
                    gui_window["-INJTIMETEXT-"].update("Injection time (sec)")
                else:
                    gui_window["-SETLTIMETEXT-"].update("Settling periods")
                    gui_window["-INJTIMETEXT-"].update("Injection periods")
                
                if prevsim_settings["Freq range"]=="custom":
                    gui_window["-FREQTYPE-"].update(disabled=False)
                    gui_window["-FREQSTART-"].update(disabled=False)
                    # gui_window["-FREQSTOP-"].update(disabled=False)
                    gui_window["-FREQSTEP-"].update(disabled=False)
                else:
                    gui_window["-FREQTYPE-"].update(disabled=True)
                    gui_window["-FREQSTART-"].update(disabled=True)
                    # gui_window["-FREQSTOP-"].update(disabled=True)
                    gui_window["-FREQSTEP-"].update(disabled=True)
                
                # Cheching if snapshot function is used or not
                if prevsim_settings["Snapshort function"]=="True":
                    
                    gui_window["-SNAPSHOTFUNCTION-"].update(True)
                    gui_window["-STARTFROMSNAPSHOT-"].update(disabled=False)
                    gui_window["-SNPPATH-"].update(disabled=False)
                    gui_window["-SNPBROWSE-"].update(disabled=False)
                    gui_window["-SNAPTIME-"].update(disabled=False)
                    gui_window["-SOLTIME-"].update(disabled=False)
                else:
                    gui_window["-SNAPSHOTFUNCTION-"].update(False)
                    gui_window["-STARTFROMSNAPSHOT-"].update(disabled=True)
                    gui_window["-SNPPATH-"].update(disabled=True)
                    gui_window["-SNPBROWSE-"].update(disabled=True)
                    gui_window["-SNAPTIME-"].update(disabled=True)
                    gui_window["-SOLTIME-"].update(disabled=True)
                
                # Cheching if start from snapshot was active
                if bool(int(prevsim_settings["Start from snapshot"])):
                    gui_window["-STARTFROMSNAPSHOT-"].update(True)
                else:
                    gui_window["-STARTFROMSNAPSHOT-"].update(False)
                    
                
                
                # Adjust according to start or not from snapshot
                if int(prevsim_settings["Start from snapshot"])==True:
                    gui_window["-SNPPATH-"].update(disabled=False)
                    gui_window["-SNPBROWSE-"].update(disabled=False)
                    gui_window["-SNAPTIME-"].update(disabled=True)
                    gui_window["-SOLTIME-"].update(disabled=True)
                    gui_window["-WARNINGSOLTIME-"].update("Must be identical to snapshot!")
                    gui_window["-WARNINGSOLVER-"].update("Must be identical to snapshot!")
                else:
                    gui_window["-SNPPATH-"].update(disabled=True)
                    gui_window["-SNPBROWSE-"].update(disabled=True)
                    gui_window["-SNAPTIME-"].update(disabled=False)
                    gui_window["-SOLTIME-"].update(disabled=False)
                    gui_window["-WARNINGSOLTIME-"].update(" ")
                    gui_window["-WARNINGSOLVER-"].update(" ")
                
                if prevsim_settings["Multiple scenarios"]=="True":
                    gui_window["-MULTSC-"].update(True)
                    gui_window["-SCBROWSE-"].update(disabled=False)
                    gui_window["-SCPATH-"].update(disabled=False)
                    gui_window["-STARTFROMSNAPSHOT-"].update(value=0, disabled=True)
                    gui_window["-SNPPATH-"].update(value="", disabled=True)
                    gui_window["-SNPBROWSE-"].update(disabled=True)
                    gui_window["-SNAPTIME-"].update(disabled=False)
                    gui_window["-SOLTIME-"].update(disabled=False)
                    gui_window["-WARNINGSOLTIME-"].update(" ")
                    gui_window["-WARNINGSOLVER-"].update(" ")
                else:
                    gui_window["-MULTSC-"].update(False)
                    gui_window["-SCBROWSE-"].update(disabled=True)
                    gui_window["-SCPATH-"].update(disabled=True)
                    gui_window["-STARTFROMSNAPSHOT-"].update(disabled=False)
                    gui_window["-SNPPATH-"].update(disabled=False)
                    
                    
                # checking if MIMO os SISO was set and changing list accordingly
                if prevsim_settings["Terminal type"] == "AC":
                    if prevsim_settings["Immitance type"] == "SISO":
                        gui_window["-INJCOMP-"].update(values=inj_types_AC_SISO,
                                                       value=prevsim_settings["Injection components"])
                    else:
                        gui_window["-INJCOMP-"].update(values=inj_types_AC_MIMO,
                                                       value=prevsim_settings["Injection components"])
                else:
                    if prevsim_settings["Immitance type"] == "SISO":
                        gui_window["-INJCOMP-"].update(values=inj_types_DC_SISO,
                                                       value=prevsim_settings["Injection components"])
                    else:
                        gui_window["-INJCOMP-"].update(values=inj_types_DC_MIMO,
                                                       value=prevsim_settings["Injection components"])
            except:
                # sg.popup("Not all settings could be loaded correctly. Old settings version is not fully compatible. Please check if settings set correct!",
                #          title="Previous settings error",
                #          icon=resource_path(ICON_FILENAME))
                pass
                
        # If fortran version is changed, also change the extension
        if event == "-FORTRAN-":
            gui_window["-FORTRANEXT-"].update(getfortran_ext(values["-FORTRAN-"]))
        
        # Changing of sweep time type, so changing text and values
        if event == "-SWEEPTIMETYPE-":
            if values["-SWEEPTIMETYPE-"]=="fixed":
                gui_window["-SETLTIMETEXT-"].update("Settling time (sec)")
                gui_window["-SETLTIME-"].update(SETTLING_TIME)
                gui_window["-INJTIMETEXT-"].update("Injection time (sec)")
                gui_window["-INJTIME-"].update(INJECTION_TIME)
            else:
                gui_window["-SETLTIMETEXT-"].update("Settling periods")
                gui_window["-SETLTIME-"].update(SETTLING_PERIODS)
                gui_window["-INJTIMETEXT-"].update("Injection periods")
                gui_window["-INJTIME-"].update(INJECTION_PERIODS)
        
        # Changing Plot step if max frequency is changed
        if event == "-FREQSTOP-" or event == "-AUTOPLOTTIME-":
            if values["-AUTOPLOTTIME-"]:
                gui_window["-PLOTTIME-"].update(disabled=True)
                
                if values["-FREQSTOP-"] != "":
                    freqmax = float(values["-FREQSTOP-"])
                else:
                    freqmax = 0
                # Look up table for plotstep choices
                if freqmax < 500:
                    gui_window["-PLOTTIME-"].update(1000.0)
                elif freqmax < 625:
                    gui_window["-PLOTTIME-"].update(800.0)
                elif freqmax < 1000:
                    gui_window["-PLOTTIME-"].update(500.0)
                elif freqmax < 1250:
                    gui_window["-PLOTTIME-"].update(400.0)
                elif freqmax < 2000:
                    gui_window["-PLOTTIME-"].update(250.0)
                elif freqmax < 2500:
                    gui_window["-PLOTTIME-"].update(200.0)
                elif freqmax < 3125:
                    gui_window["-PLOTTIME-"].update(160.0)
                elif freqmax < 4000:
                    gui_window["-PLOTTIME-"].update(125.0)
                elif freqmax < 5000:
                    gui_window["-PLOTTIME-"].update(100.0)
                elif freqmax < 6250:
                    gui_window["-PLOTTIME-"].update(80.0)
                elif freqmax < 10000:
                    gui_window["-PLOTTIME-"].update(50.0)
                else:
                    gui_window["-FREQSTOP-"].update(9000.0)
                    gui_window["-PLOTTIME-"].update(50.0)
                
            else:
                gui_window["-PLOTTIME-"].update(disabled=False)
                
        # Disabling the frequency settings if not custom
        if event == "-FREQRANGE-":
            if  values["-FREQRANGE-"]=="custom":
                gui_window["-FREQTYPE-"].update(disabled=False)
                gui_window["-FREQSTART-"].update(disabled=False)
                # gui_window["-FREQSTOP-"].update(disabled=False)
                gui_window["-FREQSTEP-"].update(disabled=False)
            else:
                gui_window["-FREQTYPE-"].update(disabled=True)
                gui_window["-FREQSTART-"].update(disabled=True)
                # gui_window["-FREQSTOP-"].update(disabled=True)
                gui_window["-FREQSTEP-"].update(disabled=True)
                
        # Change GUI settings if "Snapshot function" is choosen
        if event == "-SNAPSHOTFUNCTION-":
            if values["-SNAPSHOTFUNCTION-"] == True:
                if values["-MULTSC-"] == False:
                    gui_window["-STARTFROMSNAPSHOT-"].update(disabled=False)
                    if values["-STARTFROMSNAPSHOT-"] == True:
                        gui_window["-SNPPATH-"].update(disabled=False)
                        gui_window["-SNPBROWSE-"].update(disabled=False)
                        gui_window["-SNAPTIME-"].update(disabled=True)
                        gui_window["-SOLTIME-"].update(disabled=False)
                        gui_window["-WARNINGSOLTIME-"].update("Must be identical to snapshot!")
                        gui_window["-WARNINGSOLVER-"].update("Must be identical to snapshot!")
                    else:
                        gui_window["-SNPPATH-"].update(disabled=True)
                        gui_window["-SNPBROWSE-"].update(disabled=True)
                        gui_window["-SNAPTIME-"].update(disabled=False)
                        gui_window["-SOLTIME-"].update(disabled=False)
                        gui_window["-WARNINGSOLTIME-"].update(" ")
                        gui_window["-WARNINGSOLVER-"].update(" ")
            
            else:
                gui_window["-SNPPATH-"].update(disabled=True)
                gui_window["-SNPBROWSE-"].update(disabled=True)
                gui_window["-SNAPTIME-"].update(disabled=True)
                gui_window["-SOLTIME-"].update(disabled=True)
                gui_window["-STARTFROMSNAPSHOT-"].update(disabled=True)
                gui_window["-WARNINGSOLTIME-"].update(" ")
                gui_window["-WARNINGSOLVER-"].update(" ")
        
        # Change GUI settings if "Start from Snapshot" is choosen
        if event == "-STARTFROMSNAPSHOT-":
            if values["-STARTFROMSNAPSHOT-"] == True:
                gui_window["-SNPPATH-"].update(disabled=False)
                gui_window["-SNPBROWSE-"].update(disabled=False)
                gui_window["-SNAPTIME-"].update(disabled=True)
                gui_window["-SOLTIME-"].update(disabled=False)
                gui_window["-WARNINGSOLTIME-"].update("Must be identical to snapshot!")
                gui_window["-WARNINGSOLVER-"].update("Must be identical to snapshot!")
            else:
                gui_window["-SNPPATH-"].update(disabled=True)
                gui_window["-SNPBROWSE-"].update(disabled=True)
                gui_window["-SNAPTIME-"].update(disabled=False)
                gui_window["-SOLTIME-"].update(disabled=False)
                gui_window["-WARNINGSOLTIME-"].update(" ")
                gui_window["-WARNINGSOLVER-"].update(" ")
        
        # Change data according to Snapshot info
        if event == "-SNPPATH-":
            snppath = values["-SNPPATH-"]
            # read data from snp file
            try:
                with open(snppath, 'r') as file:
                    data = file.readlines()[1]
                
                # Reconfigure line str to numbers
                nums = [float(x) for x in data.split(" ") if x]
                # Update GUI
                gui_window["-SNAPTIME-"].update(nums[1])
                gui_window["-SOLTIME-"].update(nums[2]*1e6)
            except:
                pass
        
        # Multiple scenarios checkbox action
        if event == "-MULTSC-":
            if values["-MULTSC-"]:
                gui_window["-SCBROWSE-"].update(disabled=False)
                gui_window["-SCPATH-"].update(disabled=False)
                gui_window["-STARTFROMSNAPSHOT-"].update(value=0, disabled=True)
                gui_window["-SNPPATH-"].update(value="", disabled=True)
                gui_window["-SNPBROWSE-"].update(disabled=True)
                gui_window["-SNAPTIME-"].update(disabled=False)
                gui_window["-SOLTIME-"].update(disabled=False)
                gui_window["-WARNINGSOLTIME-"].update(" ")
                gui_window["-WARNINGSOLVER-"].update(" ")
            else:
                gui_window["-SCBROWSE-"].update(disabled=True)
                gui_window["-SCPATH-"].update(disabled=True)
                gui_window["-STARTFROMSNAPSHOT-"].update(disabled=False)
                gui_window["-SNPPATH-"].update(disabled=False)
            
        if event == "-IMTYPE-" or event == "-TERMTYPE-":
            if values["-TERMTYPE-"] == "AC":
                if values["-IMTYPE-"] == "SISO":
                    gui_window["-INJCOMP-"].update(values=inj_types_AC_SISO, value=inj_types_AC_SISO[0])
                else:
                    gui_window["-INJCOMP-"].update(values=inj_types_AC_MIMO, value=inj_types_AC_MIMO[0])
            else:
                if values["-IMTYPE-"] == "SISO":
                    gui_window["-INJCOMP-"].update(values=inj_types_DC_SISO, value=inj_types_DC_SISO[0])
                else:
                    gui_window["-INJCOMP-"].update(values=inj_types_DC_MIMO, value=inj_types_DC_MIMO[0])
        
        # =============================================================================
        # Info buttons popup windows        
        # =============================================================================
        if event == "-INFOLICTYPE-":
            sg.popup("Lock-Based License: This license type uses a combination of hardware lock and license database file to authenticate.\n-Single-User License: A single license is installed and used locally on a workstation.\n-Multi-User Licenses: One or more seats (i.e. instances of the PSCAD application) may be used locally, in addition to being shared with other workstations on the Local Area Network (LAN).\n\nCertificate License: A certificate is an electronic license that is acquired via a cloud-based license server. Certificate licenses are authorized through the customer’s MyCentre account (MyCentre Licensing).",
                     title="INFO: PSCAD License type",
                     icon=resource_path(ICON_FILENAME))
        
        if event == "-INFOFORTRAN-":
            sg.popup("Fortran versions in the list are fetched from the product file. If not same as in PSCAD settings, run the Fortran Medic -> Actions -> Generate installed products list.",
                     title="INFO: Compiler version",
                     icon=resource_path(ICON_FILENAME))
            
        if event == "-INFOFORTRANEXT-":
            sg.popup("Extension for the temporary folder created by PSCAD. Not fully automatized, so check before running!",
                     title="INFO: Compiler extension",
                     icon=resource_path(ICON_FILENAME))
                
        if event == "-INFOMAXVOLLEY-":
            sg.popup("Maximum number of simulations depends on availible PSCAD licenses and number of CPU cores on the machine. You can limit this number to optimize core usage. \nRemember that parallel computing with PSCAD requires 1 core to distribute the tasks additionally to the parallel simulations, so the number to put shall be the maximum available core minus 1.",
                     title="INFO: Max parallel simulations",
                     icon=resource_path(ICON_FILENAME))
            
        if event == "-INFOPLOTTIME-":
            sg.popup("Plot timestep needs to be sufficient to perform Discrete Fourier Transformation of all scanned frequencies of interest. It corresponds to a Nyquist frequency, and the stop frequency shall be lower than the Nyquist frequency. \n\nWhen Auto function is enabled, the plot time step will be adaptive according to the stop frequency.",
                     title="INFO: Plot timestep",
                     icon=resource_path(ICON_FILENAME))
            
        if event == "-INFOSNAPSHOT-":
            sg.popup("Snapshot function can be enabled to simulate the start-up of the model only once. Otherwise start-up time needs to be included in settling time.\nIf snapshot was already created earlier, you can reuse the snapshot file for new scan.\nSnapshot function will not work with PSCAD v5.0.1 and lower!",
                     title="INFO: Snapshot function",
                     icon=resource_path(ICON_FILENAME))
                   
        if event == "-INFOMULTSC-":
            sg.popup("Multiple scenarios can be set and the scan will loop through those. Parameters will be reset in PSCAD model.\nCheck Help for more information. Use the scenario template EXCEL file to set parameters.",
                     title="INFO: Multiple scenarios",
                     icon=resource_path(ICON_FILENAME))
            
        if event == "-INFOINJTYPE-":
            sg.popup("Series: Voltage perturbation injection in series between DUT and NET at PoC.\n \nShunt: Current perturbation injection in parallel to DUT and NET at PoC.",
                     title="INFO: Injection type",
                     icon=resource_path(ICON_FILENAME))
        
        if event == "-INFOIMTYPE-":
            sg.popup("SISO: Only one element of the immitance (impedance or admittance) will be scanned and calculated. The scan result can include interdependencies and couplings between DUT und NET.\n \nMIMO: Scans by using multiple perturbation injections and calculates full immitance or transfer matrix of DUT (and NET).",
                     title="INFO: Immitance type",
                     icon=resource_path(ICON_FILENAME))
            
        if event == "-INFOINJAMPL-":
            sg.popup("Amplitude that will be injected by the toolbox. May vary from model to model and also from injection type.The load flow condition can also have an impact. Current injection often requires higher amplitudes, but also depends on the NET impedance size.\n\nRecommendation for series inj. type:\nLow voltage: 0.02-0.05\nMid voltage: 0.01-0.02\nHigh voltage: 0.01 and lower",
                     title="INFO: Injection amplitude",
                     icon=resource_path(ICON_FILENAME))
            
        if event == "-INFOFREQRANGE-":
            sg.popup("Complete: Combine different frequency sweep types and scan both lower and higher frequency range. User can define the stop frequency. Can apply log and truelog settings.\n\nCustom: As defined by user below using single frequency sweep type. \n\nUser can click the butten See injection frequency array to check the frequency array.",
                     title="INFO: Frequency range",
                     icon=resource_path(ICON_FILENAME))
            
        if event == "-INFOFREQTYPE-":
            sg.popup("Linear: apply linear frequency sweep based on the start frequency, the stop frequency and the frequency step. \n\nLog: apply log frequency sweep based on the start frequency, the stop frequency and the points between log. \n\nTruelog: apply truelog frequency sweep based on the start frequency, the stop frequency and the total number of points.\n\nIf unclear, please use the butten See injection frequency array to check the frequency array.",
                     title="INFO: Frequency sweep type",
                     icon=resource_path(ICON_FILENAME))
            
        if event == "-INFOSETLTIME-":
            sg.popup("Settling time needs to be sufficient to let the perturbation settle to the steady state. All perturbation injection applies the same settling time.\n\nThe settling time can vary from model to model. A practical way to find the proper settling time can be through the snapshot time, or use smaller values for trial runs and increase until the frequency scan results become constant.",
                     title="INFO: Settling time",
                     icon=resource_path(ICON_FILENAME))
        
        if event == "-INFOINJTIME-":
            sg.popup("Injection time is the time window after the settling time, which is taken for frequency-domain analysis. \n\nThe injection time has to be integral periods of existing frequency components to ensure the correct Discrete Fourier Transformation.The existing frequency components include system fundamental frequency (f1), perturbed frequency (fp), and the coupled response of the perturbed frequency if the perturbation is in sequence domain (f1-2fp) ",
                     title="INFO: Injection time",
                     icon=resource_path(ICON_FILENAME))
            
        #here come other event handlers

    # Closing settings GUI    
    gui_window.close()
       
    return settings
    
# =============================================================================
# Main function for testing only
# =============================================================================

if __name__ == "__main__":
    test_settings = {"IMTB version":"XX"}
    test_settings = run(test_settings)