#!/usr/bin/env python3
"""
Immitance Measurement ToolBox (IMTB) main routine.

@author: AUF, YLI @ Energinet

Update history:
    21 Mar 2025 - v1.0 - first public version
    
"""

# =============================================================================
# IMTB version
# =============================================================================

settings = {"IMTB version":"1.0",}

# =============================================================================
# Handling of errors for packaged version (can be excluded for IDE use)
# =============================================================================

import sys

def myexcepthook(type, value, traceback, oldhook=sys.excepthook):
    oldhook(type, value, traceback)
    input("Press RETURN to exit. ")

sys.excepthook = myexcepthook  


# =============================================================================
# Main script
# =============================================================================

if __name__ == "__main__":
    
    # =============================================================================
    # IMTB Settings GUI
    # =============================================================================
    
    import IMTB_gui as gui
    settings = gui.run(settings)
    

    # =============================================================================
    # Run PSCAD simulations if needed
    # =============================================================================
    
    if settings["Start frequency-scan"] == True:
        
        import IMTB_pscad_sim as pscad_sim
        pscad_sim.run(settings)
        
        
    # =============================================================================
    # Immittance calculation    
    # =============================================================================

    if settings["Start immittance calculation"] == True:
        
        import IMTB_calculate as calculate
        calculate.run(settings)
                         
    # =============================================================================
    # Immittance plot
    # =============================================================================
    if settings["Plot impedances"] == True:
        
        import IMTB_plot as plot
        plot.run(settings)
        
    # =============================================================================
    # Deleting all raw files if set to delete
    # =============================================================================
    if settings["Clear RAW"]:
        
        import shutil
        
        rawfolder = (settings["Working folder"] + "\\IMTB_data\\" + 
                    settings["Timestamp"] + "_" + settings["simname"] + "\\raw")
        
        shutil.rmtree(rawfolder)
        print("RAW data deleted.")
    
    