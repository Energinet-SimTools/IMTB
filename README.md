# IMTB

## Description
The Immitance Measurement ToolBox (IMTB) is an automated toolbox that can be added in the electrical point of connection (PoC) of a circuit and measures the immittance (impedance or admittance) of the device under test (DUT) and/or the network (NET), as described in figure below. The toolbox is implemented in the electromagnetic transient (EMT) simulation platform using PSCAD, and automated via python scripts. It allows users to input parameters via graphic user interface (GUI), and all the immittance scan results are archived in data and plot formats. 

![Conceptual Description of Immitance Measurement ToolBox (IMTB)](https://github.com/Energinet-SimTools/IMTB/blob/main/concept_overview.png)

## Features
The IMTB v1.0 has the following key features:
- Single unit test, which means that only one toolbox can be added in simulation tests. But it can scan immittances for both sides, i.e. the device under test side and the network side.
-	AC and DC scan, which means that the toolbox supports both immittance measure-ment at AC and DC PoCs.
-	SISO and MIMO scan, which means that the toolbox can support single-input single-out (SISO) immittance scan and multi-input multi-out (MIMO) immittance ma-trix/transfer function scan.
-	Multiple core simulation, which allows users to apply multiple cores for parallel simulation and improved simulation speed. 
-	Multiple scenario simulation, which allows users to automatize scans of different operating points of the DUT.
-	Customed frequency resolution combinations, which allows users to apply customed frequency resolution combinations in different frequency ranges.  
The expected application of the IMTB is for model verification and small-signal stability analysis based on EMT models. The tool is implemented with open Python scripts, which can ease the further development and extended application with other tools.

## Limitations

## User Guide
Find the complete user guide and computation description [here](https://github.com/Energinet-SimTools/MTB/wiki).

## Contact
