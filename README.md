# Immitance Measurement ToolBox (IMTB)
The Immitance Measurement ToolBox (IMTB) is an electrical system dynamic analysis tool developed by Energinet, the Danish transmission system operator. It is an automated toolbox that can be added in the electrical point of connection (PoC) of a circuit and measures the immittance (impedance or admittance) of the device under test (DUT) and/or the network (NET), as described in figure below. The toolbox is implemented in the electromagnetic transient (EMT) simulation platform using PSCAD and automated via Python scripts. It allows users to input parameters via graphic user interface (GUI), and all the immittance scan results are archived in data and plot formats. 

![Conceptual Description of Immitance Measurement ToolBox (IMTB)](https://github.com/user-attachments/assets/19bd1c11-2e4d-4250-b2d5-7b71b2fbcf1d)

## Features
The IMTB v1.0 has the following key features:
- Single unit test, which means that only one toolbox can be added in simulation tests. But it can scan immittances for both sides, i.e. the device under test side and the network side.
-	AC and DC scan, which means that the toolbox supports both immittance measurement at AC and DC PoCs in the electrical grids.
-	SISO and MIMO scan, which means that the toolbox can support single-input single-out (SISO) immittance scan and multi-input multi-out (MIMO) immittance matrix/transfer function scan.
-	Multiple core simulation, which allows users to apply multiple cores for parallel simulation and improved simulation speed. 
-	Multiple scenario simulation, which allows users to automatize scans of different operating points.
-	Customed frequency resolution combinations, which allows users to apply customed frequency resolution combinations in different frequency ranges.  

The expected application of the IMTB is for model verification and small-signal stability analysis based on EMT models. The tool is implemented with open Python scripts, which can ease the further development and extended application with other tools.

## User Guide
Find the complete user guide and computation description in the [IMTB Wiki Home page](https://github.com/Energinet-SimTools/IMTB/wiki).

IMTB packaged version and examples are available under [IMTB release](https://github.com/Energinet-SimTools/IMTB/releases).

## Considerations and terms of use 
We welcome use of the IMTB. Considerations and terms of use are described under [License](https://github.com/Energinet-SimTools/IMTB?tab=License-1-ov-file).

## Contributions
We welcome contributions! To contribute, please file an issue via the IMTB [issues tab](https://github.com/Energinet-SimTools/IMTB/issues). You can report bugs, request features, or suggest improvements. Before submitting, please check for any known issues.

## Contact
For inquiries, please contact the Energinet simulation model team: simuleringsmodeller@energinet.dk
