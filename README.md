# EMAPEX_WAVES

Repository containing code for calculating surface wave 
spectra from EM-APEX 1Hz Votage measurements. This repository supports the work in (Stadler et al, 2025; 10.1175/JTECH-D-24-0023.1), follwing the work of D'Asaro, 2015, and Hsu, 2021. In addition to containing code for cacluting wave spectra, this repository contains notebooks for the analysis and figure generation for Stadler, 2025. Two folders within this directory contain code for different purposes. The first, for processing EM-APEX 1Hz measurements, the second
for modelling linear surface waves and the resulting predicted measurements by subsurface profiling float. A guide for applying this algorithm to an EM-APEX dataset is provided below, as well as an overview of directory structure and code in this repository.

<img width="3680" height="2085" alt="FlowChart_for_README" src="https://github.com/user-attachments/assets/a9cf5ce9-0dd6-4fb5-8b74-86b3d57f04f4" />


## WaveProcessing

The script Process_LCDRI_1Hz.ipynb in this folder provides an outline for calculating 1D wave spectra and Hs measurements from EM-APEX 1Hz files. This script calls functions spectral_processing.process_files() and spectral_processing.sig_wave_height() to make these calcuations.



### src
This directory contains two files:

1. em_apex_processing.py

    * Functions for transforming EM-APEX measurements in velocities and velocity residuals

2. spectral_processing.py

    * Functions for taking spectra of EM-APEX profiles

## WaveModelling


## Directory Structure
```bash
├── LICENSE
├── WaveProcessing/
│   ├── src
│   │   ├── em_apex_processing.py 
│   │   └── spectral_processing.py
│   ├── LCDRI_Maps.ipynb
│   │   ├── This file contains the code for figures 1 and 2
│   ├── Fits_and_Flowchart.ipynb
│   │   ├── This file contains the code for figures 3 and 4
│   ├── .ipynb
│   ├── .ipynb
│   └── .ipynb
├── WaveModelling/
│   ├── 
│   ├── 
│   └──
└── README.md
```
