# EMAPEX_WAVES

Repository containing code for calculating surface wave 
spectra from EM-APEX 1Hz Votage measurements. This repository supports the work in (Stadler, et al 2025) (10.1175/JTECH-D-24-0023.1).  Two folders within this directory contain
code for different purposes. The first, for processing EM-APEX 1Hz measurements, the second
for modelling linear surface waves and the resulting predicted measurements by subsurface profiling float.

## WaveProcessing

<img width="1000" height="800" alt="spectral_flowchart2" src="https://github.com/user-attachments/assets/30aa0348-d0ff-44a3-ba5a-5ea9c8a00631" />


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
