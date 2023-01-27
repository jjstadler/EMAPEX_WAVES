# EMAPEX_WAVES

Repository containing code for calculating surface wave 
spectra from EM-APEX 1Hz Votage measurements. Two folders within this directory contain
code for different purposes. The first, for actually processing EM-APEX measurements, the second
for modelling linear surface waves and the resulting measurements by subsurface profiling float

## WaveProcessing

### src
This directory contains two files:

#### em_apex_processing.py

  Functions for transforming EM-APEX measurements in velocities and velocity residuals

#### spectral_processing.py

  Functions for taking spectra of EM-APEX profiles

## WaveModelling


## Directory Structure
```bash
├── LICENSE
├── WaveProcessing/
│   ├── src
│   │   ├── em_apex_processing.py 
│   │   └── spectral_processing.py
│   ├── Process_LCDRI_1Hz.ipynb
│   ├── Analyze_EM_vertical_Speeds.ipynb
│   ├── Flowchart_figures.ipynb
│   ├── Visualize_Fit_Differences.ipynb
│   └──Test_Spectral_Uncertainty.ipynb
├── WaveModelling/
│   ├── 
│   ├── 
│   └──
└── README.md
```
