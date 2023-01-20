"""
Functions for use in 1D surface wave simulations and visualizations

"""
import numpy as np
import pandas as pd
from scipy.io import netcdf
import scipy.signal
from datetime import datetime
import matplotlib.pyplot as plt
import random as rand

def load_CDIP(fname):
    
    #fname = "/Volumes/TFO-exFAT-1/TFO/LCDRI/CDIPwaverider229/229p1_d01.nc";
    nc = netcdf.NetCDFFile(fname, 'r', mmap=False) #Not sure why I need to do mmap = False but it gets rid of a warning
    time = nc.variables['waveTime'][:]
    Hs = nc.variables['waveHs'][:]
    energy_density = nc.variables['waveEnergyDensity'][:]
    wave_freq = nc.variables['waveFrequency'][:]
    #dtimes = datetime(time, 'convertfrom', 'posixtime');
    nc.close()
    
    return(np.array(time.data), np.array(wave_freq.data), np.array(energy_density.data))
    

def apex_sampling_grid(t_range, z_offset=0):
    ## Returns the x-coords, t-coords, z-coords of the EM-APEX samples
    ## if len(t_range)==n, then all are length n
    ##
    #t_range = np.arange(0, 1000)
    z_range = np.linspace(0, 100, 2000)

    #Em-APEX descent speed
    em_w = 0.1 #0.1m/s

    #Get the z-position
    em_z = t_range*em_w+z_offset
    mean_u = 0.05
    u_prof = np.ones(len(z_range))*mean_u

    #First lest just try a constant mean velocity
    em_x = t_range*mean_u

    ##TO DO:
    #Get the x-position
    #First for each timestep, get the z-position
    #Then get the closest velocity value (Allowing for non-constant velocity profile)
    #vels = get_vel(em_z, u_prof)
    #em_x = 
    
    return(t_range, em_x, em_z)



#Now make a u_store given the sampling parameters, don't build a whole grid


def build_u_timeseries(t_range, em_z, em_x, test_spectra, f):

    #Normally would loop through at each time step and generate a x-z wave field
    #Lets start with t=0
    n_iter = 500

    fs = 1
    u_store = np.zeros((n_iter, len(t_range)));
    u_store_surf = np.zeros((n_iter, len(t_range)))
    #Number of iterations to run
    #zeta_store = np.zeros((n_iter, len(t_range)))
    t = 0
    x = 0
    for jj in range(0, n_iter-1):
        for i in range(0, len(f-1)):#:#len(f)-1:
            freq = f[i]
            if i == 0:
                df = f[1]-f[0]
            elif i == len(f)-1:
                df = f[i]-f[i-1]
            else:
                i
                df = (f[i+1]-f[i-1])/2
            omega = 2*np.pi*freq;
            k = np.square(omega)/9.8
            a = np.sqrt(test_spectra[i]*df*2) # Is this the right conversion to wave amplitude? 

            #Randomize phase
            phi = rand.random()*2*np.pi;

            #Randomize direction?
            #TO DO: Need to have peak in narrow directional band
            #Should we be able to input directional spectra

            u = a*omega*np.cos(k*em_x-omega*t_range + phi)*np.exp(-k*em_z)
            u_surf = a*omega*np.cos(-omega*t_range + phi)
            #zeta = a*np.cos(k*x-omega*t_range + phi)
            #print(u)
            u_store[jj, :] = u_store[jj, :] + u
            u_store_surf[jj, :] = u_store_surf[jj, :]+u_surf
            #zeta_store[jj, :] = zeta_store[jj, :] + zeta
    #how do we choose the amplitude for each frequency?



    ## Add white noise
    
    #First generate a white noise with the std of an EM-APEX float white noise situation
    #TO DO: Check if "uncertainty of 0.8-1.5 cm/s" means that's 1 Std or rms or what
    mean = 0
    std = 0.01 
    num_samples = len(u_store[:, 0])*len(u_store[0, :])
    rand_samples = 0.008*np.random.normal(loc = 0, scale = 1, size = num_samples)


#     #Plot spectra of u_noise
#     #plt.plot(rand_samples)

#     ff = np.fft.fft(rand_samples)
#     ff_freq = np.fft.fftfreq(rand_samples.shape[-1], d=1)
#     ff_freq=ff_freq[:int(len(rand_samples)/2)]
#     ff = np.delete(ff,  np.s_[int(len(rand_samples)/2):], 0)
#     ff_power = np.real(ff * np.conj(ff))

#     #Do ensemble mean and then get PSD
#     #UU = np.nanmean(np.nanmean(UUwindow, axis=1), axis=0)/(int(w/2)*fs)

    
#     ff = np.fft.fft(rand_samples)
#     ff_freq = np.fft.fftfreq(rand_samples.shape[-1], d=1)
#     ff_freq=ff_freq[:int(len(rand_samples)/2)]
#     ff = np.delete(ff,  np.s_[int(len(rand_samples)/2):], 1)
#     ff_power = np.real(ff * np.conj(ff))
#     ff_power = np.nanmean(ff_power, 0)
#     ff_en = ff_power[1:] / (np.square((2*np.pi*ff_freq[1:])))
#     plt.loglog(ff_freq[1:], ff_en)
#     plt.show()

#     print(ff_freq.shape)
#     print(ff.shape)



    #
    rand_samples = rand_samples.reshape((len(u_store[:, 0]), len(u_store[0, :])))
    u_noise = u_store + rand_samples
    u_store_surf = u_store_surf + rand_samples
    mean_u = 0.05
    #u_store = u_store + mean_u
    #u_noise = u_noise + mean_u
    return(u_store, u_noise, u_store_surf)


#Reshape the data into 120s chunks etc for taking spectra
#Here I'm doing it with a 50% overlap

def reshape_u(u, em_z, nblock, overlap, fs):
    """
    Input:
        u: array of 1Hz velocity data
           -- u should either be a 1d time-series of length m, or a 2d matrix of n length-m timeseries of shape
            (n, m)
        em_z
        nblock: Windo length for calculating spectra
        overalp: number of samples to overlap. E.G. if nblock = 120, overlap = 60 is 50% overlap, overlap = 30 is 25% overlap
        fs: sampling frequency (should be 1Hz for EM-APEX)
        
    Output: 
        u_new:
        z_new
    """
    
    #fs = 1
    #nblock = 120
    #nstep = 60 #50% overlap?
   # overlap = 60
    

    slide = nblock-overlap
    if len(u.shape)==2:
        tseries_l = len(u[1, :])
        num_of_blocks = (tseries_l//slide)-1
        n_iter = u.shape[0]
        u_new = np.zeros((n_iter, num_of_blocks, nblock))
        z_new = np.zeros((num_of_blocks, nblock))
        
        
    elif len(u.shape)==1:
        tseries_l = len(u)
        num_of_blocks = (tseries_l//slide)-1
        n_iter = 1
        u_new = np.zeros((num_of_blocks, nblock))
        z_new = np.zeros((num_of_blocks, nblock))
    #First need to rearrange the velocities into the 120s chunks w/ 50% overlap 
    nout = 0
    
    for i1 in range(0, tseries_l-nblock, slide):
        j = list(range(i1, i1+nblock))
        if len(u.shape)==2:
            u_new[:, nout, :] = u[:, j]
        elif len(u.shape)==1:
            u_new[nout, :] = u[j]

        z_new[nout, :] = em_z[j]
        nout = nout+1  

    #print(u_new.shape)
    return(u_new, z_new)


def make_vel_spectrum(u_new, fs):
    """
    Input
        u_new: numpy array of velocity profile reshaped into matrix with dims [#wave-simulations, #windows, length of window]
    output:
        UUwindow: 
        fwindow: frequency array
    
    """

    #Define window taper
    uwindow = u_new;
    w = len(uwindow[0, 0, :])
    
    #Make the hanning taper
    taper_in = np.linspace(1, w, w)*2*np.pi/w
    taper_in = taper_in - np.pi
    taper = np.sin(taper_in)
    taper2 = np.cos(taper_in/2)*np.cos(taper_in/2)
    #plt.plot(np.linspace(1, w, w), taper)
    #plt.plot(np.linspace(1, w, w), taper2)
    #plt.show()
    taper_out = np.tile(np.transpose(taper2), (u_new.shape[0], u_new.shape[1], 1))

    #plt.plot(taper_in, taper2)
    #plt.show()
    #Taper the window
    uwindowtaper = uwindow * taper_out

    #print(uwindowtaper.shape)

    #Rescale
    #Trying to take this out to test Eric's code
    factor = np.sqrt(np.var(uwindow)/np.var(uwindowtaper))
    #print(factor)
    
    uwindowready = uwindowtaper*factor
    #uwindowready = uwindowtaper
    #Take fft
    Uwindow = np.fft.fft(uwindowready)
    fwindow = np.fft.fftfreq(uwindowready.shape[-1], d=1/fs)
    fwindow=fwindow[:int(w/2)]

    Uwindow = np.delete(Uwindow,  np.s_[int(w/2):], 2)



    #Remove first entry, make last entry equal to zero?
    #NOt sure why Andy does this---Seems to skew the spectrum
    #Uwindow = np.delete(Uwindow, 0, 1)
    #Uwindow = np.append(Uwindow, np.ones((n_iter, 1)), axis=1)


    #Take power spectra

    UUwindow = np.real(Uwindow * np.conj(Uwindow))

    #Do ensemble mean and then get PSD
    #UU = np.nanmean(np.nanmean(UUwindow, axis=1), axis=0)/(int(w/2)*fs)


    #Exx = UU[1:] / (np.square((2*np.pi*fwindow[1:])))
    return(UUwindow, fwindow)



def depth_correct_Eric(UUwindow, fwindow, em_z, w, Cmax, fs):
    """
    Returns spectra modified by D'Asaro 2015 depth correction
    
    """
    
   
    #TRying the fix to the wavenumber equation (1/9/2023)
    #k_array = np.sqrt(2*np.pi*fwindow/9.8);
    k_array = np.square(2*np.pi*fwindow)/9.8
    k_mat = np.tile(k_array, (UUwindow.shape[0], UUwindow.shape[1], 1))
    
    #This is def not the easiest way to do this
    z_mat = np.tile(em_z, (UUwindow.shape[0], 1, 1))
    z_mat = np.tile(np.expand_dims(np.nanmean(z_mat, axis=2), axis=2), (1, 1, len(k_mat[0, 0, :])))
    #print(np.expand_dims(np.nanmean(z_mat, axis=2), axis=2))


    
    depth_fact = np.exp(2*k_mat*z_mat)
    depth_fact[np.exp(k_mat*z_mat)>Cmax] = np.nan
    #depth_fact[depth_fact>Cmax]=np.nan
    UUwindow_out = np.nanmean(UUwindow, axis=0)*depth_fact

    #### Now try applying the motion correction gain adjustment from D'Asaro 2015

    #k_array = np.sqrt(2*np.pi*fwindow/9.8);
    k_array = np.square(2*np.pi*fwindow)/9.8


    k_mat = np.tile(k_array, (UUwindow.shape[0], UUwindow.shape[1], 1))
    kWT = 1*k_mat*0.1*(w/fs)
    G = np.square( (np.square(np.pi)/(np.square(kWT/2)+np.square(np.pi)))*(np.sinh(kWT/2)/(kWT/2)) )
    #G_mod = np.square((1/(2*np.pi))*np.sinh(kWT/2)/(np.power(kWT/(2*np.pi), 3)+kWT/(2*np.pi)))
    UUwindow3_G1 = UUwindow_out/G
    #print(np.nanmean(np.nanmean(G_mod, axis=0), axis=0))

    ### Try the 1s sampling correction
    #These are Andy's numbers
    d1 = 0.004897
    d2 = 0.033609
    d3 = 0.999897
    #these are eric's numbers
    #d1 = 0.001642
    #d2 = 0.018517
    #d3 = 0.999528
    G2 = d1*np.square(kWT)+d2*kWT+d3
    UUwindow3_G2 = UUwindow3_G1/G2

    ###Now the final correction
    dw = 2*np.pi/w
    omega_mat = np.tile(2*np.pi*fwindow, (UUwindow.shape[0], UUwindow.shape[1], 1))
    h=0.25
    z_mat = np.tile(em_z, (UUwindow.shape[0], 1, 1))
    z_mat = np.tile(np.expand_dims(np.nanmean(z_mat, axis=2), axis=2), (1, 1, len(k_mat[0, 0, :])))

    #G3 = 1/np.square(1+2*h*(np.cosh((2*omega*dw+np.square(dw)*z_mat)/9.8)-1))
    G3 = np.square((1+2*h*(np.cosh(z_mat*(np.square(omega_mat+dw)-np.square(omega_mat))/9.8)-1)))
    UUwindow3_G3 = UUwindow3_G2/G3
    
    ##IS this one supposed to be multiplied??
    
    
    
    return(UUwindow_out, UUwindow3_G1, UUwindow3_G2, UUwindow3_G3)

def depth_correct_Andy(UUwindow, fwindow, em_z, w, Cmax):
    """
    Returns spectra modified by Hsu 2021 depth correction
 
    """
    c = 0.06
    T0 = 120
    k_array = np.sqrt(2*np.pi*fwindow/9.8);
    k_mat = np.tile(k_array, (UUwindow.shape[0], UUwindow.shape[1], 1))
    
    #This is def not the easiest way to do this
    z_mat = np.tile(em_z, (UUwindow.shape[0], 1, 1))
    z_mat = np.tile(np.expand_dims(np.nanmean(z_mat, axis=2), axis=2), (1, 1, len(k_mat[0, 0, :])))
    
    depth_fact = np.exp(2*k_mat*(1+c*np.exp(-w/T0+1))*z_mat)
    depth_fact[depth_fact>Cmax]=np.nan
    UUwindow_out = np.nanmean(UUwindow, axis=0)*depth_fact

    
    return(UUwindow_out)

def depth_correct_James(UUwindow, fwindow, em_z, w, Cmax, zeta=1/2):
    """
    Returns spectra modified by my 1/2 energy depth correction
    
    """
    
    k_array = np.sqrt(2*np.pi*fwindow/9.8);
    k_mat = np.tile(k_array, (UUwindow.shape[0], UUwindow.shape[1], 1))
    
    #This is def not the easiest way to do this
    z_mat = np.tile(em_z, (UUwindow.shape[0], 1, 1))    
    z_window_0 = z_mat[:, :, 0]
    z_mat_0 = np.tile(np.expand_dims(z_window_0, axis=2), (1, 1, len(k_mat[0, 0, :])))
    
    
    #Try the z depth I derive based on 1/2 wave energy
    dz = -1/(2*k_mat)*np.log((1/2)*np.exp(-2*k_mat*(0.1)*w)+(1-zeta))
    z = z_mat_0+dz
    depth_fact_new = np.exp(2*k_mat*(z_mat_0+dz))
    
    depth_fact_new[depth_fact_new>Cmax]=np.nan
    
    
    UUwindow_out = np.nanmean(UUwindow, axis=0)*depth_fact_new
    
    return(UUwindow_out)


def sig_wave_height(f, spec):
    """
    This function  calculates the significant wave height from a energy density spectrum, with frequencies f.
    
    Input
        f: numpy array of frequencies
        spec: energy density spectra to integrate
        
        
    Output
        swh: Significant Wave Height calculated by integrating the input spectra
    
    To Do: Does this work with 2d Spec
    """
    
    
    ##Need to remove Nan's
    num_specs = spec.shape[1]
    swhs = np.zeros(num_specs)
    for i in range(num_specs):
        spec_temp = spec[:, i]
        real_inds = np.argwhere(~np.isnan(spec_temp))
        spec_temp = spec_temp[real_inds]
        f_temp = f[real_inds]
        swh = 4*np.sqrt(np.trapz(spec_temp[:, 0], x=f_temp[:, 0]))
        swhs[i] = swh
    return(swhs)
    
