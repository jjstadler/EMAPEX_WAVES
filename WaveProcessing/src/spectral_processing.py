"""
spectral_processing_tools.py

This module contains the code for spectral processing of velocity measurements for measuring
ocean surface waves by a vertically profiling subsurface float

Tools are used by both the wave simulations code and the 1Hz processing code

Note: Maybe this should really be structured differently --> Create an object that's a velocity timeseries, and then 
that has all the different processing steps as different methods, but I don't feel like figuring out how to get all that workign right now
"""


import numpy as np
import warnings
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat, savemat
import pandas as pd
from em_apex_processing import *
import pyIGRF
from scipy import signal
import scipy





class Spectra:
    def __init__(self, fwindow, UU, VV, AZAZ, UV, UAZ, VAZ):
        self.UU = UU
        self.VV = VV
        self.AZAZ = AZAZ
        self.UV = UV
        self.UAZ = UAZ
        self.VAZ = VAZ
        self.fwindow = fwindow
        

def reshape_u(u, em_z, nblock, overlap, fs):
    """
    function for take a 1d velocity times series and reshaping it into windows for taking spectra
    #Reshape the data into 120s chunks etc for taking spectra
    #Here I'm doing it with a 50% overlap
    
    Input:
        u: array of 1Hz velocity data
           -- u should either be a 1d time-series of length m, or a 2d matrix of n length-m timeseries of shape
            (n, m)
        em_z: 1-d array of depths (meters), length m.
        nblock: Window length for calculating spectra
        overalp: number of samples to overlap. E.G. if nblock = 120, overlap = 60 is 50% overlap, overlap = 30 is 25% overlap
        fs: sampling frequency (should be 1Hz for EM-APEX)
        
    Output: 
        u_new: 2d or 3d numpy array of velocities reshaped for spectral processing
            (2d if input u is 1d, 3d if input u is 2d)
            
        z_new: 2d numpy array of depths corresponding to each measurement in u_new
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
    
    for i1 in range(0, tseries_l-nblock+1, slide):
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
    
    function for taking the power spectra from a matrix of velocity measurements (u or v), shaped so that the last two dimmensions 
    
    Input
        u_new: numpy array of velocity profile reshaped into matrix with dims [#wave-simulations, #windows, length of window]
    output:
        UUwindow: 
        fwindow: frequency array
    
    """

    #Define window taper
    uwindow = u_new;
    
    #w is length of window
    if len(uwindow.shape)==3:
        w = len(uwindow[0, 0, :])
    elif len(uwindow.shape)==2:
        w = len(uwindow[0, :])
        
    
    #Make the hanning taper
    taper_in = np.linspace(1, w, w)*2*np.pi/w
    taper_in = taper_in - np.pi
    taper = np.sin(taper_in)
    taper2 = np.cos(taper_in/2)*np.cos(taper_in/2)
 
    
    #This needs to be set correctly to match the shape we want
    if len(uwindow.shape)==3: 
        taper_out = np.tile(np.transpose(taper2), (u_new.shape[0], u_new.shape[1], 1))
    elif len(uwindow.shape)==2:
        taper_out = np.tile(np.transpose(taper2), (u_new.shape[0], 1))

    #Taper the window
    uwindowtaper = uwindow * taper_out

    #print(uwindowtaper)
    #print(np.nanvar(uwindowtaper))
    #print(np.nanvar(uwindow))
    
    #Rescale
    factor = np.sqrt(np.nanvar(uwindow)/np.nanvar(uwindowtaper))
    
    #print(factor)
    
    uwindowready = uwindowtaper*factor

    #print(uwindowready) 
    
    #test plot
    #plt.figure()
    #plt.plot(np.transpose(uwindowready))
    
    #Take fft
    Uwindow = np.fft.fft(uwindowready)
    #print(Uwindow)
    fwindow = np.fft.fftfreq(uwindowready.shape[-1], d=1/fs)
    fwindow=fwindow[:int(w/2)]

    Uwindow = np.delete(Uwindow,  np.s_[int(w/2):], -1)


    #Take power spectra
    UUwindow = np.real(Uwindow * np.conj(Uwindow))

    return(UUwindow, fwindow)


def make_az_spectrum(az, fs):
    
    """
    Function to return the power spectra and cross spectra (for wave direction calculations) based on u, v, and z acceleration.
    
    Inputs:
        u:
        v:
        az:
        fs:
    Outputs:
        Who knows??
    """
    
    
    #Define window taper
    window = az;
    
    #w is length of window
    if len(window.shape)==3:
        w = len(window[0, 0, :])
    elif len(window.shape)==2:
        w = len(window[0, :])
        
    
    #Make the hanning taper
    taper_in = np.linspace(1, w, w)*2*np.pi/w
    taper_in = taper_in - np.pi
    taper = np.sin(taper_in)
    taper2 = np.cos(taper_in/2)*np.cos(taper_in/2)
 
    
    #This needs to be set correctly to match the shape we want
    if len(window.shape)==3: 
        taper_out = np.tile(np.transpose(taper2), (az.shape[0], az.shape[1], 1))
    elif len(window.shape)==2:
        taper_out = np.tile(np.transpose(taper2), (az.shape[0], 1))

    #Taper the window
    windowtaper = window * taper_out

    #print(uwindowtaper)
    #print(np.nanvar(uwindowtaper))
    #print(np.nanvar(uwindow))
    
    #Rescale
    factor = np.sqrt(np.nanvar(window)/np.nanvar(windowtaper))
    
    #print(factor)
    
    windowready = windowtaper*factor

    #print(uwindowready) 
    
    #test plot
    #plt.figure()
    #plt.plot(np.transpose(uwindowready))
    
    #Take fft
    AZwindow = np.fft.fft(windowready)
    #print(Uwindow)
    fwindow = np.fft.fftfreq(windowready.shape[-1], d=1/fs)
    fwindow=fwindow[:int(w/2)]

    AZwindow = np.delete(AZwindow,  np.s_[int(w/2):], -1)


    #Take power spectra
    AZAZwindow = np.real(AZwindow * np.conj(AZwindow))

    
    return(AZAZwindow, fwindow)








def make_spectra(u, v, az, fs):
    """
    
    function for taking the power spectra from a matrix of vertical acceleration measurements , shaped so that the last two dimmensions 
    
    Input
        az: numpy array of velocity profile reshaped into matrix with dims [#wave-simulations, #windows, length of window]
    output:
        UUwindow: 
        fwindow: frequency array
    
    """

    #Define window tapers
    uwindow = u
    vwindow = v
    azwindow = az
    
    
    #w is length of window
    if len(uwindow.shape)==3:
        w = len(uwindow[0, 0, :])
    elif len(uwindow.shape)==2:
        w = len(uwindow[0, :])
        
    
    #Make the hanning taper
    taper_in = np.linspace(1, w, w)*2*np.pi/w
    taper_in = taper_in - np.pi
    taper = np.sin(taper_in)
    taper2 = np.cos(taper_in/2)*np.cos(taper_in/2)
 
    
    #This needs to be set correctly to match the shape we want
    if len(uwindow.shape)==3: 
        taper_out = np.tile(np.transpose(taper2), (u.shape[0], u.shape[1], 1))
    elif len(uwindow.shape)==2:
        taper_out = np.tile(np.transpose(taper2), (u.shape[0], 1))

    #Taper the window
    uwindowtaper = uwindow * taper_out
    vwindowtaper = vwindow * taper_out
    azwindowtaper = azwindow * taper_out

    
    #Rescale
    ufactor = np.sqrt(np.nanvar(uwindow)/np.nanvar(uwindowtaper))
    vfactor = np.sqrt(np.nanvar(vwindow)/np.nanvar(vwindowtaper))
    azfactor = np.sqrt(np.nanvar(azwindow)/np.nanvar(azwindowtaper))

    
    uwindowready = uwindowtaper*ufactor
    vwindowready = vwindowtaper*vfactor
    azwindowready = azwindowtaper*azfactor
    
    #Take fft
    Uwindow = np.fft.fft(uwindowready)
    Vwindow = np.fft.fft(vwindowready)
    AZwindow = np.fft.fft(azwindowready)
    
    fwindow = np.fft.fftfreq(uwindowready.shape[-1], d=1/fs)
    fwindow=fwindow[:int(w/2)]

    Uwindow = np.delete(Uwindow,  np.s_[int(w/2):], -1)
    Vwindow = np.delete(Vwindow,  np.s_[int(w/2):], -1)
    AZwindow = np.delete(AZwindow,  np.s_[int(w/2):], -1)




    #Take power spectra
    UUwindow = np.real(Uwindow * np.conj(Uwindow))
    VVwindow = np.real(Vwindow * np.conj(Vwindow))
    AZAZwindow = np.real(AZwindow * np.conj(AZwindow))
    
    #Take Cross Spectra
    UVwindow = (Uwindow * np.conj(Vwindow))
    UAZwindow = (Uwindow * np.conj(AZwindow))
    VAZwindow = (Vwindow * np.conj(AZwindow))

    specs = Spectra(fwindow, UUwindow, VVwindow, AZAZwindow, UVwindow, UAZwindow, VAZwindow)
    
    return(specs)



    

def sig_wave_height(f, spec, uncertainty=None):
    """
    This function  calculates the significant wave height from a energy density spectrum, with frequencies f.
    
    Input
        f: numpy array of frequencies
        spec: energy density spectra to integrate
        
        
    Output
        swh: Significant Wave Height calculated by integrating the input spectra
    
    To Do: Does this work with 2d Spec v
    """
    
     
    #First figure out if this is a 2d input or a 1d input
    if len(spec.shape)==1:
        #Then 1d
        spec_temp = spec

        #Need to remove Nan's
        real_inds = np.where(~np.isnan(spec_temp))[0]
        spec_temp = spec_temp[real_inds]
        f_temp = f[real_inds]
        swh = 4*np.sqrt(np.trapz(spec_temp, x=f_temp))
        swhs = np.array(swh)
        if uncertainty is not None:
            nu_s =uncertainty[2]* np.square(np.sum(spec_temp))/np.sum(np.square(spec_temp))
            llim = np.sqrt(nu_s/scipy.stats.chi2.ppf(1-.05/2, df=nu_s))*swh
            ulim = np.sqrt(nu_s/scipy.stats.chi2.ppf(.05/2, df=nu_s))*swh
            swh_upper = ulim#np.array(4*np.sqrt(np.trapz(spec_temp*uncertainty[1], x=f_temp)))-swhs
            swh_lower = llim#swhs-np.array(4*np.sqrt(np.trapz(spec_temp*uncertainty[0], x=f_temp)))
        else:
            swh_upper = None
            swh_lower = None
            
    elif len(spec.shape)==2:
        #Then 2d
        num_specs = spec.shape[1]
        swhs = np.zeros(num_specs)
        swh_upper= np.zeros(num_specs)
        swh_lower =np.zeros(num_specs)
        for i in range(num_specs):
            spec_temp = spec[:, i]
            ##Need to remove Nan's
            real_inds = np.argwhere(~np.isnan(spec_temp))
            spec_temp = spec_temp[real_inds]
            f_temp = f[real_inds]
            #print(spec_temp.shape)
            swh = 4*np.sqrt(np.trapz(spec_temp[:, 0], x=f_temp[:, 0]))
            swhs[i] = swh
            if uncertainty is not None:
                #print(np.array(4*np.sqrt(np.trapz(spec_temp[:, 0]*uncertainty[i, 1], x=f_temp))))
                #swh_upper[i] = np.array(4*np.sqrt(np.trapz(spec_temp[:, 0]*uncertainty[i, 1], x=f_temp[:, 0])))-swhs[i]
                #swh_lower[i] = swhs[i]-np.array(4*np.sqrt(np.trapz(spec_temp[:, 0]*uncertainty[i, 0], x=f_temp[:, 0])))
                
                nu_s =uncertainty[i, 2]* np.square(np.sum(spec_temp[:, 0]))/np.sum(np.square(spec_temp[:, 0]))
                #print(nu_s)
                llim = np.sqrt(nu_s/scipy.stats.chi2.ppf(1-.05/2, df=nu_s))*swh
                ulim = np.sqrt(nu_s/scipy.stats.chi2.ppf(.05/2, df=nu_s))*swh
                  
                swh_upper[i] = ulim-swh
                swh_lower[i] = swh-llim
                
            else:
                swh_upper[i] = None
                swh_lower[i] = None
                
    else:
        raise Exception("Array containing spectra to integrate must be either 1D or 2D")
     
    
    return(swhs, swh_lower, swh_upper)
        



def depth_correct_Eric(Eh, fwindow, em_z, prof_speed, nblock, Cmax, fs):
    """
    Returns spectra modified by D'Asaro 2015 depth correction for EM-APEX processed 
    

    TO DO: em_z should probably be consolidated to be shape (m,) before passing to this function for consistency with the shapes of the other parameters

    Input:
    
        Eh: (m, n) numpy array where m is the number of spectral windows, and n is the number of frequency bins
        
        fwindow: shape (n,) numpy array containing frequency bins for the spectra Eh
        
        em_z: (m, 2n) numpy array, containing depth at each 
        
        prof_speed: (m,) numpy array, containing mean profiling speed associated with each spectra
        
        nblock: length of each spectral window. It's either 2n or 2n+2, can't remember which
        
        Cmax: Max multiplicative factor for scaling up spectra
        
        fs: sampling frequency (nominally 1Hz)
        
    Output:
    
        TO DO:
    
    """
    
   
    #THIS IS WHAT IT WAS BEFORE 1/9/2023
    #k_array = np.sqrt(2*np.pi*fwindow/9.8);
    #THIS IS WHAT IT IS AFTER 1/9/2023
    k_array = np.square(2*np.pi*fwindow)/9.8
    k_mat = np.tile(k_array, (Eh.shape[0], 1))
    
    #This is def not the easiest way to do this
    #z_mat = np.tile(em_z, (Eh.shape[0], 1))
    z_mat = em_z
    z_mat = np.tile(np.expand_dims(np.nanmedian(z_mat, axis=1), axis=1), (1, len(k_mat[0, :])))
    #print(z_mat)

    
    depth_fact = np.exp(2*k_mat*z_mat)
    depth_fact[np.exp(k_mat*z_mat)>Cmax]=np.nan
    #depth_fact[depth_fact>Cmax]=np.nan
    Eh_out = Eh*depth_fact

    #### Now try applying the motion correction gain adjustment from D'Asaro 2015

    #First need to tile prof_speed
    #print(prof_speed.shape)
    prof_speed = np.tile(np.expand_dims(prof_speed, axis=1), (1, Eh.shape[1]))
    kWT2 = 1*k_mat*prof_speed*(nblock/fs)
    #print(kWT2.shape)
    #kWT = 1*k_mat*0.1*(nblock/fs)
    #Tile to be the right size
    
    
    #G = np.square((np.square(np.pi)/(np.square(kWT/2)+np.square(np.pi)))*(np.sinh(kWT/2)/(kWT/2)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        G2 = np.square((np.square(np.pi)/(np.square(kWT2/2)+np.square(np.pi)))*(np.sinh(kWT2/2)/(kWT2/2)))
    #If we get a nan value (which should occur when kWT2 = 0, just set the gain to 1 bc that's what it analytically works out to
    G2[np.isnan(G2)]=1
    #G_mod = np.square((1/(2*np.pi))*np.sinh(kWT/2)/(np.power(kWT/(2*np.pi), 3)+kWT/(2*np.pi)))
    #G_mod2 = np.square((1/(2*np.pi))*np.sinh(kWT2/2)/(np.power(kWT2/(2*np.pi), 3)+kWT2/(2*np.pi)))
    
    #print(np.nanmean(G/G2))
    Eh_G1 = Eh_out/G2
    #print(np.nanmean(np.nanmean(G_mod, axis=0), axis=0))

    ### Try the 1s sampling correction
    #These are Andy's numbers
    d1 = 0.004897
    d2 = 0.033609
    d3 = 0.999897
    
    #These are Eric's Numbers
    #d1 = 0.001642
    #d2 = 0.018517
    #d3 = 0.999528
    #G2 = d1*np.square(kWT)+d2*kWT+d3
    G2mod = d1*np.square(kWT2)+d2*kWT2+d3
    #print(np.nanmean(G2mod/G2))
    Eh_G2 = Eh_G1/G2mod

    ###Now the final correction
    dw = 2*np.pi/nblock
    omega_mat = np.tile(2*np.pi*fwindow, (Eh.shape[0], 1))
    h=0.25
   
    #G3 = 1/np.square(1+2*h*(np.cosh((2*omega*dw+np.square(dw)*z_mat)/9.8)-1))
    G3 = np.square((1+2*h*(np.cosh(z_mat*(np.square(omega_mat+dw)-np.square(omega_mat))/9.8)-1)))
    Eh_G3 = Eh_G2/G3
    
    #IS this one supposed to be multiplied or divided?
    
    
    
    
    return(Eh_out, Eh_G1, Eh_G2, Eh_G3)


def add_hf_tail(spec, f):
    """
    function to tack on an equilibrium range tail to a corrected EM-APEX spectrum
    function should work on an input that is eithe 1D (a single spectrum) or 2d (multiple spectra)
    
    4-17-23: If there's a spectrum of all nan's, it just returns a spec of all nan's instead of breaking
    
    Input:
        spec: numpy array containing spectra to add tail to. Should either be dimension (m x n), or (n,)
            where m is the # of spectra to add tail to, and n is the number of points in the spectra
            
        f: frequency bins corresponding to the spectra. Should be dimension (n,)
    """
    
    new_specs = np.copy(spec)
    
    for ind in range(spec.shape[0]):
        #Get last non-NaN value in the spectrum
        if np.isnan(spec[ind, :]).all():
            new_specs[ind] = spec[ind, :]
        else:
            end_ind = np.where(np.isnan(spec[ind, :]))[0][0]-1
            f_end = f[end_ind]
            E_end = spec[ind, end_ind]

            #Maybe E_end should be avg of last 3 points?
            E_end = np.min(spec[ind, end_ind-3:end_ind+1])
            
            c = E_end/f_end**(-4)
            extension = np.power(f, -4)*c
            extension[:end_ind]=0
            temp_spec = np.copy(spec[ind, :])
            temp_spec[end_ind:]=0

            extended_spec = temp_spec+extension
            new_specs[ind]=extended_spec
        
    return(new_specs)

def get_moving_inds(Pef):
    """
    Function returns the indicies of the EM 1Hz timegrid where the float is moving. Input the pressure reading timeseries (in the time grid of the Em measurements) and it will return a list of indicies to use for the velocity fitting.
    
    Input: 
        Pef: timesereis in the 1Hz timegrid of float pressure readings
        
    Output:
        moving_inds: list of indicies once the float has started moving during that profile
  
    """
    #initialize start_ind
    start_ind = 0
    #Loop through and find first ind where the float starts moving
    for ind in range(0, len(Pef)-1):
        if np.abs(Pef[ind+1]-Pef[ind])<0.0001:
            continue
        else:
            start_ind = ind
            break
    
    #Create array of indicies after the float starts moving
    ind_list = np.array(list(range(start_ind, len(Pef))))
    
    return(ind_list)



def get_spectral_uncertainity(E_x, E_y, Pef, u_noise, prof_speed, nblock, overlap, Cmax, fs, debug_flag=False):
    """
    This funciton outputs a range of possible spectrum based on a normal distribution of velocity noise
    
    TO DO: This takes a heck of a lot of inputs...Maybe think about how to streamline this?
    
    Input:
    
    
    Output:
        lbound
        ubound
        spec_array -
    """
    #length of time series
    npoints = len(E_x)
    
    #Do a part here due to inherent spectral uncertainity due to DOFs
    [u_x, z_x] = reshape_u(E_x, Pef, nblock, overlap, fs)
    
    #get number of blocks
    nb = u_x.shape[0]
    #This is from Percival and Walden textbook
#     DOF = 36*np.square(nb)/(19*nb - 1)
#     phi_inv = -1.96
#     phi_inv_m1 = 1.96
#     #At 95% CI
#     Qvp = DOF*( ( 1-(2/(DOF*9))+phi_inv*(((2/(DOF*9))**(1/2))))**3 )
#     Qvp_m1 = DOF*( ( 1-(2/(DOF*9))+phi_inv_m1*(((2/(DOF*9))**(1/2))))**3 )
    
#     lbound = DOF/Qvp_m1
#     ubound = DOF/Qvp
    import scipy
    nu = nb*nb*36/(19*nb-1)#*100 #why is this multiplied by 100?
    #for 95% CI
    alpha = 0.05
    cl = nu/scipy.stats.chi2.ppf(1-.05/2, df=nu)
    cu = nu/scipy.stats.chi2.ppf(.05/2, df=nu)
    
    
    if debug_flag:
        print("nb", nb)
        print("nu", nu)
        print("cl", cl)
        print("cu", cu)
    
    return(cl, cu, nu)
    
    """
    #This part is for estimating the spectral uncertainty due to velocity uncertainity
    #number of iterations to run over?
    n_iterations=50
    
    #Initialize array to store new spectra
    spec_array = np.zeros((n_iterations, 2, (nblock/2)-1))


    for niter in range(n_iterations):
        rands = np.random.normal(0, u_noise, 2*npoints)    
        E_x_new = E_x + rands[:npoints]
        E_y_new = E_y + rans[npoints:]
    
        #Now just do the spectral processing steps
        #1-Reshape
        [u_x, z_x] = reshape_u(E_x_new, Pef, nblock, overlap, fs)

        [u_y, z_y] = reshape_u(E_y_new, Pef, nblock, overlap, fs)

        #2-Calculate raw spectrum
        UUwindow, fwindow = make_vel_spectrum(u_x, fs)

        VVwindow, fwindow = make_vel_spectrum(u_y, fs)
    
        #3-Compute Eric's Correction
        
        UU = UUwindow/(int(nblock/2)*fs)
        Exx = UU[:,1:]/ (np.square((2*np.pi*fwindow[1:])))
        VV = VVwindow/(int(nblock/2)*fs)
        Eyy = VV[:,1:]/ (np.square((2*np.pi*fwindow[1:])))

        Eh = Exx+Eyy
        [Eh_Eric1, Eh_Eric2, Eh_Eric3, Eh_Eric4] = depth_correct_Eric(Eh, fwindow[1:], z_x, prof_speed, nblock, Cmax, fs)

        with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                spec_array[counter, 0, :] = np.nanmean(Eh, axis=0)
                spec_array[counter, 1, :] = np.nanmean(Eh_Eric4, axis=0) 
    """
    
    #return(spec_array)
    
def get_peak_freq(f, spec):
    """
    This function takes an array of frequency and an array of spectra, and returns the 
    peak frequency, as well as the peak spectral level of the input spectrum
    
    Input:
        f: (,N) shaped numpy array containing frequency bins
        spec: (M,N), or (,N) shaped numpy array of spectra, or a single spectrum
    Output:
        peak_freq: Numpy array of peak frequencies
        spec_level: Numpy array of peak spectral levels
    """
    if len(spec.shape)>1:
        freq_ind = np.nanargmax(spec, axis=1)
        peak_freq = np.zeros(spec.shape[0])
        spec_level = np.zeros(spec.shape[0])
        for ind in range(spec.shape[0]):
            spec_level[ind]=spec[ind, freq_ind[ind]]
            peak_freq[ind] = f[freq_ind[ind]]
    else:
        freq_ind = np.argmax(spec)
        peak_freq = f[freq_ind]
        spec_level = spec[freq_ind]
    
    return(peak_freq, spec_level)

def contains_spikes(E1, E2):
    """
    First pass at spike detection/handling...For now just returns true if one of 
    the channels has unphyscial spikes so we can ignore that profile.
    
    Ideal handling would just mask out the corrupted data.
    
    Input:
        E1: channel 1 voltages or velocties
        E2: Channel 2 voltages or Velocities
    Output:
        has_spikes: boolean type - yes is one channel has large spikes, no if they're both good
    """
    
    #Just want to check the middle ~80% of the array (away from top and bottom)
    length = len(E1)
    cut = int(np.floor(length/10))
    
    grad1 = np.gradient(E1[cut:-cut])
    grad2 = np.gradient(E2[cut:-cut])

    grad_deviation1 = np.nanstd(grad1)
    grad_deviation2 = np.nanstd(grad2)
    
    
    has_spikes = False
    if max(abs(grad1))>=abs(np.nanmean(grad1))+10*grad_deviation1:
        has_spikes = True
    if max(abs(grad2))>=abs(np.nanmean(grad2))+10*grad_deviation2:
        has_spikes = True
        
    #print(has_spikes, max(abs(grad1)), abs(np.nanmean(grad1))+5*grad_deviation1)
    return(has_spikes)



def datenum_to_datetime(datenum_arr):
    """
    This is just for processing swift data. Maybe doesn't bleong in this file, but nowhere else to put it right now
    Convert Matlab datenum into Python datetime.
    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.
    """
    #If its a pandas array loop through and do each elments
    dt_out = np.empty(len(datenum_arr), dtype=object)
    for n in range(len(datenum_arr)):
        datenum = datenum_arr[n]
        days = datenum % 1
        hours = days % 1 * 24
        minutes = hours % 1 * 60
        seconds = minutes % 1 * 60
        dtime = datetime.datetime.fromordinal(int(datenum)) \
               + datetime.timedelta(days=int(days)) \
               + datetime.timedelta(hours=int(hours)) \
               + datetime.timedelta(minutes=int(minutes)) \
               + datetime.timedelta(seconds=round(seconds)) \
               - datetime.timedelta(days=366)
        dt_out[n] = dtime
    return dt_out




def process_files(fname_base, Cmax=10, navg=120, nstep=60, sim=False):
    #########################
    #####Define Constants####
    #########################
    plot_count=0
    
    Cmax=10
    uVpc = 1e6 * (10e-3 / (2**24))

    #Degree of polynomial fit
    Nfit=1

    nstep_off=25
    navg_off=50
    base_fr=0.02

    ch=0.06
    cz=0.09

    nstep = 60
    navg = 120 
    Zmax=100
    Dirmax=80
    #electrode separation
    esep1 = (8+5/8)*0.0254 # m
    esep2 = (8+5/8)*0.0254 # m
    c1 = 0.5
    c2 = -0.2
    # angles between compass and electrode axes
    alpha2 = 1.95
    alpha1 = alpha2 - np.pi/2
    
    
    
    float_list = os.listdir(fname_base)

    #Initialize arrays for storing stuff
    big_spec_store = []
    big_time_store = []
    big_up_down_store = []
    big_prof_store = []
    big_uncertainty_store = []
    big_loc_store = []
    resid_store = np.array([])

    #For investigating down profiles
    #down_min_z = []
    #up_min_z =[]


    up = True

    float_id_counter = 0
    ignore_count = 0
    too_deep_counter = 0
    min_bin = []
    first_bot = []

    ##Testing for nans -- TO DO: Think we can delete this nan stuff
    nancounter = 0
    nanstorer = []
    for float_id in float_list:

        if "grid" in float_id:
            continue
        if ".DS_" in float_id:
            continue

        dec_name = fname_base+float_id+"/dec/"

        #Loop through each profile for that float
        files = os.listdir(dec_name)
        if sim==True:
            continue
        else:
            efr_files = [file for file in files if "efr.mat" in file and not file.startswith('.')]

        spec_store = np.zeros((len(efr_files), 2 , 59))
        time_store = np.zeros(len(efr_files))
        up_down_store = np.zeros(len(efr_files))
        uncertainty_store = np.zeros((len(efr_files), 2))
        loc_store = np.zeros((len(efr_files), 2))
        prof_store = np.empty(len(efr_files), dtype=object)
        counter=0
        #Load each profiling file, and then calculate the 1D spectrum
        for file in efr_files:
            fname = dec_name + file
            
            EFR = loadmat(fname)


            prof_num = int(file.split('-')[2])
            #Load the UXT times, and correct

            efr_times = EFR['UXT'] - EFR['AGE']
            efr_times = efr_times[0, :]
            seqno = EFR['SEQNO'][0, :]

            #Fit UXT times to sequence number (measurement # for that profile) to make sure monotonically increasing
            p = np.polyfit(seqno,efr_times,1)
            pfit = np.poly1d(p)
            mlt_efr = pfit(seqno);


            #Load GPS file for calculating 
            # gps files are only on up profiles (even)
            if prof_num%2==0:
                up = True
                cut = fname.find("efr")
                gpsfname = fname[:cut]+"gps.mat"
            else:
                up = False
                new_file = file.split('-')[0]+'-'+file.split('-')[1]+'-{:04d}'.format(prof_num+1)+"-"+file.split('-')[3]+'-gps.mat'
                gpsfname = dec_name+new_file
            GPS = loadmat(gpsfname)

            #Load CTD file 
            cut = fname.find("efr")
            ctdfname = fname[:cut]+"ctd.mat"
            CTD = loadmat(ctdfname)

            ctd_time = CTD["UXT"][0, :]
            P = CTD["P"][0, :]
            Pef = np.interp(mlt_efr, ctd_time, P)

            tim_pd = pd.to_datetime(GPS["UXT_GPS"][0, :],unit='s', utc=True,)
            #Convert time to fractional year for use in the igrf function
            frac_yrs = np.array([year_fraction(dt) for dt in tim_pd])
            avg_lat = np.nanmean(GPS["LAT"][0, :])
            avg_lon = np.nanmean(GPS["LON"][0, :])
            avg_frac_yrs = np.nanmean(frac_yrs)

            #Get magnetic field values
            [Bx, By, Bz, f] = pyIGRF.calculate.igrf12syn(avg_frac_yrs, 1, 0, avg_lat, avg_lon)
            #get conversion factor for converting to velocity
            fz=-np.nanmean(Bz);
            fh=np.nanmean(np.sqrt(Bx**2+By**2));
            sfv1 = 1e3/(fz*esep1*(1.0+c1));
            sfv2 = 1e3/(fz*esep2*(1.0+c1));

            #Convert from counts to microvolts
            E1 = (EFR["E1"][0, :]-2**23) * uVpc;
            E2 = (EFR["E2"][0, :]-2**23) * uVpc;

            #print(EFR["E1"].shape)
            #pull out compass values
            HZ = EFR["HZ"][0, :];
            HY = EFR["HY"][0, :];
            HX = EFR["HX"][0, :];


            #If up is true, flip everything
            if up:
                E1 = np.flip(E1)
                E2 = np.flip(E2)
                HZ = np.flip(HZ)
                HX = np.flip(HX)
                HY = np.flip(HY)
                Pef = np.flip(Pef)
                mlt_efr = np.flip(mlt_efr)

            else:
                #Do Nothing
                pass


            #Remove the beginning of the timesereis before the float starts moving
            #Need to do this for all of E1, E2, HX, HY, mlt_efr
            moving_inds = get_moving_inds(Pef)

            #Uncomment this to print out the filenames where EM starts before CTD
            #if len(moving_inds)<len(Pef):
                #print(len(Pef)-len(moving_inds), fname)
                #continue

            #Apply moving_inds to the EM timeseries
            E1 = E1[moving_inds]
            E2 = E2[moving_inds]
            HX = HX[moving_inds]
            HY = HY[moving_inds]
            mlt_efr = mlt_efr[moving_inds]
            Pef = Pef[moving_inds]

            #Do the 50s fits 
            [e1offs,e2offs,e1fits,e2fits,anghxhy, resid] = em_offset(Nfit,mlt_efr,nstep_off,navg_off,E1,E2,HX,HY);

            #Get overall fi and calculate the residuals
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                e1off=np.nanmean(e1offs,1);
                e2off=np.nanmean(e2offs,1);
                e1fit=np.nanmean(e1fits,1);
                e2fit=np.nanmean(e2fits,1);
                resid = np.nanmean(resid,1);

            #Calculate the residual
            e1r = E1 - e1fit
            e2r = E2 - e2fit


            if plot_count==0:
                plt.figure()
                plt.plot(E1[0:100])
                plt.plot(e1fit[0:100])
                plot_count=1
            #plt.figure()
            #plt.plot(E1[:600])
            #plt.plot(e1fit[:600])

            
            
            ## Do spike detection
            #If either channel has spikes, ignore the profile.
            #spikes=contains_spikes(E1, E2)
            #if spikes:
            #    print(fname)
            #    continue


            #Now need to convert to velocity (m/s)
            e1r = e1r*sfv1
            e2r = e2r*sfv2

            #plt.plot(e1r)
            
            #Now use the angles to rotate to x-y coordinates
            avg_angs = np.copy(anghxhy)
            avg_angs[~np.isnan(avg_angs)] = np.unwrap(avg_angs[~np.isnan(avg_angs)])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                avg_angs = np.nanmean(avg_angs, axis=1)


            E2_r=e2r*np.cos(avg_angs)+e1r*np.sin(avg_angs);
            E1_r=-e2r*np.sin(avg_angs)+e1r*np.cos(avg_angs);

            E_x = E1_r*np.cos(alpha1)-E2_r*np.sin(alpha1)
            E_y = E1_r*np.sin(alpha1)+E2_r*np.cos(alpha1)

            #Now try highpass filtering the data
            sos = signal.butter(10, 0.04, 'hp', fs=1, output='sos')
            E_x_filtered = signal.sosfilt(sos, E_x)
            E_y_filtered = signal.sosfilt(sos, E_y)

            E_x = E_x_filtered
            E_y = E_y_filtered



            #Now take the spectra
            nblock = 120
            fs = 1
            overlap = 60

            #in the case where the float doesn't actually move for nblock measurements, need to skip it
            if len(E_x)<nblock:
                continue


            [u_x, z_x] = reshape_u(E_x, Pef, nblock, overlap, fs)
            [u_y, z_y] = reshape_u(E_y, Pef, nblock, overlap, fs)
            #prof_speed is a numpy array that stores the mean vertical velocity (in m/s)
            #for each spectral window
            #Throwaway to get t_new in the same shape as u_x and z_x for calculating mean profiling speed
            [u_x_temp, t_new] = reshape_u(E_x, mlt_efr, nblock, overlap, fs)

            prof_speed = np.abs(z_x[:, 0]-z_x[:, -1])/np.abs(t_new[:, 0]-t_new[:, -1])

            prof_speed_new = np.zeros(len(prof_speed))
            for block_ind in range(z_x.shape[0]):
                prof_speed_try = np.abs(np.gradient(z_x[block_ind, :], t_new[block_ind, :]))
                z_inds = prof_speed_try<0.001
                prof_speed_try[z_inds]=np.nan
                prof_speed_removed_zeros = np.nanmedian(prof_speed_try)
                prof_speed_new[block_ind]=prof_speed_removed_zeros

            prof_speed = prof_speed_new


            zero_inds = np.where(prof_speed==0)[0]
            #if len(zero_inds)>0:
                #print(z_x[0, :])

            UUwindow, fwindow = make_vel_spectrum(u_x, fs)

            VVwindow, fwindow = make_vel_spectrum(u_y, fs)

            min_z = np.min(Pef)

            if min_z>20:
                too_deep_counter+=1
            else:  
                min_bin = np.append(min_bin, min_z)
                first_bot = np.append(first_bot, z_x[0, -1])

                UU = UUwindow/(int(nblock/2)*fs)
                Exx = UU[:,1:]/ (np.square((2*np.pi*fwindow[1:])))
                VV = VVwindow/(int(nblock/2)*fs)
                Eyy = VV[:,1:]/ (np.square((2*np.pi*fwindow[1:])))

                Eh = Exx+Eyy

                if np.isnan(np.nanmean(np.nanmean(Eh, axis=0))):
                    #print("All NaNs!!")
                    if nancounter==0:
                        #plt.loglog(fwindow, np.transpose(VVwindow))
                        #plt.plot(np.transpose(u_y))
                        #plt.loglog(fwindow[1:], np.transpose(Eh))
                        print(make_vel_spectrum(u_y, fs))

                    temp1 = np.expand_dims(np.array(E_x), axis=0)
                    temp2 = np.expand_dims(np.array(E_y), axis=0)
                    temp3 = np.append(temp1, temp2, axis=0)
                    nanstorer.append(temp3)
                    #if nancounter<10:
                    #    plt.figure()
                    #    plt.plot(E_x)
                    #    plt.plot(E_y)
                    nancounter+=1

                [Eh_Eric1, Eh_Eric2, Eh_Eric3, Eh_Eric4] = depth_correct_Eric(Eh, fwindow[1:], z_x, prof_speed, nblock, Cmax, fs)
                if np.isnan(Eh_Eric4).all():
                    continue
                    #Then what happens is the float never moved?
                    #What if we just try applying the depth correction
                    k_array = np.square(2*np.pi*fwindow[1:])/9.8
                    k_mat = np.tile(k_array, (Eh.shape[0], 1))

                    #This is def not the easiest way to do this
                    #z_mat = np.tile(em_z, (Eh.shape[0], 1))
                    z_mat = z_x
                    z_mat = np.tile(np.expand_dims(np.nanmedian(z_mat, axis=1), axis=1), (1, len(k_mat[0, :])))
                    #print(z_mat)


                    depth_fact = np.exp(2*k_mat*z_mat)
                    depth_fact[np.exp(k_mat*z_mat)>Cmax]=np.nan
                    #depth_fact[depth_fact>Cmax]=np.nan
                    Eh_Eric4 = Eh*depth_fact

                u_noise = 0 #This is just for testing...
                [lbound, ubound, nu] = get_spectral_uncertainity(E_x, E_y, Pef, u_noise, prof_speed, nblock, overlap, Cmax, fs)
                
                #Try propogating errors through and see if this has a different result
                Eh_lbound = Eh*lbound
                Eh_ubound = Eh*ubound
                #[_, _, _, Eh_Eric4_lbound] = depth_correct_Eric(Eh_lbound, fwindow[1:], z_x, prof_speed, nblock, Cmax, fs)
                #[_, _, _, Eh_Eric4_ubound] = depth_correct_Eric(Eh_ubound, fwindow[1:], z_x, prof_speed, nblock, Cmax, fs)


                #lbound = Eh_Eric4_lbound/Eh_Eric4
                #ubound = Eh_Eric4_ubound/Eh_Eric4

                #Average each profile spec and add to the storer
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    spec_store[counter, 0, :] = np.nanmean(Eh, axis=0)
                    spec_store[counter, 1, :] = np.nanmean(Eh_Eric4, axis=0)

                    time_store[counter] = np.nanmean(mlt_efr)
                    prof_store[counter] = float_id+"_"+str(prof_num)

                    uncertainty_store[counter, :] = np.array([lbound, ubound])
                    loc_store[counter, :] = np.array([avg_lat, avg_lon])
                    if prof_num%2==0:
                        #Then it's even and its an up profile
                        up_down_store[counter] = 1
                    else:
                        up_down_store[counter] = 0



                counter+=1



        # Now store each one in a big array for sorting etc later
        if float_id_counter==0:
            big_spec_store=spec_store
            big_uncertainty_store = uncertainty_store
            big_time_store = time_store
            big_up_down_store = up_down_store
            big_prof_store = prof_store
            big_loc_store = loc_store
        else:
            big_spec_store = np.append(big_spec_store, spec_store, axis=0)
            big_uncertainty_store = np.append(big_uncertainty_store,uncertainty_store, axis=0)
            big_time_store = np.append(big_time_store, time_store)
            big_up_down_store = np.append(big_up_down_store, up_down_store)
            big_prof_store = np.append(big_prof_store, prof_store)
            big_loc_store = np.append(big_loc_store, loc_store, axis=0)



        float_id_counter+=1

     #Getting rid of the profiles where minimum depth was below 20m
    kill = np.where(big_spec_store[:, 0, 5]==0)


    spec_store_shallow = np.delete(big_spec_store, kill[0], axis=0)
    time_store_shallow = np.delete(big_time_store, kill[0], axis=0)
    up_down_store_shallow = np.delete(big_up_down_store, kill[0], axis=0)
    prof_store_shallow = np.delete(big_prof_store, kill[0], axis=0)
    uncertainty_store_shallow = np.delete(big_uncertainty_store, kill[0], axis=0)
    loc_store_shallow = np.delete(big_loc_store, kill[0], axis=0)

    #Sort all the arrays by time
    out = zip(spec_store_shallow, up_down_store_shallow, time_store_shallow, prof_store_shallow)
    out2 = zip(uncertainty_store_shallow, time_store_shallow, loc_store_shallow)
    #list(out)[0]
    sorted_array = sorted(out, key=lambda tup: tup[2])
    sorted_array2 = sorted(out2, key=lambda tup: tup[1])

    unzipped = ([ a for a,b,c,d in sorted_array ], [ b for a,b,c,d in sorted_array ], [c for a,b,c,d in sorted_array], [d for a,b,c,d in sorted_array])
    unzipped2 = ([ a for a,b,c in sorted_array2 ], [ b for a,b,c in sorted_array2 ], [c for a,b,c in sorted_array2])

    spec_store_sorted = np.array(unzipped[0])
    up_down_store_sorted = np.array(unzipped[1])
    time_store_sorted = np.array(unzipped[2])
    prof_store_sorted = np.array(unzipped[3])

    uncertainty_store_sorted = np.array(unzipped2[0])
    loc_store_sorted = np.array(unzipped2[2])
    
    return(spec_store_sorted, time_store_sorted, uncertainty_store_sorted, up_down_store_sorted, loc_store_sorted, prof_store_sorted, fwindow[1:])