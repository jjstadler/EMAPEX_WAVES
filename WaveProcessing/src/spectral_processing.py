"""
spectral_processing_tools.py

This module contains the code for spectral processing of velocity measurements for measuring
ocean surface waves by a vertically profiling subsurface float

Tools are used by both the wave simulations code and the 1Hz processing code


Note: Maybe this should really be structured differently --> Create an object that's a velocity timeseries, and then 
that has all the different processing steps as different methods, but I don't feel like figuring out how to get all that workign right now
"""


import numpy as np

#Reshape the data into 120s chunks etc for taking spectra
#Here I'm doing it with a 50% overlap

def reshape_u(u, em_z, nblock, overlap, fs):
    """
    function for take a 1d velocity times series and reshaping it into windows for taking spectra
    
    
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


    #Rescale
    factor = np.sqrt(np.var(uwindow)/np.var(uwindowtaper))
    
    uwindowready = uwindowtaper*factor

    #Take fft
    Uwindow = np.fft.fft(uwindowready)
    fwindow = np.fft.fftfreq(uwindowready.shape[-1], d=1/fs)
    fwindow=fwindow[:int(w/2)]

    Uwindow = np.delete(Uwindow,  np.s_[int(w/2):], -1)


    #Take power spectra
    UUwindow = np.real(Uwindow * np.conj(Uwindow))

    
    return(UUwindow, fwindow)



def sig_wave_height(f, spec):
    """
    This function  calculates the significant wave height from a energy density spectrum, with frequencies f.
    
    Input
        f: numpy array of frequencies
        spec: energy density spectra to integrate
        
        
    Output
        swh: Significant Wave Height calculated by integrating the input spectra
    
    To Do: Does this work with 2d Spec v
    """
    
    
    ##Need to remove Nan's
    num_specs = spec.shape[1]
    print(spec.shape)
    swhs = np.zeros(num_specs)
    for i in range(num_specs):
        spec_temp = spec[:, i]
        real_inds = np.argwhere(~np.isnan(spec_temp))
        spec_temp = spec_temp[real_inds]
        f_temp = f[real_inds]
        swh = 4*np.sqrt(np.trapz(spec_temp[:, 0], x=f_temp[:, 0]))
        swhs[i] = swh
    return(swhs)



def depth_correct_Eric(Eh, fwindow, em_z, nblock, Cmax, fs):
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
    z_mat = np.tile(np.expand_dims(np.nanmean(z_mat, axis=1), axis=1), (1, len(k_mat[0, :])))
    #print(np.expand_dims(np.nanmean(z_mat, axis=2), axis=2))


    
    depth_fact = np.exp(2*k_mat*z_mat)
    depth_fact[np.exp(k_mat*z_mat)>Cmax]=np.nan
    #depth_fact[depth_fact>Cmax]=np.nan
    Eh_out = Eh*depth_fact

    #### Now try applying the motion correction gain adjustment from D'Asaro 2015

    #First need to tile prof_speed
    #prof_speed = np.tile(np.expand_dims(prof_speed, axis=1), (1, Eh.shape[1]))
    #kWT2 = 1*k_mat*prof_speed*(nblock/fs)
    kWT = 1*k_mat*0.1*(nblock/fs)
    #Tile to be the right size
    
    
    #G = np.square((np.square(np.pi)/(np.square(kWT/2)+np.square(np.pi)))*(np.sinh(kWT/2)/(kWT)))
    G_mod = np.square((1/(2*np.pi))*np.sinh(kWT/2)/(np.power(kWT/(2*np.pi), 3)+kWT/(2*np.pi)))
    #G_mod2 = np.square((1/(2*np.pi))*np.sinh(kWT2/2)/(np.power(kWT2/(2*np.pi), 3)+kWT2/(2*np.pi)))
    
    #print(np.nanmean(G_mod2/G_mod))


    Eh_G1 = Eh_out/G_mod
    #print(np.nanmean(np.nanmean(G_mod, axis=0), axis=0))

    ### Try the 1s sampling correction
    #These are Andy's numbers
    #d1 = 0.004897
    #d2 = 0.033609
    #d3 = 0.999897
    
    d1 = 0.001642
    d2 = 0.018517
    d3 = 0.999528
    G2 = d1*np.square(kWT)+d2*kWT+d3
    Eh_G2 = Eh_G1/G2

    ###Now the final correction
    dw = 2*np.pi/nblock
    omega_mat = np.tile(2*np.pi*fwindow, (Eh.shape[0], 1))
    h=0.25
   
    #G3 = 1/np.square(1+2*h*(np.cosh((2*omega*dw+np.square(dw)*z_mat)/9.8)-1))
    G3 = 1/(1+2*h*(np.cosh((z_mat*np.square(omega_mat+dw)-np.square(omega_mat))/9.8)-1))
    Eh_G3 = Eh_G2*G3
    
    
    
    
    return(Eh_out, Eh_G1, Eh_G2, Eh_G3)