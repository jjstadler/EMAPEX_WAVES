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
        
    elif len(spec.shape)==2:
        #Then 2d
        num_specs = spec.shape[1]
        swhs = np.zeros(num_specs)
        for i in range(num_specs):
            spec_temp = spec[:, i]
            ##Need to remove Nan's
            real_inds = np.argwhere(~np.isnan(spec_temp))
            spec_temp = spec_temp[real_inds]
            f_temp = f[real_inds]
            swh = 4*np.sqrt(np.trapz(spec_temp[:, 0], x=f_temp[:, 0]))
            swhs[i] = swh
    
    else:
        raise Exception("Array containing spectra to integrate must be either 1D or 2D")
        
    return(swhs)



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
    z_mat = np.tile(np.expand_dims(np.nanmean(z_mat, axis=1), axis=1), (1, len(k_mat[0, :])))
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
    kWT = 1*k_mat*0.1*(nblock/fs)
    #Tile to be the right size
    
    
    G = np.square((np.square(np.pi)/(np.square(kWT/2)+np.square(np.pi)))*(np.sinh(kWT/2)/(kWT/2)))
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
    G2 = d1*np.square(kWT)+d2*kWT+d3
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
    
    Input:
        spec: numpy array containing spectra to add tail to. Should either be dimension (m x n), or (n,)
            where m is the # of spectra to add tail to, and n is the number of points in the spectra
            
        f: frequency bins corresponding to the spectra. Should be dimension (n,)
    """
    
    new_specs = np.copy(spec)
    
    for ind in range(spec.shape[0]):
        #Get last non-NaN value in the spectrum
        end_ind = np.where(np.isnan(spec[ind, :]))[0][0]-1
        f_end = f[end_ind]
        E_end = spec[ind, end_ind]
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



def get_spectral_uncertainity(E_x, E_y, Pef, u_noise, prof_speed, nblock, overlap, Cmax, fs):
    """
    This funciton outputs a range of possible spectrum based on a normal distribution of velocity noise
    
    TO DO: This takes a heck of a lot of inputs...Maybe think about how to streamline this?
    
    Input:
    
    
    Output:
        spec_array -
    """
    spec_array = []
    npoints = len(E_x)
    n_iterations=50
    #TO DO: This frequency bin size should not be hard-coded
    spec_store = np.zeros((n_iterations, 2, (nblock/2)-1))


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
                spec_array[counter, 0, :] = np.nanmean(Eh_Eric4, axis=0) 

    return(spec_array)