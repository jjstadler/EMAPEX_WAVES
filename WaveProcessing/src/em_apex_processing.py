
"""
Functions for use for processing 1Hz EM-APEX raw data

"""
import datetime
import numpy as np
import matplotlib.pyplot as plt


def em_offset(N,t,nstep,navg,E1,E2,Hx,Hy):

              
    """
    Input:
        N: Order of polynomial fit
        t: time series corresponding to E1, E2
        nstep: step of window movement for overlapping windows
        navg: window of each individual window
        E1: raw voltage on channel 1
        E2: raw voltage on channel 2
        Hx: Magnetometer x-direction
        Hy: Magnetometer y-direction
        
    
    Output:
        e1off: offset voltage on E1
        e2off: Offset voltage on E2
        e1fit: fitted E1 voltage
        e2fit: fitted E2 voltage
        anghxhy: orientation of magnetometer
        
    
    """
    #EM_OFFSET_EST is used to do the least-squarcs fit within small windows,
    #in order to estimate the time series of voltage offset
    #The concept is the same as the default data processing method
    #but focus on estimating the offset


    #Adapted to python from matlab codes are written by Dr. J-Y Hsu in National Taiwan University
    #on 11/23/2021



    nout=0;
    nblock = (len(t)//nstep)-1
    e1off = np.empty((len(t), nblock))
    e2off = np.empty((len(t), nblock))
    e1fit = np.empty((len(t), nblock))
    e2fit = np.empty((len(t), nblock))
    anghxhy = np.empty((len(t), nblock))
    resids = np.empty((len(t), nblock))

    e1off.fill(np.nan)
    e2off.fill(np.nan)
    e1fit.fill(np.nan)
    e2fit.fill(np.nan)
    anghxhy.fill(np.nan)
    resids.fill(np.nan)





    for i1 in range(0, len(t)-navg, nstep):
        j = list(range(i1, i1+navg))
        tj = t[j];
        e1 = E1[j];
        e2 = E2[j];
        ##
        hx = Hx[j];
        hy = Hy[j];
        hx = hx - np.nanmean(hx)
        hx = (hx / np.std(hx)) / np.sqrt(2)
        hy = hy - np.nanmean(hy)
        hy = (hy / np.std(hy)) / np.sqrt(2)

        angs=np.arctan2(hx,hy);
        anghxhy[j,nout]=angs;      
        ##
        con = np.ones(tj.shape);
        trnd = tj - tj[1]; # trend
        trnd = trnd / (max(trnd)-min(trnd)) 
        BASIS=np.stack((hx, hy, con), axis=1)  
        for k in range(1, N+1):
            BASIS=np.concatenate((BASIS,np.expand_dims(trnd**k, axis=1)), axis=1)        

        [COEF1, resid1, rank, s] = np.linalg.lstsq(BASIS, e1, rcond=None) # least squares fit
        [COEF2, resid2, rank, s] = np.linalg.lstsq(BASIS, e2, rcond=None)
    
        #Calculate the norm2 error
        #lstsq returns the sum of the squares of the individual residuals
        resid = np.sqrt(resid1+resid2)
        #Put residual into an array of same value with length nstep
        resid = np.ones(navg)*resid 
        
        #plt.figure()
        #plt.plot(np.matmul(BASIS, COEF2))
        #plt.plot(e2)
        
        e1b=con*COEF1[2]; #Fitted result of background constant
        e2b=con*COEF2[2];    
        e1f=hx*COEF1[0]+hy*COEF1[1]+con*COEF1[2]; #Fitted result of the curve
        e2f=hx*COEF2[0]+hy*COEF2[1]+con*COEF2[2];

        for k in range(1, N+1):
            e1b=e1b+COEF1[2+k]*trnd**k; #the revised results of voltage offset by adding trend
            e2b=e2b+COEF2[2+k]*trnd**k;
            e1f=e1f+COEF1[2+k]*trnd**k;
            e2f=e2f+COEF2[2+k]*trnd**k;        

        e1off[j,nout]=e1b;
        e2off[j,nout]=e2b;
        e1fit[j,nout]=e1f;
        e2fit[j,nout]=e2f;
        resids[j,nout] = resid;


        nout +=1

    return e1off, e2off, e1fit, e2fit, anghxhy, resids




def year_fraction(date):
    """
    Helper function for converting to fractional date from datetime
    
    Input:
        date: python datetime data type
        
    Output:
        fractional year (e.g. 2017.24)
    """
    start = datetime.date(date.year, 1, 1).toordinal()
    year_length = datetime.date(date.year+1, 1, 1).toordinal() - start
    return date.year + float(date.toordinal() - start) / year_length


