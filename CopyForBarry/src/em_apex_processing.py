
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
        trnd = tj - tj[0]; # trend
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


def correct_magnetometer(HX, HY):
    
    """
    Function to correct the hard/soft magnetometer offsets.
    """
    
    hx = HX-np.nanmean(HX)
    hy = HY-np.nanmean(HY)

    hx = (hx / np.std(hx)) / np.sqrt(2)
    hy = (hy / np.std(hy)) / np.sqrt(2)

    X = np.expand_dims(np.transpose(hx), axis=1)
    Y = np.expand_dims(np.transpose(hy), axis=1)
    F = np.ones_like(X)
    # Formulate and solve the least squares problem ||Ax - b ||^2
    A = np.hstack([X**2, X * Y, Y**2, X, Y])
    b = np.ones_like(X)
    x = np.linalg.lstsq(A, b)[0].squeeze()


    plt.scatter(X, Y, label='Raw Magnetometer')

    # Plot the least squares ellipse
    x_coord = np.linspace(-1.5,1.5,300)
    y_coord = np.linspace(-1.5,1.5,300)
    X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
    Z_coord = x[0] * X_coord ** 2 + x[1] * X_coord * Y_coord + x[2] * Y_coord**2 + x[3] * X_coord + x[4] * Y_coord
    plt.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('k'), linewidths=2)
    plt.xlabel('X')
    plt.ylabel('Y')


    A=x[0]
    B=x[1]
    C=x[2]
    D=x[3]
    E=x[4]
    F=-1

    #eqn for theta
    theta = np.arctan((1/x[1])*(x[2]-x[0]-np.sqrt((x[0]-x[2])**2+x[1]**2)))

    #eqn for major and minor axes of ellipse
    a = - np.sqrt(2*(A*(E**2)+C*(D**2)-B*D*E+((B**2)-4*A*C)*F)*((A+C)+np.sqrt(((A-C)**2)+(B**2))))/((B**2)-4*A*C)
    b = - np.sqrt(2*(A*(E**2)+C*(D**2)-B*D*E+((B**2)-4*A*C)*F)*((A+C)-np.sqrt(((A-C)**2)+(B**2))))/((B**2)-4*A*C)

    #eqn for hard iron offset
    x0 = (2*C*D - B*E)/(B**2 - 4*A*C)
    y0 = (2*A*E - B*D)/(B**2 - 4*A*C)


    ellipse_coords = np.transpose(np.hstack([X, Y]))


    #Apply hard iron correction
    ellipse_coords[0, :] = ellipse_coords[0, :]-x0
    ellipse_coords[1, :] = ellipse_coords[1, :]-x0


    #Apply rotation matrix
    R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    ellipse_out=np.matmul(R, ellipse_coords)

    #Rescale elipse
    ellipse_out[0, :] = ellipse_out[0, :]/a
    ellipse_out[1, :] = ellipse_out[1, :]/b


    #Rotate back to correct orientation
    R2 = np.array([[np.cos(-theta), np.sin(-theta)], [-np.sin(-theta), np.cos(-theta)]])
    ellipse_out2=np.matmul(R2, ellipse_out)

    hx_fixed = ellipse_out2[0, :]
    hy_fixed = ellipse_out2[1, :]
    
    return(hx_fixed, hy_fixed)
