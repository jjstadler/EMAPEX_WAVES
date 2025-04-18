
def exp_func(x, a, k, c):
     return a * np.log(-k * x) + c

def residual_waveheight(all_profs):
    """
    Inputs:
        z: Depth (>0)
        Verr: velocitiy residuals for a profile
    Output:
        Hs: Waveheight value for a profile
    """


    err1 = all_profs['Verr1']
    err2 = all_profs['Verr2']

    z = all_profs['Pef']

    for prof in err1.shape
    curve_fit(func, z, err1)