#!/usr/bin/env python

"""
[summary]
"""

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import gamma_err as ger
import two_tail_err as tte

"""
[area for file details]
"""

def main():
    """
    [method summary]
    """
    # scope variables
    filename = r'./cluster.csv'
    to_rm = 'dr2_radial_velocity'
    binsize = 100

    # reading file for specified location
    dataframe = pd.read_csv(filename)

    # debug
    print(len(dataframe))
    # print(dataframe.head())
    
    # check output with DR2 data removal
    if to_rm in dataframe.columns:
        temp = dataframe.dropna()
        print(len(temp))    # debug
        dataframe.drop(['dr2_radial_velocity', 'dr2_radial_velocity_error'], axis='columns', inplace=True)
    
    # remove rows with missing data
    dataframe.dropna(inplace=True)
    print(len(dataframe))   # debug

    # parameter variables for gamma errors
    p, v = 0.60, 0
    dataframe = ger.remove_error(data=dataframe, param='ra_error', bins=binsize, pthresh=p, verbose=v)
    dataframe = ger.remove_error(data=dataframe, param='dec_error', bins=binsize, pthresh=p, verbose=v)
    dataframe = ger.remove_error(data=dataframe, param='pmra_error', bins=binsize, pthresh=p, verbose=v)
    dataframe = ger.remove_error(data=dataframe, param='pmdec_error', bins=binsize, pthresh=p, verbose=v)
    dataframe = ger.remove_error(data=dataframe, param='parallax_error', bins=binsize, pthresh=p, verbose=v)
    print(len(dataframe))   # debug

    p, v = 2, 0
    dataframe = tte.chop_tails(data=dataframe, param='ra', bins=binsize, factor=p, verbose=v)
    dataframe = tte.chop_tails(data=dataframe, param='dec', bins=binsize, factor=p, verbose=v)
    dataframe = tte.chop_tails(data=dataframe, param='pmra', bins=binsize, factor=p, verbose=v)
    dataframe = tte.chop_tails(data=dataframe, param='pmdec', bins=binsize, factor=p, verbose=v)
    dataframe = tte.chop_tails(data=dataframe, param='parallax', bins=binsize, factor=p, verbose=v)
    print(len(dataframe))   # debug

    x = np.array(dataframe['pmra'])
    y = np.array(dataframe['pmdec'])
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    sc = plt.scatter(dataframe['pmra'], dataframe['pmdec'], marker='.', c=z)
    plt.colorbar(sc)
    plt.show()
    plt.hist(z, bins=100, histtype='step')
    plt.axvline(np.mean(z) + np.std(z))
    plt.show()
    dataframe['kde'] = z
    plt.hist(dataframe['parallax'], bins=100, histtype='step')
    plt.show()
    dataframe = dataframe[dataframe['kde'] >= np.mean(z) + np.std(z)]
    x = np.array(dataframe['pmra'])
    y = np.array(dataframe['pmdec'])
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    sc = plt.scatter(dataframe['pmra'], dataframe['pmdec'], marker='.', c=z)
    plt.colorbar(sc)
    plt.show()
    dataframe = dataframe[dataframe['parallax'] >= 0]
    plt.gca().invert_yaxis()
    plt.scatter(dataframe['bp_rp'], dataframe['phot_g_mean_mag'], marker='.', c=dataframe['kde'])
    plt.show()
    print(len(dataframe))

if __name__=='__main__':
    main()