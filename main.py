#!/usr/bin/env python

"""
[summary]
"""

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import gaussian_kde

import gamma_err as ger
import two_tail_err as tte
import visualize as vis

"""
[area for file details]
"""

def main():
    """
    [method summary]
    """
    # scope variables
    filename = r'./ic4665.csv'
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
    p, v = 0.68, 0
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

    # vis.generate_plots(dataframe)

    x = np.array(dataframe['pmra'])
    y = np.array(dataframe['pmdec'])
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    max_x, max_y = x[np.where(z == max(z))], y[np.where(z == max(z))]
    sc = plt.scatter(dataframe['pmra'], dataframe['pmdec'], marker='.', c=z)
    plt.axvline(max_x, color='chocolate', linestyle='--')
    plt.axhline(max_y, color='chocolate', linestyle='--')
    plt.colorbar(sc)
    plt.show()

    plt.plot(x, y, 'c.')
    plt.axvline(max_x, color='chocolate', linestyle='--')
    plt.axhline(max_y, color='chocolate', linestyle='--')
    plt.show()

    dist = list()
    for i in range(len(dataframe)):
        x1, y1 = x[i], y[i]
        dist.append(math.dist([x1, y1], [max_x, max_y]))

    dataframe['dist'] = dist
    dataframe = ger.remove_error(data=dataframe, param='dist', bins=binsize, pthresh=.99, verbose=1)

    x = np.array(dataframe['pmra'])
    y = np.array(dataframe['pmdec'])
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    max_x, max_y = x[np.where(z == max(z))], y[np.where(z == max(z))]
    sc = plt.scatter(dataframe['pmra'], dataframe['pmdec'], marker='.', c=z)
    plt.axvline(max_x, color='chocolate', linestyle='--')
    plt.axhline(max_y, color='chocolate', linestyle='--')
    plt.colorbar(sc)
    plt.show()

    vis.generate_plots(dataframe)
    
    """ v OLD CODE v """
    # plt.hist(z, bins=100, histtype='step')
    # plt.axvline(np.mean(z) + np.std(z))
    # plt.show()
    # dataframe['kde'] = z
    # plt.hist(dataframe['parallax'], bins=100, histtype='step')
    # plt.show()
    # dataframe = dataframe[dataframe['kde'] >= np.mean(z) + 2 * np.std(z)]
    # x = np.array(dataframe['pmra'])
    # y = np.array(dataframe['pmdec'])
    # xy = np.vstack([x, y])
    # z = gaussian_kde(xy)(xy)
    # sc = plt.scatter(dataframe['pmra'], dataframe['pmdec'], marker='.', c=z)
    # plt.colorbar(sc)
    # plt.show()
    # dataframe = dataframe[dataframe['parallax'] >= 0]
    # plt.gca().invert_yaxis()
    # plt.scatter(dataframe['bp_rp'], dataframe['phot_g_mean_mag'], marker='.', c=dataframe['kde'])
    # plt.show()
    # print(len(dataframe))

if __name__=='__main__':
    main()