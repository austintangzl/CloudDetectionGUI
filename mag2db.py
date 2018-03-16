# This function will convert magnitude to decibels
# By Austin Tang 10 Jan 2018
#
# Usage:

import numpy as np


def mag2db(mag):
    if np.isscalar(mag) is False:
        db = []         # Initalise the list
        for i in range(len(mag)):
            if mag[i] < 0:
                mag[i] = np.nan

            db.append(20 * np.log10(mag[i]))
        db = np.array(db)
    elif np.isscalar(mag) is True:
        if mag < 0:
            mag = np.nan

        db = 20 * np.log10(mag)

    return db
