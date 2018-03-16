#from osgeo import gdal
import osgeo.gdal as gdal
from osgeo.gdalconst import *
import numpy
import struct
import sys
import os
import subprocess
from extractKML import *
import utm
from PIL import ImageTk, Image
import cv2
import numpy as np
import tkinter as tk


def readKML(kmlfilename):
    # Read the KML File
    try:
        for polygons in readPoly(kmlfilename):
            polygons, desc = polygons
            print(polygons)
            outputText = '\nLoaded KML/KMZ file'
    except:
        print('Unable to load KML/KMZ file')
        outputText = '\nUnable to load KML/KMZ file'

    return outputText, polygons


def createNpArray(imgfilename, polygons, AOISelection):
    # Read in the geotiff file
    dataset = gdal.Open(imgfilename, gdal.GA_ReadOnly)

    outputText = ''
    if not dataset:
        print('Dataset empty')
        outputText = outputText + '\nDataset empty'
    else:
        # GDAL affine transform parameters, According to gdal documentation xoff/yoff are image left corner, a/e are pixel wight/height and b/d is rotation and is zero if image is north up.
        xoff, a, b, yoff, d, e = dataset.GetGeoTransform()

        if AOISelection is 1:
            polyarray = [[None for _ in range(2)] for _ in range(len(polygons))]  # Declare a polygon array

            # Convert the latitude and longitude from the kml/kmz file into pixel coordinates
            for i in range(len(polygons)):
                lon = polygons[i][0]
                lat = polygons[i][1]
                [easting, northing, zoneNum, zoneLet] = utm.from_latlon(lat, lon)
                pixelX = int(round((easting - xoff) / a))
                pixelY = int(round((northing - yoff) / e))

                polyarray[i][0] = pixelX
                polyarray[i][1] = pixelY

            # Change the poly array list to np.array
            npArrayPoly = numpy.array(polyarray, numpy.int32)
            npArrayPoly = npArrayPoly.reshape((-1, 1, 2))  # Reshape the array

            # Get the image size
            row_len = dataset.RasterYSize
            col_len = dataset.RasterXSize

            # Create the cloud_map to work with
            cloud_map = np.zeros(shape=(row_len, col_len))

            # Convert the Polygon array into a mask file
            aoiFinal = cv2.fillPoly(cloud_map, [npArrayPoly], 1)

            outputText = outputText + '\nAOI Mask Created'
        elif AOISelection is 2:
            # Get the first band
            band1 = dataset.GetRasterBand(1).ReadAsArray()

            # Get the image size
            xlen = dataset.RasterXSize
            ylen = dataset.RasterYSize

            for i in range(ylen):
                findpixel = np.nonzero(band1[i, :])  # Find the first row
                findpixel = findpixel[0]
                if findpixel.any():
                    polygons.append([findpixel[0], i])
                    break

            for i in reversed(range(xlen)):
                findpixel = np.nonzero(band1[:, i])  # Find the first row
                findpixel = findpixel[0]
                if findpixel.any():
                    polygons.append([i, findpixel[0]])
                    break

            for i in reversed(range(ylen)):
                findpixel = np.nonzero(band1[i, :])  # Find the first row
                findpixel = findpixel[0]
                if findpixel.any():
                    polygons.append([findpixel[0], i])
                    break

            for i in range(xlen):
                findpixel = np.nonzero(band1[:, i])  # Find the first row
                findpixel = findpixel[0]
                if findpixel.any():
                    polygons.append([i, findpixel[0]])
                    break

            # Complete the loop in the polygon
            polygons.append(polygons[0])

            polyarray = [[None for _ in range(2)] for _ in range(len(polygons))]  # Declare a polygon array

            # Convert the latitude and longitude from the kml/kmz file into pixel coordinates
            for i in range(len(polygons)):
                polyarray[i][0] = polygons[i][0]
                polyarray[i][1] = polygons[i][1]

            # Change the poly array list to np.array
            npArrayPoly = numpy.array(polyarray, numpy.int32)
            npArrayPoly = npArrayPoly.reshape((-1, 1, 2))  # Reshape the array

            # Get the image size
            row_len = dataset.RasterYSize
            col_len = dataset.RasterXSize

            # Create the cloud_map to work with
            cloud_map = np.zeros(shape=(row_len, col_len))

            # Convert the Polygon array into a mask file
            aoiFinal = cv2.fillPoly(cloud_map, [npArrayPoly], 1)

            # JUST DETECT WHOLE IMAGE LA
            aoiFinal = np.ones(shape=(row_len, col_len))

            outputText = outputText + '\nAOI Mask Created'
        if AOISelection is 3:
            # Get all three (3) bands
            try:
                redband = dataset.GetRasterBand(1).ReadAsArray()
            except IOError:
                print('The red band cannot be opened')
            except:
                print('Unable to read red band')
                return
            try:
                greenband = dataset.GetRasterBand(2).ReadAsArray()
            except IOError:
                print('The greenband cannot be opened')
            except:
                print('Unable to read green band')
                return
            try:
                blueband = dataset.GetRasterBand(3).ReadAsArray()
            except IOError:
                print('The blueband cannot be opened')
            except:
                print('Unable to read blue band')
                return

            # Magenta R = >140, G = <50, B = >140
            ind1 = np.asarray(np.where(redband >= 140))
            ind2 = np.asarray(np.where(greenband <= 50))
            ind3 = np.asarray(np.where(blueband >= 140))

            # Get the image size
            row_len = dataset.RasterYSize
            col_len = dataset.RasterXSize

            # Create/Initialise zero array matrix
            aoi1 = np.zeros(shape=(row_len, col_len), dtype=np.uint16)
            aoi2 = np.zeros(shape=(row_len, col_len), dtype=np.uint16)
            aoi3 = np.zeros(shape=(row_len, col_len), dtype=np.uint16)

            # Make values in aoi 100 for those detected colour values
            for i in range(len(ind1[0])):
                aoi1[ind1[0, i], ind1[1, i]] = 100
            for i in range(len(ind2[0])):
                aoi2[ind2[0, i], ind2[1, i]] = 100
            for i in range(len(ind3[0])):
                aoi3[ind3[0, i], ind3[1, i]] = 100

            # Combine all the values into a detected aoi array matrix
            combined_aoi = np.multiply(np.multiply(aoi1, aoi2), aoi3)

            # Clear unused variables
            del ind1, ind2, ind3, aoi1, aoi2, aoi3

            # Initialise detection arrays
            test1 = np.zeros(shape=(row_len, col_len), dtype=np.uint16)
            test2 = np.zeros(shape=(row_len, col_len), dtype=np.uint16)
            test3 = np.zeros(shape=(row_len, col_len), dtype=np.uint16)
            test4 = np.zeros(shape=(row_len, col_len), dtype=np.uint16)

            thickness = 5   # Set the thickness value of the detection

            for n in range(row_len):
                ind = np.asarray(np.where(combined_aoi[n, :] >= 1))
                if ind.size != 0:
                    test1[n, ind[0][0]+thickness:] = 100
                    test2[n, :ind[0][-1]-thickness] = 100

            for n in range(col_len):
                ind = np.asarray(np.where(combined_aoi[:, n] >= 1))
                if ind.size != 0:
                    test3[ind[0][0]+thickness:, n] = 100
                    test4[:ind[0][-1]-thickness, n] = 100

            aoiFinal = np.multiply(np.multiply(test1, test2), np.multiply(test3, test4))
            aoiFinal[aoiFinal > 0] = 1


            outputText = outputText + '\nAOI Mask Created'

    return aoiFinal, dataset, row_len, col_len, outputText


def createCloudMap(row_len, col_len):
    cloud_map = np.zeros(shape=(row_len, col_len), dtype=np.uint16)
    outputText = '\nCloud Map created'
    return cloud_map, outputText


def getPanchroBand(dataset):
    panchroband = dataset.GetRasterBand(1).ReadAsArray()
    combinedmapvalues = np.uint16(panchroband)
    return combinedmapvalues


def getMultiSpecBands(dataset):
    redband = dataset.GetRasterBand(1).ReadAsArray()
    greenband = dataset.GetRasterBand(2).ReadAsArray()
    blueband = dataset.GetRasterBand(3).ReadAsArray()
    combinedmapvalues = np.uint16(redband) + np.uint16(greenband) + np.uint16(blueband)
    return combinedmapvalues


def getRGBBands(dataset):
    redband = dataset.GetRasterBand(1).ReadAsArray()
    greenband = dataset.GetRasterBand(2).ReadAsArray()
    blueband = dataset.GetRasterBand(3).ReadAsArray()
    return np.array(redband), np.array(greenband), np.array(blueband)


def getHSVBands(dataset):
    redband = dataset.GetRasterBand(1).ReadAsArray()
    greenband = dataset.GetRasterBand(2).ReadAsArray()
    blueband = dataset.GetRasterBand(3).ReadAsArray()

    # "Normalize"

    # Get the image size
    row_len = dataset.RasterYSize
    col_len = dataset.RasterXSize

    combinedmapvalues = np.zeros(shape=(3, row_len, col_len), dtype=np.uint8)
    combinedmapvalues[0, ...] = np.uint8(redband)
    combinedmapvalues[1, ...] = np.uint8(greenband)
    combinedmapvalues[2, ...] = np.uint8(blueband)

    # Transpose the matrix to prepare for conversion
    combinedmapvalues = np.transpose(combinedmapvalues, (1, 2, 0))

    # Convert to HSV
    combinedmapvalues = cv2.cvtColor(combinedmapvalues, cv2.COLOR_RGB2HSV)

    # Transpose it back
    combinedmapvalues = np.transpose(combinedmapvalues, (2, 0, 1))

    # Pull out Saturation
    combinedmapvalues = combinedmapvalues[1, ...]

    return combinedmapvalues


def getHSLBands(dataset):
    redband = dataset.GetRasterBand(1).ReadAsArray()
    greenband = dataset.GetRasterBand(2).ReadAsArray()
    blueband = dataset.GetRasterBand(3).ReadAsArray()

    # "Normalize"

    # Get the image size
    row_len = dataset.RasterYSize
    col_len = dataset.RasterXSize

    combinedmapvalues = np.zeros(shape=(3, row_len, col_len), dtype=np.uint8)
    combinedmapvalues[0, ...] = np.uint8(redband)
    combinedmapvalues[1, ...] = np.uint8(greenband)
    combinedmapvalues[2, ...] = np.uint8(blueband)

    # Transpose the matrix for conversion
    combinedmapvalues = np.transpose(combinedmapvalues, (1, 2, 0))

    # Convert to HSL Space
    combinedmapvalues = cv2.cvtColor(combinedmapvalues, cv2.COLOR_RGB2HLS)

    # Transpose the maxtrix back
    combinedmapvalues = np.transpose(combinedmapvalues, (2, 0, 1))

    # Pull out Lightness?
    combinedmapvalues = combinedmapvalues[1, ...]

    return combinedmapvalues


def getYCrCbBands(dataset):

    redband = dataset.GetRasterBand(1).ReadAsArray()
    greenband = dataset.GetRasterBand(2).ReadAsArray()
    blueband = dataset.GetRasterBand(3).ReadAsArray()

    # "Normalize"

    # Get the image size
    row_len = dataset.RasterYSize
    col_len = dataset.RasterXSize

    combinedmapvalues = np.zeros(shape=(3, row_len, col_len), dtype=np.uint8)
    combinedmapvalues[0, ...] = np.uint8(redband)
    combinedmapvalues[1, ...] = np.uint8(greenband)
    combinedmapvalues[2, ...] = np.uint8(blueband)

    combinedmapvalues = np.transpose(combinedmapvalues, (1, 2, 0))

    # combinedmapvalues = cv2.cvtColor(combinedmapvalues, cv2.COLOR_RGB2HSV)
    # Convert to YCrCb Space
    combinedmapvalues = cv2.cvtColor(combinedmapvalues, cv2.COLOR_RGB2YCrCb)

    combinedmapvalues = np.transpose(combinedmapvalues, (2, 0, 1))

    Y = combinedmapvalues[0, ...]
    Cr = combinedmapvalues[1, ...]
    Cb = combinedmapvalues[2, ...]

    return np.array(Y), np.array(Cr), np.array(Cb)


def computeX(dataset, cloud_map, row_len, col_len, peak_reflec_gain, upper_thresh, color_scheme):
    # Consolidate the shit

    # Check how many raster bands there is
    if dataset.RasterCount == 3 or dataset.RasterCount == 4:
        if color_scheme == 'RGB':
            combinedmapvalues = getMultiSpecBands(dataset)
        elif color_scheme == 'HSL':
            combinedmapvalues = getHSVBands(dataset)  # Changed 12 March 2018
        elif color_scheme == 'YCrCb':
            [combinedmapvalues, Cr, Cb] = getYCrCbBands(dataset)  # Changed 14 March 2018
        # [redband, greenband, blueband] = getRGBBands(dataset)
    elif dataset.RasterCount == 1:
        combinedmapvalues = getPanchroBand(dataset)

    topPercent = 100 - upper_thresh
    N = int(np.round((col_len / 100) * topPercent))
    tempMat = np.sort(combinedmapvalues, 0)
    peak_reflec = np.mean(tempMat[(len(tempMat) - N):, :])
    print('peak_reflect1: %f' % peak_reflec)
    outputText = '\npeak_reflect1: %f' % peak_reflec
    tempMat = None

    # whiteness = 5

    poscheck = (0.04*128)+128
    negcheck = (-0.04*128)+128

    for x in range(row_len):
        temp = combinedmapvalues[x, :]
        # checkwhite = ((greenband[x, :]-whiteness <= redband[x, :]) & (redband[x, :] <= greenband[x, :]+whiteness)) & ((blueband[x, :]-whiteness <= redband[x, :]) & (redband[x, :] <= blueband[x, :]+whiteness))
        # ind = np.asarray(np.where((temp >= peak_reflec*peak_reflec_gain) & checkwhite))
        if color_scheme == 'RGB':
            ind = np.asarray(np.where(temp >= peak_reflec * peak_reflec_gain))
        elif color_scheme == 'HSL':
            ind = np.nan
        elif color_scheme == 'YCrCb':
            checkwhite = (((Cr[x, :] <= poscheck) & (Cr[x, :] >= negcheck)) & ((Cb[x, :] <= poscheck) & (Cb[x, :] >= negcheck)))
            ind = np.asarray(np.where((temp >= peak_reflec * peak_reflec_gain) & checkwhite))

        temp1 = np.copy(cloud_map[x, :])
        for n in range(len(ind[0])):
            temp1[ind[0, n]] = 1

        tempd = np.power(np.diff(temp), 2)
        m = np.mean(tempd)
        ind = np.asarray(np.where(tempd <= m * 0.7))
        temp3 = np.copy(cloud_map[x, :])
        for n in range(len(ind[0])):
            temp3[ind[0, n]] = 1

        cloud_map[x, :] = cloud_map[x, :] + np.multiply(temp1, temp3)
    print('x direction done')
    outputText = outputText + '\nx direction done'

    if color_scheme == 'RGB':
        Band1 = 0
        Band2 = 0
        Band3 = 0
    elif color_scheme == 'HSL':
        Band1 = 0
        Band2 = 0
        Band3 = 0
    elif color_scheme == 'YCrCb':
        Band1 = 0
        Band2 = Cr
        Band3 = Cb

    return cloud_map, combinedmapvalues, peak_reflec, outputText, Band1, Band2, Band3


def computeY(cloud_map, combinedmapvalues, col_len, peak_reflec_gain, peak_reflec, color_scheme, Band1, Band2, Band3):
    # whiteness = 5

    poscheck = (0.04 * 128) + 128
    negcheck = (-0.04 * 128) + 128

    for y in range(col_len):
        temp = combinedmapvalues[:, y]
        # checkwhite = ((greenband[:, y] - whiteness <= redband[:, y]) & (redband[:, y] <= greenband[:, y] + whiteness)) & ((blueband[:, y] - whiteness <= redband[:, y]) & (redband[:, y] <= blueband[:, y] + whiteness))
        if color_scheme == 'RGB':
            ind = np.asarray(np.where(temp >= peak_reflec * peak_reflec_gain))
        elif color_scheme == 'HSL':
            ind = np.nan
        elif color_scheme == 'YCrCb':
            checkwhite = (((Band2[:, y] <= poscheck) & (Band2[:, y] >= negcheck)) & ((Band3[:, y] <= poscheck) & (Band3[:, y] >= negcheck)))
            ind = np.asarray(np.where((temp >= peak_reflec * peak_reflec_gain) & checkwhite))

        temp1 = np.copy(cloud_map[:, y])
        for n in range(len(ind[0])):
            temp1[ind[0, n]] = 1

        tempd = np.power(np.diff(temp), 2)
        m = np.mean(tempd)
        ind = np.asarray(np.where(tempd <= m * 0.7))
        temp3 = np.copy(cloud_map[:, y])
        for n in range(len(ind[0])):
            temp3[ind[0, n]] = 1

        cloud_map[:, y] = cloud_map[:, y] + np.multiply(temp1, temp3)
    print('y direction done')
    outputText = '\ny direction done'

    return cloud_map, outputText


def erodeAndDilate(cloud_map, aoiFinal):
    for x in range(cloud_map.shape[0]):
        for y in range(cloud_map.shape[1]):
            if cloud_map[x, y] >= 1:
                cloud_map[x, y] = 1

    cloud_map = np.multiply(cloud_map, aoiFinal)

    # https: // www.packtpub.com / mapt / book / application_development / 9781785283932 / 2 / ch02lvl1sec24 / erosion - and -dilation

    # A Closing of 5x5 is done first to close all gaps in the detections
    kernel = np.ones((5, 5), np.uint8)
    cloud_map = cv2.dilate(cloud_map, kernel, iterations=1)
    cloud_map = cv2.erode(cloud_map, kernel, iterations=1)
    # Then a Opening of 2x2 is done to remove speckles in the detections
    kernel = np.ones((2, 2), np.uint8)
    cloud_map = cv2.erode(cloud_map, kernel, iterations=1)
    cloud_map = cv2.dilate(cloud_map, kernel, iterations=1)

    return cloud_map


def calculateCloudPercentage(cloud_map, aoiFinal):
    # Find the percentage of cloud cover
    ind1 = np.where(cloud_map > 0)
    ind2 = np.where(aoiFinal > 0)
    percent = (np.divide(len(ind1[0]), len(ind2[0]))) * 100
    print('The cloud cover is: %3.2f' % percent)
    outputText = '\nThe cloud cover is: %3.2f' % percent
    return percent, outputText


def Cloud_Detection_Algo(imgfilename, kmlfilename, reflec_gain, outputText):
    # Read the KML File
    try:
        for polygons in readPoly(kmlfilename):
            polygons, desc = polygons
            print(polygons)
            outputText = '\nLoaded KML/KMZ file'
    except:
        print('Unable to load KML/KMZ file')
        outputText = '\nUnable to load KML/KMZ file'

    # Read in the geotiff file
    dataset = gdal.Open(imgfilename, gdal.GA_ReadOnly)

    if not dataset:
        print('Dataset empty')
        outputText.insert(tk.END, '\nDataset empty')
    else:
        # GDAL affine transform parameters, According to gdal documentation xoff/yoff are image left corner, a/e are pixel wight/height and b/d is rotation and is zero if image is north up.
        xoff, a, b, yoff, d, e = dataset.GetGeoTransform()

        polyarray = [[None for _ in range(2)] for _ in range(len(polygons))]  # Declare a polygon array

        # Convert the latitude and longitude from the kml/kmz file into pixel coordinates
        for i in range(len(polygons)):
            lon = polygons[i][0]
            lat = polygons[i][1]
            [easting, northing, zoneNum, zoneLet] = utm.from_latlon(lat, lon)
            pixelX = int(round((easting - xoff) / a))
            pixelY = int(round((northing - yoff) / e))

            polyarray[i][0] = pixelX
            polyarray[i][1] = pixelY

        # Change the poly array list to np.array
        npArrayPoly = numpy.array(polyarray, numpy.int32)
        npArrayPoly = npArrayPoly.reshape((-1, 1, 2))  # Reshape the array

        # Get the image size
        row_len = dataset.RasterYSize
        col_len = dataset.RasterXSize

        # Create the cloud_map to work with
        cloud_map = np.zeros(shape=(row_len, col_len))

        # Convert the Polygon array into a mask file
        aoiFinal = cv2.fillPoly(cloud_map, [npArrayPoly], 1)
        cloud_map = np.zeros(shape=(row_len, col_len), dtype=np.uint16)

        # Consolidate the shit
        redband = dataset.GetRasterBand(1).ReadAsArray()
        greenband = dataset.GetRasterBand(2).ReadAsArray()
        blueband = dataset.GetRasterBand(3).ReadAsArray()

        combinedmapvalues = np.uint16(redband) + np.uint16(greenband) + np.uint16(blueband)

        N = int(np.round(col_len / 10))
        tempMat = np.sort(combinedmapvalues, 0)
        peak_reflec = np.mean(tempMat[(len(tempMat) - N):, :])
        print('peak_reflect1: %f' % peak_reflec)
        outputText.insert(tk.END, '\npeak_reflect1: %f' % peak_reflec)
        peak_reflec_gain = reflec_gain
        tempMat = None

        for x in range(row_len):
            temp = combinedmapvalues[x, :]

            ind = np.asarray(np.where(temp >= peak_reflec * peak_reflec_gain))

            temp1 = np.copy(cloud_map[x, :])
            for n in range(len(ind[0])):
                temp1[ind[0, n]] = 1

            tempd = np.power(np.diff(temp), 2)
            m = np.mean(tempd)
            ind = np.asarray(np.where(tempd <= m * 0.7))
            temp3 = np.copy(cloud_map[x, :])
            for n in range(len(ind[0])):
                temp3[ind[0, n]] = 1

            cloud_map[x, :] = cloud_map[x, :] + np.multiply(temp1, temp3)
        print('x direction done')
        outputText.insert(tk.END, '\nx direction done')
        temp = None
        temp1 = None
        temp3 = None
        ind = None
        m = None

        for y in range(col_len):
            temp = combinedmapvalues[:, y]
            ind = np.asarray(np.where(temp >= peak_reflec * peak_reflec_gain))

            temp1 = np.copy(cloud_map[:, y])
            for n in range(len(ind[0])):
                temp1[ind[0, n]] = 1

            tempd = np.power(np.diff(temp), 2)
            m = np.mean(tempd)
            ind = np.asarray(np.where(tempd <= m * 0.7))
            temp3 = np.copy(cloud_map[:, y])
            for n in range(len(ind[0])):
                temp3[ind[0, n]] = 1

            cloud_map[:, y] = cloud_map[:, y] + np.multiply(temp1, temp3)
        print('y direction done')
        outputText.insert(tk.END, '\ny direction done')
        temp = None
        temp1 = None
        temp3 = None
        ind = None
        m = None

        for x in range(cloud_map.shape[0]):
            for y in range(cloud_map.shape[1]):
                if cloud_map[x, y] >= 1:
                    cloud_map[x, y] = 1

        cloud_map = np.multiply(cloud_map, aoiFinal)

        print("Eroding and Dilating")
        outputText.insert(tk.END, '\nEroding and Dilating')

        # https: // www.packtpub.com / mapt / book / application_development / 9781785283932 / 2 / ch02lvl1sec24 / erosion - and -dilation
        kernel = np.ones((1, 1), np.uint8)
        cloud_map = cv2.erode(cloud_map, kernel, iterations=1)
        cloud_map = cv2.dilate(cloud_map, kernel, iterations=1)
        kernel = np.ones((5, 5), np.uint8)
        cloud_map = cv2.dilate(cloud_map, kernel, iterations=1)
        cloud_map = cv2.erode(cloud_map, kernel, iterations=1)

        # Find the percentage of cloud cover
        ind1 = np.where(cloud_map > 0)
        ind2 = np.where(aoiFinal > 0)
        percent = (np.divide(len(ind1[0]), len(ind2[0]))) * 100
        print('The cloud cover is: %3.2f' % percent)
        outputText.insert(tk.END, '\nThe cloud cover is: %3.2f' % percent)

        # # Show the image
        # cv2.namedWindow("image",
        #                 cv2.WINDOW_KEEPRATIO)               # Create a new window (Similar to creating a new figure in matlab)
        # cv2.namedWindow("aoi_final", cv2.WINDOW_KEEPRATIO)
        # cv2.imshow("image", cloud_map)                      # Show the image
        # cv2.imshow("aoi_final", aoiFinal)                   # Show the image
        # cv2.waitKey(0)                                      # Tells the cv2 to wait for a keypress
        # cv2.destroyAllWindows()                             # Close all windows

        return cloud_map, percent
