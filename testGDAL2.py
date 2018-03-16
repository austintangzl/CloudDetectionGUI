from osgeo import gdal
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
# import scipy

# filename = "/Users/austintang/PycharmProjects/CloudDetectionGUI3.6/For Brenda/GeoTiff/Mosiac_Johor_13Sep17_5Oct17_16Jun17_8bit.tif"
# filename = "C:\CloudDetectionGUI3.6\For Brenda\GeoTiff\Mosiac_Johor_13Sep17_5Oct17_16Jun17_8bit.tif"
# filename = "C:\CloudDetectionGUI3.6\For Brenda\GeoTiff\johor_downsample.jp2"
# filename = "C:\CloudDetectionGUI3.6\For Brenda\LandSat\\test.tif"

filename = "C:\CloudDetectionGUI3.6\output.tif"
filename = "C:\CloudDetectionGUI3.5\output.tif"
filename = "C:\CloudDetectionGUI3.6\For Brenda\T1_00006197\T012017101306250892A_L2.tif"

# Load the GeoTiff image
dataset = gdal.Open(filename, gdal.GA_ReadOnly)

# Read KML File
kmlfilename = "C:\\CloudDetectionGUI3.6\\For Brenda\\GeoTiff\\PUB Johor AOI.kmz"
# Extract the points and make a polygon
try:
    for polygons in readPoly(kmlfilename):
        polygons, desc = polygons
        print(polygons)
except:
    print('Unable to load KML/KMZ file')


# Get file size
filesize = os.path.getsize(filename)

print(filesize/1e9)
if filesize/1e9 >= 1:
    print('Compressing Image')
    # Get the pixel information
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    dataset = None
    # Delete temp image file if it exist
    if os.path.isfile(os.getcwd()+'\output.tif') is True:
        os.remove(os.getcwd()+'\output.tif')
    # Reduce the filesize (Use -ts to resize with pixels, use -tr to resize with resolution) (http://www.gdal.org/gdalwarp.html)
    if filesize/1e9 >= 4:
        compressfactor = 10
    elif filesize/1e9 >= 3:
        compressfactor = 7
    elif filesize/1e9 >= 2:
        compressfactor = 5
    elif filesize/1e9 >= 1:
        compressfactor = 1
    command = ['gdalwarp -ts ' + str(int(width/compressfactor)) + ' ' + str(int(height/compressfactor)) + ' "' + filename + '" "' + os.getcwd() + '\output.tif"']
    # command = ['gdalwarp -tr 100 100 "' + filename + '" "' + os.getcwd() + '\output.tif"']
    # subprocess.run(command, shell=True)
    command = ''.join(command)
    os.system(command)
    filename = os.getcwd()+'\output.tif'
    dataset = gdal.Open(filename, gdal.GA_ReadOnly)
    print('Compress Image Done')

# If there is no KML/KMZ file
eoband = dataset.GetRasterBand(1).ReadAsArray()
# eoband = np.uint16(eoband)

xlen = dataset.RasterXSize
ylen = dataset.RasterYSize

polygons = []
print("\nSize is {} x {} x {}".format(dataset.RasterXSize,
                                      dataset.RasterYSize,
                                      dataset.RasterCount))


for i in range(ylen):
    findpixel = np.nonzero(eoband[i, :])  # Find the first row
    findpixel = findpixel[0]
    if findpixel.any():
        polygons.append([findpixel[0], i])
        break

for i in reversed(range(xlen)):
    findpixel = np.nonzero(eoband[:, i])  # Find the first row
    findpixel = findpixel[0]
    if findpixel.any():
        polygons.append([i, findpixel[0]])
        break

for i in reversed(range(ylen)):
    findpixel = np.nonzero(eoband[i, :])  # Find the first row
    findpixel = findpixel[0]
    if findpixel.any():
        polygons.append([findpixel[0], i])
        break

for i in range(xlen):
    findpixel = np.nonzero(eoband[:, i])  # Find the first row
    findpixel = findpixel[0]
    if findpixel.any():
        polygons.append([i, findpixel[0]])
        break

# Complete the loop in the polygon
polygons.append(polygons[0])

print('\nMy Polygons are:')
print(polygons)


if not dataset:
    print('Dataset empty')
else:
    # Getting Dataset Information
    print("Driver: {}/{}".format(dataset.GetDriver().ShortName,
                                 dataset.GetDriver().LongName))
    print("Size is {} x {} x {}".format(dataset.RasterXSize,
                                        dataset.RasterYSize,
                                        dataset.RasterCount))
    print("Projection is {}".format(dataset.GetProjection()))
    geotransform = dataset.GetGeoTransform()
    if geotransform:
        print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
        print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))

    # Fetching a Raster Band
    band = dataset.GetRasterBand(1)
    print("Band Type={}".format(gdal.GetDataTypeName(band.DataType)))

    min1 = band.GetMinimum()
    max1 = band.GetMaximum()
    if not min1 or not max1:
        (min1, max1) = band.ComputeRasterMinMax(True)
    print("Min={:.3f}, Max={:.3f}".format(min1, max1))

    if band.GetOverviewCount() > 0:
        print("Band has {} overviews".format(band.GetOverviewCount()))

    if band.GetRasterColorTable():
        print("Band has a color table with {} entries".format(band.GetRasterColorTable().GetCount()))

    scanline = band.ReadRaster(xoff=0, yoff=0,
                               xsize=band.XSize, ysize=1,
                               buf_xsize=band.XSize, buf_ysize=1,
                               buf_type=gdal.GDT_Float32)

    tuple_of_floats = struct.unpack('f' * band.XSize, scanline)

    # GDAL affine transform parameters, According to gdal documentation xoff/yoff are image left corner, a/e are pixel wight/height and b/d is rotation and is zero if image is north up.
    xoff, a, b, yoff, d, e = dataset.GetGeoTransform()


    def pixel2coord(x, y):
        """Returns global coordinates from pixel x, y coords"""
        xp = a * x + b * y + xoff
        yp = d * x + e * y + yoff
        return xp, yp


    def pt2fmt(pt):
        fmttypes = {
            GDT_Byte: 'B',
            GDT_Int16: 'h',
            GDT_UInt16: 'H',
            GDT_Int32: 'i',
            GDT_UInt32: 'I',
            GDT_Float32: 'f',
            GDT_Float64: 'f'
        }
        return fmttypes.get(pt, 'x')

    # Get columns and rows of your image from gdalinfo
    transf = dataset.GetGeoTransform()
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    bands = dataset.RasterCount
    band = dataset.GetRasterBand(1)  # Repeated code
    bandtype = gdal.GetDataTypeName(band.DataType)
    driver = dataset.GetDriver().LongName

    # polyarray = polygons
    polyarray = [[None for _ in range(2)] for _ in range(len(polygons))]    # Declare a polygon array

    for i in range(len(polygons)):
        polyarray[i][0] = polygons[i][0]
        polyarray[i][1] = polygons[i][1]

    # Convert the latitude and longitude from the kml/kmz file into pixel coordinates
    # for i in range(len(polygons)):
    #     lon = polygons[i][0]
    #     lat = polygons[i][1]
    #     [easting, northing, zoneNum, zoneLet] = utm.from_latlon(lat, lon)
    #     pixelX = int(round((easting - xoff) / a))
    #     pixelY = int(round((northing - yoff) / e))
    #
    #     polyarray[i][0] = pixelX
    #     polyarray[i][1] = pixelY



    # # Set the pixels for the polygons points to be magenta
    # for i in range(len(polyarray)):
    #     redband[polyarray[i][1]][polyarray[i][0]] = 255
    #     greenband[polyarray[i][1]][polyarray[i][0]] = 0
    #     blueband[polyarray[i][1]][polyarray[i][0]] = 255

    # Create the output image
    driver = dataset.GetDriver()    # Get the driver from the original image
    # print driver
    if os.path.isfile(os.getcwd()+'\\newimg.tif') is True:
        os.remove(os.getcwd()+'\\newimg.tif')
    outDs = driver.Create(os.getcwd() + '\\newimg.tif', cols, rows, dataset.RasterCount, dataset.GetRasterBand(1).DataType)   # Create the new image driver
    if outDs is None:
        print('Could not create newimg.tif')
        sys.exit(1)

    # Pull out the individual image bands from the newly created driver
    if dataset.RasterCount == 3:
        redband = dataset.GetRasterBand(1)  # Get the red band raster data out
        redband = redband.ReadAsArray()  # Convert the red band data into array
        greenband = dataset.GetRasterBand(2)  # Get the green band raster data out
        greenband = greenband.ReadAsArray()  # Convert the green band data into array
        blueband = dataset.GetRasterBand(3)  # Get the blue band raster data out
        blueband = blueband.ReadAsArray()  # Convert the blue band data into array

        outBand1 = outDs.GetRasterBand(1)
        outBand2 = outDs.GetRasterBand(2)
        outBand3 = outDs.GetRasterBand(3)
        outData = np.zeros((rows, cols, 3), np.uint16)

        # Write the band dats into the numpy array
        outData[..., 0] = redband
        outData[..., 1] = greenband
        outData[..., 2] = blueband
    elif dataset.RasterCount == 1:
        outBand1 = outDs.GetRasterBand(1)
        outData = np.zeros((rows, cols, 1), np.uint16)

        outData[..., 0] = dataset.GetRasterBand(1).ReadAsArray()



    # Show image (For testing)
    # img = Image.fromarray(outData, 'RGB')
    # img.show()

    # Change the poly array list to np.array
    npArrayPoly = numpy.array(polyarray, numpy.int32)
    npArrayPoly = npArrayPoly.reshape((-1, 1, 2))       # Reshape the array

    if dataset.RasterCount == 3:
        outDataBGR = cv2.cvtColor(outData, cv2.COLOR_RGB2BGR)           # Convert the RGB to BGR
        cv2.polylines(outDataBGR, [npArrayPoly], True, (255, 0, 255), 5)   # Draw the polygon in the numpy array (CV2 color is in BGR)
        # # cv2.namedWindow("image", cv2.WINDOW_NORMAL)     # Create a new window (Similar to creating a new figure in matlab)
        # # cv2.imshow("image", outDataBGR)                 # Show the image
        # # cv2.waitKey(0)                                  # Tells the cv2 to wait for a keypress
        # # cv2.destroyAllWindows()                         # Close all windows

        # Convert the image back to RGB
        outDataRGB = cv2.cvtColor(outDataBGR, cv2.COLOR_BGR2RGB)
        # Clear the data
        outDataBGR = None
        outData = None

        # Write the data into the new file
        outBand1.WriteArray(outDataRGB[..., 0], 0, 0)
        outBand2.WriteArray(outDataRGB[..., 1], 0, 0)
        outBand3.WriteArray(outDataRGB[..., 2], 0, 0)

        # flush data to disk, set the NoData value and calculate stats
        outBand1.FlushCache()
        outBand1.SetNoDataValue(-99)
        outBand2.FlushCache()
        outBand2.SetNoDataValue(-99)
        outBand3.FlushCache()
        outBand3.SetNoDataValue(-99)
    elif dataset.RasterCount == 1:
        # outDataBGR = cv2.cvtColor(outData, cv2.COLOR_RGB2BGR)  # Convert the RGB to BGR
        # cv2.polylines(outDataBGR, [npArrayPoly], True, (255, 0, 255), 5)  # Draw the polygon in the numpy array (CV2 color is in BGR)
        # # Convert the image back to RGB
        # outDataRGB = cv2.cvtColor(outDataBGR, cv2.COLOR_BGR2RGB)
        # # Clear the data
        # outDataBGR = None
        # outData = None
        # Write the data into the new file
        outBand1.WriteArray(outData[..., 0], 0, 0)
        # flush data to disk, set the NoData value and calculate stats
        outBand1.FlushCache()
        outBand1.SetNoDataValue(-99)

    # georeference the image and set the projection
    outDs.SetGeoTransform(dataset.GetGeoTransform())
    outDs.SetProjection(dataset.GetProjection())

    # del redband, greenband, blueband

    outDs = None    # Release the memory of the map

    # Open the new map
    workingmap = gdal.Open(os.getcwd() + '\\newimg.tif', gdal.GA_ReadOnly)
    # workingmap = gdal.Open(filename, gdal.GA_ReadOnly)

    print("\nSize is {} x {} x {}".format(workingmap.RasterXSize,
                                          workingmap.RasterYSize,
                                          workingmap.RasterCount))
    # Fetching a Raster Band
    band = workingmap.GetRasterBand(1)
    print("Band Type={}".format(gdal.GetDataTypeName(band.DataType)))

    row_len = workingmap.RasterYSize
    col_len = workingmap.RasterXSize

    # Create the cloud_map to work with
    cloud_map = np.zeros(shape=(row_len, col_len))

    # Convert the Polygon array into a mask file
    mask_color = [255, 0, 255]
    aoiFinal = cv2.fillPoly(cloud_map, [npArrayPoly], 1)
    cloud_map = np.zeros(shape=(row_len, col_len), dtype=np.uint16)

    # Consolidate the shit
    # mahmapvalues = np.zeros(shape=(xlen, ylen, 3))

    if workingmap.RasterCount == 3:
        redband = workingmap.GetRasterBand(1).ReadAsArray()
        greenband = workingmap.GetRasterBand(2).ReadAsArray()
        blueband = workingmap.GetRasterBand(3).ReadAsArray()
        combinedmapvalues = np.uint16(redband) + np.uint16(greenband) + np.uint16(blueband)
    elif workingmap.RasterCount == 1:
        eoband = workingmap.GetRasterBand(1).ReadAsArray()
        combinedmapvalues = np.uint16(eoband)

    N = int(np.round(col_len/10))
    tempMat = np.sort(combinedmapvalues, 0)
    peak_reflec = np.mean(tempMat[(len(tempMat)-N):, :])
    print('peak_reflect1: %f' % peak_reflec)
    # tempMat = np.ndarray.flatten(combinedmapvalues)
    # tempMat = np.sort(combinedmapvalues)
    # peak_reflec = np.mean(tempMat[(len(tempMat)-int(np.round(len(tempMat)/10))):, :])
    # print('peak_reflect2: %f' % peak_reflec)
    # peak_reflec = 135.309769
    peak_reflec_gain = 0.8
    tempMat = None

    for x in range(row_len):
        temp = combinedmapvalues[x, :]

        ind = np.asarray(np.where(temp >= peak_reflec*peak_reflec_gain))

        temp1 = np.copy(cloud_map[x, :])
        for n in range(len(ind[0])):
            temp1[ind[0, n]] = 1

        tempd = np.power(np.diff(temp), 2)
        m = np.mean(tempd)
        ind = np.asarray(np.where(tempd <= m*0.7))
        temp3 = np.copy(cloud_map[x, :])
        for n in range(len(ind[0])):
            temp3[ind[0, n]] = 1

        cloud_map[x, :] = cloud_map[x, :] + np.multiply(temp1, temp3)
    print('x direction done')

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

    temp = None
    temp1 = None
    temp3 = None
    ind = None
    m = None

    # for i in range(cloud_map.shape[0]):
    #     ind = np.asarray(np.where(cloud_map[i,:] >= 1))
    #
    #     for n in range(len(ind[0])):
    #         cloud_map[i, n] = cloud_map[i]


    for x in range(cloud_map.shape[0]):
        for y in range(cloud_map.shape[1]):
            if cloud_map[x, y] >= 1:
                cloud_map[x, y] = 1

    cloud_map = np.multiply(cloud_map, aoiFinal)

    print("Eroding and Dilating")

    # https: // www.packtpub.com / mapt / book / application_development / 9781785283932 / 2 / ch02lvl1sec24 / erosion - and -dilation
    kernel = np.ones((1, 1), np.uint8)
    cloud_map = cv2.erode(cloud_map, kernel, iterations=1)
    cloud_map = cv2.dilate(cloud_map, kernel, iterations=1)
    kernel = np.ones((5, 5), np.uint8)
    cloud_map = cv2.dilate(cloud_map, kernel, iterations=1)
    cloud_map = cv2.erode(cloud_map, kernel, iterations=1)

    # Invert Cloud_map
    # cloud_map = 1 - cloud_map
    # cloud_map = np.multiply(cloud_map, aoiFinal)

    # Find the percentage of cloud cover
    ind1 = np.where(cloud_map > 0)
    ind2 = np.where(aoiFinal > 0)
    percent = (np.divide(len(ind1[0]), len(ind2[0])))*100
    print('The cloud cover is: %3.2f' % percent)

    # Show the image
    cv2.namedWindow("image", cv2.WINDOW_KEEPRATIO)     # Create a new window (Similar to creating a new figure in matlab)
    cv2.namedWindow("aoi_final", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("image", cloud_map)                      # Show the image
    cv2.imshow("aoi_final", aoiFinal)                   # Show the image
    cv2.waitKey(0)                                      # Tells the cv2 to wait for a keypress
    cv2.destroyAllWindows()                             # Close all windows

dataset = None  # Close the dataset (Release the memory)
print('DONEDEDED FOOL')
