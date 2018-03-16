from osgeo import gdal
from osgeo.gdalconst import *
import sys
import numpy as np


filename = "C:\CloudDetectionGUI3.6\For Brenda\T1_00006197\T012017101306250892A_L2.tif"

# Load the GeoTiff image
dataset = gdal.Open(filename, gdal.GA_ReadOnly)

if not dataset:
    print('Dataset empty')
    sys.exit()

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

eoband = dataset.GetRasterBand(1).ReadAsArray()
# eoband = np.uint16(eoband)

lonlen = dataset.RasterXSize
latlen = dataset.RasterYSize

imageborder = []

for i in range(latlen):
    findpixel = np.nonzero(eoband[i, :])  # Find the first row
    findpixel = findpixel[0]
    if findpixel.any():
        imageborder.append([i, findpixel[0]])
        break

for i in reversed(range(lonlen)):
    findpixel = np.nonzero(eoband[:, i])  # Find the first row
    findpixel = findpixel[0]
    if findpixel.any():
        imageborder.append([findpixel[0], i])
        break

for i in reversed(range(latlen)):
    findpixel = np.nonzero(eoband[i, :])  # Find the first row
    findpixel = findpixel[0]
    if findpixel.any():
        imageborder.append([i, findpixel[0]])
        break

for i in range(lonlen):
    findpixel = np.nonzero(eoband[:, i])  # Find the first row
    findpixel = findpixel[0]
    if findpixel.any():
        imageborder.append([findpixel[0], i])
        break

print(imageborder)

print(eoband[imageborder[0][0], imageborder[0][1]])
