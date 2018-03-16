from PIL import ImageTk, Image

filename = "C:\CloudDetectionGUI3.6\For Brenda\GeoTiff\Mosiac_Johor_13Sep17_5Oct17_16Jun17_8bit.tif"

try:
    myimage = Image.open(filename)  # Open the image
except:
    print('Unable to load image')

myimage.load()
