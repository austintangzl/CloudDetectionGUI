#from osgeo import _gdal as gdal
# import osgeo._gdal
import osgeo.gdal as gdal
from osgeo.gdalconst import *
from Cloud_Detection_Algo import *
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import ImageTk, Image
from extractKML import *
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

class CloudDetectionMain(tk.Tk):
    def __init__(self, parent):
        tk.Tk.__init__(self, parent)

        # Set the window icon
        # self.iconbitmap('C:\CloudDetectionGUI3.6\Icons\cloud_icon.ico')
        self.iconbitmap(os.getcwd() + '\Icons\\austin_icon.ico')

        # Set the GUI Default Background color
        bgcolor = '#%02x%02x%02x' % (249, 249, 249)  # Setting the background color
        self.tk_setPalette(background=bgcolor, foreground='black', activeBackground='black', activeForeground='white')

        self.requestImgTagVar = tk.StringVar()
        self.requestImgEntryVar = tk.StringVar()
        self.requestImgEntry = tk.Entry(self,
                                        bg='white',
                                        textvariable=self.requestImgEntryVar)
        self.requestKmlTagVar = tk.StringVar()
        self.requestKmlEntryVar = tk.StringVar()  # Declare the entry box variable
        self.requestKmlEntry = tk.Entry(self,
                                        bg='white',
                                        textvariable=self.requestKmlEntryVar)  # Create the entry box
        self.labelVariable = tk.StringVar()  # Declare the label's variable

        # Create a Radio button to select whether to use KML
        self.selectKMLVar = tk.IntVar()

        # Initialise some variables/guiobjects
        self.img_copy = []
        self.img_copy2 = []
        self.imgPanel = tk.Label(self)
        self.sliderTagVar = tk.StringVar()
        self.reflec_gain_var = tk.DoubleVar()
        self.scale = tk.Scale(self)
        self.imgPanel2 = tk.Label(self)
        self.outputText = tk.Text(self)

        self.parent = parent
        self.initialize()

    def initialize(self):
        self.grid()  # Set the GUI to a grid formation

        # Create a label for image selection
        requestImgTag = tk.Label(self,
                                 textvariable=self.requestImgTagVar,
                                 anchor='w',
                                 fg='black')
        requestImgTag.grid(column=0, row=0, columnspan=3, stick='EW')
        self.requestImgTagVar.set(u'Please select a image file')

        self.requestImgEntry.grid(column=0,
                                  row=1,
                                  sticky='EW')

        requestImgEntryButt = tk.Button(self,
                                        text=u'...',
                                        command=self.requestImgEntryButtClick)
        requestImgEntryButt.grid(column=1, row=1)

        # Create a label for kml/kmz explanation
        requestkmltag = tk.Label(self,
                                 textvariable=self.requestKmlTagVar,
                                 anchor='w',
                                 fg='black')  # Creating the label
        requestkmltag.grid(column=0, row=2, columnspan=3, stick='EW')
        self.requestKmlTagVar.set(u'Please select a kml/kmz file')

        # Create a entry/edit box
        self.requestKmlEntry.grid(column=0,
                                  row=3,
                                  sticky='EW')  # Place the entry box into the grid
        self.requestKmlEntry.bind('<Return>',
                                  self.OnPressEnter)  # Set a callback for a key press
        self.requestKmlEntryVar.set(u"")  # Initialise the entry variable

        # Create the button
        self.requestKmlEntryButt = tk.Button(self,
                                        text=u'...',
                                        command=self.OnButtonClick)  # Create the button
        self.requestKmlEntryButt.grid(column=1, row=3)  # Place the button into the grid

        # Create the label
        label = tk.Label(self,
                         textvariable=self.labelVariable,
                         anchor='w',
                         fg='white',
                         bg='blue')  # Create the label
        label.grid(column=0, row=4, columnspan=3, sticky='EW')  # Place the label into the grid
        self.labelVariable.set(u'Hello!')  # Initialise the label variable

        # Create Button to load images and read kml
        loadImgButt = tk.Button(self,
                                text=u'Load Image',
                                command=self.loadImgButtClick)
        loadImgButt.grid(column=2, row=1, columnspan=1, sticky='EW')

        self.loadKmlButt = tk.Button(self,
                                text=u'Load KML/KMZ',
                                command=self.loadKmlButtClick)
        self.loadKmlButt.grid(column=2, row=3, columnspan=1, sticky='EW')

        # Create the Radio button for AOI selection
        self.selectKMLRadio1 = tk.Radiobutton(self, text='KML/KMZ', variable=self.selectKMLVar, value=1,
                                              command=self.selectKML)
        self.selectKMLRadio1.grid(column=0, row=5, columnspan=1, sticky='EW')
        self.selectKMLRadio2 = tk.Radiobutton(self, text='Whole Image', variable=self.selectKMLVar, value=2,
                                              command=self.selectKML)
        self.selectKMLRadio2.grid(column=1, row=5, columnspan=1, sticky='EW')
        self.selectKMLRadio3 = tk.Radiobutton(self, text='Detect Edge', variable=self.selectKMLVar, value=3,
                                              command=self.selectKML)
        self.selectKMLRadio3.grid(column=2, row=5, columnspan=1, sticky='EW')

        # Call the image
        # imgpath = '/Users/austintang/Documents/CloudDetectionGUI/Images/python_icon2.jpg'  # Set the image path
        # imgpath = 'C:\CloudDetectionGUI3.6\Icons\python_icon.png'  # Set the image path
        imgpath = os.getcwd() + '\Icons\\austin.png'  # Set the image path
        img = Image.open(imgpath)
        width, height = img.size
        width = int(width/6)
        height = int(height/6)
        maxsize = (width, height)  # Create a tuple for the image panel diamensions
        img = img.resize(maxsize)
        self.img_copy = img.copy()
        img = ImageTk.PhotoImage(img)
        self.imgPanel = tk.Label(self, image=img)
        self.imgPanel.image = img
        self.imgPanel.grid(column=3, row=1, columnspan=1, rowspan=6, sticky='NSEW', padx=5, pady=5)

        # Create a slider bar for the reflec_gain
        # self.sliderTagVar = tk.StringVar()
        sliderTag = tk.Label(self,
                             textvariable=self.sliderTagVar,
                             anchor='s',
                             fg='black')
        sliderTag.grid(column=0, row=7, columnspan=1, stick='EW')
        self.sliderTagVar.set(u'Reflectance Gain:')
        # self.reflec_gain_var = tk.DoubleVar()
        self.scale = tk.Scale(self, variable=self.reflec_gain_var, orient=tk.HORIZONTAL, from_=0, to=1, resolution=0.05)
        self.scale.set(0.8)
        self.scale.grid(column=1, row=7, columnspan=2, rowspan=1, sticky='EW')

        # Create a button to start the cloud detection process
        startCloudDetectionButt = tk.Button(self,
                                            text=u'Start Cloud Detection',
                                            command=self.startCloudDetection)
        startCloudDetectionButt.grid(column=0, row=8, columnspan=1, sticky='EW')

        # Create a button to save images
        saveOutputImagesButt = tk.Button(self,
                                         text=u'Save Output',
                                         command=self.saveOutputImages)
        saveOutputImagesButt.grid(column=2, row=8, columnspan=1, sticky='EW')

        # Create a image panel for the final image
        # imgpath2 = 'C:\CloudDetectionGUI3.6\Icons\Empty.png'
        imgpath2 = os.getcwd() + '\Icons\\austin.png'
        img2 = Image.open(imgpath2)
        self.img_copy2 = img2.copy()
        maxsize = (width, width)  # Create a tuple for the image panel diamensions
        img2 = img2.resize(maxsize)  # Resize the image to fit the image panel
        img2 = ImageTk.PhotoImage(img2)
        self.imgPanel2 = tk.Label(self, image=img2)
        self.imgPanel2.image = img2
        self.imgPanel2.grid(column=3, row=7, columnspan=1, rowspan=6, sticky='NSEW', padx=5, pady=5)

        # Add a text box for the program output
        # self.outputVar = tk.StringVar()
        self.outputText = tk.Text(self, width=45, height=10)
        self.outputText.grid(column=0, row=9, columnspan=3, rowspan=4, sticky='NSEW')
        # self.outputVar.set(u'Welcome to the Cloud Detection 1.0 program. \nPlease select the image (GeoTIFF) and KML/KMZ file and run the program.')
        self.outputText.insert(tk.END,
                               'Welcome to the Cloud Detection 1.0 program. \nPlease select the image (GeoTIFF) and KML/KMZ file and run the program.')

        # GUI Settings
        self.grid_columnconfigure(0, weight=1, minsize=200)
        self.grid_columnconfigure(1, weight=1, minsize=20)
        self.grid_columnconfigure(2, weight=1, minsize=100)
        self.grid_columnconfigure(3, weight=2, minsize=400)
        self.grid_rowconfigure(6, weight=1, minsize=120)
        self.grid_rowconfigure(11, weight=1, minsize=100)
        self.resizable(True, True)
        self.update()
        self.geometry(self.geometry())
        self.requestKmlEntry.focus_set()
        self.requestKmlEntry.selection_range(0, tk.END)

        # Bind the image panel to a resizing gui callback
        self.imgPanel.bind('<Configure>', self.gui_resize)
        self.imgPanel2.bind('<Configure>', self.gui_resize2)
        self.update()

    def selectKML(self):
        self.labelVariable.set('You\'ve selected option: ' + str(self.selectKMLVar.get()))
        if self.selectKMLVar.get() is 1:
            self.requestKmlEntry.config(state='normal')
            self.requestKmlEntryButt.config(state='normal')
            self.loadKmlButt.config(state='normal')
        elif self.selectKMLVar.get() is 2:
            self.requestKmlEntry.config(state='disabled')
            self.requestKmlEntryButt.config(state='disabled')
            self.loadKmlButt.config(state='disabled')
        elif self.selectKMLVar.get() is 3:
            self.requestKmlEntry.config(state='disabled')
            self.requestKmlEntryButt.config(state='disabled')
            self.loadKmlButt.config(state='disabled')

    def OnButtonClick(self):
        # self.labelVariable.set(self.entryVariable.get()+" (You clicked the button!)")
        # print "You clicked the button!"
        # self.entry.focus_set()
        # self.entry.selection_range(0, tk.END)
        options = {'filetypes': [('Keyhole Markup Language', '.kml'), ('Keyhole Markup Language', '.kmz')]}
        kmlfile = filedialog.askopenfilename(**options)
        self.requestKmlEntryVar.set(kmlfile)

    def OnPressEnter(self):
        self.labelVariable.set(self.requestKmlEntryVar.get() + " (You pressed enter!)")
        print("You pressed enter!")
        self.requestKmlEntry.focus_set()
        self.requestKmlEntry.selection_range(0, tk.END)

    def requestImgEntryButtClick(self):
        options = {'filetypes': [('JPEG', '.jpg'), ('Portable Graphics Network', '.png'), ('GeoTIFF', '.tif')]}
        imgfile = filedialog.askopenfilename(**options)
        self.requestImgEntryVar.set(imgfile)

    def resizeImage(self, imgpath):
        dataset = gdal.Open(imgpath, gdal.GA_ReadOnly)
        print('Compressing Image')
        self.outputText.insert(tk.END, '\nCompressing Image')
        self.outputText.see(tk.END)
        # Get the pixel information
        width = dataset.RasterXSize
        height = dataset.RasterYSize

        # Release the memory
        # noinspection PyUnusedLocal
        dataset = None

        # Delete temp image file if it exist
        if os.path.isfile(os.getcwd() + '\output.tif') is True:
            os.remove(os.getcwd() + '\output.tif')

        # Reduce the filesize (Use -ts to resize with pixels, use -tr to resize with resolution) (http://www.gdal.org/gdalwarp.html)
        compressfactor = 10
        command = ['gdalwarp -ts ' + str(int(width / compressfactor)) + ' ' + str(
            int(height / compressfactor)) + ' "' + imgpath + '" "' + os.getcwd() + '\output.tif"']

        command = ''.join(command)
        os.system(command)
        imgpath = os.getcwd() + '\output.tif'
        self.requestImgEntryVar.set(imgpath)
        print('Compress Image Done')
        self.outputText.insert(tk.END, '\nCompressing Image Done')
        self.outputText.see(tk.END)
        self.update()
        return imgpath

    # noinspection PyUnusedLocal
    def loadImgButtClick(self):
        print('loadImgButtClick button pressed')
        self.outputText.insert(tk.END, '\nloadImgButtClick button pressed')
        self.outputText.see(tk.END)
        imgpath = self.requestImgEntryVar.get()
        if len(self.requestImgEntry.get()) != 0:
            # Get file size
            filesize = os.path.getsize(imgpath)
            print('The image filesize is: %f' % filesize)
            self.outputText.insert(tk.END, '\nThe image filesize is: %f' % filesize)
            if filesize / 1e9 >= 4:
                answer = messagebox.askquestion("Resize Image", "The current image size is too large to display/process. Do you want to resize image?", icon='warning')
                if answer == 'yes':
                    imgpath = self.resizeImage(imgpath)
                elif answer == 'no':
                    print('Did not compress and display image.')
                    self.outputText.insert(tk.END, '\nDid not compress and display image.')
                    self.outputText.see(tk.END)
                    return

            try:
                if imgpath[-4:] == '.tif':
                    # this allows GDAL to throw Python Exceptions
                    gdal.UseExceptions()

                    try:
                        src_ds = gdal.Open(imgpath)
                    except RuntimeError as e:
                        print('Unable to open ( %s )' % imgpath)
                        mystring = '\nUnable to open ( %s )' % imgpath
                        self.outputText.insert(tk.END, mystring)
                        self.outputText.see(tk.END)
                        print(e)
                        sys.exit(1)

                    try:
                        redband = src_ds.GetRasterBand(1)       # Get the red band raster data out
                        redband = redband.ReadAsArray()         # Convert the red band data into array
                    except RuntimeError as e:
                        print('Band 1 ( Red ) not found')
                        self.outputText.insert(tk.END, '\nBand 1 ( Red ) not found')
                        self.outputText.see(tk.END)
                        print(e)
                        sys.exit(1)

                    try:
                        greenband = src_ds.GetRasterBand(2)     # Get the green band raster data out
                        greenband = greenband.ReadAsArray()     # Convert the green band data into array
                    except RuntimeError as e:
                        print('Band 2 ( Green ) not found')
                        self.outputText.insert(tk.END, '\nBand 2 ( Green ) not found')
                        self.outputText.see(tk.END)
                        print(e)
                        sys.exit(1)

                    try:
                        blueband = src_ds.GetRasterBand(3)      # Get the blue band raster data out
                        blueband = blueband.ReadAsArray()       # Convert the blue band data into array
                    except RuntimeError as e:
                        print('Band 3 ( Blue ) not found')
                        self.outputText.insert(tk.END, '\nBand 3 ( Blue ) not found')
                        self.outputText.see(tk.END)
                        print(e)
                        sys.exit(1)

                    # Show the image in GUI
                    img = Image.open(imgpath)                       # Convert the tiff image
                    self.img_copy = img.copy()
                    labelWidth = self.imgPanel.winfo_width()-100        # Get the image panel Width
                    labelHeight = self.imgPanel.winfo_height()-100      # Get the image panel Height
                    maxsize = (labelWidth, labelHeight)             # Create a tuple for the image panel diamensions
                    img = img.resize(maxsize)                       # Resize the image to fit the image panel
                    img = ImageTk.PhotoImage(img)                   # Convert the image into a tkinker image
                    self.imgPanel.configure(image=img)              # Set the image into the image panel
                    self.imgPanel.image = img                       # Set the image into the image panel 2 (MUST DO)
                    self.update()

                elif imgpath[-4:] == '.jpg' or imgpath[-4:] == '.JPG' or imgpath[-4:] == '.png' or imgpath[-4:] == '.PNG':
                    img = Image.open(imgpath)                   # Open the image
                    self.img_copy = img.copy()
                    labelWidth = self.imgPanel.winfo_width()    # Get the image panel Width
                    labelHeight = self.imgPanel.winfo_height()  # Get the image panel Height
                    maxsize = (labelWidth, labelHeight)         # Create a tuple for the image panel diamensions
                    img = img.resize(maxsize)                   # Resize the image to fit the image panel
                    img = ImageTk.PhotoImage(img)               # Convert the image into a tkinker image
                    self.imgPanel.configure(image=img)          # Set the image into the image panel
                    self.imgPanel.image = img                   # Set the image into the image panel 2 (MUST DO)

            except Exception as e:
                # logger.error('Unable to load image: ' + str(e))
                print('Unable to load image: ' + str(e))
                mystring = '\nUnable to load image: ' + str(e)
                self.outputText.insert(tk.END, mystring)
                self.outputText.see(tk.END)
                return

        else:
            print('No Image to load')

    def loadKmlButtClick(self):
        print('loadKmlButtClick button pressed')
        self.outputText.insert(tk.END, '\nloadKmlButtClick button pressed')
        self.outputText.see(tk.END)
        filename = self.requestKmlEntryVar.get()
        if len(self.requestKmlEntry.get()) != 0:
            try:
                for polygons in readPoly(filename):
                    polygons, desc = polygons
                    print(polygons)
            except:
                print('Unable to load KML/KMZ file')
                self.outputText.insert(tk.END, '\nUnable to load KML/KMZ file')
                self.outputText.see(tk.END)
        else:
            print('Nothing to load')
            self.outputText.insert(tk.END, '\nNothing to load')
            self.outputText.see(tk.END)

    def startCloudDetection(self):
        imgFilename = self.requestImgEntryVar.get()     # Get the image filename for checking
        if not imgFilename:                             # Check if any image file was input
            print('Please input a image file')
            self.outputText.insert(tk.END, '\nPlease input an image file')
            self.outputText.see(tk.END)
            return
        if not(imgFilename[-4:] == '.jpg' or imgFilename[-4:] == '.JPG' or imgFilename[-4:] == '.png' or imgFilename[-4:] == '.PNG' or imgFilename[-4:] == '.tif' or imgFilename[-4:] == '.TIF'):
            # Check if the image file is the correct format (jpg,png,tif)
            print('Image is not the correct format')
            self.outputText.insert(tk.END, '\nImage is not the correct format')
            self.outputText.see(tk.END)
            return

        if len(self.requestImgEntry.get()) != 0:
            # Get file size
            filesize = os.path.getsize(imgFilename)
            print('The image filesize is: %f' % filesize)
            self.outputText.insert(tk.END, '\nThe image filesize is: %f' % filesize)
            if filesize / 1e9 >= 4:     # Check if the file size is more than 4gb
                answer = messagebox.askquestion("Resize Image", "The current image size is enormous and will slow down detection process. Do you want to resize image?", icon='warning')
                if answer == 'yes':
                    self.resizeImage(imgFilename)
                elif answer == 'no':
                    print('Did not compress and continuing.')
                    self.outputText.insert(tk.END, '\nDid not compress and continuing. (Don\'t blame me if the program hangs)')
                    self.outputText.see(tk.END)

        if self.selectKMLVar.get() is 1:  # If KML/KMZ is selected as the input
            kmlFilename = self.requestKmlEntryVar.get()     # Get the kml/kmz filename for checking
            if not kmlFilename:                             # Check if the kml/kmz file was input
                print('Please input a kml/kmz file')
                self.outputText.insert(tk.END, '\nPlease input a kml/kmz file')
                self.outputText.see(tk.END)
                return
            if not(kmlFilename[-4:] == '.kml' or kmlFilename[-4:] == '.KML' or kmlFilename[-4:] == '.kmz' or kmlFilename[-4:] == '.KMZ'):
                # Check if the kml/kmz file is the correct format
                print('KML/KMZ file is not the correct format')
                self.outputText.insert(tk.END, '\nKML/KMZ file is not the correct format')
                self.outputText.see(tk.END)
                return

            del kmlFilename
        del imgFilename

        print('Starting Cloud Detection')
        self.outputText.insert(tk.END, '\nStarting Cloud Detection')
        self.outputText.see(tk.END)
        # cloud_map, cloud_percent = Cloud_Detection_Algo(self.requestImgEntryVar.get(), self.requestKmlEntryVar.get(), self.reflec_gain_var.get(), self.outputText)

        if self.selectKMLVar.get() is 1:
            outputText, polygons = readKML(self.requestKmlEntryVar.get())   # Read the KML/KMZ file
            self.outputText.insert(tk.END, outputText)
            self.outputText.see(tk.END)
            self.update()
        elif self.selectKMLVar.get() is 2:
            outputText = '\nUsing the whole image as AOI'
            self.outputText.insert(tk.END, outputText)
            self.outputText.see(tk.END)
            self.update()
            polygons = []   # Leave as empty for now
        elif self.selectKMLVar.get() is 3:
            outputText = '\nDetecting drawn AOI'
            self.outputText.insert(tk.END, outputText)
            self.outputText.see(tk.END)
            self.update()
            polygons = []   # Leave as empty for now

        aoiFinal, dataset, row_len, col_len, outputText = createNpArray(self.requestImgEntryVar.get(), polygons, self.selectKMLVar.get())    # Create the aoi mask/boundary from the kml file
        self.outputText.insert(tk.END, outputText)
        self.outputText.see(tk.END)
        self.update()

        cloud_map, outputText = createCloudMap(row_len, col_len)    # Create empty cloud map
        self.outputText.insert(tk.END, outputText)
        self.outputText.see(tk.END)
        self.update()

        cloud_map, combinedmapvalues, peak_reflec, outputText = computeX(dataset, cloud_map, row_len, col_len, self.reflec_gain_var.get())
        self.outputText.insert(tk.END, outputText)
        self.outputText.see(tk.END)
        self.update()

        cloud_map, outputText = computeY(cloud_map, combinedmapvalues, col_len, self.reflec_gain_var.get(), peak_reflec)
        self.outputText.insert(tk.END, outputText)
        self.outputText.see(tk.END)
        self.update()

        self.outputText.insert(tk.END, '\nEroding and Dilating')
        self.outputText.see(tk.END)
        self.update()
        cloud_map = erodeAndDilate(cloud_map, aoiFinal)

        percent, outputText = calculateCloudPercentage(cloud_map, aoiFinal)
        self.outputText.insert(tk.END, outputText)
        self.outputText.see(tk.END)
        self.update()

        # plt.imsave('filename.png', np.array(data).reshape(50, 50), cmap=cm.gray)
        plt.imsave(os.getcwd() + 'cloud_output.png', np.array(cloud_map), cmap=cm.gray)

        img2 = Image.open(os.getcwd() + 'cloud_output.png')
        self.img_copy2 = img2.copy()
        labelWidth2 = self.imgPanel2.winfo_width()          # Get the image panel Width
        labelHeight2 = self.imgPanel2.winfo_height()        # Get the image panel Height
        maxsize2 = (labelWidth2, labelHeight2)              # Create a tuple for the image panel diamensions
        img2 = img2.resize(maxsize2)                        # Resize the image to fit the image panel
        img2 = ImageTk.PhotoImage(img2)                     # Convert the image into a tkinker image
        self.imgPanel2.configure(image=img2)                # Set the image into the image panel
        self.imgPanel2.image = img2                         # Set the image into the image panel 2 (MUST DO)

    def saveOutputImages(self):
        # f = filedialog.asksaveasfile(initialdir=os.getcwd(), mode='w', defaultextension=".png")
        mydir = filedialog.askdirectory()
        if mydir is None:  # asksaveasfile return `None` if dialog closed with "cancel".
            return
        result = self.img_copy
        result.save(mydir+'/compressed.png')
        result = self.img_copy2
        result.save(mydir + '/cloudmap.png')

    def gui_resize(self, event):
        # w, h = event.width - 100, event.height - 100
        # self.c.config(width=w, height=h)
        img = self.img_copy
        labelWidth = self.imgPanel.winfo_width()    # Get the image panel Width
        labelHeight = self.imgPanel.winfo_height()  # Get the image panel Height
        maxsize = (labelWidth, labelHeight)         # Create a tuple for the image panel diamensions
        img = img.resize(maxsize)                   # Resize the image to fit the image panel
        img = ImageTk.PhotoImage(img)               # Convert the image into a tkinker image
        self.imgPanel.configure(image=img)          # Set the image into the image panel
        self.imgPanel.image = img                   # Set the image into the image panel 2 (MUST DO)

    def gui_resize2(self, event):
        img2 = self.img_copy2
        labelWidth2 = self.imgPanel2.winfo_width()  # Get the image panel Width
        labelHeight2 = self.imgPanel2.winfo_height()  # Get the image panel Height
        maxsize2 = (labelWidth2, labelHeight2)  # Create a tuple for the image panel diamensions
        img2 = img2.resize(maxsize2)  # Resize the image to fit the image panel
        img2 = ImageTk.PhotoImage(img2)  # Convert the image into a tkinker image
        self.imgPanel2.configure(image=img2)  # Set the image into the image panel
        self.imgPanel2.image = img2  # Set the image into the image panel 2 (MUST DO)


if __name__ == "__main__":
    app = CloudDetectionMain(None)
    app.title('Cloud Detection')
    app.mainloop()

# root = tk.Tk()
#
# logo = tk.PhotoImage(file='/Users/austintang/Documents/CloudDetectionGUI/Images/python_icon4.gif')
#
# w = tk.Label(root, text='Hello Tkinter!!')
# w.pack()
#
# w1 = tk.Label(root, image=logo).pack(side='right')
#
# explanation = """shuadgias shuba shuba shuba nom nom wubw wubw wub wub was
# asjdohfoasdhfoosaidjf
# shimmu shimmy ya shimmy yay shimmy yo
# shibidopapdepapapka
# skrrrrt pap skiddy pip pap pap. SKIIIYAAA."""
#
# w2 = tk.Label(root,
#               justify=tk.LEFT,
#               padx=10,
#               text=explanation).pack(side='left')
#
# tk.Label(root,
#          text="Red Text in Times Font",
#          fg="red",
#          font="Times").pack(side='bottom')
#
# tk.Label(root,
#          text="Green Text in Helvetica Font",
#          fg="light green",
#          bg="dark green",
#          font="Helvetica 16 bold italic").pack(side='bottom')
#
# tk.Label(root,
#          text="Blue Text in Verdana bold",
#          fg="blue",
#          bg="yellow",
#          font="Verdana 10 bold").pack(side='bottom')
#
# counter = 0
#
#
# def counter_label(label):
#     def count():
#         global counter
#         counter += 1
#         label.config(text=str(counter))
#         label.after(1000, count)
#
#     count()
#
#
# # root = tk.Tk()
# root.title("Counting Seconds")
# label = tk.Label(root, fg="green")
# label.pack()
# counter_label(label)
# button = tk.Button(root, text='Stop', width=25, command=root.destroy)
# button.pack()
#
# # master = tk.Tk()
# whatever_you_do = "Whatever you do will be insignificant, but it is very important that you do it.\n(Mahatma Gandhi)"
# msg = tk.Message(root, text = whatever_you_do)
# msg.config(bg='lightgreen', font=('times', 24, 'italic'))
# msg.pack()
# tk.mainloop()
#
#
# def write_slogan():
#     print("Tkinter is easy to use!")
#
# # root = tk.Tk()
# frame = tk.Frame(root)
# frame.pack()
#
# button = tk.Button(frame,
#                    text="QUIT",
#                    fg="red",
#                    command=quit)
# button.pack(side=tk.LEFT)
# slogan = tk.Button(frame,
#                    text="Hello",
#                    command=write_slogan)
# slogan.pack(side=tk.LEFT)
#
#
# root.mainloop()
