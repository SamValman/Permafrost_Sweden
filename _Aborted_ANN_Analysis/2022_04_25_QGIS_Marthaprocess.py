from qgis.PyQt import QtGui
from qgis import processing
###### Section One: load in rasters ####
# UAV (just using one for now: change fn (file name) to local drive location
UAVx = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\2022_03_martha\Swedish_permafrost_project_files\SD_vegetation.tif"

band1_fn = '' # can be made into a list and therefore iterated through each other section.


# use line below if you want to plot any of these
# iface.addRasterLayer(fn, 'layername')


##### Section 2: Vectorise ####
outfn = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\01_RA2021_2022\2022_03_arctic\Permafrost_Sweden\QGIS\QWorkingOutputs\tempSD.tif"
def polygonize(infn, outfn):
    processing.run("native:pixelstopolygons", \
    {'INPUT_RASTER': fn, \
    'RASTER_BAND':1,'FIELD_NAME':'VALUE','OUTPUT':'outfn'})

polygonize(UAVx, outfn)


##### Section Three: clip raster(bands) to layer (vectorUAV)####

### section Two: clip bands to UAV 

def cliptoUAV(band_fn, output):
    processing.run("gdal:cliprasterbymasklayer",\
    {'INPUT': band_fn ,
        clipping layer = outfn,
    \n 'PROJWIN':'419292.319800000,420217.319800000,7583374.519400000,7584233.519400000 [EPSG:32634]','NODATA':None,'OPTIONS':'','DATA_TYPE':0,'EXTRA':'','OUTPUT':'TEMPORARY_OUTPUT'})
    # above line diff

b1output = ''
cliptoUAV(band1_fn, b1output)

##### Section 4: vectorise these new smaller bandsizes ####
b1_vectorOutput = ''
polygonize(boutput, bvectoroutput)

##### Section 5: spatially join ####
def spatialJoinPerBand(uav_vector, band_vector):
    processing.run("qgis:joinattributesbylocation", {'INPUT':uav_vector, \
    'JOIN':band_vector, 'PREDICATE':[0], 'JOIN_FIELDS':[],'METHOD':0, 'DISCARD_NONMATCHING':TRUE, \
    'PREFIX':'', 'OUTPUT':TEMPORARY_OUTPUT'})

spatialJoinPerBand(UAV_vector, band_Vector)
spatialJoinPerBand(TEMPORARY_OUTPUT, Band_Vector)
''
SpatialJoinPerBand(penultimate_output, Band_vector)

##### Section 6: work out area

Simple calculator - need to look into 






##### MAY BE USEFUL ####
#processing.algorithmHelp("attributename")
    
