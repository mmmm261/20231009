import arcpy
import os

Path = r'F:\cgwx'
RasterPath = r'F:\cgwx'
mxdName = r'aa.mxd'
replaceLayerName = 'train_000051.tif'

arcpy.env.workspace = RasterPath
mxd = arcpy.mapping.MapDocument(os.path.join(Path, mxdName))
df = arcpy.mapping.ListDataFrames(mxd)[0]
layer = arcpy.mapping.ListLayers(mxd,replaceLayerName, df)[0]
rasters = arcpy.ListRasters()

for i, raster in enumerate(rasters):
    print(raster)
    layer.replaceDataSource(RasterPath, 'RASTER_WORKSPACE', raster)
    layer.name = raster

    arcpy.mapping.ExportToPNG(mxd, os.path.join(Path, raster[:6]), df, df_export_width=512,
                              df_export_height=512)

del mxd
