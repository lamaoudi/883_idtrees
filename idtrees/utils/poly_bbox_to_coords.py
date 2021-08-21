import numpy as np
import pandas as pd

import read_tifs # To import all image paths from data utils

"""
# Reads in data from IDtrees challenge and creates csv file
"""
# Errors: 
# MLBS_7 and OSBS_16, _35 seem to not have bboxs?

# Define names of csv training data 
# TODO: read values from data_train_description.csvc
cols = ['uid', 'rgb_path', 'hsi_path', 'las_path', 'chm_path',
            'class_id', 'xmin', 'ymin', 'xmax', 'ymax',
            'width', 'height', 'class_code', 
            'scientific_name', 'site_id', 'taxon_rank', 'utm_zone', # Metadata on each tree:
            'nlcd_class', 
            'elevation', 'growth_form', 'plant_status', 
            'stem_diameter', 'tree_height', 'max_crown_diameter', 
            'ninety_crown_diameter', 'canopy_position']

# Initialize output dataframe
# TODO: set all data types right
df = pd.DataFrame(columns=cols)
df = df.set_index('uid') # Set uid as unique identifier

# Read in tree species data
# contains: ['indvdID', 'siteID', 'taxonID', 'taxonRank', 'utmZone', 'nlcdClass', 'elevation', 'growthForm', 'plantStatus', 'stemDiameter', 'height', 'maxCrownDiameter', 'ninetyCrownDiameter', 'canopyPosition']
path_in = 'data\\train\\Field\\train_data.csv'
df_in = pd.read_csv(path_in) 
cols_in = ['indvdID', 'siteID', 'taxonCode', 'taxonID', 'scientificName', 'taxonRank', 'utmZone', 'nlcdClass', 'elevation', 'growthForm', 'plantStatus', 'stemDiameter', 'height', 'maxCrownDiameter', 'ninetyCrownDiameter', 'canopyPosition']
cols_rn = ['uid', 'site_id', 'class_id', 'class_code','scientific_name', 'taxon_rank', 'utm_zone', 'nlcd_class', 'elevation', 'growth_form', 'plant_status', 'stem_diameter', 'tree_height', 'max_crown_diameter', 'ninety_crown_diameter', 'canopy_position'] 
df_in = df_in.rename(columns={col_in:col for col, col_in in zip(cols_rn, cols_in)}) # Rename column names 
df = pd.concat([df_in, df], axis=1) # Concatenate columsn of df and df_in
df = df.loc[:,~df.columns.duplicated()] # Remove duplicate columns from df, because they are empty 
df = df.set_index('uid') # Set uid as unique identifier # TODO: fix bug that i have to set uid twice; this shrinks dim of df by 1

# Read in tree species integer id
path_in = "data\\train\\Field\\taxonID_ScientificName.csv" 
df_in = pd.read_csv(path_in) 
df_in = df_in.rename(columns={col_in:col for col, col_in in zip(cols_rn, cols_in)}) # Rename column names
df = df.drop(['scientific_name', 'class_id'], axis=1) # Drop scientific name that occurs in both tables
df = df.join(df_in.set_index(['class_code']), on=['class_code']) # Concatenate to existing dataframe


import arcpy
from arcpy import env
from arcpy.sa import *

env.overwriteOutput = True

# Define paths
env.workspace = "C:\\Users\\Bjoern\\Documents\\MIT\\ai4earth\\idtrees\\"
#env.workspace = "\\"

# Print coordinates of each bounding box in bbox_poly shapefile
# Import database
def get_bboxs(bbox_poly, df):
    # Input: 
    # bbox_poly - string: Name of shape file with bbox polygons
    # df - pd.DataFrame: df in which to insert data
    # Output:
    # df - pd.DataFrame: df in which to insert data
    cols_bbox_poly = ['FID', "SHAPE@", 'indvdID']
    with arcpy.da.SearchCursor(bbox_poly, (cols_bbox_poly)) as cursor:
        non_exist_ids = []
        for row in cursor:
            # Copy and transform bounding box values
            # ArcGIS has origin in bottom-left image corner, x-axis along width, y-axis along height
            # Ours has origin in top-lef image corner, x-axis along width, y-axis along height

            # TODO transfrom from lat, lon to im coordinates 
            uid = str(row[2])
            ext = row[1].extent

            ids = ['xmin', 'xmax', 'ymin', 'ymax']
            vals = [ext.XMin, ext.XMax, ext.YMax, ext.YMin]
            try:
                df.loc[uid, ids] = vals
            except:
                # Throws away ~30 data entries
                print('Tree with UID '+uid+' not found, skipping entry.')
                non_exist_ids.append(uid)
        print('List of ids that have not been found: ', len(non_exist_ids), non_exist_ids)
    del cursor
    return df

bbox_path = 'data\\train\\ITC\\train'
sites = ['_MLBS','_OSBS']
bbox_polys = [bbox_path + site + '.shp' for site in sites]
for bbox_poly in bbox_polys:
    df = get_bboxs(bbox_poly,df)

# Read in properties of .tif images and store in dictionary

# Assing bounding boxes to images and convert lat lon to im coordinates, 
# TODO: Do this for rgb, hsi, las, and chm
def read_ims(df):
    modes = ['RGB', 'HSI', 'LAS', 'CHM']
    im_keys = ['rgb_path', 'hsi_path', 'las_path', 'chm_path']
    im_paths = {mode:read_tifs.get_im_paths(mode) for mode in modes}
    n_ims = len(im_paths) # Number of images

    # Get image coordinates and other metadata for each image
    for i, im_path in enumerate(im_paths['RGB']):
        print('converting im ', i, im_path)

        # (Assumption!: All images' orientation is aligned with lat, lon.)
        xmin_im = float(arcpy.GetRasterProperties_management(im_path, "LEFT").getOutput(0))
        xmax_im = float(arcpy.GetRasterProperties_management(im_path, "RIGHT").getOutput(0))
        ymin_im = float(arcpy.GetRasterProperties_management(im_path, "TOP").getOutput(0))
        ymax_im = float(arcpy.GetRasterProperties_management(im_path, "BOTTOM").getOutput(0))
        #Sample Output: ('TOP', <Result '4134999'>)

        # Set condition to find all trees who's bounding box is inside the image
        correct_ims = True
        slk = .5 if correct_ims else 0. # ASSUMPTION: there can be up to .5 degree slack in the boundary boxes
        # TODO: Calculate y and x offset per image.
        #y_off = .323 # Offset of bboxs in y-axis (bbox are too high) in degree, bc of misalignment of bbox and images, visible in arcgis
        is_bbox = (df['xmin'] >= xmin_im-slk) & (df['xmax'] <= xmax_im+slk) & (df['ymin'] <= ymin_im+slk) & (df['ymax'] >= ymax_im-slk)
        n_bboxs = int(df.loc[is_bbox].shape[0])
        if n_bboxs == 0: print('[Warning] No bounding boxes found for image ', im_path)
        # print('found ' + str(nbboxs) + ' bboxs')

        # Set image width and height
        width = int(arcpy.GetRasterProperties_management(im_path, "COLUMNCOUNT").getOutput(0))
        height = int(arcpy.GetRasterProperties_management(im_path, "ROWCOUNT").getOutput(0))
        df.loc[is_bbox, 'width'] = int(width)
        df.loc[is_bbox, 'height'] = int(height)# * np.ones(df_cp.shape[0],dtype=int) 

        # Transform bbox coords from lat., lon. to im. coords. (see data_train_description for im coords.)
        df.loc[is_bbox, 'xmin'] = (df.loc[is_bbox].xmin - xmin_im) / (xmax_im - xmin_im)
        df.loc[is_bbox, 'xmax'] = (df.loc[is_bbox].xmax - xmin_im) / (xmax_im - xmin_im)
        df.loc[is_bbox, 'ymin'] = (ymin_im - df.loc[is_bbox].ymin) / (ymin_im - ymax_im)
        df.loc[is_bbox, 'ymax'] = (ymin_im - df.loc[is_bbox].ymax) / (ymin_im - ymax_im)

        # Normalize from [0,1] float to [0,width] integer
        x_cols, y_cols = ['xmin', 'xmax'], ['ymin', 'ymax']
        df.loc[is_bbox, x_cols] = (width * df.loc[is_bbox, x_cols]).round(0).astype(int)
        df.loc[is_bbox, y_cols] = (height * df.loc[is_bbox, y_cols]).round(0).astype(int)

        # Lower and upper bound bbox coordinates. Necessary because of offset.
        # TODO: Properly correct the bbox offset.
        if correct_ims:
            df.loc[is_bbox, x_cols] = df.loc[is_bbox, x_cols].clip(0, width)
            df.loc[is_bbox, y_cols] = df.loc[is_bbox, y_cols].clip(0, height)

        # Assign image paths
        for im_key, mode in zip(im_keys, modes):
            df.loc[is_bbox, im_key] = im_paths[mode][i]

    # TODO: find all bboxs that have not been found in an image
    print('Number of bounding boxes not found: ', df.width.isnull().sum(), ' for: ',  df[df.rgb_path.isnull()].index.values)
    df.drop(df[df.rgb_path.isnull()].index, inplace = True)
    return df

df = read_ims(df)

# Dump dataframe to csv utf-8 file
df.to_csv(r'data/data_train_mitree.csv', index=True)
import pdb; pdb.set_trace()

"""
# Additional metadata in tif.
# (RGB and HSI, band 1 has unknown sensorname, acquisition date, wavelength)
# prop_names = ["TOP", "RIGHT", "BOTTOM", "LEFT", # Corresponds to ymin, xmax, ymax, xmin in im coords
         #"CELLSIZEX", "CELLSIZEY",
         "VALUETYPE", # ArcGIS valuetype, e.g., 9 = 32-bit floating point
         "COLUMNCOUNT", "ROWCOUNT"] # I.e., width, height of image
         #"BANDCOUNT","SENSORNAME", "ACQUISITIONDATE", "WAVELENGTH", "MINIMUM", "MAXIMUM",],
"""
# TODO: Delete. Describe() is not necessary
"""
desc = arcpy.Describe(im)
print("desc")
print([(child.name, child.dataType) for child in desc.children])
"""

