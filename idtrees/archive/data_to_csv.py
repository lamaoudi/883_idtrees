" This script needs to be executed prior to training a model. The script puts the raw data into a csv file that can be more easiliy loaded by the dataloader function of the pytorch model."


import pandas as pd
import geopandas as gpd

def load_ids_from_csv(path, header=None, skiprows=None):
    
    df = pd.read_csv(path, header=header)
    labelsID = labels_frame.taxonCode
    labelsname = labels_frame.scientificName

    return df, labelsID, labelsname


if __name__ == '__main__':

    df, classID, className = load_ids_from_csv('/n/home00/nwendt/projects/idtrees/data/train/Field/taxonID_ScientificName.csv')
    a,b,c = load_ids_from_csv('/n/home00/nwendt/projects/idtrees/data/data_train_description.csv', header=[1,3,4])
    a,b,c = load_ids_from_csv('/n/home00/nwendt/projects/idtrees/data/data_train_description.csv', skiprows=[0,2,4])
    
    path = '/n/home00/nwendt/projects/idtrees/data/data_train_description.csv'
    df = pd.read_csv(path, skiprows=[0,2,4], encoding = "ISO-8859-1", nrows=1)   
    
    
def load_bboxes_from_shp(path)
    
    bbox_MLBS = gpd.read_file("/n/home00/nwendt/projects/idtrees/data/train/ITC/train_MLBS.shp")
    bbox_OSBS = gpd.read_file("/n/home00/nwendt/projects/idtrees/data/train/ITC/train_OSBS.shp")

    bbox_OSBS.geometry
    bbox_MLBS.geometry

    
    bbox_OSBS
    bbox_MLBS
