'''
Purpose: Generates csv file of annotations from .txts
'''
import pandas as pd
import os
from tqdm import tqdm
import argparse

argparser = argparse.ArgumentParser(description='Generate annotations csv file from .txts')
argparser.add_argument('-i', '--input',
                       help='input dir name')
argparser.add_argument('-o', '--output',
                       help='output file name')

args = argparser.parse_args()

# parse arguments
INPUTDIR = args.input
FILENAME = args.output

'''
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
'''

df = pd.DataFrame(columns=['filename', 'class', 'truncated', 'occluded', 'observation angle', \
                           'xmin', 'ymin', 'xmax', 'ymax', 'height', 'width', 'length', \
                           'xloc', 'yloc', 'zloc', 'rot_y'])

def assign_values(filename, idx, list_to_assign):
    df.at[idx, 'filename'] = filename

    df.at[idx, 'class'] = list_to_assign[0]
    df.at[idx, 'truncated'] = list_to_assign[1]
    df.at[idx, 'occluded'] = list_to_assign[2]
    df.at[idx, 'observation angle'] = list_to_assign[3]

    # bbox coordinates
    df.at[idx, 'xmin'] = list_to_assign[4]
    df.at[idx, 'ymin'] = list_to_assign[5]
    df.at[idx, 'xmax'] = list_to_assign[6]
    df.at[idx, 'ymax'] = list_to_assign[7]

    # 3D object dimensions
    df.at[idx, 'height'] = list_to_assign[8]
    df.at[idx, 'width'] = list_to_assign[9]
    df.at[idx, 'length'] = list_to_assign[10]

    # 3D object location
    df.at[idx, 'xloc'] = list_to_assign[11]
    df.at[idx, 'yloc'] = list_to_assign[12]
    df.at[idx, 'zloc'] = list_to_assign[13]

    # rotation around y-axis in camera coordinates
    df.at[idx, 'rot_y'] = list_to_assign[14]

all_files = sorted(os.listdir(INPUTDIR))
pbar = tqdm(total=len(all_files), position=1)

count = 0
for idx, f in enumerate(all_files):
    pbar.update(1)
    file_object = open(INPUTDIR + f, 'r')
    file_content = [x.strip() for x in file_object.readlines()]

    for line in file_content:
        elements = line.split()
        if elements[0] == 'DontCare':
            continue

        assign_values(f, count, elements)
        count += 1

df.to_csv(FILENAME, index=False)
