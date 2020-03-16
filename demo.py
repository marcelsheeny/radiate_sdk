import radiate
import numpy as np
import cv2

folder_path = '/media/marcel/df5725dc-6216-424d-842c-30fff5c71c5d/Dropbox/RES_EPS_PathCad/datasets/radiate/city_3_2'

seq = radiate.Sequence(folder_path)

for t in np.arange(seq.init_timestamp, seq.end_timestamp, 0.1):
    output = seq.get_from_timestamp(t)

    seq.vis_all(output)