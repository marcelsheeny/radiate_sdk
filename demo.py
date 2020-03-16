import radiate
import numpy as np
import os

ROOT_PATH = 'path/to/dataset/root/'
sequence_name = 'city_3_2'

folder_path = os.path.join(ROOT_PATH, sequence_name)

seq = radiate.Sequence(folder_path)

for t in np.arange(seq.init_timestamp, seq.end_timestamp, 0.1):
    output = seq.get_from_timestamp(t)

    seq.vis_all(output)