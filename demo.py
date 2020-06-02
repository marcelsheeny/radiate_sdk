import radiate
import numpy as np
import os

# path to the sequence
root_path = '/media/marcel/df5725dc-6216-424d-842c-30fff5c71c5d/datasets/radiate/'
sequence_name = 'fog_6_0'

# time (s) to retrieve next frame
dt = 0.25

# load sequence
seq = radiate.Sequence(os.path.join(root_path, sequence_name))

# play sequence
for t in np.arange(seq.init_timestamp, seq.end_timestamp, dt):
    output = seq.get_from_timestamp(t)
    seq.vis_all(output)