import os
import numpy as np 
import sys
import time
import random


def preprocess(input_file, scale, cube_size, min_num) :

    prefix=input_file.split('/')[-1].split('_')[0]+str(random.randint(1,100))
    print('==== Preprocess ====')

    #scaling
    start =time.time()
    if scale == 1:
        scaling_file = input_file
    else:
        pc = load_ply_data(input_file)
        pc_down = np.round(pc.astype('float32')*scale)
        pc_down = np.unique(pc_down, axis=0)
        scaling_file = prefix+'downscaling.ply'
        write_ply_data(scaling_file, pc_down)
    print("Scaling: {}s".format(round(time.time()-start,4)))        