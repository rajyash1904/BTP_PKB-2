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




    #Partitioning
    start = time.time()
    partitioned_points, cube_positions = load_points(scaling_file, cube_size, min_num)
    print("Partition: {}s".format(round(time.time()-start,4)))
    if scale != 1:
        os.system("rm "+scaling_file) 



    # Voxelization
    start = time.time()
    cubes = points2voxels(partitioned_points, cube_size)
    points_numbers = np.sum(cubes, axis=(1,2,3,4)).astype(np.uint16)
    print("Voxelisation: {}s".format(round(time.time()-start,4)))


    print('Cubes shape: {}'.format(cubes.shape))
    print('points numbers (sum/mean/max/min): {} {} {} {}'.format(points_numbers.sum(), round(points_numbers.mean()), points_numbers.max(), points_numbers.min()))


    return cubes, cube_positions, points_numbers


def postprocess(output_file, cubes, points_numbers, cube_positions, scaling_file, cube_size, rho, fixed_thres=None):
    """Classify voxels to occupied or free, then extract points and write to file.
    Input:  deocded cubes, cube positions, points numbers, cube size and rho=ouput numbers/input numbers.
    """
    