import argparse
import open3d as o3d

parser = argparse.ArgumentParser()

parser.add_argument("--filename",type= str,default='',dest="filename", help="filename for input and output ply file")
args = parser.parse_args()

input_file = './figs/'+args.filename+'.ply'
output_file = args.filename+'_rec.ply'

#print(input_file)
#print(output_file)
in_pcd = o3d.read_point_cloud(input_file)
out_pcd = o3d.read_point_cloud(output_file)

o3d.draw_geometries([in_pcd])
o3d.draw_geometries([out_pcd])