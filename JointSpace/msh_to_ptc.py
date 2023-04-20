import open3d as o3d
import numpy as np
import os
from os import path

path_to_obj = '/projects/0/vvasileiou4/ABO/3dmodels/pointclouds'
for folder in os.listdir('/projects/0/vvasileiou4/ABO/3dmodels/original'):
    for file in os.listdir('/projects/0/vvasileiou4/ABO/3dmodels/original/'+folder):
        path_to_glb = '/projects/0/vvasileiou4/ABO/3dmodels/original/'+folder+'/'+file
        glb_prefix = path_to_glb.split('/')[-2]+'/'+path_to_glb.split('/')[-1][:-4]
        target_obj = path_to_obj + '/' + glb_prefix + ".txt"
        if path.exists(target_obj):
            continue
        else:
            mesh = o3d.io.read_triangle_mesh(path_to_glb)
            mesh.compute_vertex_normals()
            #o3d.visualization.draw_geometries([mesh])
            pcd = mesh.sample_points_uniformly(number_of_points=10000)
            #o3d.visualization.draw_geometries([pcd])
            array=np.asarray(pcd.points)

            with open(target_obj, mode='w') as f:
                for i in range(len(array)):
                    f.write("{}, {}, {}\n".format(array[i][0], array[i][1], array[i][2]))
        
            print("Done: ", target_obj)