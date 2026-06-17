import open3d as o3d
import numpy as np

ld = o3d.data.LivingRoomPointClouds()

pcds = []
for pcd_path in ld.paths:
    pcds.append(o3d.io.read_point_cloud(pcd_path))
temp = o3d.io.read_point_cloud(ld.paths[0])
o3d.visualization.draw_geometries([temp])

# Testing generating normal on bunny mesh
bunny = o3d.data.BunnyMesh()
gt_mesh = o3d.io.read_triangle_mesh(bunny.path)

pcd = gt_mesh.sample_points_poisson_disk(5000)
pcd.normals = o3d.utility.Vector3dVector(np.zeros(
    (1, 3)))  # invalidate existing normals"

pcd.estimate_normals()
pcd.orient_normals_consistent_tangent_plane(100)

with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9)
    
print(mesh)
o3d.visualization.draw_geometries([mesh])

o3d.visualization.draw_geometries([pcd], point_show_normal=True)