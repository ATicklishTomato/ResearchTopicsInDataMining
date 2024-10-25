import open3d as o3d
import numpy as np

# Load the point cloud data from the .xyz file
xyz_file = "bunny_reconstructed.xyz"
points = []

# Read the .xyz file
with open(xyz_file, 'r') as f:
    for line in f:
        x, y, z, nx, ny, nz = map(float, line.split())
        points.append([x, y, z, nx, ny, nz])

# Convert to a NumPy array
points = np.array(points)

# Extract coordinates and normals
coords = points[:, :3]
normals = points[:, 3:6]

# Create an Open3D point cloud object and assign points and normals
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(coords)
pcd.normals = o3d.utility.Vector3dVector(normals)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd], window_name="Bunny Point Cloud", width=800, height=600)
