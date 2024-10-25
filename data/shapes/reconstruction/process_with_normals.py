import open3d as o3d
import numpy as np
import urllib.request
import os

# Step 1: Download a sample point cloud with normals
url = "https://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz"  # Example point cloud (Stanford Bunny)
output_file = "bunny.tar.gz"
if not os.path.exists(output_file):
    print("Downloading point cloud data...")
    urllib.request.urlretrieve(url, output_file)
    os.system(f"tar -xzf {output_file}")

# Extracted PLY file name (adjust this based on the contents of the downloaded file)
ply_file = "bunny/reconstruction/bun_zipper.ply"

# Step 2: Load the PLY file using Open3D
print("Loading point cloud...")
pcd = o3d.io.read_point_cloud(ply_file)

# Step 3: Check for and estimate normals if they are not present
if not pcd.has_normals():
    print("No normals found in the file. Estimating normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
else:
    print("Normals are already present in the point cloud.")

# Step 4: Normalize and scale the coordinates to fit within the range [-1, 1]
coords = np.asarray(pcd.points)
normals = np.asarray(pcd.normals)

coords -= np.mean(coords, axis=0, keepdims=True)
coord_max = np.amax(coords, axis=0, keepdims=True)
coord_min = np.amin(coords, axis=0, keepdims=True)
coords = (coords - coord_min) / (coord_max - coord_min)
coords -= 0.5
coords *= 2.0

# Step 5: Save the point cloud as an XYZ file with normals
xyz_output = "bunny_with_normals.xyz"
with open(xyz_output, 'w') as f:
    for point, normal in zip(coords, normals):
        f.write(f"{point[0]} {point[1]} {point[2]} {normal[0]} {normal[1]} {normal[2]}\n")

print(f"Point cloud with normals saved as {xyz_output}")

# Step 6: Visualize the point cloud
print("Visualizing point cloud...")
pcd.points = o3d.utility.Vector3dVector(coords)
pcd.normals = o3d.utility.Vector3dVector(normals)
o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Visualization")
