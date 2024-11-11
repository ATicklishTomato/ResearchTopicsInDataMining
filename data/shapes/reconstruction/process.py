import open3d as o3d
import urllib.request
import os

# Step 1: Download a sample point cloud file (in PLY format)
url = "https://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz"  # Example point cloud (Stanford Bunny)
output_file = "bunny.tar.gz"
if not os.path.exists(output_file):
    print("Downloading point cloud data...")
    urllib.request.urlretrieve(url, output_file)
    os.system(f"tar -xzf {output_file}")

# Extracted PLY file name (adjust this based on the contents of the downloaded file)
ply_file = "bunny/reconstruction/bun_zipper.ply"

# Step 2: Read the PLY file using Open3D
print("Loading point cloud...")
pcd = o3d.io.read_point_cloud(ply_file)

# Step 3: Estimate normals if they are not present
print("Estimating normals...")
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

# Step 4: Save the point cloud as an XYZ file with normals
xyz_output = "output_pointcloud.xyz"
o3d.io.write_point_cloud(xyz_output, pcd)
print(f"Point cloud saved as {xyz_output}")

# Step 5: Visualize the point cloud
print("Visualizing point cloud...")
o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Visualization")
