import logging
import numpy as np
import plyfile
import skimage.measure
import time
import torch


class SDFDecoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, coords):
        model_in = {'coords': coords}
        return self.model(model_in)['model_out']

def create_mesh_with_pointcloud(
    model, filename, N=256, max_batch=64 ** 3, offset=None, scale=None, threshold=0.01
):
    start = time.time()
    pointcloud_filename = filename + ".xyz"

    decoder = SDFDecoder(model)
    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()

        samples[head : min(head + max_batch, num_samples), 3] = (
            decoder(sample_subset)
            .squeeze()
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    print("Sampling took: %f" % (end - start))

    # Convert the SDF samples to a point cloud
    convert_sdf_samples_to_pointcloud(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        pointcloud_filename,
        threshold,
        offset,
        scale
    )

def convert_sdf_samples_to_pointcloud(
    pytorch_3d_sdf_tensor, voxel_grid_origin, voxel_size, pointcloud_filename_out, threshold=0.01, offset=None, scale=None
):
    """
    Convert sdf samples to a point cloud file with normals.

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n, n, n)
    :param voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :param voxel_size: float, the size of the voxels
    :param pointcloud_filename_out: string, path of the filename to save to
    :param threshold: float, the absolute threshold around zero to include points in the point cloud
    :param offset: Optional offset to apply to the point cloud coordinates.
    :param scale: Optional scale to apply to the point cloud coordinates.
    """
    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    # Find points where the SDF is close to zero (within a threshold)
    sdf_mask = np.abs(numpy_3d_sdf_tensor) < threshold
    indices = np.argwhere(sdf_mask)

    # Convert voxel indices to coordinates
    points = voxel_grid_origin + indices * voxel_size

    # Compute normals using the gradient of the SDF
    gradients = np.gradient(numpy_3d_sdf_tensor, voxel_size, axis=(0, 1, 2))
    normals = np.stack(gradients, axis=-1)
    normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)

    # Extract normals for points close to zero level set
    normals = normals[sdf_mask]

    # Apply offset and scale to points if provided
    if scale is not None:
        points /= scale
    if offset is not None:
        points -= offset

    # Save the points with normals as an .xyz file
    with open(pointcloud_filename_out, 'w') as f:
        for point, normal in zip(points, normals):
            f.write(f"{point[0]} {point[1]} {point[2]} {normal[0]} {normal[1]} {normal[2]}\n")

    print(f"Point cloud with normals saved as {pointcloud_filename_out}")