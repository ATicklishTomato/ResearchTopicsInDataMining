import logging
import time

import numpy as np
import plyfile
import torch
import wandb
from matplotlib import pyplot as plt
from skimage import measure
# import open3d as o3d

# from data.metrics import chamfer_hausdorff_distance, intersection_over_union
from data.utils import get_mgrid, lin2img

logger = logging.getLogger(__name__)

def make_contour_plot(array_2d,mode='log'):
    fig, ax = plt.subplots(figsize=(2.75, 2.75), dpi=300)

    levels = None
    colors = None

    if mode== 'log':
        num_levels = 6
        levels_pos = np.logspace(-2, 0, num=num_levels) # logspace
        levels_neg = -1. * levels_pos[::-1]
        levels = np.concatenate((levels_neg, np.zeros((0)), levels_pos), axis=0)
        colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=num_levels*2+1))
    elif mode== 'lin':
        num_levels = 10
        levels = np.linspace(-.5,.5,num=num_levels)
        colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=num_levels))

    sample = np.flipud(array_2d)
    CS = ax.contourf(sample, levels=levels, colors=colors)
    cbar = fig.colorbar(CS)

    ax.contour(sample, levels=levels, colors='k', linewidths=0.1)
    ax.contour(sample, levels=[0], colors='k', linewidths=0.3)
    ax.axis('off')
    return fig



def sdf_summary(model, model_input, total_steps, test=False):
    if test:
        label = 'test'
    else:
        label = total_steps

    create_mesh(model, f'./out/{model.__class__.__name__}_{label}')

    # Get the predicted coordinates from the created mesh
    # predicted_data = o3d.t.io.read_point_cloud(f'./out/{model.__class__.__name__}_{label}.ply')

    # Calculate the PSNR between the ground truth and predicted distance
    # iou = intersection_over_union(predicted_data.point.positions, model_input['coords'].squeeze())
    # chamfer, hausdorff, _, _, _, _ = chamfer_hausdorff_distance(model_input['coords'].squeeze(), predicted_data.point.positions)

    # if wandb.run is not None:
    #     if not test:
    #         wandb.log({'iou': iou, 'chamfer': chamfer, 'hausdorff': hausdorff}, step=total_steps)
    #     else:
    #         wandb.log({'iou': iou, 'chamfer': chamfer, 'hausdorff': hausdorff, 'test': True})
    # else:
    #     logger.info(f"IoU: {iou}, Chamfer: {chamfer}, Hausdorff: {hausdorff}")

    slice_coords_2d = get_mgrid(512)

    yz_slice_coords = torch.cat((torch.zeros_like(slice_coords_2d[:, :1]), slice_coords_2d), dim=-1)
    yz_slice_model_input = {'coords': yz_slice_coords.cuda()[None, ...]}

    yz_model_out = model(yz_slice_model_input)
    sdf_values = yz_model_out['model_out']
    sdf_values = lin2img(sdf_values).squeeze().cpu().numpy()
    fig = make_contour_plot(sdf_values)

    if wandb.run is not None:
        wandb.log({model.__class__.__name__ + "_" + label + "_" + 'yz_sdf_slice': wandb.Image(fig)})
    else:
        plt.savefig(f'./out/{model.__class__.__name__}_{label}_yz_sdf_slice.png')

    xz_slice_coords = torch.cat((slice_coords_2d[:, :1],
                                 torch.zeros_like(slice_coords_2d[:, :1]),
                                 slice_coords_2d[:, -1:]), dim=-1)
    xz_slice_model_input = {'coords': xz_slice_coords.cuda()[None, ...]}

    xz_model_out = model(xz_slice_model_input)
    sdf_values = xz_model_out['model_out']
    sdf_values = lin2img(sdf_values).squeeze().cpu().numpy()
    fig = make_contour_plot(sdf_values)

    if wandb.run is not None:
        wandb.log({model.__class__.__name__ + "_" + label + "_" + 'xz_sdf_slice': wandb.Image(fig)})
    else:
        plt.savefig(f'./out/{model.__class__.__name__}_{label}_xz_sdf_slice.png')

    xy_slice_coords = torch.cat((slice_coords_2d[:, :2],
                                 -0.75 * torch.ones_like(slice_coords_2d[:, :1])), dim=-1)
    xy_slice_model_input = {'coords': xy_slice_coords.cuda()[None, ...]}

    xy_model_out = model(xy_slice_model_input)
    sdf_values = xy_model_out['model_out']
    sdf_values = lin2img(sdf_values).squeeze().cpu().numpy()
    fig = make_contour_plot(sdf_values)

    if wandb.run is not None:
        wandb.log({model.__class__.__name__ + "_" + label + "_" + 'xy_sdf_slice': wandb.Image(fig)})
    else:
        plt.savefig(f'./out/{model.__class__.__name__}_{label}_xy_sdf_slice.png')

def create_mesh(
        decoder, filename, N=256, max_batch=64 ** 3, offset=None, scale=None
):
    start = time.time()
    ply_filename = filename

    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = {'coords': samples[head: min(head + max_batch, num_samples), 0:3].cuda()}

        samples[head: min(head + max_batch, num_samples), 3] = (
            decoder(sample_subset)['model_out']
            .squeeze()  # .squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    logger.info("sampling takes: %f" % (end - start))

    convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
        offset,
        scale,
    )

def convert_sdf_samples_to_ply(
        pytorch_3d_sdf_tensor,
        voxel_grid_origin,
        voxel_size,
        ply_filename_out,
        offset=None,
        scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    start_time = time.time()
    logger.debug(f"Input shape: {pytorch_3d_sdf_tensor.shape}")
    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    try:
        verts, faces, normals, values = measure.marching_cubes(
            numpy_3d_sdf_tensor, spacing=[voxel_size] * 3, method="lewiner"
        )
    except Exception as error:
        logger.warning("marching cubes failed: %s" % error)
        pass

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )

    if wandb.run is not None:
        wandb.log({ply_filename_out: wandb.Object3D(open(ply_filename_out))})