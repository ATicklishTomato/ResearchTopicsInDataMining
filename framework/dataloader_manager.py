import logging
import os

from torch.utils.data import DataLoader

from data.audio.reconstruction.dataset import Reconstruction as AudioReconstruction
from data.images.reconstruction.dataset import Reconstruction as ImageReconstruction
from data.shapes.reconstruction.dataset import Reconstruction as ShapeReconstruction
from data.utils import Implicit2DWrapper, ImplicitAudioWrapper

logger = logging.getLogger(__name__)

def get_dataloader(args):
    logger.setLevel(args.verbose)
    logger.info("Getting relevant dataloaders")

    if args.data == "images":
        logger.debug(f"Getting dataloader for image reconstruction")

        data_path = f"data/images/reconstruction/files/{args.data_fidelity}"

        if os.path.isdir(data_path):
            # Sort the .JPEG files in the data_path
            image_files = sorted([f for f in os.listdir(data_path) if f.endswith(".JPEG")])

            # Ensure there are enough files for the specified data_point
            if args.data_point < len(image_files):
                selected_image = image_files[args.data_point]
                print(f"Selected image file: {selected_image}")

                img_dataset = ImageReconstruction(
                    path=os.path.join(data_path, selected_image)
                )
                coord_dataset = Implicit2DWrapper(img_dataset, sidelength=500, compute_diff='all')
                return DataLoader(coord_dataset, shuffle=True, batch_size=args.batch_size, pin_memory=True, num_workers=0)
            else:
                logger.error(f"Data point {args.data_point} is out of range. Only {len(image_files)} files found.")
                raise ValueError(f"Data point {args.data_point} is out of range.")
    elif args.data == "audio":
        logger.debug(f"Getting dataloader for audio reconstruction")

        audio_dir = "data/audio/reconstruction/files/"

        if args.data_point < len(os.listdir(audio_dir)):
            audio_file = os.path.join(audio_dir, os.listdir(audio_dir)[args.data_point])
        else:
            # Choose the first audio file
            audio_file = os.path.join(audio_dir, os.listdir(audio_dir)[0])

        if os.path.isfile(audio_file):
            audio_dataset = AudioReconstruction(
                path=audio_file
            )
            coord_dataset = ImplicitAudioWrapper(audio_dataset)
            return DataLoader(coord_dataset, shuffle=True, batch_size=args.batch_size, pin_memory=True,
                              num_workers=0)
        else:
            logger.error(f"Data path {audio_file} does not exist.")
            raise FileNotFoundError(f"Data path {audio_file} does not exist.")
    elif args.data == "shapes":
        logger.debug(f"Getting dataloader for shapes data")

        data_path = f"data/shapes/reconstruction/files/"

        if os.path.isdir(data_path):
            # Sort the .xyz files in the data_path
            shape_files = sorted([f for f in os.listdir(data_path) if f.endswith(".xyz")])

            logger.debug(f"Found {len(shape_files)} shape files: {shape_files}")

            # Ensure there are enough files for the specified data_point
            if args.data_point < len(shape_files):
                selected_sdf = shape_files[args.data_point]
                logger.info(f"Selected SDF file: {selected_sdf}")
            else:
                # Choose the first file
                selected_sdf = shape_files[0]

            sdf_dataset = ShapeReconstruction(
                pointcloud_path=os.path.join(data_path, selected_sdf), 
                on_surface_points=1500
            )

            logger.debug(f"Data point {args.data_point} selected, using shape file {selected_sdf}")

            return DataLoader(sdf_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)
    else:
        logger.error(f"Data type {args.data} not recognized")
        raise ValueError(f"Data type {args.data} not recognized")