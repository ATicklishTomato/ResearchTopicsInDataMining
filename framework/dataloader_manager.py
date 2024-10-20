import logging
import os

from torch.utils.data import DataLoader

from data.audio.data import AudioFile
from data.images.reconstruction.dataset import Reconstruction
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

                img_dataset = Reconstruction(
                    path=os.path.join(data_path, selected_image)
                )
                coord_dataset = Implicit2DWrapper(img_dataset, sidelength=500, compute_diff='all')
                return DataLoader(coord_dataset, shuffle=True, batch_size=args.batch_size, pin_memory=True, num_workers=0)
            else:
                logger.error(f"Data point {args.data_point} is out of range. Only {len(image_files)} files found.")
                raise ValueError(f"Data point {args.data_point} is out of range.")
        else:
            logger.error(f"Data path {data_path} does not exist.")
            raise FileNotFoundError(f"Data path {data_path} does not exist.")
    elif args.data == "audio":
        logger.debug(f"Getting dataloader for audio reconstruction")

        data_path = f"data/audio/files/"

        if os.path.isdir(data_path):
            # Sort the .WAV files in the data_path
            audio_files = sorted([f for f in os.listdir(data_path) if f.endswith(".wav")])

            logger.debug(f"Found {len(audio_files)} audio files: {audio_files}")

            # Ensure there are enough files for the specified data_point
            if args.data_point < len(audio_files):
                selected_audio = audio_files[args.data_point]
                print(f"Selected audio file: {selected_audio}")
            else:
                # Choose the first audio file
                selected_audio = audio_files[0]

            audio_dataset = AudioFile(filename=os.path.join(data_path, selected_audio))
            coord_dataset = ImplicitAudioWrapper(audio_dataset)

            logger.debug(f"Data point {args.data_point} selected, using audio file {selected_audio}")

            return DataLoader(coord_dataset, shuffle=True, batch_size=args.batch_size, pin_memory=True,
                                    num_workers=0)

    else:
        logger.error(f"Data type {args.data} not recognized")
        raise ValueError(f"Data type {args.data} not recognized")