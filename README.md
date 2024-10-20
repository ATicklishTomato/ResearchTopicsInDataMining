# ResearchTopicsInDataMining
Code for the course 2AMM20 - Research Topics in Data Mining at TU Eindhoven. The code was written for the research group project in the Neural Fields topic. Please check the authors in the README to find the group members.

## Setup
Start by setting up a conda environment using the command `conda create -n RTDM python=3.10` and activate it using `conda activate RTDM`.
Install the dependencies using the command `pip install -r requirements.txt`.

## Usage
The code can be run through `run.py`. A series of arguments can be passed to the script to specify the model, dataset, and other parameters. The full help text is shown below. It can be accessed by running `python run.py -h` or `python run.py --help`.
```
usage: run.py [-h] [--data {images}] [--model {siren,mfn,fourier,kan,basic}] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--lr LR] [--seed SEED] [--device DEVICE]
              [--verbose {10,20,30,40}] [--save] [--load] [--save_dir SAVE_DIR] [--skip_train] [--skip_test]

Train and test a neural fields model on a chosen dataset with certain parameters

options:
  -h, --help            show this help message and exit
  --data {images}       Type of data to train and test on. Default is images
  --data_point DATA_POINT
                        Choose the index of the data_point to train on.
  --data_fidelity {low,medium,high}
                        Choose the fidelity of the data point to train on.
  --model {siren,mfn,fourier,kan,basic}
                        Type of model to use. Options are for SIREN, Multiplicative Filter Networks, Fourier Filter Banks, Kolmogorov-Arnold Networks, and a basic coordinate-   
                        MLP, respectively. Default is basic
  --sweep               Run a hyperparameter sweep. Default is False. Note: This will override any arguments passed related to sweep parameters
  --sweep_runs SWEEP_RUNS
                        Number of random runs to perform in the hyperparameter sweep. Default is 25
  --epochs EPOCHS       Number of epochs to train for. Default is 1001
  --batch_size BATCH_SIZE
                        Batch size for training. Default is 1
  --lr LR               Learning rate for training. Default is 1e-3
  --seed SEED           Seed for random number generation. Default is 42
  --device DEVICE       PyTorch device to train on. Default is cuda
  --verbose {10,20,30,40}
                        Verbosity level for logging. Options are for DEBUG, INFO, WARNING, and ERROR, respectively. Default is INFO
  --save                Save the model and optimizer state_dicts (if applicable) after training. Default is False
  --load                Load the stored model and optimizer state_dicts (if applicable) before training and skip training. Default is False
  --skip_train          Skip training and only evaluate the model. Default is False
  --skip_test           Skip testing and only train the model. Default is False
  --wandb_api_key WANDB_API_KEY
                        Your personal API key for Weights and Biases. Default is None. Alternatively, you can leave this empty and store the key in a file in the project root   
                        called "wandb.login". This file will be ignored by git. NOTE: Make sure to keep this key private and secure. Do not share it or upload it to a public    
                        repository.
```

### SIREN
To train the SIREN model on image data, use:
```
python .\run.py --model siren
```

## Analyzing results
The training loop creates model summaries every couple hundred epochs.
This summary can be viewed using tensorboard, with the command:
```
tensorboard --logdir=./logs
```

## Authors
 - Minas Chamamtzoglou
 - Seppe Hannen
 - Nikos Perdikogiannis
 - David Wang
 - Luuk Wubben
