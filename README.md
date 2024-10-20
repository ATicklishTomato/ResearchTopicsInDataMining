# ResearchTopicsInDataMining
Code for the course 2AMM20 - Research Topics in Data Mining at TU Eindhoven. The code was written for the research group project in the Neural Fields topic. Please check the authors in the README to find the group members.

## Setup
Start by setting up a conda environment using the command `conda create -n RTDM python=3.10` and activate it using `conda activate RTDM`.
Install the dependencies using the command `pip install -r requirements.txt`.

## Usage
The code can be run through `run.py`. A series of arguments can be passed to the script to specify the model, dataset, and other parameters. The full help text is shown below. It can be accessed by running `python run.py -h` or `python run.py --help`.
```
usage: run.py [-h] [--data {images}]
[--data_point DATA_POINT]
[--data_fidelity {low,medium,high}]
[--model {siren,mfn,fourier,kan,basic}]
[--sweep] 
[--sweep_runs SWEEP_RUNS]
[--epochs EPOCHS] 
[--batch_size BATCH_SIZE]
[--lr LR] 
[--seed SEED]
[--device DEVICE]
[--verbose {10,20,30,40}]
[--save]
[--load] 
[--save_dir SAVE_DIR] 
[--skip_train] 
[--skip_test]
[--wandb_api_key WANDB_API_KEY]

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

### Run a model
To run a model on image data, use the following command:
```
python run.py --model siren
```
Other models and configurations can be set up using the CLI arguments defined above.

### Hyperparameter sweep
To run a hyperparameter sweep, use the following command:
```
python run.py --model siren --sweep
```
Other models can be used for the sweep by changing the `--model` argument.
It is also possible to choose the number of random runs to perform in the sweep using the `--sweep_runs` argument.

## Analyzing results
The results are either stored in the `out` directory or in the Weights and Biases dashboard. 
To analyze the results, you can check the figures and logs in the `out` directory or the Weights and Biases dashboard.
Weights & Biases results are also stored locally in the `wandb` directory for every run, 
but the dashboard provides a more user-friendly interface.

## Snellius supercomputer usage
To run the code on the Snellius supercomputer, you need to copy the code to the supercomputer using `scp`.
After copying the code, you can run the code using the following command:
```bash
sbatch snellius_job.bash
```
**NOTE:** Before running, ensure you've updated the relevant `SBATCH` flags in the `snellius_job.bash` script, as well as the python execution command.

You can then check the status of jobs started by your user using the `squeue` command.
```bash
squeue -u <username>
```

You can then check the status of the specific job with the `-j` flag.
```bash
squeue -j <job_id>
```

You can cancel the job using the `scancel` command.
```bash
scancel <job_id>
```

### script setup
The `snellius_job.bash` script is set up to run the code on the Snellius supercomputer.
To pass parameters, we use `#SBATCH` flags in the script:
- `#SBATCH --account=my_snellius_account` to specify the account to use
- `#SBATCH --time=2:00:00` to specify the maximum time the job can run
- `#SBATCH -p gpu_mig` to specify the partition to use. `gpu_mig` uses GPU partitions. `gpu`uses whole GPUs.
- `#SBATCH -N 1` to specify the number of nodes to use
- `#SBATCH --tasks-per-node 1` to specify the number of tasks per node
- `#SBATCH --gpus=1` to specify the number of GPUs to use
- `#SBATCH --output=R-%x.%j.out` to specify the output file

More information on how to set up the script for different environments, e.g. using one or multiple CPUs, can be found in the [Snellius documentation](https://servicedesk.surf.nl/wiki/display/WIKI/Example+job+scripts).

## Authors
 - Minas Chamamtzoglou
 - Seppe Hannen
 - Nikos Perdikogiannis
 - David Wang
 - Luuk Wubben
