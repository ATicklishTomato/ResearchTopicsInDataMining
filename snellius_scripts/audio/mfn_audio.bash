#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --tasks-per-node 1
#SBATCH --gpus=1
#SBATCH --output=R-%x.%j.out

module load 2022
module load Miniconda3/4.12.0
module load Python/3.10.4-GCCcore-11.3.0


# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/sw/arch/RHEL8/EB_production/2022/software/Miniconda3/4.12.0/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/sw/arch/RHEL8/EB_production/2022/software/Miniconda3/4.12.0/etc/profile.d/conda.sh" ]; then
        . "/sw/arch/RHEL8/EB_production/2022/software/Miniconda3/4.12.0/etc/profile.d/conda.sh"
    else
        export PATH="/sw/arch/RHEL8/EB_production/2022/software/Miniconda3/4.12.0/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda init bash python=3.10
conda activate RTDM  # conda environment name

echo 'Conda environment activated, installing requirements';
pip install -r requirements.txt --user
echo 'Requirements installed, starting experiment';

echo "Print Python Version";
python --version

#echo 'Starting new experiment';
python run.py --model mfn --data audio --data_point 1 --epochs 1000 --lr 0.000001