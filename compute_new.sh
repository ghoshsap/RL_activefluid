#!/bin/bash
# FILENAME:  vae2.sh

#SBATCH -J 128
#SBATCH --nodes=1
##SBATCH -n 12
#SBATCH --ntasks-per-node=6
#SBATCH --account=hagan-lab
#SBATCH --partition=hagan-gpu
#SBATCH --qos=medium
#SBATCH --nodelist=gpu-l40-4-0
#SBATCH --gres=gpu:L40
##SBATCH --gres=gpu:RTX2:1
#SBATCH --time=3-00:00:00
##SBATCH --exclusive 
# export NCCL_DEBUG=INFO

echo "NODELIST="${SLURM_NODELIST}


#source ~/.bashrc

echo $TMP
echo $TMPDIR
echo $TMP


DIR_NAME=$1

# Check if the directory name was provided
if [ -z "$DIR_NAME" ]; then
    echo "No directory name provided. Exiting."
    exit 1
fi

# Create the directory if it doesn't exist
mkdir -p $DIR_NAME


#export RAY_SESSION_DIR=/scratch0/ray_sessions

echo "activating cupss environment"
module load cuda/12.4
module load share_modules/FFTW/3.3.8_gcc_sp
#nvcc solver_server.cpp -I./ -I/opt/ohpc/pub/libs/gnu7/openmpi3/boost/1.67.0/include/ -I/share/software/utilities/FFTW/3.3.8_gcc_sp/include/ -L../cuPSS/src/ -lcufft -lfftw3f -lcurand -lcupss -lstdc++fs -O2
./flat_init > $DIR_NAME/cupss_output_$SLURM_NODELIST.txt 2> $DIR_NAME/cupss_error_$SLURM_NODELIST.txt &

#module purge
sleep 20

echo "activating rl environment"
source /work/saptorshighosh/miniconda3/bin/activate
conda activate rl2
python3 ray_flow_noise.py > $DIR_NAME/ray_o.txt 2> $DIR_NAME/ray_e.txt


