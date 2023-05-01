# Proximal Policy Gradient Arborescence 
The official repo of PPGA! Implemented in PyTorch and run with [Brax](https://github.com/google/brax), a GPU-Accelerated
high-throughput simulator for rigid bodies. This project also contains a modified version of [pyribs](https://github.com/icaros-usc/pyribs),
a QD library, and implements a modified multi-objective, vectorized version of Proximal Policy Optimization (PPO) based off
of [cleanrl](https://github.com/vwxyzjn/cleanrl).

## Requirements
We use Anaconda to manage dependencies. 
```bash
conda env create -f environment.yml
conda activate ppga  
```
Then install this project's custom version of pyribs.
```bash
cd pyribs && pip install -e. && cd..
```
### CUDA 
This project has been tested on Ubuntu 20.04 with an NVIDIA RTX 3090 GPU. In order to enable GPU-Acceleration, your machine must support 
CUDA 11.X with minimum driver version 450.80.02 (Linux x86_64). See [here](https://docs.nvidia.com/deploy/cuda-compatibility/)
for more details on cuda compatibility. 

The environment.yml file intentionally contains no CUDA dependencies since this is a machine dependent property, and so 
jax-cuda and related CUDA packages must be installed by the user. We recommend installing one of the following jaxlib-cuda packages:
```bash
# for CUDA 11 and cuDNN 8.2 or newer
wget https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.3.25+cuda11.cudnn82-cp39-cp39-manylinux2014_x86_64.whl
pip install jaxlib-0.3.25+cuda11.cudnn82-cp39-cp39-manylinux2014_x86_64.whl

# OR 

# for CUDA 11 and cuDNN 8.0.5 or newer 
wget https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.3.25+cuda11.cudnn805-cp39-cp39-manylinux2014_x86_64.whl
pip install jaxlib-0.3.25+cuda11.cudnn805-cp39-cp39-manylinux2014_x86_64.whl
```

If you run into issues getting cuda-accelerated jax to work, please see the [jax github](https://github.com/google/jax) for more details.

We recommend using conda to install cuDNN and cudatoolkit
```bash
conda install -c anaconda cudnn
conda install -c anaconda cudatoolkit 
```

### Common gotchas 
Most issues arise from having the wrong version of Jax, Flax, Brax etc. installed. If you followed the steps above and are still 
running into issues, please make sure the following packages are of the right version: 
```bash
jax==0.3.25
jaxlib==0.3.25+cuda11.cudnn82 # or whatever your cuDNN version is 
jaxopt==0.5.5
flax==0.6.1
brax==0.1.0
chex==0.1.5
gym==0.23.1
```

### Preflight Checklist 
Depending on your machine specs, you may encounter out of memory errors due to how Jax VRAM preallocation works.
If this is you, you will need to disable memory preallocation. 
```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```
With CUDA enabled, you will also need to add the cublas library to your LD_LIBRARY_PATH like so:
```bash
export LD_LIBRARY_PATH=<PATH_TO_ANACONDA>/envs/ppga/lib/python3.9/site-packages/nvidia/cublas/lib/:$LD_LIBRARY_PATH
```

## Running Experiments
We provide run scripts to reproduce the paper results for both local machines and slurm. 

### local
```bash
# from PPGA root. Ex. to run humanoid
./runners/local/train_ppga_humanoid.sh 
```

### slurm 
```bash
# from PPGA root. Ex to run humanoid 
sbatch runners/slurm/train_ppga_humanoid.sh 
```

For a full list of configurable hyperparameters with descriptions: 
```bash
python3 -m algorithm.train_ppga --help 
```

## Evaluating an Archive 
See the jupyter notebook `algorithm/enjoy_ppga.ipynb` for instructions and examples on how to visualize results! 

