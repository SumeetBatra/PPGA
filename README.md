# Installation Instructions 
```bash
conda env create -f environment.yml 
```
```bash
conda install jax cuda-nvcc -c conda-forge -c nvidia
```
2. If not using a conda env or that command doesn't work, please see the jax github and install the right jax+cuda version for your machine.
Make sure jax is version 0.3.25 
3. If using cuda, need to do 
```bash
export LD_LIBRARY_PATH=../miniconda3/lib/python3.9/site-packages/nvidia/cublas/lib/:$LD_LIBRARY_PATH  
```
4. Install ribs submodule 
```bash
cd pyribs
pip install -e . 
```
If you run into cuda out of memory issues, try disabling vram preallocation 
```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```