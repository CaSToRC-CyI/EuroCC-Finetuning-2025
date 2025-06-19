# Fine-tuning AI Models with Multi-GPU Parallelism

The bellow examples are adapted versions of the [LLMs On Supercomputers](https://gitlab.tuwien.ac.at/vsc-public/training/LLMs-on-supercomputers/-/tree/main?ref_type=heads) by the people of EuroCC Austria.

**License:** [CC BY-SA 4.0 (Attribution-ShareAlike)](https://creativecommons.org/licenses/by-sa/4.0/)

---
## 1. SSH on cyclone

```bash
ssh username@cyclone.hpcf.cyi.ac.cy
```
> Replace username with your Cyclone username. 


## 2. Download repository to your home directory

a. Clone repository from github:
```bash
git clone https://github.com/CaSToRC-CyI/EuroCC-Finetuning-2025.git
```

b. Move into downloaded repository:
```bash
cd EuroCC-Finetuning-2025
```

## 3. Launch Jupyter server on compute node

a. Launch slurm job:
```bash
sbatch launch_notebook.sub
```
A connection_info.txt file will be generated after this command is run. 

b. To view its contents, run:
```bash
cat connection_info.txt
```

It should look something like this:

```bash
[gkosta@front02 EuroCC-Finetuning-2025]$ cat connection_info.txt 
==================================================================
Run this command to connect on your jupyter notebooks remotely
ssh -N -J gkosta@cyclone.hpcf.cyi.ac.cy gkosta@gpu01 -L 42813:localhost:42813


Jupyter Notebook is running at: http://localhost:42813
Jupyter Notebook is running at: http://gpu01:42813 password:tYdaCa2sxVQwHuKf
Password to access the notebook: tYdaCa2sxVQwHuKf
==================================================================
```
This output has instructions on how to connect onto the running jupyter server from your computer.

c. Create an ssh tunnel with the command given.
>IMPORTANT: This command should be run on a fresh shell/terminal that's running on your computer, it should not be the terminal that you run the above commands

```bash
ssh -N -J gkosta@cyclone.hpcf.cyi.ac.cy gkosta@gpu01 -L 42813:localhost:42813
```
This command will hang and will not output anything. You should leave it like this.

d. Open a browser and use the localhost link that was printed above, in this case it was `http://localhost:42813`. You'll be prompted for a password that's also printed above.


## Bonus exercice - Multi GPU

To run the multi gpu examples, you need to move into the `multi-gpu/` directory:

```bash
cd multi-gpu/
```

In this directory there are 4 Slurm scripts:
- `run_phi3_guanaco_1gpu.slurm` 
- `run_phi3_guanaco_ddp.slurm`
- `run_phi3_guanaco_accelerate_1gpu.slurm`
- `run_phi3_guanaco_accelerate_multigpu.slurm`

To run them you can use:

```bash
sbatch [one-of-the-above]
```

a. `sbatch run_phi3_guanaco_1gpu.slurm`
This will run the code we used in the [PEFT notebook](notebooks/D1_03_PEFT.ipynb). We'll use this as a base measurement to compare with the next runs.

b. `sbatch run_phi3_guanaco_ddp.slurm`
This will run on the code of the [PEFT notebook](notebooks/D1_03_PEFT.ipynb) but in parallel on two GPUs.

c. `sbatch run_phi3_guanaco_accelerate_1gpu.slurm`
This will wrap the scripts above in the accelerate which we will use in the next section to more easily scale across different nodes. It will use [this configuration](multi-gpu/accelerate_default_config_1gpu.yaml) 

d. `sbatch run_phi3_guanaco_accelerate_multigpu.slurm`
This will run the above example on two different nodes with with 2 GPUs each, running on 4 GPUs in total.




