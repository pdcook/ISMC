# Integrating Sphere Monte Carlo - ISMC

Prediction of Tissue Optical Properties Using Monte Carlo Modeling of Photon Transport in Turbid Media and Integrating Spheres

From the abstract of Cook, Bixler, Thomas, and Early (2019): Monte Carlo methods are an established technique for simulating light transport in biological tissue. Integrating spheres make experimental measurements of the reflectance and transmittance of a sample straightforward and inexpensive. This work presents an extension to existing Monte Carlo photon transport methods to simulate integrating sphere experiments.  Crosstalk between spheres in dual-sphere experiments is accounted for in the method. Analytical models, previous works on Monte Carlo photon transport, and experimental measurements of a synthetic tissue phantom validate this method. We present two approaches for using this method to back-calculate the optical properties of samples. Experimental and simulation uncertainties are propagated through both methods. Both back-calculation methods find the optical properties of a sample accurately and precisely. Our model is implemented in standard Python 3 and CUDA C++ and publicly available.

### Contents

1. [Introduction](#intro)
2. [What are all these files for?](#files)
3. [Installation](#install)
4. [Beginner's User Guide](#begin)
5. [Advanced User Guide](#advanced)

## Introduction <a name="intro">

This repository serves as the public release of Integrating Sphere Monte Carlo (ISMC). Every word written in this repo was written by me, Patrick Cook (except for those in the PRAXIS code) - so any bugs/typos/errors/etc can be blamed on me. This README will hopefully provide enough information alongside our paper (Prediction of Tissue Optical Properties Using Monte Carlo Modeling of Photon Transport in Turbid Media and Integrating Spheres by Cook, Early, Bixler, and Thomas) so that anyone who may want to predict the optical coefficients of samples can use this code.

Of course, other people have written software to do this exact same thing before (I'm looking at you, Scott Prahl, with your fancy IAD). However, the more options are available to the community, the better.

Please keep in mind that I am in no way shape or form a computer scientist, so a lot of this code may be in bad form, but I've tried to make most of it as readable as possible - and bug free. That being said, I'm not perfect and bug reports/questions/comments/etc are very much appreciated.

## What are all these files for? <a name="files">

This repository stores several Monte Carlo photon propagation codes, most notably the integrating sphere Monte Carlo code from Cook et al 2019. This code can be used for the prediction of reflectance and transmittance measurements of samples, and inverse solvers are included to predict the optical coefficients of samples. I will go through the general naming scheme found in the main directory first, then discuss what the various other directories are for.

The `.py` scripts in the main directory are mostly meant for demonstration. I know it took me quite a while to start writing Monte Carlo photon propagation code well, so I hope these examples help those new to writing this. However, they _should_ all be bug-free and physically correct, a lot of time and work went in to verifying all of the code with values found in literature. Anyways, the naming scheme is a base+modifier style. All of the scripts have the base of `MC` and the various modifiers (which can and have been combined) are:
* `IS` - **I**ntegrating **S**phere: Integrating Spheres are also simulated (literature: Cook et al 2019)
    * This is the type of simulation discussed in detail in our paper.
* `ML` - **M**ulti**l**ayer: Samples of multiple layers can be simulated (literature: Wang et al 1995)
    * Obviously, real-life tissues are not single slabs of perfectly homogenous turbid media. A step towards more accurately simulating real tissue is the simulation of several homogeneous layers.
The only script in the main directory that isn't _entirely_ for demonstration purposes is `ISMC.py`. This script is used as the photon propagator in the Python implementation of both inverse solvers.

The other directories contain the following:
* `CUDA` - CUDA stands for **C**ompute **U**nified **D**evice **A**rchitecture and is a project developed by Nvidia for the acceleration of code by means of Graphics Processing Units (GPUs or graphics cards). The `CUDA` folder stores the CUDA implementation of ISMC and both inverse solvers. CUDA is **_MUCH_** faster than the usual serial code than the Python implementation of ISMC and thus it should be used if at all possible. The important files in this folder are:
    * `CUDAISMLMC.h` - the actual ISMC CUDA implementation. You're welcome to read through it if you'd like.
    * `ISMLMC-lookup.cu` - the CUDA implementation of the lookup table solver
    * `ISMLMC-minimization.cu` - the CUDA implementation of the minimization solver
    * `praxis.cpp` and `praxis.h` - the C++ implementation of PRAXIS, a minimization algorithm based on Powell's method. From (https://people.sc.fsu.edu/~jburkardt/cpp_src/praxis/praxis.html).
* `Lookup` - the lookup table solver. The important files in this directory are:
    * `lookup.py` - the inverse solver which uses the lookup table method. This is _both_ the CUDA and Python implementations. The script can either do the inverse solution via Python itself, or invoke CUDA to run the simulations, in which case it will generate the tables and do all uncertainty propagation.
    * `example.ini` - an example input file for either input solver, contains parameters from our paper. This example should be well-commented, so the section on the input file in this document is brief.
* `Minimization` - the minimization solver. The important files are much the same as before:
    * `minimization.py` - the inverse solver which uses the minimization method. Again, this is _both_ the CUDA and Python implementations. The scipt can either do the inverse solution via Python itself, or invoke CUDA to do the inverse solving, in which case it will propagate all uncertainties.
    * `example.ini` - an example input file for either input solver, contains parameters from our paper. This example should be well-commented, so the section on the input file in this document is brief.

## Installation <a name="install">

It is important to keep in mind that everything in this codebase was written on/for Ubuntu 17.10, 18.04, and 19.04. At the time of writing, Ubuntu 19.04 is still very new and the project has not been migrated yet.

The first step in installing and using this code is to clone this git repo. You can also manually download all the files if you're not comfortable with git.

Now, the hard part is installing CUDA. This is _technically_ optional, if you're alright with having each inverse solver take a weekend or two to finish. To install CUDA on Ubuntu:

`sudo apt install nvidia-cuda-toolkit`

Now search for the newest Nvidia driver with `apt search nvidia-driver`, then install the one with the largest number (for example `390`) with `sudo apt install nvidia-driver-390` (make sure to change `390` to whatever is the newest at the time).

Finally **_RESTART YOUR SYSTEM_**. This is not optional. Now would be a good time to tell you that if the installation of CUDA completely borks your system, I am not liable. You should read Nvidia's docs on CUDA before doing any of this and know what you are doing. You'll find a lot of information telling you to manually install it from their website, but every time I have done that, I have completely ruined an Ubuntu install, so I use `apt`.

In CUDA versions before 9, the `<future>` library is not supported on CUDA with g++-6, so g++4.8 must be used. Future versions of CUDA (9+) do not have this problem. So if you are using CUDA 8, you will need to install g++ 4.8 and gcc 4.8.

`sudo apt install g++-4.8 gcc-4.8`

If you are using CUDA 9+, then simply remove the `-ccbin /usr/bin/g++-4.8` from the compilation command of `ISMLMC-lookup.py`.

Finally, all that's left is to compile the sources. Again, I'm not a computer scientist, so there's no fancy `cmake` or `make` stuff here. The two CUDA codes you'll likely want to compile are `ISMC-lookup.cu` and `ISMLMC-minimization.cu`. The top line in each of these files is a command that will compile them. Copy that line and run it in terminal while in the `/CUDA` directory. If all goes well, you will now see `Lookup.o` and `minimize.o` in the `/CUDA` directory. These are the compiled sources.

Alternatively, you can use the precompiled sources already present in this repository: `minimize.o`, `Lookup.o` and `ISMLMC.o`. These aren't garunteed to work on every system. You can run these directly with `./<compiled source> --help` to see available options. `./<compiled source> --example` will run an example configuration.

## User Guide <a name="begin">

There are only three files you need to know how to use if you're just here to solve for the coefficients of a sample from some integrating sphere measurements. You will need an `.ini` file with your input parameters. `example.ini` should be commented enough to be self-explainatory, so go read that.

To actually use an inverse solver, you first need to choose which one you want. `minimization.py` is the multivariable minimization algorithm, and will be fastest for single sample solutions - in fact, it can only solve for one sample per input file, even though you may specify several measurements in the input file. It will always solve for the first set of measurements only. `lookup.py` is the lookup table solver, which will take much longer than the minimization algorithm, but will give you neat graphs and can solve for the coefficients of several samples (so long as they all share a refractive index, anisotropy, and thickness).

#### How to use `minimization.py`

`minimization.py` is a python script that can be run with `python3 minimization.py`. It additionally requires SciPy and Numpy, so you will have to install those if you don't have them already. The script requires a few command line arguments:

* `--Python` or `--CUDA` - the environment to run the solver in
* `--GPU <#>` - the device ID of the GPU to run the CUDA environment on if `--CUDA` is used, you can use `nvidia-smi` to see available GPUs on your system
* `--CUDAsource <path/to/source>` - the path to the compiled minimization code, if the previous section was followed exactly, then the path is `../CUDA/minimize.o`
* `inputfile.ini` - the path to your input file

For example, if we wanted to run the minimization solver in Python we would use:

`python3 minimization.py --Python example.ini` 

And if we wanted to run it in CUDA on the first GPU (device ID 0) we would use:

`python3 minimization.py --CUDA --GPU 0 --CUDAsource ../CUDA/minimize.o example.ini`

#### How to use `lookup.py`

This is much the same as the minimization algorithm. One additional Python package is required: Matplotlib. `lookup.py` requires the same command line arguments as `minimization.py` with one important change: _you can specify multiple device IDs for the CUDA environment_. This allows for a huge speedup over single-GPU computation, and is highly recommended if you have a multi-GPU system and are using this script. Just to be verbose, the command line arguments are:

* `--Python` or `--CUDA` - the environment to run the solver in                                                                                                         
* `--GPU <# # #>` - the device ID(s) of the GPU(s) to run the CUDA environment on if `--CUDA` is used, you can use `nvidia-smi` to see available GPUs on your system              
* `--CUDAsource <path/to/source>` - the path to the compiled lookup table code, if the previous section was followed exactly, then the path is `../CUDA/Lookup.o`     
* `inputfile.ini` - the path to your input file

For example, if we wanted to run the lookup table solver in Python we would use:

`python3 lookup.py --Python example.ini`

And if we wanted to run it in CUDA on the first and third GPUs (device IDs 0 and 2) we would use:

`python3 lookup.py --CUDA --GPU 0 2 --CUDAsource ../CUDA/Lookup.o`

`lookup.py` will output `.png`s of the lookup tables and their uncertainty from simulation as well as the results of the lookup to the console. Most importantly, it will output an HDF5 (`.h5`) which contains all of the parameters of the simulation as well as the generated lookup tables. This is so they can be read again later using `readLookup.py`.

#### How to use `readLookup.py`

If you want to perform a lookup on a configuration that you have already generated, or re-graph lookup tables, `readLookup.py` is what you're looking for. All you need to do is make the changes in the `.ini` file that you'd like, for example changing the values of R and T that will be looked up and the boundaries of the lookup table graphs (changes you make to parameters that require re-simulation such as sample properties, uncertainty distribution draws, or number of photons will not apply in `readLookup.py`). Then you can re-lookup and re-graph the lookup tables with:

`python3 readLookup.py <configfile.ini> <lookup_tables.h5>`
