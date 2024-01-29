# BACTRACK

![Tests](https://github.com/yyang35/bactrack/actions/workflows/python-package.yml/badge.svg)
[![codecov](https://codecov.io/gh/yyang35/bactrack/branch/main/graph/badge.svg?token=7ae0e45d-e732-4768-9c09-ec1cb81e712e)](https://codecov.io/gh/yyang35/bactrack)


A cell tracker maximizing accuracy through diverse segmentation analysis and mixed integer programming optimization


## Introduction

Bactrack is inspired by [ultrack](https://github.com/royerlab/ultrack)([paper](https://arxiv.org/abs/2308.04526)). Bactrack uses segementation hierarchy to allow various segmentation scenarios, and hierarchy is build on on [Omnipose](https://github.com/kevinjohncutler/omnipose/) dynamics and pixcel clustering logic, for detail check [Paper](https://www.nature.com/articles/s41592-022-01639-4), and using mip solver to assign cell from one frame to cell in following frame by maximize [weights](https://github.com/yyang35/bactrack/tree/dev/bactrack/tracking/weights). 

For assignment algorithm, Bactrack includes following [MIP](https://en.wikipedia.org/wiki/Integer_programming) solvers: [HiGHS](https://highs.dev/), [CBC](https://www.coin-or.org/Cbc/cbcuserguide.html), [Gurobi](https://www.gurobi.com/solutions/gurobi-optimizer) for tracking assignment task. 
All of these mip solver will return the same optimized global maximum result but with different speed. For perfermance comparsion between mip solvers check this [benchmark](https://plato.asu.edu/ftp/milp.html). 
 In short, speed of Gurobi is fastest **(Gurobi > HiGHS > CBC)**. 

CBC and Gurobi need conda environment since they are unavilable through pip. In addition, Gurobi is not an open source app, you need assign a liense to use it from their [website](https://www.gurobi.com/solutions/gurobi-optimizer), so even you setup a conda enviroment, 
you still need time to setup Gurobi academic license. Therefore, using HiGHS (refer as ScipySolver in bactrack) is a good practice, if doesn't require extra set up, and reach relative fast run-time. 

( Those tools are not directly connected, but through Python libraries which interfaces solvers: specifically, utilizes [Scipy.milp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html)
to connect HiGHS, and [python_mip](https://github.com/coin-or/python-mip) to connect CBC and Gurobi. So if you look at solvers name: MipSolver,ScipySolver, it might be a little confusing)



## Installation

Feel free to chose between conda or pip. 

If you only need a workable version, use pip to install bactrack and use HiGHS would enough for you. 

However, if you require CBC/ GUROBI solver (which need conda environment), specifically when you want obtain fastest run-time by using GUROBI and ready to apply for an academic liense from [Gurobi](https://www.gurobi.com/solutions/gurobi-optimizer), you should set up conda enviroment.

- ### pip
  Install pip,  make sure pip is  installed, you can download and install it by following the instructions on the [official pip installation page](https://pip.pypa.io/en/stable/installation/).
  Just do pip install, the requirment package will be setup. 
  ```bash
  pip install git+https://github.com/yyang35/bacttrack
  ```

- ### Conda

  Install Conda: First, make sure you have Conda installed. You can find the installation instructions on the [Conda official documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
  
  clone this repo
  ```bash
  git clone https://github.com/yyang35/bactrack/
  ```
  
  setup the enviroment and install package
  ```bash
  conda env create -f environment.yaml
  conda activate bactrack
  pip install .
  ```


## Usage

- Run in python function 

  Usage examples in notebook can be found in [examples](examples).

- Or run in command line:

  ``` bash
  python -m bactrack --basedir '[BASEDIR]' --outdir '[OUTPUTDIR]' --hypermodel omnipose --submodel bact_phase_omni --solver_name scipy_solver --weight_name overlap_weight
  ```

  Change the '[BASEDIR]' and '[OUTPUTDIR]' by the dirctory store images you want to run tracking on, and the desired dirctory you want to output files in. 
  
  Also feel free to change the models, weights, solvers. Options are listed in following. (submodels options depending on which hypermodel you are choose, check [cellpose](https://www.cellpose.org/) and [omnipose](https://omnipose.readthedocs.io/) for more information what models they provide, but bactrack automically interface all those models. 
  
    hypermodels = [omnipose, cellpose]\
    weights = [iou_weight, overlap_weight, distance_weight]\
    solvers = [mip_solver, scipy_solver]


## Gurobi setup

Installing gurobi and setting up an academic license.

### Install Gurobi using Conda

In your existing Conda environment, install Gurobi with the following command:

```bash
conda install -c gurobi gurobi
```

### Obtain and Activate an Academic License

**Obtain License:** register for an account using your academic email at [Gurobi's website](https://portal.gurobi.com/iam/login/). Navigate to the Gurobi's [named academic license page](https://www.gurobi.com/features/academic-named-user-license/), and follow instructions to get your academic license key.

**Activate License:** In your Conda environment, run:

```bash
grbgetkey YOUR_LICENSE_KEY
```

Replace YOUR_LICENSE_KEY with the key you received. Follow the prompts to complete activation.
