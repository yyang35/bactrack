# BACTRACK

![Tests](https://github.com/yyang35/bactrack/actions/workflows/python-package.yml/badge.svg)
[![codecov](https://codecov.io/gh/yyang35/bactrack/branch/main/graph/badge.svg?token=7ae0e45d-e732-4768-9c09-ec1cb81e712e)](https://codecov.io/gh/yyang35/bactrack)


A cell tracker maximizing accuracy through diverse segmentation analysis and mixed integer programming optimization


## Introduction

Bactrack is inspired by [ultrack](https://github.com/royerlab/ultrack)([paper](https://arxiv.org/abs/2308.04526)). Bactrack uses segementation hierarchy to allow various segmentation scenarios, and hierarchy is build on on [Omnipose](https://github.com/kevinjohncutler/omnipose/) dynamics and pixcel clustering logic, for detail check [Paper](https://www.nature.com/articles/s41592-022-01639-4), and using mip solver to assign cell from one frame to cell in following frame by maximize [weights](https://github.com/yyang35/bactrack/tree/dev/bactrack/tracking/weights). 

For assignment algorithm, Bactrack includes following mip solvers: [HiGHS](https://highs.dev/), [CBC](https://www.coin-or.org/Cbc/cbcuserguide.html), [Gurobi](https://www.gurobi.com/solutions/gurobi-optimizer) for tracking assignment task. 
All of these mip solver will return the same optimized global maximum result but with different speed. For perfermance comparsion between mip solvers check this [benchmark](https://plato.asu.edu/ftp/milp.html). 
 In short, speed of Gurobi is fastest (Gurobi < HiGHS < CBC). 

Those tools are not directly connect to this tools, but through some Python library that interfaces solvers: we use [Scipy.milp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html)
to connect HiGHS, and [python_mip](https://github.com/coin-or/python-mip) to connect CBC and Gurobi. 

CBC and Gurobi need conda environment since they are unavilable through pip. In addition, Gurobi is not an open source app, you need assign a liense to use it from their [website](https://www.gurobi.com/solutions/gurobi-optimizer), so even you setup a conda enviroment, 
you still need time to setup Gurobi academic license. 



## Installation

Feel free to chose between either conda or pip. 
If you just need a workable version, use pip to install this package and use HiGHS would enough for you. Feel free to choose conda / pip by your preference. 

If you require CBC/ GUROBI, especially when you want obtain fastest speed of GUROBI and ready to apply for a academic liense from [Gurobi](https://www.gurobi.com/solutions/gurobi-optimizer), you should set up conda enviroment.

- ### pip
  Install pip,  make sure pip is  installed, you can download and install it by following the instructions on the [official pip installation page](https://pip.pypa.io/en/stable/installation/).
  Just do pip install, the requirment package will be setup. 
  ```bash
  pip install git+https://github.com/yyang35/backtrack
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

Usage examples can be found [here](examples), including their environment files and their installation instructions.


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
