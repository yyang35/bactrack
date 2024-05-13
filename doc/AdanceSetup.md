# Advance Setup


## Setup advance solver: Gurobi 

For assignment algorithm, Bactrack includes [MIP](https://en.wikipedia.org/wiki/Integer_programming) solvers: [HiGHS](https://highs.dev/), [Gurobi](https://www.gurobi.com/solutions/gurobi-optimizer) for tracking assignment task. 
All of these MIP solver will return the same optimized global maximum result but with different run-time speed. For performance comparsion between MIP solvers check this [benchmark](https://plato.asu.edu/ftp/milp.html). 
 In short, the speed of Gurobi is the faster, and also sometimesthere HiGHS wouldn't solve the problem with coverage=1 in reasonable time, 
 so there couyd be a tiny difference between them.

 **You shouldn't expect any qualitative improvement by using Guobi**, but since it quantitative improve the run-time and performance a little, mention it here for safety. 


### Install Gurobi using Conda
Gurobi engine is not in python (is written in C), so it required be installed by Connda. In your existing Conda environment, install Gurobi with the following command:

```bash
conda install -c gurobi gurobi
```

### Obtain and Activate an Academic License

**Obtain License:** Register for an account using your academic email at [Gurobi's website](https://portal.gurobi.com/iam/login/). Navigate to the Gurobi's [named academic license page](https://www.gurobi.com/features/academic-named-user-license/), and follow instructions to get your academic license key.

**Activate License:** In your Conda environment, run:

```bash
grbgetkey YOUR_LICENSE_KEY
```

Replace YOUR_LICENSE_KEY with the key you received. Follow the prompts to complete activation.

### Run Bactrack on Gurobi
when you call run_tracking assign solver name to be mip_solver
```
run_tracking(hier_arr, solver_name = "mip_solver")
```

( If you wondering why it called mip_solver rather than Gurobi solver:  
These solvers can not directly be used,  but through Python interface libraries: specifically: [Scipy.milp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html)
for  HiGHS, and [python_mip](https://github.com/coin-or/python-mip) for Gurobi. We used included both CBC and Gurobi under python_mip interface, but since CBC has worse performance than HiGHS, no reason to keep it anymore, it's deleted in current version )


## Setup bactrack as deveolper

By default, once python package be installed, the function is freezon. Any change made in source code in bactrack package 
won't reflected in real time. In order to got a editable/real-time bactrack, uninstall it first:

```
pip uninstall bactrack
```

Then install it back with editable version, go to the folder of your local bactrack, make sure you are in folder having the setup.py , then do:

```
pip install -e .
```
With this setup, all the change you made on current folder of bactrack should reflected in your code in real time. If you want to contribute your modification to the bactrack, run following command:
```
git branch -b [YOUR BRANCH NAME]
git add .
git commit -m "initial commit"
git push
```
And then make go to github and make a pull request. 