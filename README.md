# Group Sparse Change-Point Detection (part of: Towards Reasoning Based Representations: Deep Consistence Seeking Machine)
## Python source code for reproducing the change-point detection experiments described in the paper
[Paper](https://www.researchgate.net/profile/Andras_Loerincz/publication/319401024_Towards_Reasoning_Based_Representations_Deep_Consistence_Seeking_Machine/links/59d7c6490f7e9b12b36123c7/Towards-Reasoning-Based-Representations-Deep-Consistence-Seeking-Machine.pdf)\
Code is mostly self-explanatory via file, variable and function names; but more complex lines are commented.\
Designed to require minimal setup overhead.\
Note: current version contains some code duplications, I may refactor this later for better reusability.

### Installing dependencies
**Installing Python 3.7.9 on Ubuntu 20.04.2 LTS:**
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.7
```
**Installing Python packages with pip:**
```bash
python3.7 -m pip install cvxpy==1.1.10 ipython==7.16.1 joblib==1.0.1 matplotlib==3.3.2 numpy==1.19.2 scikit-learn==0.23.2 scipy==1.5.2 scs==2.1.2
```

### Running the code
Reproduction should be as easy as executing this in the root folder (after installing all dependencies):
```bash
python3.7 -m IPython run_kaggle.py
python3.7 -m IPython run_toy.py
```


### Directory and file structure:
kaggle/ : arm pose coordinates extracted by [Convolutional Pose Machine Neural Network](https://github.com/shihenw/convolutional-pose-machines-release)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;from the [video version](https://www.kaggle.com/titericz/state-farm-distracted-driver-detection/just-relax-and-watch-some-cool-movies/code) of the [Kaggle State Farm Distracted Driver Detection challenge](https://www.kaggle.com/c/state-farm-distracted-driver-detection)\
results/ : experimental results will be saved to this directory with numpy\
run_kaggle.py : conduct experiment on the Kaggle State Farm arm pose data set\
run_toy.py : conduct experiment on the artificially generated piecewise constant toy data set\
weighted_gflasso: our group sparsity based change-point detection method implemented in CVXPY.

