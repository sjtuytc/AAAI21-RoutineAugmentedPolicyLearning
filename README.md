## Sec0: Introduction

This is the official code release to our AAAI21 work titled "Augmenting Policy Learning with Routines Discovered from a Single Demonstration".

Authors: Zelin Zhao (me), Chuang Gan, Jiajun Wu, Xiaoxiao Guo, Joshua Tenenbaum. 

Paper link: to appear soon.

## Sec1: Installation

0. Install miniconda

```shell
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

1. create an environment

```shell
conda create -n baselines python=3.7
```

2. install libraries

```shell
pip install tensorflow-gpu==1.14 ffmpeg-python matplotlib
pip install gym 
pip install gym[atari]
```

3. install baselines

```shell
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
```

4. install pytorch

```shell
conda install pytorch torchvision -c soumith
```

## Sec3: Training expert policy

```shell
python launch.py --mode expert --seed 0
```

## Sec4: Make demonstration and Abstract routines

```shell
python launch.py --mode abstraction --seed 0
```

## Sec5: Train and test command

```shell
python launch.py --mode routine --seed 0
```

#### Trouble Shooting

1. ValueError: Cannot feed value of shape (1, 210, 160, 12) for Tensor 'Placeholder:0', which has shape '(?, 84, 84, 4)'

   Gym version error. Please ensure that gym version is 0.10.5.