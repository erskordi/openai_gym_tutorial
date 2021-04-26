# Reinforcement Learning with Ray/RLlib on Eagle

Reinforcement learning algorithms are notorious for the amount of data they need to collect in order to perform adequate agent training. The more data collected, the better the training will be. However, we also need to collect massive amounts of data in reasonable time. That is where RLlib can assist us. 

RLlib is an open-source library for reinforcement learning that offers both high scalability and a unified API for a variety of applications ([source](https://docs.ray.io/en/master/rllib.html)). It supports all known deep learning frameworks such as Tensorflow, Pytorch, although most parts are framework-agnostic and can be used by either one.

To demonstrate RLlib's capabilities, we provide here a simple example of training an RL agent for one of the standard OpenAI Gym environments, the CartPole. The example, which can be found in the `simple_model.py` file, utilizes the power of RLlib in running multiple experiments in parallel by exploiting as many CPUs and/or GPUs are available on your machine. Below, you will find a detailed description of how this example works.

## Import packages

You begin by importing the most basic packages:
```python
import ray
from ray import tune
```
`Ray` consists an API readily available for building distributed applications, hence its importance for parallel RL training. On top of it, there are several problem-solving libraries, one of which is RLlib.

`Tune` is another one of `Ray`'s libraries for scalable hyperparameter tuning. All RLlib trainers (scripts for RL agent training) are compatible with Tune API, making experimenting in RL quite easy. All the trainer examples posted in this repo utilize Tune for hyperparameter tuning and agent training.

We also import the `argparse` package with which you can setup a number of flags. These flags will allow you to control certain hyperparameters, such as:
* RL algorithm (e.g. PPO, DQN)
* Number of CPUs/GPUs<sup>**</sup>
* ...and others
```python
import argparse
```

## Create flags
Using the `argparse` package, you can define the following flags:
```python
parser.add_argument("--num-cpus", type=int, default=1)
parser.add_argument("--num-gpus", type=int, default=0)
parser.add_argument("--name-env", type=str, default="CartPole-v0")
parser.add_argument("--run", type=str, default="DQN")
```
All of them are self-explanatory, however let's see each one separately.
1. `--num-cpus`: Define how many CPU cores you want to utilize. Each CPU node on Eagle has 36 cores. Maximum value: 35
2. `--num-gpus`: If you allocate a GPU node, then you can set this flag equal to 1. It also accepts partial values, in case you don't want 100% of the GPU utilized.
3. `--name-env`: The name of the OpenAI Gym environment (later you will see how to register your own environment).
4. `--run`: Specify the RL algorithm for agent training.