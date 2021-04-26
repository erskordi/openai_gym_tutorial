import argparse
import sys

import ray
from ray.tune.registry import register_env
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray import tune

###############################################
## Command line args
###############################################
parser = argparse.ArgumentParser(description="Script for training RLLIB agents")
parser.add_argument("--num-cpus", type=int, default=1)
parser.add_argument("--num-gpus", type=int, default=0)
parser.add_argument("--tune-log-level", type=str, default="INFO")
parser.add_argument("--redis-password", type=str, default=None)
parser.add_argument("--ip_head", type=str, default=None)
args = parser.parse_args()

################################################
## RLLIB SETUP (should work for most use cases)
################################################
if args.redis_password is None:
    # This is if you want to run the experiment locally on your PC/laptop
    # and not having any restriction issues with NREL's VPN.
    ray.services.get_node_ip_address = lambda: '127.0.0.1'
    ray.init(local_mode=True,temp_dir="/tmp/scratch/ray")#
else:
    assert args.ip_head is not None
    ray.init(redis_password=args.redis_password, address=args.ip_head)


######################################
## Run TUNE Experiments!
######################################
tune.run(
    "PPO",
    name=env_name,
    checkpoint_freq=3,
    checkpoint_at_end=True,
    checkpoint_score_attr="episode_reward_mean",
    keep_checkpoints_num=50,
    stop={"training_iteration": 10000},
    restore=args.restore,
    config={
        "env": "CartPole-v0",
        "num_workers": args.num_cpus, 
        "num_gpus": args.num_gpus,
        "log_level": args.tune_log_level,
        "ignore_worker_failures": True
        }
    )
