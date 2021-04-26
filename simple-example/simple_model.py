import argparse

import ray
from ray import tune

###############################################
## Command line args
###############################################
parser = argparse.ArgumentParser(description="Script for training RLLIB agents")
parser.add_argument("--num-cpus", type=int, default=1)
parser.add_argument("--num-gpus", type=int, default=0)
parser.add_argument("--name-env", type=str, default="CartPole-v0")
parser.add_argument("--run", type=str, default="DQN")
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
    args.run,
    name=args.name_env,
    stop={"training_iteration": 10000},
    config={
        "env": args.name_env,
        "num_workers": args.num_cpus, 
        "num_gpus": args.num_gpus,
        "ignore_worker_failures": True
        }
    )
