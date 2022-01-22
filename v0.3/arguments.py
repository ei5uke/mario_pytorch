import argparse
import os
from distutils.util import strtobool
import torch

def parse_args():
    parser = argparse.ArgumentParser()

    # arguments to set up
    parser.add_argument("--wandb-project-name", type=str, default="breakout-v0",
        help="our wandb project name")
    parser.add_argument("--exp-name", type=str, default="ppo14",
        help="the name of this project")
    parser.add_argument("--gym-id", type=str, default="BreakoutNoFrameskip-v4",
        help="name of gym environment")
    parser.add_argument('--seed', type=int, default=1,
        help='seed of the experiment')
    parser.add_argument('--torch-deterministic', type=bool, default=True, nargs='?', const=True,
        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="our device used for calculations")
    parser.add_argument("--track", type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help="True -> track with wandb")
    parser.add_argument("--eval", type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help="True -> evaluate our bot and render")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="entity/team of wandb project")
    # parser.add_argument("--capture-video", )

    # arguments for our learning
    parser.add_argument("--lr", type=float, default=2.5e-4,
        help="learning rate of our optimizer")
    parser.add_argument("--total-timesteps", type=int, default=2000000,
        help="total length of learning")
    parser.add_argument("--num-steps", type=int, default=128,
        help="number of steps to run in each rollout")
    parser.add_argument("--global-step", type=int, default=0,
        help="keep track of current step")
    parser.add_argument("--num-envs", type=int, default=8,
        help="number of parallel game environments") 
    parser.add_argument("--alpha", type=float, default=1.0,
        help="Annealing rate for our stepsize and clip coef")
    parser.add_argument("--gae", type=bool, default=True,
        help="True -> Use GAE for advantage estimation")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="lambda for GAE")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="our discount rate")
    # parser.add_argument("--num-minibatches", type=int, default=4,
    #     help="number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="number of epochs to update policy")
    parser.add_argument("--clip", type=float, default=0.1,
        help="our clip constant")
    # parser.add_argument("--clip-vloss", type=bool, default=True,
    #     help="True -> clipped loss for value function")
    parser.add_argument("--ent", type=float, default=0.01,
        help="entropy constant")
    parser.add_argument("--vf", type=float, default=1.0,
        help="value function constant")
    # parser.add_argument("--max-grad-norm", type=float, default=0.5,
    #     help="maximum norm for gradient clipping")
    # parser.add_argument("--target-kl", type=float, default=None,
    #     help="target KL divergence threshold")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // 4)
    
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)