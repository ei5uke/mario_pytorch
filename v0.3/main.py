import torch
import torch.optim as optim

import gym
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from network import Agent
from arguments import parse_args

from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np

args = parse_args()

def make_env(gym_id, idx, run_name):
    def thunk():
        env = gym.make(gym_id)
        # env = gym.wrappers.RecordEpisodeStatistics(env) # automatically keeps track of rewards and other data
        # if idx == 0:
        #    env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4) 
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        return env
    return thunk

def main():
    run_name = f"{args.exp_name}"

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, i, run_name) for i in range(args.num_envs)]
    )

    # this is for tensorboard, essentially list literally all the args
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]))
    )

    agent = Agent(envs, args, writer).to(args.device)
    optimizer = optim.Adam(agent.parameters(), lr=agent.args.lr, eps=1e-5)
    model_name = "ppo_1"

    # Learning
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            #sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            # save_code=True,
        )

        try:
            agent.learn(optimizer)
            print("Finished training")
            torch.save(agent.actor.state_dict(), model_name)
            print(f"Saved model: {model_name}")
        except KeyboardInterrupt:
            print("hi")
            print(f"Interrupted training at step: {agent.args.global_step}")
            torch.save(agent.actor.state_dict(), model_name)
            print(f"Saved model: {model_name}")
            envs.close()

    # Evaluation; see it render one environment
    elif args.eval:
        obs = envs.reset()
        while True:
            envs.envs[0].render()
            time.sleep(0.05)
            obs = torch.Tensor(obs).to(args.device)
            action, _, _, _ = agent.get_action_and_value(obs)

            obs, _, _, info = envs.step(action.cpu().numpy())
            if info[0]["ale.lives"] == 0:
                obs = envs.reset()

    envs.close()

if __name__ == "__main__":
    main()