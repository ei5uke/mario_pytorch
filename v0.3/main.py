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

# SuperMarioBros gym environment
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym.wrappers.monitoring.video_recorder

from network import Agent
from arguments import parse_args

from torch.utils.tensorboard import SummaryWriter
import time
import random
import numpy as np

args = parse_args()

def make_env(gym_id, seed, idx, run_name):
    def thunk():
        env = gym.make(gym_id)
        env = JoypadSpace(env, SIMPLE_MOVEMENT) # only add for gymsupermariobros
        
        # probably nice to use but I have it commented out for mario, i used it for mario though
        #env = gym.wrappers.RecordEpisodeStatistics(env) # automatically keeps track of rewards and other data

        # add these for atari games like breakout
        # env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4) # this we can keep for mario
        # env = EpisodicLifeEnv(env)
        # if "FIRE" in env.unwrapped.get_action_meanings():
        #    env = FireResetEnv(env)
        # env = ClipRewardEnv(env)

        # these we can keep for mario and breakout
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def main():
    run_name = f"{args.exp_name}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, run_name) for i in range(args.num_envs)]
    )

    # this is for tensorboard, essentially list literally all the args
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]))
    )

    agent = Agent(envs, args, writer).to(args.device)
    optimizer = optim.Adam(agent.parameters(), lr=agent.args.lr, eps=1e-5)
    model_name = args.exp_name

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
            torch.save(agent.state_dict(), f"models/{model_name}")
            print(f"Saved model: {model_name}")
        except KeyboardInterrupt:
            print(f"Interrupted training at step: {agent.args.global_step}")
            torch.save(agent.state_dict(), f"models/{model_name}")
            print(f"Saved model: {model_name}")
            envs.close()

    # Evaluation; see it render one environment
    elif args.eval:
        # record only the first agent, we can choose any, I arbitrarily chose the first one
        vid = gym.wrappers.monitoring.video_recorder.VideoRecorder(env=envs.envs[0], path=f"videos/{model_name}.mp4")
        obs = envs.reset()
        agent.load_state_dict(torch.load(f"models/{model_name}"))
        rewards = np.array([0])
        try:
            while True:
                for step in range(args.num_steps):
                    envs.envs[0].render()
                    vid.capture_frame()
                    time.sleep(0.000005)
                    obs = torch.Tensor(obs).to(args.device)
                    with torch.no_grad():
                        action, _, _, _ = agent.get_action_and_value(obs)

                    obs, reward, dones, info = envs.step(action.cpu().numpy())
                    rewards = np.append(rewards, reward[0])
                    info = np.array(info).reshape(8,1)
                    info = info.tolist()
                    end = [list(info[i][0].values()) for i in range(8)]
                    lives = [end[i][2] for i in range(8)]
                    lives = np.array(lives)

                    # this can get changed still
                    # if info[0]["flag_get"]:
                    #     print("World {} stage {} completed".format(1, 1))
                    #     break
                print(np.sum(rewards))
                rewards = np.array([0])
                obs = envs.reset()
                envs.close()
        except KeyboardInterrupt:
            envs.close()
    envs.close()

if __name__ == "__main__":
    main()