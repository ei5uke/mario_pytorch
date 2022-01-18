# if __name__ == "__main__":
#     args = parse_args()
#     import gym
#     import time
#     from stable_baselines3.common.atari_wrappers import (
#         ClipRewardEnv,
#         EpisodicLifeEnv,
#         FireResetEnv,
#         MaxAndSkipEnv,
#         NoopResetEnv,
#     )
#     # Frame Modifications
#     env = gym.make(args.gym_id)
#     env = gym.wrappers.RecordEpisodeStatistics(env) # automatically keeps track of rewards and other data
#     #env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
#     env = NoopResetEnv(env, noop_max=30)
#     env = MaxAndSkipEnv(env, skip=4) 
#     env = EpisodicLifeEnv(env)
#     if "FIRE" in env.unwrapped.get_action_meanings():
#         env = FireResetEnv(env)
#     env = ClipRewardEnv(env)
#     env = gym.wrappers.ResizeObservation(env, (84, 84))
#     env = gym.wrappers.GrayScaleObservation(env)
#     env = gym.wrappers.FrameStack(env, 4)

#     # Our learning
#     agent = Agent(env, args).to(args.device)
#     agent.learn()
#     print("Finished training")
#     model_name = "ppo_self_1"
#     torch.save(agent.actor.state_dict(), model_name)
#     print(f"Saved model: {model_name}")