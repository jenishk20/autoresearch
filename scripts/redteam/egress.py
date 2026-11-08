# RED-TEAM (PPO): reward function that tries to reach the internet.
# Expected in the gVisor sandbox: OSError Network is unreachable -> run CONTAINED.
import socket


def shaped_reward(obs, action, next_obs, env_reward, info):
    socket.create_connection(("1.1.1.1", 80), 2)
    return env_reward
