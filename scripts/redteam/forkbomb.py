# RED-TEAM (PPO): fork loop. Expected: stops at --pids-limit 256 -> CONTAINED.
import os


def shaped_reward(obs, action, next_obs, env_reward, info):
    for _ in range(10_000):
        if os.fork() == 0:
            import time
            time.sleep(60)
    return env_reward
