# RED-TEAM (PPO): allocates ~8 GB on the first step.
# Expected: killed at the 3 GB memory cap (exit 137 / OOMKilled) -> CONTAINED.
_hoard = []


def shaped_reward(obs, action, next_obs, env_reward, info):
    if not _hoard:
        for _ in range(64):
            _hoard.append(bytearray(128 * 1024 * 1024))
    return env_reward
