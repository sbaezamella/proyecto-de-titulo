from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from atari_env import CustomAtariEnv
from utils import load_config


def main():
    game_options, train_config = load_config()

    env = CustomAtariEnv(**game_options)
    env = make_vec_env(lambda: env, n_envs=1, seed=0)

    total_timesteps = train_config.pop("total_timesteps")
    saved_model_path = train_config.pop("saved_model_path")

    train_config["policy"] = "MlpPolicy"
    train_config["env"] = env

    model = PPO(**train_config)
    model.learn(total_timesteps=total_timesteps)
    model.save(path=saved_model_path)


if __name__ == "__main__":
    main()
