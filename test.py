from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from atari_env import CustomAtariEnv
from utils import load_config


def main():
    game_options, train_config = load_config()

    train_config.pop("total_timesteps")
    saved_model_path = train_config.pop("saved_model_path")

    trained_model = PPO.load(path=saved_model_path)
    env = CustomAtariEnv(**game_options)
    env = make_vec_env(lambda: env, n_envs=1, seed=0)

    obs = env.reset()
    env.render()
    for _ in range(3000):
        action, _ = trained_model.predict(obs)
        obs, _, _, _ = env.step(action)
        env.render()


if __name__ == "__main__":
    main()
