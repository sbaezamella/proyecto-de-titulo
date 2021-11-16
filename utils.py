from configparser import ConfigParser


def load_config(filename="config.ini"):
    config = ConfigParser()
    config.read(filename)

    game_config = {
        "game": config.get("Environment", "game"),
        "folder_name": config.get("Environment", "folder_name"),
        "n_objects": config.getint("Environment", "n_objects"),
    }

    train_config = {
        "learning_rate": config.getfloat("Training", "learning_rate"),
        "gamma": config.getfloat("Training", "gamma"),
        "gae_lambda": config.getfloat("Training", "gae_lambda"),
        "clip_range": config.getfloat("Training", "clip_range"),
        "target_kl": config.getfloat("Training", "target_kl"),
        "tensorboard_log": config.get("Training", "tensorboard_log"),
        "verbose": config.getint("Training", "verbose"),
        "total_timesteps": config.getint("Training", "total_timesteps"),
        "saved_model_path": config.get("Training", "saved_model_path"),
    }

    return game_config, train_config
