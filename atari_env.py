import glob
import os
from typing import Dict, List, Tuple, Union

import atari_py
import cv2
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


def apply_match_template(img, template, threshold=0.8):
    # Apply template matching
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    return np.argwhere(res >= threshold)


class CustomAtariEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        game: str,
        folder_name: str,
        n_objects: int,
        obs_type: str = "distance",
        frameskip: Union[int, Tuple[int, int], List[int]] = 1,
        repeat_action_probability: Union[int, float] = 0.0,
        full_action_space: bool = False,
    ):
        """Frameskip should be either a tuple (indicating a random range to
        choose from, with the top value exclude), or an int."""

        assert obs_type in ("distance")

        self.game = game
        self.game_path = atari_py.get_game_path(game)

        if not os.path.exists(self.game_path):
            msg = "You asked for game %s but path %s does not exist"
            raise IOError(msg % (game, self.game_path))
        self._obs_type = obs_type
        self.frameskip = frameskip
        self.ale = atari_py.ALEInterface()
        self.viewer = None

        # Tune (or disable) ALE's action repeat:
        # https://github.com/openai/gym/issues/349
        assert isinstance(
            repeat_action_probability, (float, int)
        ), "Invalid repeat_action_probability: {!r}".format(repeat_action_probability)
        self.ale.setFloat(
            "repeat_action_probability".encode("utf-8"), repeat_action_probability
        )

        self.seed()

        self.images_folder = os.path.join(
            os.path.dirname(__file__),
            "images",
            folder_name,
        )
        self.objects_array = self._get_objects_from_imgs()
        self.agent_array = self._get_agent_from_img()

        self._action_set = (
            self.ale.getLegalActionSet()
            if full_action_space
            else self.ale.getMinimalActionSet()
        )

        self.n_objects = n_objects
        self.vector_size = (
            n_objects * 3
        )  # objectos a retornar por sus coordenadas (x, y) más distancia relativa

        self.action_space = spaces.Discrete(len(self._action_set))
        self.observation_space = spaces.Box(
            low=-200.0,
            high=200.0,
            dtype=np.float64,
            shape=(self.vector_size,),
        )

    def seed(self, seed=None) -> List:
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        # Empirically, we need to seed before loading the ROM.
        self.ale.setInt(b"random_seed", seed2)
        self.ale.loadROM(self.game_path)

        return [seed1, seed2]

    def step(self, a) -> Tuple[List, int, bool, Dict]:
        reward = 0.0
        action = self._action_set[a]

        if isinstance(self.frameskip, int):
            num_steps = self.frameskip
        else:
            num_steps = self.np_random.randint(self.frameskip[0], self.frameskip[1])
        for _ in range(num_steps):
            reward += self.ale.act(action)
        ob = self._get_obs()

        return ob, reward, self.ale.game_over(), {"ale.lives": self.ale.lives()}

    def _get_objects_from_imgs(self) -> List[np.ndarray]:
        return [
            cv2.imread(object_filename, 0)
            for object_filename in glob.glob(f"{self.images_folder}/object*.png")
        ]

    def _get_agent_from_img(self) -> List[np.ndarray]:
        return [
            cv2.imread(agent_filename, 0)
            for agent_filename in glob.glob(f"{self.images_folder}/agent*.png")
        ]

    def _get_obs(self) -> np.ndarray:
        return self._get_coords_and_distances()

    def _get_coords_and_distances(self) -> np.ndarray:
        # Obtener tablero en gris
        gray_board = self._get_gray_board()
        # Buscar coordenadas de objetos y retornarlas
        objects_coords_and_distances = self._get_objects_coords(gray_board)
        return objects_coords_and_distances

    def _get_gray_board(self) -> np.ndarray:
        # Return gray image and reshape ndarray
        gray_screen = self.ale.getScreenGrayscale()
        if self.game == "ms_pacman":
            return gray_screen.reshape(210, 160)[:173, ::]
        return gray_screen.reshape(210, 160)

    def _get_objects_coords(self, board: np.ndarray) -> np.ndarray:
        agent_centroid, found_agent = self._find_agent(board)
        objects_centroids = self._find_objects(board)

        if found_agent:
            self.agent_centroid = agent_centroid
        elif not hasattr(self, "agent_centroid"):
            self.agent_centroid = (37.5, 190.0)

        return self._calculate_objects_centroids(self.agent_centroid, objects_centroids)

    def _find_agent(
        self, board: np.ndarray
    ) -> Tuple[Union[Tuple[float, float], None], bool]:
        agents = self.agent_array

        for agent_repr in agents:
            loc = apply_match_template(board, agent_repr)

            if loc.any():
                w, h = agent_repr.shape[::-1]
                y, x = loc[0]
                centroid = (x + (w / 2), y + (h / 2))
                return centroid, True

        return None, False

    def _find_objects(self, board: np.ndarray) -> np.ndarray:
        objects = self.objects_array
        temp = []

        for obj in objects:
            loc = apply_match_template(board, obj)

            if not loc.any():
                continue

            w, h = obj.shape[::-1]

            loc[:, 0] = loc[:, 0] + (h / 2)
            loc[:, 1] = loc[:, 1] + (w / 2)
            temp.append(loc)

        if not temp:
            return np.array([])

        centroids = np.vstack(temp)

        # Swap columns
        centroids[:, 0], centroids[:, 1] = centroids[:, 1], centroids[:, 0].copy()
        return centroids

    def _calculate_objects_centroids(
        self, agent_centroid: Tuple[float, float], objects_centroids: np.ndarray
    ) -> np.ndarray:
        if objects_centroids.size <= 0:
            return np.array([0.0, 0.0, 0.0] * self.n_objects).reshape(self.vector_size)

        agent_x, agent_y = agent_centroid
        coords = objects_centroids.copy()
        coords[:, 0], coords[:, 1] = coords[:, 0] - agent_x, agent_y - coords[:, 1]

        distances = np.sqrt(np.sum(np.square(coords), axis=1, keepdims=True))

        coords = np.append(coords, distances, axis=1)

        sorted_objects_coords = coords[np.argsort(coords[:, 2])]

        sorted_objects_coords = np.vstack(
            (np.array(agent_centroid + (0.0,)), sorted_objects_coords)
        )

        found = sorted_objects_coords.shape[0]
        if found < self.n_objects:
            missing = self.n_objects - found
            dummy = np.array([[0.0, 0.0, 0.0]] * missing)
            sorted_objects_coords = np.vstack((sorted_objects_coords, dummy))

        return sorted_objects_coords[: self.n_objects, :].reshape(self.vector_size)

    def _get_image(self) -> np.ndarray:
        return self.ale.getScreenRGB2()

    @property
    def _n_actions(self) -> int:
        return len(self._action_set)

    def reset(self) -> np.ndarray:
        self.ale.reset_game()
        return self._get_obs()

    def render(self, mode="human"):
        img_rgb = self._get_image()
        # img = self._get_gray_board()

        # agent_gray_array = self.agent_array
        # for agent_repr in agent_gray_array:

        #     agent_w, agent_h = agent_repr.shape[::-1]
        #     agent_res = cv2.matchTemplate(img, agent_repr, cv2.TM_CCOEFF_NORMED)
        #     agent_loc = np.argwhere(agent_res >= 0.8)

        #     if not agent_loc.any():
        #         continue

        #     y, x = agent_loc[0]
        #     centroid = (x + (agent_w / 2), y + (agent_h / 2))
        #     temp = []
        #     temp2 = []

        #     for enemy in self.objects_array:
        #         enemy_res = cv2.matchTemplate(img, enemy, cv2.TM_CCOEFF_NORMED)
        #         enemy_loc = np.argwhere(enemy_res >= 0.8)

        #         if not enemy_loc.any():
        #             continue

        #         loc = enemy_loc.copy()
        #         loc2 = enemy_loc.copy()
        #         w, h = enemy.shape[::-1]
        #         temp2.append(loc2)
        #         loc[:, 0] = loc[:, 0] + (h / 2)
        #         loc[:, 1] = loc[:, 1] + (w / 2)
        #         temp.append(loc)

        #     if not temp:
        #         return

        #     centroids = np.vstack(temp)
        #     locs = np.vstack(temp2)

        #     # Swap columns
        #     centroids[:, 0], centroids[:, 1] = (
        #         centroids[:, 1],
        #         centroids[:, 0].copy(),
        #     )

        #     agent_x, agent_y = centroid
        #     coords = centroids.copy()
        #     coords[:, 0], coords[:, 1] = (
        #         coords[:, 0] - agent_x,
        #         agent_y - coords[:, 1],
        #     )

        #     distances = np.square(coords)
        #     distances = np.sqrt(np.sum(distances, axis=1, keepdims=True))

        #     coords = np.append(coords, distances, axis=1)
        #     coords = np.append(coords, locs, axis=1)

        #     sorted_enemy_coords = coords[np.argsort(coords[:, 2])]
        #     sorted_enemy_coords = np.vstack(
        #         (np.append([0, 0, 0], agent_loc[0]), sorted_enemy_coords)
        #     )
        #     nearest_enemies = sorted_enemy_coords[: self.n_objects, :]
        #     loc2 = (
        #         nearest_enemies[:, 3].astype("uint8"),
        #         nearest_enemies[:, 4].astype("uint8"),
        #     )

        #     for pt in zip(*loc2[::-1]):
        #         cv2.rectangle(
        #             img_rgb,
        #             pt,
        #             (pt[0] + w, pt[1] + h),
        #             (255, 0, 0),
        #             1,
        #         )

        if mode == "rgb_array":
            return img_rgb
        elif mode == "human":
            from gym.envs.classic_control import rendering

            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img_rgb)
            return self.viewer.isopen

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
