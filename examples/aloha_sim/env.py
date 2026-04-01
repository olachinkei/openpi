import gym_aloha  # noqa: F401
import gymnasium
import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override


class AlohaSimEnvironment(_environment.Environment):
    """An environment for an Aloha robot in simulation."""

    def __init__(
        self,
        task: str,
        obs_type: str = "pixels_agent_pos",
        seed: int = 0,
        render_mode: str | None = None,
        visualization_width: int = 640,
        visualization_height: int = 336,
        visualization_camera_id: str = "angle",
    ) -> None:
        np.random.seed(seed)
        self._rng = np.random.default_rng(seed)

        gym_kwargs: dict[str, object] = {"obs_type": obs_type}
        if render_mode is not None:
            gym_kwargs["render_mode"] = render_mode
            gym_kwargs["visualization_width"] = visualization_width
            gym_kwargs["visualization_height"] = visualization_height
        self._gym = gymnasium.make(task, **gym_kwargs)

        self._last_obs = None
        self._done = True
        self._episode_reward = 0.0
        self._last_render_frame: np.ndarray | None = None
        self._last_raw_top_frame: np.ndarray | None = None
        self._visualization_width = visualization_width
        self._visualization_height = visualization_height
        self._visualization_camera_id = visualization_camera_id

    @override
    def reset(self) -> None:
        gym_obs, _ = self._gym.reset(seed=int(self._rng.integers(2**32 - 1)))
        self._last_obs = self._convert_observation(gym_obs)  # type: ignore
        self._done = False
        self._episode_reward = 0.0
        self._update_render_frame()

    @override
    def is_episode_complete(self) -> bool:
        return self._done

    @override
    def get_observation(self) -> dict:
        if self._last_obs is None:
            raise RuntimeError("Observation is not set. Call reset() first.")

        return self._last_obs  # type: ignore

    @override
    def apply_action(self, action: dict) -> None:
        gym_obs, reward, terminated, truncated, info = self._gym.step(action["actions"])
        self._last_obs = self._convert_observation(gym_obs)  # type: ignore
        self._done = terminated or truncated
        self._episode_reward = max(self._episode_reward, reward)
        self._update_render_frame()

    @property
    def episode_reward(self) -> float:
        return float(self._episode_reward)

    def is_success(self, success_reward_threshold: float = 4.0) -> bool:
        return self.episode_reward >= success_reward_threshold

    def get_video_frame(self) -> np.ndarray:
        if self._last_render_frame is not None:
            return self._last_render_frame
        if self._last_raw_top_frame is not None:
            return self._last_raw_top_frame
        raise RuntimeError("Video frame is not available. Call reset() first.")

    def _convert_observation(self, gym_obs: dict) -> dict:
        img = gym_obs["pixels"]["top"]
        self._last_raw_top_frame = image_tools.convert_to_uint8(np.asarray(img))
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
        # Convert axis order from [H, W, C] --> [C, H, W]
        img = np.transpose(img, (2, 0, 1))

        return {
            "state": gym_obs["agent_pos"],
            "images": {"cam_high": img},
        }

    def _update_render_frame(self) -> None:
        frame = None
        try:
            physics = getattr(getattr(self._gym, "unwrapped", self._gym), "_env", None)
            if physics is not None and hasattr(physics, "physics"):
                frame = physics.physics.render(
                    height=self._visualization_height,
                    width=self._visualization_width,
                    camera_id=self._visualization_camera_id,
                )
        except (AttributeError, TypeError, ValueError):
            frame = None

        if frame is None:
            try:
                frame = self._gym.render()
            except (AttributeError, TypeError):
                frame = None

        self._last_render_frame = None if frame is None else image_tools.convert_to_uint8(np.asarray(frame))
