import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers.frame_stack import FrameStack
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from gym.wrappers.resize_observation import ResizeObservation
from gym import Wrapper


class SkipFrame(Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward, done = 0.0, False
        for _ in range(self.skip):
            next_state, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return next_state, total_reward, done, trunc, info


def create_mario_env(env_id: str = "SuperMarioBros-1-1-v0", render_mode: str = "human") -> FrameStack:
    """Create a Mario environment with specific configurations."""

    env = gym_super_mario_bros.make(env_id, apply_api_compatibility=True, render_mode=render_mode)

    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    env = SkipFrame(env, skip=4)
    env = ResizeObservation(env, (84, 84))
    env = GrayScaleObservation(env)
    env = FrameStack(env, 4, lz4_compress=True)

    return env
