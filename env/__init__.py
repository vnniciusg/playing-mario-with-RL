import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers.frame_stack import FrameStack
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from gym.wrappers.resize_observation import ResizeObservation


def create_mario_env(env_id: str = "SuperMarioBros-1-1-v0", render_mode: str = "rgb_array") -> FrameStack:
    """Create a Mario environment with specific configurations."""

    env = gym_super_mario_bros.make(env_id, apply_api_compatibility=True, render_mode=render_mode)

    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, (84, 84))
    env = FrameStack(env, 4)

    return env


if __name__ == "__main__":
    __import__("warnings").filterwarnings("ignore")

    env = create_mario_env(render_mode="human")

    done = True
    for step in range(100_000):
        if done:
            env.reset()
        state, reward, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated
        env.render()
    env.close()
