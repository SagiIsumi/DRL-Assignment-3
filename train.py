from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
import gym
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from gym.vector import SyncVectorEnv
import imageio
import numpy as np
import matplotlib.pyplot as plt
from DQN_agent import *
from itertools import count
from collections import deque
import sys


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class EpisodicLifeMario(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped._life
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped._life
        return obs


# env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v0',
#                                stages=['1-1','1-2','1-3','1-4'])
def make_env():
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    # env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v0',
    #                             stages=['1-1','1-2',])
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env =MaxAndSkipEnv(env, 4)
    env = GrayScaleObservation(env, keep_dim=True)       # (1,H,W)    
    env = ResizeObservation(env, (84, 84) )               # (1,84,84)    
    env= EpisodicLifeMario(env)
    env = FrameStack(env, 4)                             # (4,84,84)
    return env
env = make_env()
output_index=2
video_path = f"random_agent_{output_index}.mp4"
fps = 30
writer = imageio.get_writer(video_path, fps=fps)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_path="./checkpoint/v1/qnet_4400.pth"
agent = DQN_Mario(12,load_path)
# TODO: Determine the number of episodes for training
# agent.q_net.load_state_dict(torch.load('./checkpoint/v1/qnet_500.pth',map_location=device))
# agent.target_net.load_state_dict(agent.q_net.state_dict())


def train():
    try:
        num_episodes = 8000
        reward_history = []
        return_record = [0]
        for episode in range(num_episodes):
            state = env.reset()
            state = np.squeeze(state, axis=-1)
            state = np.ascontiguousarray(state)
            total_reward = 0
            for t in count():
                # if episode%3==0:
                #     action= agent.get_action(state, True)
                # else:
                action= agent.get_action(state, False)
                # print(action)
                obs, reward, done, info = env.step(action)
                total_reward+=reward
                if done:
                    next_state = None
                else:
                    obs = np.squeeze(obs, axis=-1)
                    obs = np.ascontiguousarray(obs)
                    next_state = obs
                # shaped_reward = np.sign(reward) * (np.sqrt(abs(reward) + 1) - 1) + 0.001 * reward
                priority = min(abs(reward) + 0.1, 15.0)
                agent.replaybuffer.add(priority, state, action, reward,next_state,)
                agent.train()
                state = next_state
                frame = env.render(mode='rgb_array')
                frame = np.asarray(frame)
                if frame.dtype != np.uint8:
                    frame = (255 * np.clip(frame, 0, 1)).astype(np.uint8)
                writer.append_data(frame)

                if done:
                    break
            print(f"Episode {episode}, Total Reward: {total_reward}, buffer: {len(agent.replaybuffer)}")
            if episode % 200 == 0 and episode != 0:
                torch.save(agent.q_net.state_dict(), f'./checkpoint/v{output_index}/qnet_{episode}.pth')
            reward_history.append(total_reward)
            if episode % 30 == 0 and episode != 0:
                print(f"Episode {episode}, Reward: {np.mean(reward_history[-30:])}, buffer: {len(agent.replaybuffer)}")
                return_record.append(np.mean(reward_history[-30:]))
    finally:
        return return_record

if __name__ == "__main__":
    try:
        reward_history=train()
    finally:        
        plt.plot([30*i for i in range(len(reward_history))],reward_history)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Training History")
        plt.savefig("episodes-reward_figure.jpg")
        #env.close()
        print("Video saved at:", video_path)


    #755090  noisy_layer std=0.5 lr = 2.5e-4 batch_size=256 hard_update_step=10000
    #2072841  noisy_layer std=0.5 lr = 2.5e-4 tau=0.99 batch_size=256 soft_update_step=10 
    #947841  noisy_layer std=0.5 lr = 1e-4 tau=0.99 batch_size=256 Hard_update_step=3000 Qnet_noisy withour priority 
    #v1   noisy_layer std=0.8 lr = 2.5e-4 batch_size=256 hard_update_step=10000 
    #v2   noisy_layer std=2.5 lr = 2e-5 batch_size=256 hard_update_step=3000 
    #v3   noisy_layer std=0.8 lr = 1e-5 batch_size=256 hard_update_step=10000 #epsilon degree
    #v4   std=2.0