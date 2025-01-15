import math

import gym
import numpy as np

from snake_game import SnakeGame

class SnakeEnv(gym.Env):
    """
    初始化Snake环境。

    参数:
    - seed: 随机种子，用于确保环境的可重复性。
    - board_size: 棋盘的大小，决定了游戏区域的宽度和高度。
    - silent_mode: 是否启用静默模式，如果启用，游戏运行时不会显示图形界面。
    - limit_step: 是否限制步骤数量，用于控制游戏的最大步数。

    该环境用于模拟贪吃蛇游戏，提供与 Gym 兼容的接口。
    """
    def __init__(self, seed=0, board_size=12, silent_mode=True, limit_step=True):
        super().__init__()
        # 初始化SnakeGame实例，这是实际运行游戏的内部引擎。
        self.game = SnakeGame(seed=seed, board_size=board_size, silent_mode=silent_mode)
        # 重置游戏状态，准备开始新的游戏。
        self.game.reset()

        # 记录是否启用静默模式。
        self.silent_mode = silent_mode

        # 定义动作空间，贪吃蛇可以向上、左、右、下四个方向移动。
        self.action_space = gym.spaces.Discrete(4) # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN

        # 定义观测空间，表示游戏画面的大小和颜色深度。
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(84, 84, 3),
            dtype=np.uint8
        )

        # 设置随机种子，用于环境的可重复性。
        self.seed_val = 5201314

        # 记录棋盘大小，用于计算最大步数和其他参数。
        self.board_size = board_size
        # 计算棋盘的格子总数，决定蛇的最大长度。
        self.grid_size = board_size ** 2 # Max length of snake is board_size^2
        # 计算游戏初始时蛇的长度。
        self.init_snake_size = len(self.game.snake)
        # 计算蛇的最大增长长度。
        self.max_growth = self.grid_size - self.init_snake_size

        # 初始化游戏结束标志为False。
        self.done = False

        # 根据是否限制步骤，设置步数限制。
        if limit_step:
            # 如果限制步骤，设置步数限制为棋盘格子数乘以4，提供充足的机会获取食物。
            self.step_limit = self.grid_size * 4 # More than enough steps to get the food.
        else:
            # 如果不限制步骤，设置极大的步数限制，几乎等于不限制。
            self.step_limit = 1e9 # Basically no limit.
        # 初始化奖励计数器，用于跟踪连续行动的次数。
        self.reward_step_counter = 0

    def reset(self):
        """
        重置游戏环境。

        此方法在游戏或训练回合开始时调用，以初始化游戏状态。
        它不仅重置了游戏本身，还重置了一些内部变量，为新的训练回合做准备。
        """
        # 重置游戏状态
        self.game.reset()

        # 初始化完成标志为False，表示游戏尚未结束
        self.done = False
        # 初始化奖励计数器，用于计算步数奖励
        self.reward_step_counter = 0

        # 生成并返回初始观察值
        obs = self._generate_observation()
        return obs
    
    def step(self, action):
        self.done, info = self.game.step(action) # info = {"snake_size": int, "snake_head_pos": np.array, "prev_snake_head_pos": np.array, "food_pos": np.array, "food_obtained": bool}
        obs = self._generate_observation()

        reward = 0.0
        self.reward_step_counter += 1

        if info["snake_size"] == self.grid_size: # Snake fills up the entire board. Game over.
            reward = self.max_growth * 0.1 # Victory reward
            self.done = True
            if not self.silent_mode:
                self.game.sound_victory.play()
            return obs, reward, self.done, info
        
        if self.reward_step_counter > self.step_limit: # Step limit reached, game over.
            self.reward_step_counter = 0
            self.done = True
        
        if self.done: # Snake bumps into wall or itself. Episode is over.
            # Game Over penalty is based on snake size.
            reward = - math.pow(self.max_growth, (self.grid_size - info["snake_size"]) / self.max_growth) # (-max_growth, -1)            
            reward = reward * 0.1
            return obs, reward, self.done, info
          
        elif info["food_obtained"]: # Food eaten. Reward boost on snake size.
            reward = info["snake_size"] / self.grid_size
            self.reward_step_counter = 0 # Reset reward step counter
        
        else:
            # Give a tiny reward/penalty to the agent based on whether it is heading towards the food or not.
            # Not competing with game over penalty or the food eaten reward.
            if np.linalg.norm(info["snake_head_pos"] - info["food_pos"]) < np.linalg.norm(info["prev_snake_head_pos"] - info["food_pos"]):
                reward = 1 / info["snake_size"]
            else:
                reward = - 1 / info["snake_size"]
            reward = reward * 0.1

        # max_score: 72 + 14.1 = 86.1
        # min_score: -14.1

        return obs, reward, self.done, info
    
    def render(self):
        self.game.render()

    def seed(self, seed=None):
        self.seed_val = seed
        return [seed]

    def get_action_mask(self):
        return np.array([[self._check_action_validity(a) for a in range(self.action_space.n)]])
    
    # Check if the action is against the current direction of the snake or is ending the game.
    def _check_action_validity(self, action):
        current_direction = self.game.direction
        snake_list = self.game.snake
        row, col = snake_list[0]
        if action == 0: # UP
            if current_direction == "DOWN":
                return False
            else:
                row -= 1

        elif action == 1: # LEFT
            if current_direction == "RIGHT":
                return False
            else:
                col -= 1

        elif action == 2: # RIGHT 
            if current_direction == "LEFT":
                return False
            else:
                col += 1     
        
        elif action == 3: # DOWN 
            if current_direction == "UP":
                return False
            else:
                row += 1

        # Check if snake collided with itself or the wall. Note that the tail of the snake would be poped if the snake did not eat food in the current step.
        if (row, col) == self.game.food:
            game_over = (
                (row, col) in snake_list # The snake won't pop the last cell if it ate food.
                or row < 0
                or row >= self.board_size
                or col < 0
                or col >= self.board_size
            )
        else:
            game_over = (
                (row, col) in snake_list[:-1] # The snake will pop the last cell if it did not eat food.
                or row < 0
                or row >= self.board_size
                or col < 0
                or col >= self.board_size
            )

        if game_over:
            return False
        else:
            return True

    # EMPTY: BLACK; SnakeBODY: GRAY; SnakeHEAD: GREEN; FOOD: RED;
    def _generate_observation(self):
        obs = np.zeros((self.game.board_size, self.game.board_size), dtype=np.uint8)

        # Set the snake body to gray with linearly decreasing intensity from head to tail.
        obs[tuple(np.transpose(self.game.snake))] = np.linspace(200, 50, len(self.game.snake), dtype=np.uint8)
        
        # Stack single layer into 3-channel-image.
        obs = np.stack((obs, obs, obs), axis=-1)
        
        # Set the snake head to green and the tail to blue
        obs[tuple(self.game.snake[0])] = [0, 255, 0]
        obs[tuple(self.game.snake[-1])] = [255, 0, 0]

        # Set the food to red
        obs[self.game.food] = [0, 0, 255]

        # Enlarge the observation to 84x84
        obs = np.repeat(np.repeat(obs, 7, axis=0), 7, axis=1)

        return obs

# Test the environment using random actions
# NUM_EPISODES = 100
# RENDER_DELAY = 0.001
# from matplotlib import pyplot as plt

# if __name__ == "__main__":
#     env = SnakeEnv(silent_mode=False)
    
    # # Test Init Efficiency
    # print(MODEL_PATH_S)
    # print(MODEL_PATH_L)
    # num_success = 0
    # for i in range(NUM_EPISODES):
    #     num_success += env.reset()
    # print(f"Success rate: {num_success/NUM_EPISODES}")

    # sum_reward = 0

    # # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN
    # action_list = [1, 1, 1, 0, 0, 0, 2, 2, 2, 3, 3, 3]
    
    # for _ in range(NUM_EPISODES):
    #     obs = env.reset()
    #     done = False
    #     i = 0
    #     while not done:
    #         plt.imshow(obs, interpolation='nearest')
    #         plt.show()
    #         action = env.action_space.sample()
    #         # action = action_list[i]
    #         i = (i + 1) % len(action_list)
    #         obs, reward, done, info = env.step(action)
    #         sum_reward += reward
    #         if np.absolute(reward) > 0.001:
    #             print(reward)
    #         env.render()
            
    #         time.sleep(RENDER_DELAY)
    #     # print(info["snake_length"])
    #     # print(info["food_pos"])
    #     # print(obs)
    #     print("sum_reward: %f" % sum_reward)
    #     print("episode done")
    #     # time.sleep(100)
    
    # env.close()
    # print("Average episode reward for random strategy: {}".format(sum_reward/NUM_EPISODES))
