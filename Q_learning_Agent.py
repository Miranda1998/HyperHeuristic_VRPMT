import random
import numpy as np

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    def choose_action(self, state):
        """
        选择动作
        :param state: 当前环境状态
        :return: 选择的动作
        """
        raise NotImplementedError("choose_action方法需要在子类中实现")

    def learn(self, state, action, reward, next_state, done):
        """
        学习过程，根据当前经验更新策略
        :param state: 当前状态
        :param action: 当前动作
        :param reward: 当前奖励
        :param next_state: 下一个状态
        :param done: 是否结束
        """
        raise NotImplementedError("learn方法需要在子类中实现")



class QLearningAgent(Agent):
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995,
                 min_epsilon=0.01):
        super().__init__(state_size, action_size)
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.epsilon_decay = epsilon_decay  # 探索率衰减
        self.min_epsilon = min_epsilon  # 最小探索率
        self.q_table = np.zeros((state_size, action_size))  # Q表，存储每个状态-动作对的价值

    def choose_action(self, state):
        """
        epsilon-greedy策略
        :param state: 当前状态
        :return: 选择的动作
        """
        if random.uniform(0, 1) < self.epsilon:
            # 随机选择一个动作（探索）
            return random.choice(range(self.action_size))
        else:
            # 选择Q表中最大值对应的动作（利用）
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        """
        更新Q表
        :param state: 当前状态
        :param action: 当前动作
        :param reward: 当前奖励
        :param next_state: 下一个状态
        :param done: 是否结束
        """
        # Q学习更新规则
        best_next_action = np.argmax(self.q_table[next_state])
        target = reward + self.gamma * self.q_table[next_state, best_next_action] * (1 - done)
        self.q_table[state, action] += self.alpha * (target - self.q_table[state, action])

        # 衰减探索率
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
