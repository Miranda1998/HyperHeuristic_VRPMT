import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


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

class PPOAgent(Agent):
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=0.2, alpha=0.0003, batch_size=32, memory_size=10000,
                 tau=0.95):
        super().__init__(state_size, action_size)

        # 超参数
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.tau = tau

        # 策略网络（Actor）和价值网络（Critic）
        self.actor_network = self._build_actor_network(state_size, action_size)
        self.critic_network = self._build_critic_network(state_size)

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=self.alpha)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=self.alpha)

        # 经验回放
        self.memory = deque(maxlen=self.memory_size)

    def _build_actor_network(self, state_size, action_size):
        """
        构建策略网络（Actor）
        """
        model = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Softmax(dim=-1)  # 输出概率分布
        )
        return model

    def _build_critic_network(self, state_size):
        """
        构建价值网络（Critic）
        """
        model = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        return model

    def choose_action(self, state):
        """
        从策略网络中选择动作
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.actor_network(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()  # 从概率分布中采样
        return action.item(), dist.log_prob(action)

    def learn(self, state, action, reward, next_state, done, old_log_prob):
        """
        更新策略网络和价值网络
        """
        # 存储经验
        self.memory.append((state, action, reward, next_state, done, old_log_prob))

        # 当经验足够多时，进行学习
        if len(self.memory) >= self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones, old_log_probs = zip(*batch)

            # 转换为 PyTorch 张量
            states_tensor = torch.FloatTensor(states)
            actions_tensor = torch.LongTensor(actions)
            rewards_tensor = torch.FloatTensor(rewards)
            next_states_tensor = torch.FloatTensor(next_states)
            dones_tensor = torch.FloatTensor(dones)
            old_log_probs_tensor = torch.stack(old_log_probs)

            # 计算优势函数
            values = self.critic_network(states_tensor).squeeze(-1)
            next_values = self.critic_network(next_states_tensor).squeeze(-1)
            deltas = rewards_tensor + self.gamma * next_values * (1 - dones_tensor) - values
            advantages = deltas.detach()

            # 计算当前的价值函数
            new_values = self.critic_network(states_tensor).squeeze(-1)

            #

