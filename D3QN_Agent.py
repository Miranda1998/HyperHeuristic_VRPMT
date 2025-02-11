import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

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



class D3QNAgent(Agent):
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01,
                 alpha=0.001, batch_size=32, memory_size=10000):
        super().__init__(state_size, action_size)

        # 超参数
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.alpha = alpha
        self.batch_size = batch_size
        self.memory_size = memory_size

        # 创建 Q 网络和目标网络
        self.q_network = self._build_network(state_size, action_size)
        self.target_network = self._build_network(state_size, action_size)

        # 初始化优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.alpha)

        # 经验回放
        self.memory = deque(maxlen=self.memory_size)

        # 初始化目标网络
        self._update_target_network()

    def _build_network(self, state_size, action_size):
        """
        构建一个简单的多层感知机（MLP）神经网络来预测 Q 值
        """
        model = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        return model

    def choose_action(self, state):
        """
        epsilon-greedy 策略，选择动作
        """
        if random.random() < self.epsilon:
            return random.choice(range(self.action_size))  # 探索
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # 转为张量
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()  # 利用

    def learn(self, state, action, reward, next_state, done):
        """
        更新 Q 网络
        """
        # 存储经验
        self.memory.append((state, action, reward, next_state, done))

        # 当经验足够多时，进行学习
        if len(self.memory) >= self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            # 转换为 PyTorch 张量
            states_tensor = torch.FloatTensor(states)
            next_states_tensor = torch.FloatTensor(next_states)
            actions_tensor = torch.LongTensor(actions)
            rewards_tensor = torch.FloatTensor(rewards)
            dones_tensor = torch.FloatTensor(dones)

            # 计算 Q 网络的 Q 值
            q_values = self.q_network(states_tensor)
            q_values_next = self.target_network(next_states_tensor)

            # 计算目标 Q 值
            next_q_values = torch.max(q_values_next, dim=1)[0]
            target_q_values = rewards_tensor + self.gamma * next_q_values * (1 - dones_tensor)

            # 获取当前状态的 Q 值
            action_q_values = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

            # 计算损失
            loss = nn.MSELoss()(action_q_values, target_q_values)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # 更新 epsilon（探索率衰减）
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

        # 定期更新目标网络
        self._update_target_network()

    def _update_target_network(self):
        """
        将 Q 网络的权重复制到目标网络
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
