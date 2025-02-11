import numpy as np


class TaskSchedulingEnv:
    def __init__(self, num_vehicles, num_tasks):
        self.num_vehicles = num_vehicles
        self.num_tasks = num_tasks
        self.state = None
        self.done = False
        self.reset()

    def reset(self):
        """
        重置环境状态
        """
        # 初始化车辆的任务调度情况（每辆车的任务ID和对应时刻）
        self.state = np.zeros((self.num_vehicles, self.num_tasks, 2))  # 每辆车有num_tasks个任务，任务ID和执行时刻
        self.done = False
        return self.state

    def step(self, action):
        """
        执行动作并更新环境状态
        :param action: 强化学习代理选择的动作
        :return: 下一个状态，奖励，是否结束
        """
        # 执行动作，更新任务调度（邻域搜索）
        self._apply_action(action)

        # 计算奖励（如路径时间、任务完成时间等）
        reward = self._calculate_reward()

        # 判断是否达到结束条件（例如所有任务已完成，或达到最大迭代次数）
        self.done = self._check_done()

        return self.state, reward, self.done

    def _apply_action(self, action):
        """
        根据动作更新当前状态（执行邻域搜索）
        """
        # 动作一般是对调度方案的修改，可以是交换任务、调整时刻等
        vehicle, task1, task2 = action  # 假设action表示车辆ID和需要交换的两个任务ID
        self.state[vehicle, task1, 1], self.state[vehicle, task2, 1] = self.state[vehicle, task2, 1], self.state[
            vehicle, task1, 1]

    def _calculate_reward(self):
        """
        计算当前调度的奖励，通常是基于目标函数的（如路径时间、任务延迟等）
        :return: 奖励值
        """
        total_time = np.sum(self.state[:, :, 1])  # 假设奖励与总时间有关，越小越好
        return -total_time  # 最小化路径时间为目标

    def _check_done(self):
        """
        判断是否结束
        """
        # 任务完成，或者达到最大迭代次数
        return False  # 这里可以根据实际需求设置结束条件
