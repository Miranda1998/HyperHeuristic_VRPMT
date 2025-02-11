from Env import *
from Q_learning_Agent import *



if __name__ == '__main__':
    def train():
        num_vehicles = 2  # 车辆数量
        num_tasks = 5  # 任务数量
        env = TaskSchedulingEnv(num_vehicles, num_tasks)
        agent = QLearningAgent(state_size=10, action_size=20)  # 假设有10个状态，20个可能的动作（需要根据具体情况调整）

        episodes = 1000  # 训练回合数
        for ep in range(episodes):
            state = env.reset()  # 重置环境
            total_reward = 0
            done = False
            while not done:
                action = agent.choose_action(state)  # 选择动作
                next_state, reward, done = env.step(action)  # 执行动作并获取反馈
                agent.learn(state, action, reward, next_state, done)  # 学习
                state = next_state  # 更新状态
                total_reward += reward

            print(f"Episode {ep + 1}, Total Reward: {total_reward}")
