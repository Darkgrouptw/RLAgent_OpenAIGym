from RLAgent_DeepQNetwork import DeepQNetwork
import gym

# 創建環境
env = gym.make('MountainCar-v0')

# EpsilonFunction
# https://www.desmos.com/calculator/qgg3tdayyt
memory_size = 2000
Agent = DeepQNetwork(
    env.observation_space.shape[0],
    env.action_space.n,
    learningRate=1e-3,
    gamma=0.95,
    # decayRate=5e-5,
    decayRate=0.0002,
    batchSize=128,
    memorySize=memory_size,
    targetReplaceIter=100,
    IsOutputGraph=True
)
Agent.model.load_weights("MountainCar-v0.h5")

# Test Part
def TestModel():
    # 歸零
    state = env.reset()
    totalReward = 0

    # 開始模擬
    while True:
        # redner 畫面
        # if(i > EpochNumber * 0.75):
        env.render()

        # 選擇的動作
        actionValue = Agent.chooseAction(state, IsTrainning=True)

        # 選擇動作後 的結果
        nextState, reward, IsDone, Info = env.step(actionValue)

        # 修改一下 Reward
        # 根據高度修改
        position, velocity = nextState
        reward = abs(position - (-0.5))


        totalReward += reward

        # 學習
        Agent.learn()

        if IsDone:
            print("TotalReward:", totalReward)
            break

        state = nextState


# Main
TestModel()
exit()