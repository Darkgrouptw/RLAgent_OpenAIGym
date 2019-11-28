from RLAgent_DeepQNetwork import DeepQNetwork
import matplotlib.pyplot as plt
import gym
import numpy as np
import gc

# # 設定環境
# 
# ## Observation
# | Num | Observation | Min   | Max  |
# |-----|-------------|-------|------|
# | 0   | position    | -1.2  | 0.6  |
# | 1   | velocity    | -0.07 | 0.07 |
# 
# ## Action
# | Num | Action     |
# |-----|------------|
# | 0   | push left  |
# | 1   | no push    |
# | 2   | push right |

# 創建環境
env = gym.make('MountainCar-v0')
gc.enable()
# gc.set_debug(gc.DEBUG_STATS|gc.DEBUG_LEAK)

# EpsilonFunction
# https://www.desmos.com/calculator/qgg3tdayyt
memory_size = 2000
Agent = DeepQNetwork(
    env.observation_space.shape[0],
    env.action_space.n,
    learningRate=1e-3,
    gamma=0.95,
    decayRate=5e-5,
    # decayRate=0.0002,
    batchSize=128,
    memorySize=memory_size,
    targetReplaceIter=100,
    IsOutputGraph=True
)

# # 開始訓練
# 主要有兩個步驟：
# 1. 產生 random 資料，塞滿 memorySize
# 2. 開始按照 explore or exploit 的策略下去 Try

# ## Helper Function
def GenerateRandomData():
    state = env.reset()
    for i in range(memory_size):
        action = env.action_space.sample()
        nextState, reward, IsDone, _ = env.step(action)
        
        Agent.storeMemory(state, action, reward, nextState)
        if IsDone:
            state = env.reset()
        state = nextState

# Training Part
# TotalReward = []
def TrainModel(EpochNumber = 300):
    env.seed(3)
    for i in range(EpochNumber):
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
            # 根據高度修改 (加快收斂)
            position, velocity = nextState
            reward = abs(position - (-0.5))
            
            
            totalReward += reward
            
            # 存進記憶庫裡
            Agent.storeMemory(
                state=state,
                action=actionValue,
                reward=reward,
                nextState=nextState
            )

            # 學習
            Agent.learn()
                
            if IsDone:
                print("Epoch:",(i+1)," TotalReward:", totalReward, " P:", Agent._EpsilonFunction())
                # TotalReward.append(totalReward)

                if i % 100 == 0:
                    Agent.model.save("MountainCarV0." + str(i) + ".h5")
                    gc.collect()
                break

            state = nextState
            
        # 判斷是否完成
        # if np.mean(TotalReward[-10:]) > 50:
        #     break

    # 儲存模型
    Agent.model.save("MountainCar-v0.h5")
    env.close()

# Main
GenerateRandomData()
TrainModel(10000)
exit()