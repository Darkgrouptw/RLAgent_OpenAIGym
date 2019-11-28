import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.activations import relu
from tensorflow.keras import Model
import numpy as np
from Memory import Memory

np.random.seed(3)
tf.random.set_seed(3)

# Deep Q Network
class DeepQNetwork:
    def __init__(
            self,

            # 初始
            numFeatures,                                    # Features 數目
            numActions,                                     # Action 數目
            learningRate = 0.01,                            # Learning Rate
            batchSize = 32,                                 # 每次要抓多少
            gamma = 0.99,                                   # 獎賞每經過一次要下降多少 (Reward Decay)

            # Explore or Exploit
            exploreStart = 1,                               # 一開始要 Explore 的機率
            exploreEnd = 0.01,                              # 最後 Explore 會遞減的機率
            decayRate = 1e-4,                               # 每次測試完之後，會下降的比例

            # Memory & 其他
            memorySize = 10000,                             # 記憶庫大小
            targetReplaceIter = 100,                        # 多少次之後要把 evaluate 蓋掉 target net
            IsOutputGraph = False                           # 是否要 print Graph
    ):
        # 清除之前所有使用過的記憶體
        tf.keras.backend.clear_session()
        # gpus = tf.config.experimental.list_physical_devices('GPU')
        # tf.config.experimental.set_virtual_device_configuration(
        #     gpus[0],
        #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8096)])

        # 儲存需要用到的參數
        self.numFeatures = numFeatures
        self.numActions = numActions
        self.learningRate = learningRate
        self.batchSize = batchSize
        self.gamma = gamma
        self.exploreStart = exploreStart
        self.exploreEnd = exploreEnd
        self.decayRate = decayRate
        self.targetReplaceIter = targetReplaceIter

        # 建造網路
        self._BuildNet(IsOutputGraph)

        # 創建 [State (size => n_features),
        #       Action (size => 1 (0 1 2 3)),
        #       Reward (size => 1),
        #       Next State (Size => n_features)]
        self.memory = Memory(memorySize, numFeatures * 2 + 2)

        # 設定 learning Counter
        self.learnStepCounter = 0
        # self.lossArray = []

    # 建構網路
    def _BuildNet(
            self,
            IsOutputGraph
    ):
        # 設定的參數
        layer1_HiddenUnits = 64

        #############################################################
        # Evaluate Net
        # 要即時更新，跟以前做比對
        #############################################################
        state = Input([self.numFeatures], dtype = tf.float32, name="State")
        net = Dense(layer1_HiddenUnits,
                    activation=relu,
                    name="EvalNet_layer1")(state)
        net = Dense(self.numActions,
                    name="EvalNet_output")(net)
        q_eval = net

        # 設定 Model
        self.model = Model(inputs=state, outputs=q_eval)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(self.learningRate),
                           loss=tf.keras.losses.MeanSquaredError())

        #############################################################
        # Target Net
        # 一段時間才會更新 (此網路是要預測在下一個狀態下，應該要什麼樣的判斷，然後在乘上 gamma 來更新 Evaluate Net)
        #############################################################
        nextState = Input([self.numFeatures], dtype=tf.float32, name="TargetState")
        net = Dense(layer1_HiddenUnits,
                    activation=relu,
                    name="Target_layer1")(nextState)
        net = Dense(self.numActions,
                    name="Target_output")(net)
        q_next = net

        # 設定 Model
        self.Tmodel = Model(inputs=nextState, outputs=q_next)

        # 接者產生 Tensorboard callback
        # logdir = os.path.join("logs")
        # if not os.path.exists(logdir):
        #     os.mkdir(logdir)
        # self.tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_graph=IsOutputGraph)

    def _EpsilonFunction(self):
        return (self.exploreStart - self.exploreEnd) * np.exp(-self.decayRate * self.learnStepCounter) + self.exploreEnd
        # return np.max([self.exploreStart - self.decayRate * self.learnStepCounter, self.exploreEnd])

    # 選擇
    def chooseAction(
            self,
            state,                                              # 目前狀況
            IsTrainning = False                                 # 是否使用於訓練
    ):
        # Probability
        exploreP = self._EpsilonFunction()

        # 接著下去判斷要執行哪一個結果
        action = 0
        if IsTrainning and exploreP > np.random.uniform():
            action = np.random.randint(0, self.numActions)
        else:
            actionValue = self.model.predict(state.reshape([-1, self.numFeatures]))
            action = np.argmax(actionValue)
        return action

    # 儲存記憶庫
    def storeMemory(
            self,
            state,                                          # 目前狀態
            action,                                         # 採取動作
            reward,                                         # 獎賞
            nextState                                       # 下一個狀態
    ):

        dataArray = np.zeros([state.shape[0] + 2 + nextState.shape[0]], dtype=np.float32)
        dataArray[:state.shape[0] + 0] = state
        dataArray[state.shape[0] + 0: state.shape[0] + 1] = action
        dataArray[state.shape[0] + 1:state.shape[0] + 2] = reward
        dataArray[state.shape[0] + 2:] = nextState

        self.memory.store(dataArray)

    # 學習
    def learn(self):
        # 先判斷是否要覆蓋 Target
        if self.learnStepCounter % self.targetReplaceIter == 0:
            self.Tmodel.set_weights(self.model.get_weights())

        # 拿出資料之後，丟進 Network 去預測
        batchMemoryArray = self.memory.sample(self.batchSize)
        stateArray = batchMemoryArray[:, :self.numFeatures]
        nextStateArray = batchMemoryArray[:, -self.numFeatures:]
        actionArray = batchMemoryArray[:, self.numFeatures].astype(int)             # 哪一個 Action 做的
        rewardArray = batchMemoryArray[:, self.numFeatures + 1]
        q_eval = self.model.predict(stateArray)
        q_next = self.Tmodel.predict(nextStateArray)                                 # 從後面取

        # 要去算差拒並乘上 gamma
        '''
        公式：
        Q(s, a) = r + gamma * max(Q(s', a'))
        
        假設
        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]
         
        然後實際上是
        q_next =
        [[-1, 2, 3,],
         [4, 5, -2]]
        action = 
        [[0, 2]]
        
        所以根據上方的公式，正確的 q_eval 應該是：
        [[(-1) - (1), x, x],
         [x, x, (-2) - (6)]]
        '''
        q_target = q_eval.copy()
        q_target[:, actionArray] = rewardArray + self.gamma * np.max(q_next, axis=1)

        loss = self.model.fit(
            x=stateArray,
            y=q_target,
            # initial_epoch=self.learnStepCounter,
            # epochs=1,
            # batch_size=self.batchSize,
            verbose=0 #,
            # callbacks=[self.tb_callback]
        )
        self.learnStepCounter += 1
        self.learnStepCounter = np.min([self.learnStepCounter, 100000])
        # self.lossArray.append(loss.history['loss'])