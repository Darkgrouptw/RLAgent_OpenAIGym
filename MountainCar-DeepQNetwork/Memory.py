import numpy as np

# 管理記憶庫的
class Memory:
    def __init__(
            self,
            memorySize,                                     # 記憶庫大小
            numSize                                         # 每一筆資料要多少
    ):
        self.memorySize = memorySize
        self.memory = np.zeros([memorySize, numSize], dtype=np.float32)
        self.memoryIndex = 0

    # sample 出 Batchsize 個
    def sample(
            self,
            batchSize = 32                                      # Sample 個數
    ):
        chooseIndex = np.random.choice(self.memorySize, size=batchSize)
        return self.memory[chooseIndex]

    # 存入記憶庫
    def store(
            self,
            data                                            # 資料
    ):
        index = self.memoryIndex % self.memorySize

        self.memory[index, :] = data
        self.memoryIndex += 1