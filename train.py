import tensorflow.keras as keras
from preprocess import generate_training_sequences, SEQUENCE_LENGTH
import json
# JSON文件的文件名
json_file_name = 'mapping.json'
# 打开JSON文件并加载数据
with open(json_file_name, 'r') as json_file:
    data = json.load(json_file)
# 计算JSON文件中键的数量
OUTPUT_UNITS = len(data)    #来自mapping.json里units的个数(这个数据集里有38个类), vocabulary_size
# 打印输出，验证结果
print(f"The number of OUTPUT_UNITS is: {OUTPUT_UNITS}")

NUM_UNITS = [512, 512]   #2层LSTM模型，可以增加，构建多层
LOSS = "sparse_categorical_crossentropy"    #error function
LEARNING_RATE = 0.001
EPOCHS = 50 # 40-100 
BATCH_SIZE = 64 #一批的数量
SAVE_MODEL_PATH = "model.h5"

"""
    每次运行会清理掉之前生成的
    “.h5文件”
"""

def build_model(output_units, num_units, loss, learning_rate):
    """构建并编译模型
    :参数 output_units (int): 输出单元的数量
    :参数 num_units (list of int): 隐藏层中单元的数量
    :参数 loss (str): 所使用的损失函数类型
    :参数 learning_rate (float): 应用的学习率
    :返回 model (tf model): 魔法发生的地方 :D
    """

    #构建模型架构
    input = keras.layers.Input(shape=(None, output_units))  #None是第一个维度，有多少time steps are passing to the model
    # (None表示想要尽可能多的time steps); output_units -> how many elements we have for each time step
    # (等于 vocabulary_size, 这里是38)
    #x = keras.layers.LSTM(num_units[0])(input)  #passing the inputs into the LSTM
    #x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.LSTM(num_units[0], return_sequences=True if len(num_units) > 1 else False)(input)  #将输入传递给 LSTM
    x = keras.layers.Dropout(0.2)(x)

   # 新添加的LSTM和Dropout层
    for i in range(1, len(num_units)):
       x = keras.layers.LSTM(num_units[i], return_sequences=True if i != len(num_units) - 1 else False)(x)
       x = keras.layers.Dropout(0.2)(x)

    output = keras.layers.Dense(output_units, activation="softmax")(x)

    model = keras.Model(input, output)

    # 编译模型
    model.compile(loss=loss,
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=["accuracy"])

    model.summary()

    return model


def train(output_units=OUTPUT_UNITS, num_units=NUM_UNITS, loss=LOSS, learning_rate=LEARNING_RATE):
    """训练并保存 TF 模型。
    :参数 output_units (int): 输出单元的数量
    :参数 num_units (list of int): 隐藏层中单元的数量
    :参数 loss (str): 所使用的损失函数类型
    :参数 learning_rate (float): 应用的学习率
    """

    # 生成训练序列
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)

    # 构建网络
    model = build_model(output_units, num_units, loss, learning_rate)

    # 训练模型
    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # 保存模型
    model.save(SAVE_MODEL_PATH)


if __name__ == "__main__":
    train()
