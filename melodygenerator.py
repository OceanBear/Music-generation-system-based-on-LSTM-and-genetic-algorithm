import json
import numpy as np
import tensorflow.keras as keras
import music21 as m21
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH    #从上一个文件夹里倒入

class MelodyGenerator:
    """这是一个封装了LSTM模型并提供生成旋律工具的类"""
    def save_to_txt(self, melody, file_name='mel.txt'):
        """将旋律列表保存到txt文件中"""
        with open(file_name, 'w') as f:
            for note in melody:
                f.write(note + ' ')

    def __init__(self, model_path="model-han.h5"):  #之前训练好的模型的地址
        """这是一个初始化TensorFlow模型的构造器。"""

        self.model_path = model_path    #模型的路径
        self.model = keras.models.load_model(model_path)    #

        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)

        self._start_symbols = ["/"] * SEQUENCE_LENGTH


    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
        # 输入melody seed, 如 "64 _ 63 _ _ "
        # temperature (可以是0到正无穷，但一般只取0-1)
        """使用深度学习模型生成旋律，并返回一个MIDI文件
        :参数 seed (str): 用于编码数据集的旋律种子
        :参数 num_steps (int): 生成的步数
        :参数 max_sequence_len (int): 种子中考虑生成的最大步数
        :参数 temperature (float): 在[0，1]区间的浮点数。接近0的数值使模型更具确定性
            更接近1的数使得生成更加不可预测
        :返回 melody (字符列表): 代表旋律的符号列表
        """

        # 创建带有起始符号的种子
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed

        # map seed to int  转化为整数
        seed = [self._mappings[symbol] for symbol in seed]  #

        for _ in range(num_steps):

            # limit the seed to max_sequence_length 限制种子序列的长度
            seed = seed[-max_sequence_length:]

            # one-hot encode the seed 对种子进行独热编码
            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))
            # (1, max_sequence_length, num of symbols in the vocabulary) 创建三维，符合keras
            onehot_seed = onehot_seed[np.newaxis, ...] #把原本的bi-dimensional array二维序列变成三维

            # 做出预测
            probabilities = self.model.predict(onehot_seed)[0]
            # [0.1, 0.2, 0.1, 0.6] -> 1 #每一个值表示样本出现的可能性
            output_int = self._sample_with_temperature(probabilities, temperature)

            # 更新种子
            seed.append(output_int)

            # 将整数映射到编码
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]

            # 检测旋律是否结束
            if output_symbol == "/":
                break

            # 如果没有结束, 就更新旋律
            melody.append(output_symbol)

        return melody

    def _sample_with_temperature(self, probabilites, temperature):
        """从概率数组中采样一个索引，使用温度重新应用softmax
        :参数 predictions (nd.array): 包含每个可能输出的概率的数组。
        :参数 temperature (float): 浮点数在区间[0，1]中。接近0的数字使模型更具确定性。接近1的数字使生成更不可预测。
        :返回 index (int): 选定的输出符号
        """
        # temperature -> infinity (probablity distribution tend to be the same, 会过于趋同变得死板)
        # temperature -> 0 (原本概率最高的值会被百分百选中, 变成确定事件, 变得可以预测)
        # temperature = 1
        predictions = np.log(probabilites) / temperature    #temperature变大，distribution会变得趋同
        probabilites = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilites)) # [0, 1, 2, 3]
        index = np.random.choice(choices, p=probabilites)

        return index    #返回值


    def save_melody(self, melody, step_duration=0.25, format="midi", file_name="mel.mid"): #step_duration=0.25, 指一个四分音符
        """将旋律转换为MIDI文件
        :参数 melody (str列表):
        :参数 min_duration (float): 每个时间步的四分之一长度的持续时间
        :参数 file_name (str): midi文件的名称
        :返回
        """

        # 创建一个music21stream（stream是一个包含所有音符的容器）
        stream = m21.stream.Stream()

        start_symbol = None
        step_counter = 1

        # 解析旋律中的所有符号，并创建音符/休止符对象
        # 60 _ _ _ r _ 62 _ (MIDI的音符/休止符被称为一个事件)
        for i, symbol in enumerate(melody):

            # 处理我们有音符/休止符的情况
            if symbol != "_" or i + 1 == len(melody): #判断是否为 _ 或者 结束

                # 确保我们处理的是第一个之后的音符/休止符
                if start_symbol is not None:

                    quarter_length_duration = step_duration * step_counter # 0.25 * 4 = 1

                    # 处理休止符
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)

                    # 处理音符
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)

                    stream.append(m21_event)

                    # 重置步数计数器
                    step_counter = 1

                start_symbol = symbol # start_symbol 变为现在的symbol

            # 处理遇到延长符号"_"的情况
            else:
                step_counter += 1

        # 将m21 stream写入midi文件（写文件）
        stream.write(format, file_name)


if __name__ == "__main__":
    mg = MelodyGenerator()
    seed = "67 _ 67 _ 67 _ _ 65 64 _ 64 _ 64 _ _ _"
    seed2 = "67 _ _ _ _ _ 65 _ 64 _ 62 _ 60 _ _ _"
    seed21 = "67 _ _ _ 67 _ _ _ 69 _ 67 _ 64 _ 62 _ 60 _ _ _ 60 _ 57 _ 55 _ _ _ _ _ "
    seed22 = "64 _ 62 _ 64 _ 62 _ 60 _ 57 _ _ _ 60 _ 62 _ 64 _ _ 62 60 _ 57 _ 57 _ _ _ _ _ _ _ "
    seed3 = "67 _ 65 _ 64 _ 62 _"
    seed4 = "69 _ 72 69 67 _ 69 _ 67"
    seed5 = "69 72 69 67 64"
    seed6 = "62 60 _ _ _ 57 _ 62 _ _ _ 57 _ 55 _"
    melody = mg.generate_melody(seed21, 2000, SEQUENCE_LENGTH, 0.75)
    # seed, num_steps(步骤数), max_sequence_length(序列最大长度), temperature(温度参数)
    print(melody)
    mg.save_to_txt(melody)
    mg.save_melody(melody)
