import os
import json
import numpy as np
import tensorflow.keras as keras
from music21.analysis.discrete import DiscreteAnalysisException
import music21 as m21
import music21
music21.environment.UserSettings()['debug'] = False

KERN_DATASET_PATH = "c-han"
SAVE_DIR = "dataset"
SINGLE_FILE_DATASET = "file_dataset"
MAPPING_PATH = "mapping.json"
SEQUENCE_LENGTH = 64

from music21 import environment
# 创建用户环境设置对象
env = environment.Environment()

# 设置MuseScore路径
environment.Environment()['musicxmlPath'] = r'E:\Program Files\MuseScore 3\bin\MuseScore3.exe'  #设置音乐展示
environment.Environment()['musescoreDirectPNGPath'] = r'E:\Program Files\MuseScore 3\bin\MuseScore3.exe'    #设置图形展示
# 将'/path/to/your/musescore'替换为MuseScore的实际路径
"""
    每次运行会清理掉之前生成的
    “dataset”、“file_dataset”、“mapping.json”
"""

# 持续时间以四分音符长度表示
ACCEPTABLE_DURATIONS = [
    0.25, # 十六分音符
    0.5, # 八分音符
    0.75,
    1.0, # 四分音符
    1.5,
    2, # 二分音符
    3,
    4 # 全音符
]


def load_songs_in_kern(dataset_path):
    """使用music21加载数据集中的所有kern片段。
    :参数 dataset_path (str): 数据集路径
    :返回 songs (m21流列表): 包含所有片段的列表
    """
    songs = []

    # 遍历数据集中的所有文件并使用music21进行加载
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:

            # 只考虑kern文件
            if file[-3:] == "krn":
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs


def has_acceptable_durations(song, acceptable_durations):
    """一个布尔例程，如果歌曲的所有持续时间都可接受，则返回True，否则返回False。
    :参数 song (m21 流)：
    :参数 acceptable_durations (列表)：四分音符长度的可接受持续时间列表
    :返回 (布尔值)：
    """
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True

def transpose(song):
    """将歌曲调整为C大调/A小调
    :参数 piece (m21 流): 需要调整的曲子
    :返回 transposed_song (m21 流):
    """

    # 从歌曲中获取键值
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]

    # estimate key using music21
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")

    # 获取移调的间隔。例如，Bmaj -> Cmaj
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    # 根据计算的间隔移调歌曲
    tranposed_song = song.transpose(interval)
    return tranposed_song

def encode_song(song, time_step=0.25):
    """将乐谱转换为时间序列型的音乐表示。
    编码列表中的每一个项目代表 'min_duration' 四分音符长度。
    每一步使用的符号有：整数代表MIDI音符，'r' 代表有休止符，
    以及 '_' 代表笔记/休止符被延续到一个新的时间步。以下是一个编码样例：
    ["r", "", "60", "", "", "", "72" "_"]
    :参数 song (m21 stream): 需要编码的作品
    :参数 time_step (float): 每个时间步骤的长度，以四分音符计
    :返回
    """

    encoded_song = []

    for event in song.flat.notesAndRests:

        # 处理音符
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi # 60
        # 处理休止符
        elif isinstance(event, m21.note.Rest):
            symbol = "r"

        # 将音符/休止符转换为时间序列符号注释
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):

            # 如果这是我们第一次看到一个音符/休止符，那就对它进行编码。
            # 否则，这意味着我们在新的时间步中承载相同的符号
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    # 将编码过的歌曲转换为字符串
    encoded_song = " ".join(map(str, encoded_song))

    return encoded_song


def preprocess(dataset_path):

    # 加载民谣歌曲
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")
    # 检查目录是否存在
    if not os.path.exists(SAVE_DIR):
    # 若不存在则创建
        os.makedirs(SAVE_DIR)
        print('Directory created.')
    else:
        print('Directory already exists.')

    for i, song in enumerate(songs):
        try:
            # 过滤掉那些持续时间不可接受的歌曲
            if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
                continue

            # 将歌曲调整到C大调/A小调
            song = transpose(song)

            # 用音乐时间序列表示法对歌曲进行编码
            encoded_song = encode_song(song)

            # 将歌曲保存到文本文件中
            save_path = os.path.join(SAVE_DIR, str(i))

            with open(save_path, "w") as fp:
                fp.write(encoded_song)

            if i % 10 == 0:
                print(f"Song {i} out of {len(songs)} processed")
        except Exception as e:
            print(f"Error processing song {i}: {e}")
            continue  # 跳过这首歌，进入下一首

def load(file_path):
    with open(file_path, "r") as fp:
        song = fp.read()
    return song


def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    """生成一个整理所有编码歌曲并添加新作品分隔符的文件。
    :参数 dataset_path (str): 存有编码歌曲的文件夹路径
    :参数 file_dataset_path (str): 用于在单个文件中保存歌曲的文件路径
    :参数 sequence_length (int): 考虑用于训练的时间步数
    :返回 songs (str): 包含数据集中所有歌曲及分隔符的字符串
    """

    new_song_delimiter = "/ " * sequence_length
    songs = ""

    # 加载编码的歌曲并添加分隔符
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimiter

    # 移除字符串最后一个字符的空格
    songs = songs[:-1]

    # 保存包含所有数据集的字符串
    with open(file_dataset_path, "w") as fp:
        fp.write(songs)

    return songs


def create_mapping(songs, mapping_path):
    """创建一个json文件，将歌曲数据集中的符号映射到整数上
    :参数 songs (str): 包含所有歌曲的字符串
    :参数 mapping_path (str): 保存映射的路径
    :返回:
    """
    mappings = {}

    # 识别词汇表
    songs = songs.split()
    vocabulary = list(set(songs))

    # 创建映射
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i

    # 将词汇保存到json文件中
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4)


def convert_songs_to_int(songs):
    int_songs = []

    # 加载映射
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)

    # 将歌曲字符串转换为列表
    songs = songs.split()

    # 将歌曲映射为整数
    for symbol in songs:
        int_songs.append(mappings[symbol])

    return int_songs


def generate_training_sequences(sequence_length):
    """为训练创建输入和输出数据样本。每个样本都是一个序列。
    :参数 sequence_length (int): 每个序列的长度。以16分音符为量化，64个音符等于4个小节
    :返回 inputs (ndarray): 训练输入
    :返回 targets (ndarray): 训练目标
    """

    # 加载歌曲并将它们映射为整数
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)

    inputs = []
    targets = []

    # 生成训练序列
    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])

    # 对序列进行一键编码
    vocabulary_size = len(set(int_songs))
    # 输入大小：（序列数，序列长度，词汇量大小）
    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    targets = np.array(targets)

    print(f"There are {len(inputs)} sequences.")

    return inputs, targets


def main():
    preprocess(KERN_DATASET_PATH)
    songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    create_mapping(songs, MAPPING_PATH)
    #inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)


if __name__ == "__main__":
    main()
