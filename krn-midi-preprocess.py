import os
import music21 as m21

from music21 import environment
# 创建用户环境设置对象
env = environment.Environment()

# 设置MuseScore路径
environment.Environment()['musicxmlPath'] = r'E:\Program Files\MuseScore 3\bin\MuseScore3.exe'  #设置音乐展示
environment.Environment()['musescoreDirectPNGPath'] = r'E:\Program Files\MuseScore 3\bin\MuseScore3.exe'    #设置图形展示
# 将'/path/to/your/musescore'替换为MuseScore的实际路径

us = environment.UserSettings()
KERN_DATASET_PATH = "deutschl/test"

# durations are expressed in quarter length，这些是acceptable-duration
ACCEPTABLE_DURATIONS = [
    0.25, # 16th note 16分音符
    0.5, # 8th note 8分音符
    0.75,
    1.0, # quarter note 4分音符
    1.5,
    2, # half note 半音符
    3,
    4 # whole note 全音符
]

    #load the folk songs
    #filter out songs that have non-acceptable durations
    #transpose songs to Cmaj/Amin
    #encode songs with music time series representation
    #save songs to text file

    #load the folk songs
def load_songs_in_kern(dataset_path):
    """Loads all kern pieces in dataset using music21.

    :param dataset_path (str): Path to dataset
    :return songs (list of m21 streams): List containing all pieces
    """
    songs = []

    # go through all the files in dataset and load them with music21
    # path路径，subdirs 下属的路径
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:

            # consider only kern files
            if file[-3:] == "krn":  #判断最后三个字母是否是krn，判断是否是krn格式
                song = m21.converter.parse(os.path.join(path, file)) #导入歌曲path是歌曲路径，而file是歌曲的文件
                songs.append(song) #music21 是一个允许操作的特征音乐数据，可以转化音乐文件的格式。kern,MIDI,MusicXML->m21->kern,MIDI....
                #music21可以从一种面向对象的方式描述音乐
    return songs


#判断这首歌里的音符和休止符是否符合要求
def has_acceptable_durations(song, acceptable_durations):
    """Boolean routine that returns True if piece has all acceptable duration, False otherwise.

    :param song (m21 stream):
    :param acceptable_durations (list): List of acceptable duration in quarter length
    :return (bool):
    """
    for note in song.flat.notesAndRests:    #这里保留notes和rests，而不研究key 和 time signature
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True


#把音乐转换成C maj / A min
def transpose(song):
    """Transposes song to C maj/A min

    :param piece (m21 stream): Piece to transpose
    :return transposed_song (m21 stream):
    """

    # get key from the song, 寻回,在key都以符号表示的情况下
    parts = song.getElementsByClass(m21.stream.Part)    #
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure) #获取第一part的measure（）
    key = measures_part0[0][4]  #

    # estimate key using music21, 评估,在key没有以符号表示的情况下
    if not isinstance(key, m21.key.Key):    #如果读取的key不符合m21的例子
        key = song.analyze("key")   #让music21去预测这首歌的key，并且将其转递给变量key

    # get interval for transposition. E.g., Bmaj -> Cmaj，得到换位的间隔（如Bmaj到Cmaj的间隔）
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    # transpose song by calculated interval，按计算的音程调换歌曲
    tranposed_song = song.transpose(interval) #music21的特有语句
    return tranposed_song


#预处理
def preprocess(dataset_path):

    # load folk songs
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.") #输出加载的歌曲的个数

    for song in songs:

        # filter out songs that have non-acceptable durations 筛除duration non-acceptable的歌曲
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            continue    #跳过

        # transpose songs to Cmaj/Amin

        song = transpose(song)

        # encode songs with music time series representation

        # save songs to text file


if __name__ == "__main__":

    # load songs
    songs = load_songs_in_kern(KERN_DATASET_PATH) #导入歌曲的路径
    print(f"Loaded {len(songs)} songs.")    #输出加载歌曲得到个数
    song = songs[0] #拿出第一首歌

    print(f"Has acceptable duration? 持续时间是否可接受？ {has_acceptable_durations(song, ACCEPTABLE_DURATIONS)}")

    # transpose song
    transposed_song = transpose(song)

    #song.show()     #展示第一首歌
    transposed_song.show()  #展示转换后的歌



