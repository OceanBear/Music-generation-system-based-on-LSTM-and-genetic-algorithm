程序使用方法：

首先，按照压缩包内文件“requirement.txt”安装所需Python库

此项目所需Python==3.6，个人强烈建议在Anaconda等软件所创建的虚拟机中运行本项目。
除此之外，此程序还使用到了musescore 3和pycharm这两个软件。

在环境安装完毕后，首先启动目录中的“preprocess.py”，这一程序将读取根目录中的Kernscores数据集中的歌曲，并且对其进行数据与处理。
在进行完预处理后，将得到一个文件夹和两个文件。其中会创建“dataset”文件夹
，其中的每一个文件都对应原本数据集中的一首歌曲文件。
程序会创建一个名为“file_dataset”的文件，此文件的内容就是将上述“dataset”文件夹中预处理得到的歌曲合并成一个完整的文件。
除此之外，还会创建一个名为“mapping.json”的映射文件，这一文件的内容是统计上述“file_dataset”文件中序列的所有符号类型。

接下来，运行目录中的“train.py”程序，这一程序将使用预处理后的数据进行模型训练，并且最终得到一个名为“model.h5”的模型文件。

之后就需要运行“melodygenerator.py”， 在其中，可以通过操作：“seed(旋律种子), num_steps(步骤数), max_sequence_length(序列最大长度), temperature(温度参数)”
这些超参数来控制旋律的生成。在程序执行完毕后，会生成两个文件，一个是“mel.mid”，即所生成旋律的midi文件。
另一个是“mel.txt”，即以文本形式保存的旋律序列。

接下来，需要打开“Mel-harm-GA”这一文件夹，并且将上述所生成的“mel.mid”文件放入其中的
“original-midi”文件夹中。接着，回到“Mel-harm-GA”文件夹，运行“midi-encode.py”这一程序，之前的“mel.mid”文件将被转化为“melody_corrected.json”文件。
这一文件内部是以“音高-时值对”格式保存的“mel.mid”的旋律序列。

在此之后运行同文件夹下的“geneticmelodyharmonizer.py”，并且选择所需要的父代选择方式（这里建议使用“适应度比例选择”方法）。
在程序运行完毕后，musescore应该会被启动，并且展示所生成的含有旋律与和弦伴奏的歌曲的五线谱。
除此之外，这一歌曲还将以midi文件和txt文件的形式分别被保存在“mel-chord.mid”与“chord-progress.txt”中。


额外注意事项，对于所有程序中所出现的如下代码：

from music21 import environment
env = environment.Environment()
environment.Environment()['musicxmlPath'] = r'E:\Program Files\MuseScore 3\bin\MuseScore3.exe'  #设置音乐展示
environment.Environment()['musescoreDirectPNGPath'] = r'E:\Program Files\MuseScore 3\bin\MuseScore3.exe'    #设置图形展示

请将其中的路径替换为你电脑上musescore的安装路径，否则musescore将无法正常启动。

对于“Mel-harm-GA/chords-dict”中的“chord_mappings.json”和“preferred_transitions.json”这两个文件。
其分别表示“和弦字典库”与“常用的和弦转换字典库”。用户可以根据自己的需求重写其中的字典库，以实现更加多样化的和弦效果。
