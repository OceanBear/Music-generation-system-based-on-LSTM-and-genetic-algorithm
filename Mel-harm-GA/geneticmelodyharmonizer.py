import random
import json
from dataclasses import dataclass

import music21
from music21 import environment

# 创建用户环境设置对象
env = environment.Environment()
# 设置MuseScore路径
environment.Environment()['musicxmlPath'] = r'E:\Program Files\MuseScore 3\bin\MuseScore3.exe'  #设置音乐展示
environment.Environment()['musescoreDirectPNGPath'] = r'E:\Program Files\MuseScore 3\bin\MuseScore3.exe'    #设置图形展示
# 将'/path/to/your/musescore'替换为MuseScore的实际路径

@dataclass(frozen=True)
class MelodyData:
    """
    这个类封装了旋律的详情，包括其音符、总时长和小节数。
    音符被表示为一个元组的列表，每个元组包含一个音高和它的时长。
    总时长和小节数根据提供的音符计算得出。

    属性:
    notes (元组的列表): 列表中每个元组代表旋律的一个音符，格式为(音高, 时长)。
    duration (int): 旋律的总时长，根据音符计算得出。
    number_of_bars (int): 旋律的总小节数，根据时长计算，假设一个4/4拍子符号。

    方法:
    __post_init__: 数据类初始化后调用的一个方法，基于提供的音符来计算并设置时长和小节数。

    """

    notes: list
    duration: int = None  # 计算属性
    number_of_bars: int = None  # 计算属性

    def __post_init__(self):
        object.__setattr__(
            self, "duration", sum(duration for _, duration in self.notes)
        )
        object.__setattr__(self, "number_of_bars", self.duration // 16) #修改对应16分音符


class GeneticMelodyHarmonizer:
    """
    使用遗传算法为给定的旋律生成和弦伴奏。
    它通过适应度函数演化一群和弦序列，以找到最适合旋律的和弦。

    属性:
        melody_data (MusicData): 包含旋律信息的数据。
        chords (list): 用于生成序列的可用和弦。
        population_size (int): 和弦序列种群的大小。
        mutation_rate (float): 遗传算法中发生突变的概率。
        fitness_evaluator (FitnessEvaluator): 用于评估适应度的实例。
    """

    def __init__(
        self,
        melody_data,
        chords,
        population_size,
        mutation_rate,
        fitness_evaluator,
        selection_method,
    ):
        """
    使用旋律数据、和弦、种群大小、突变率和适应度评估器初始化生成器。

    参数:
    melody_data (MusicData): 旋律信息。
    chords (list): 可用和弦。
    population_size (int): 算法中的种群大小。
    mutation_rate (float): 每个和弦的突变概率。
    fitness_evaluator (FitnessEvaluator): 和弦适应度的评估器。
        """
        self.melody_data = melody_data
        self.chords = chords
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.fitness_evaluator = fitness_evaluator
        self._population = []
        self.selection_method = selection_method

    def generate(self, generations=1000):
        """
        使用遗传算法生成和旋律和谐的和弦序列。

        参数:
            generations (int): 进化的代数。

        返回:
            best_chord_sequence (list): 在最后一代中找到的适应度最高的和弦和谐化序列。
        """
        self._population = self._initialise_population()
        for _ in range(generations):
            parents = self._select_parents()
            new_population = self._create_new_population(parents)
            self._population = new_population
        best_chord_sequence = (
            self.fitness_evaluator.get_chord_sequence_with_highest_fitness(
                self._population
            )
        )
        return best_chord_sequence

    def _initialise_population(self):
        """
        使用随机和弦序列初始化种群。

        返回:
            list: 随机生成和弦序列的列表。
        """
        return [
            self._generate_random_chord_sequence()
            for _ in range(self.population_size)
        ]

    def _generate_random_chord_sequence(self):
        """
        生成与旋律中的小节数量相同的随机和弦序列。

        返回:
            list: 随机生成的和弦列表。
        """

        return [
            random.choice(self.chords)
            for _ in range(self.melody_data.number_of_bars)
        ]

    def _tournament_selection(self):
        """
        通过锦标赛选择法从当前种群中选择一位父代。

        锦标赛选择通过随机选择一定数量的个体（称为竞争者），然后根据适应度从这些竞争者中选择一位胜出者作为父代。这个方法旨在模仿自然选择中的“竞争”过程。

        返回:
            选定的父代和弦序列。
        """
        tournament_size = 2  # 或者其他您选定的锦标赛大小
        contenders = random.sample(self._population, tournament_size)
        fit_scores = [self.fitness_evaluator.evaluate(c) for c in contenders]
        winner_index = fit_scores.index(max(fit_scores))
        return contenders[winner_index]


    def _select_parents(self):
        """
        选择用于繁殖的父代序列
        (1)适应度比例选择、(2)锦标赛选择、(3)随机选择
        返回:
            list: 选定的父代和弦序列。
        """
        if self.selection_method == "fitness_proportionate":
            # 进行适应度比例选择
            fitness_values = [self.fitness_evaluator.evaluate(seq) for seq in self._population]
            parents = random.choices(
                self._population, weights=fitness_values, k=self.population_size
            )
        elif self.selection_method == "tournament":
            # 进行锦标赛选择
            parents = [self._tournament_selection() for _ in range(self.population_size)]
        elif self.selection_method == "random":
            # 进行随机选择
            parents = random.sample(self._population, self.population_size)
        else:
            raise ValueError("未知的父代选择方法")

        return parents

    def _create_new_population(self, parents):
        """
        从提供的父代中生成新的和弦序列种群。

        这个方法使用交叉和变异操作创建新一代的和弦序列。对于每对父代和弦序列，
        它会生成两个子代。每个子代都是父代配对交叉操作的结果，然后可能经过
        变异。通过收集所有这些子代来形成新种群。

        该方法确保新的种群大小等于生成器预定义的种群大小。它按对处理父代，对于
        每一对，都会生成两个子代。

        参数:
            parents (list): 用于生成新种群的父代和弦序列列表。

        返回:
            list: 一组新的和弦序列种群，由父代生成。

        注意：
            这个方法假设种群大小为偶数，且父代数量等于预定义的种群大小。
        """
        new_population = []
        for i in range(0, self.population_size, 2):
            child1, child2 = self._crossover(
                parents[i], parents[i + 1]
            ), self._crossover(parents[i + 1], parents[i])
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            new_population.extend([child1, child2])
        return new_population

    def _crossover(self, parent1, parent2):
        """
        随机使用不同的交叉方式将两个父代序列合并为新的子代序列。

        参数:
            parent1 (list): 第一个父代和弦序列。
            parent2 (list): 第二个父代和弦序列。

        返回:
            list: 结果的子代和弦序列。
        """
        # 随机选择四种交叉方式
        crossover_method = random.choice(['single_point', 'multi_point', 'uniform', 'ox1'])
        child = []

        if crossover_method == 'single_point':
            # 单点交叉
            cut_index = random.randint(1, len(parent1) - 1)
            child = parent1[:cut_index] + parent2[cut_index:]
        elif crossover_method == 'multi_point':
            # 多点交叉
            cut_indices = sorted(random.sample(range(1, len(parent1)), 2))
            child = parent1[:cut_indices[0]] + parent2[cut_indices[0]:cut_indices[1]] + parent1[cut_indices[1]:]
        elif crossover_method == 'uniform':
            # 均匀交叉
            child = [random.choice(pair) for pair in zip(parent1, parent2)]
        elif crossover_method == 'ox1':
            # OX1交叉（简化版本：取父代1的一段，然后在父代2中插入不在此段中的元素）
            start, end = sorted(random.sample(range(len(parent1)), 2))
            middle_part = parent1[start:end]
            rest_part = [item for item in parent2 if item not in middle_part]
            child = rest_part[:start] + middle_part + rest_part[start:]

        return child

    def _mutate(self, chord_sequence):
        """
        根据突变率在序列中突变一个和弦。

        参数:
            chord_sequence (list): 需要发生突变的和弦序列。

        返回:
            list: 发生突变后的和弦序列。
        """

        mutation_type = random.choice(['random_reset', 'swap', 'shuffle', 'inversion'])
        if mutation_type == 'random_reset':
            # 随机重置
            mutation_index = random.randint(0, len(chord_sequence) - 1)
            chord_sequence[mutation_index] = random.choice(self.chords)
        elif mutation_type == 'swap':
            # 交换变异
            idx1, idx2 = random.sample(range(len(chord_sequence)), 2)
            chord_sequence[idx1], chord_sequence[idx2] = chord_sequence[idx2], chord_sequence[idx1]
        elif mutation_type == 'shuffle':
            # 混洗变异
            indices = random.sample(range(len(chord_sequence)), random.randint(2, min(5, len(chord_sequence))))
            subsequence = [chord_sequence[i] for i in indices]
            random.shuffle(subsequence)
            for i, idx in enumerate(indices):
                chord_sequence[idx] = subsequence[i]
        elif mutation_type == 'inversion':
            # 反转变异
            start, end = sorted(random.sample(range(len(chord_sequence)), 2))
            if end - start > 1:
                chord_sequence[start:end] = reversed(chord_sequence[start:end])

        return chord_sequence

'''
        if random.random() < self.mutation_rate:
            mutation_index = random.randint(0, len(chord_sequence) - 1)
            chord_sequence[mutation_index] = random.choice(self.chords)
        return chord_sequence
'''

class FitnessEvaluator:
    """
        基于各种音乐标准评估和弦序列的适应度。

        属性:
            melody (list): 表示音符的元组列表，形式为（音高，持续时间）。
            chords (dict): 与其对应音符的和弦字典。
            weights (dict): 不同适应度评估函数的权重。
            preferred_transitions (dict): 首选和弦转换。
    """

    def __init__(
        self, melody_data, chord_mappings, weights, preferred_transitions
    ):
        """
        使用旋律、和弦、权重和首选过渡初始化适应度评估器。

        参数:
            melody_data (MelodyData): 旋律信息。
            chord_mappings (dict): 可用和弦映射到它们的音符。
            weights (dict): 每个适应度评估函数的权重。
            preferred_transitions (dict): 首选的和弦转换。
        """
        self.melody_data = melody_data
        self.chord_mappings = chord_mappings
        self.weights = weights
        self.preferred_transitions = preferred_transitions


    def get_chord_sequence_with_highest_fitness(self, chord_sequences):
        """
        返回适应度得分最高的和弦序列。

        参数:
            chord_sequences (list): 需要评估的和弦序列列表。

        返回:
            list: 适应度得分最高的和弦序列。
        """
        return max(chord_sequences, key=self.evaluate)

    def evaluate(self, chord_sequence):
        """
        评估给定和弦序列的适应度。

        参数:
            chord_sequence (list): 需要评估的和弦序列。

        返回:
            float: 和弦序列的整体适应度得分。
        """
        return sum(
            self.weights[func] * getattr(self, f"_{func}")(chord_sequence)
            for func in self.weights
        )

    def _chord_melody_congruence(self, chord_sequence):
        """
        计算和弦序列与旋律之间的一致性。
        此函数评估序列中的每个和弦与旋律的相应部分的对齐程度如何。
        通过检查旋律中的音符是否出现在同时演奏的和弦中来测量对齐度
        如果旋律音符与和弦匹配得好，则给予奖励。

        参数:
            chord_sequence (list): 一个将要与旋律进行评估的和弦列表。

        返回:
            float: 一个分数，代表和弦序列与旋律之间一致性的程度，该分数已经通过旋律的持续时间进行了标准化。
        """
        score, melody_index = 0, 0
        for chord in chord_sequence:
            bar_duration = 0
            while bar_duration < 16 and melody_index < len(
                self.melody_data.notes
            ):
                pitch, duration = self.melody_data.notes[melody_index]
                if pitch[0] in self.chord_mappings[chord]:
                    score += duration
                bar_duration += duration
                melody_index += 1
        return score / self.melody_data.duration

    def _chord_variety(self, chord_sequence):
        """
        评估序列中使用的和弦的多样性。
        此函数根据序列中存在的独特和弦数量与可用和弦总数进行比较，计算出一个分数。
        和弦序列中的高度多样性将导致更高的分数，有助于提升音乐的复杂性和趣味性。

        参数:
            chord_sequence (list): 需要评估的和弦序列。

        返回:
            float: 一个表示和弦序列中和弦多样性与所有可用和弦总数的相对比例的标准化得分。
        """
        unique_chords = len(set(chord_sequence))
        total_chords = len(self.chord_mappings)
        return unique_chords / total_chords

    def _harmonic_flow(self, chord_sequence):
        """
        通过检查连续和弦之间的过渡来评估和弦序列的和声流动。
        此函数根据和弦转换与预定义的首选转换相吻合的频率对序列进行评分。
        平滑且在音乐上愉悦的过渡将得到更高的分数。

        参数:
            chord_sequence (list): 需要评估的和弦序列。

        返回:
            float: 一个基于序列中首选和弦过渡频率的标准化得分。
        """
        score = 0
        for i in range(len(chord_sequence) - 1):
            next_chord = chord_sequence[i + 1]
            if next_chord in self.preferred_transitions[chord_sequence[i]]:
                score += 1
        return score / (len(chord_sequence) - 1)

    def _functional_harmony(self, chord_sequence):
        """
        根据功能和声原则评估和弦序列。
        此功能检查序列开头和结尾的主音以及次属和主导和弦的存在等关键和声功能。
        符合这些和规则定会得到更高的分数。

        参数：
            chord_sequence (list)：要评估的和弦序列。

        返回：
            float：一个分数，表示序列遵守传统功能和声的程度，已经通过执行的检查次数进行了标准化。
        """
        score = 0
        # 维持对主和弦的检查，但也适应一些其他常见的开始和弦
        if chord_sequence[0] in ["C", "Am", "Cmaj7", "Cadd9"]:
            score += 1
        # 通常的结束和弦，增加一些变化和丰富性
        if chord_sequence[-1] in ["C", "G", "Cmaj7", "G7"]:
            score += 1
        # 检查是否有良好的和声流动，如从F到G，或者从Fmaj7到G7，添加一些和弦变体以丰富过渡的多样性
        if any(chord in ["F", "Fmaj7", "Dm", "Dm7"] for chord in chord_sequence) and any(chord in ["G", "G7", "C", "Cmaj7"] for chord in chord_sequence):
            score += 1

        return score / 3


def create_score(melody, chord_sequence, chord_mappings):
    """
    根据给定的旋律和和弦序列创建一个music21乐谱。

    参数:
        melody (list): 一个元组列表，表示以 (音符名, 持续时间) 的格式表示的音符。
        chord_sequence (list): 一个和弦名列表。

    返回:
        music21.stream.Score: 包含旋律和和弦序列的音乐乐谱。
    """
    # 创建一个乐谱对象
    score = music21.stream.Score()

    # 创建旋律部分并向其中添加音符
    melody_part = music21.stream.Part()
    for note_name, duration in melody:
        # 将时值从十六分音符转换为四分音符
        quarter_note_duration = duration / 4
    # 判断是否为休止符
        if note_name == "R":
            # 创建一个休止符对象
            rest = music21.note.Rest(quarterLength=quarter_note_duration)
            melody_part.append(rest)
        else:
            # 若不是休止符，继续处理为普通音符
            melody_note = music21.note.Note(note_name, quarterLength=quarter_note_duration)
            melody_part.append(melody_note)

    # 创建和弦部分并向其中添加和弦
    chord_part = music21.stream.Part()
    current_duration = 0  # 追踪放置和弦的持续时间

    for chord_name in chord_sequence:
        # 将和弦名称翻译成音符列表
        chord_notes_list = chord_mappings.get(chord_name, [])
        # 创建一个music21和弦
        chord_notes = music21.chord.Chord(chord_notes_list, quarterLength=4)  # 假设是4/4拍子符号
        chord_notes.offset = current_duration
        chord_part.append(chord_notes)
        current_duration += 4  # 假设每个和弦占满全拍（4个四分音符）

    # 将上述部分添加到和弦中
    score.append(melody_part)
    score.append(chord_part)

    return score


def main():

    # 读取 chord_mappings 和 preferred_transitions
    with open('chords-dict/chord_mappings.json', 'r') as cm_file:
        chord_mappings = json.load(cm_file)

    with open('chords-dict/preferred_transitions.json', 'r') as pt_file:
        preferred_transitions = json.load(pt_file)

    # 读取 melody_corrected.json 中的旋律数据
    with open('melody_corrected.json', 'r') as melody_file:
        melody_data = json.load(melody_file)
    twinkle_twinkle_melody = melody_data['twinkle_twinkle_melody']

    weights = {
        "chord_melody_congruence": 0.4,
        "chord_variety": 0.1,
        "harmonic_flow": 0.3,
        "functional_harmony": 0.2
    }

    # 原来字典的位置

    # 提示用户选择父代选择方法
    print("父代选择方式，请输入对应的数字：")
    print("(1)适应度比例选择、(2)锦标赛选择、(3)随机选择")
    selection_method_input = input("请输入您选择的方法对应的数字：")
    try:
        selection_method = int(selection_method_input)
    except ValueError:
        print("输入无效，请输入有效数字（1，2，3）。")
        return  # 如果输入无效，直接返回并终止程序

    # 根据用户的选择设置选择方法
    if selection_method == 1:
        selection_method_name = "fitness_proportionate"
    elif selection_method == 2:
        selection_method_name = "tournament"
    elif selection_method == 3:
        selection_method_name = "random"
    else:
        print("输入无效，请输入有效数字（1，2，3，4）。")
        return  # 如果输入无效，直接返回并终止程序

    # 实例化生成和弦的对象
    melody_data = MelodyData(twinkle_twinkle_melody)
    fitness_evaluator = FitnessEvaluator(
        melody_data=melody_data,
        weights=weights,
        chord_mappings=chord_mappings,
        preferred_transitions=preferred_transitions,
    )
    harmonizer = GeneticMelodyHarmonizer(
        melody_data=melody_data,
        chords=list(chord_mappings.keys()),
        population_size=100,
        mutation_rate=0.05,
        fitness_evaluator=fitness_evaluator,
        selection_method=selection_method_name  # 传入用户选择的方法
    )

    # 用遗传算法生成和弦
    generated_chords = harmonizer.generate(generations=1000)

    # 渲染到music21的乐谱并显示
    music21_score = create_score(
        twinkle_twinkle_melody, generated_chords, chord_mappings
    )
    music21_score.write('midi', fp='mel-chord.mid')

    # 保存所生成的和弦进行
    with open('chord-progress.txt', 'w') as f:
        f.write('-'.join(generated_chords))

    # 在musescore里进行展示
    music21_score.show()


if __name__ == "__main__":
    main()
