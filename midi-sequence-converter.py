import mido
from mido import MidiFile

def midi_to_sequence(midi_file_path):
    mid = MidiFile(midi_file_path)

    ticks_per_beat = mid.ticks_per_beat
    tempo = mido.bpm2tempo(120)  # 设置默认的 BPM 为 120

    sequence = []
    for i, track in enumerate(mid.tracks):
        for j, msg in enumerate(track):
            if msg.type == 'note_on':
                sequence.append(str(msg.note))
                if j < len(track) - 1:
                    next_msg = track[j + 1]
                    duration_in_beats = mido.tick2second(next_msg.time, ticks_per_beat, tempo) / (60 / 120)
                    duration_in_quarters = int(round(duration_in_beats / 0.25))
                    sequence[-1] += '_ ' * (duration_in_quarters - 1)
            #elif msg.type == 'note_off':
                #sequence.append('r')
    with open('path/german-1145-out', 'w') as f:
        f.write(' '.join(sequence))

midi_to_sequence('path/german-1145.mid')

#midi_to_sequence('path/mel.mid', 'path/output.txt')
