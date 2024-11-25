import mido
from mido import MidiFile
import json

# Mapping of MIDI pitch to note names
note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Function to convert MIDI note number to note name
def note_number_to_name(number):
    return note_names[number % 12] + str(number // 12 - 1)

# Function to convert MIDI ticks to the number of sixteenth notes
def ticks_to_sixteenth_notes(ticks, ticks_per_beat):
    return ticks / (ticks_per_beat / 4)

# Load the MIDI file and initialize the melody list
mid = MidiFile('original-midi/mel.mid', clip=True)
melody = []

# Iterate over all messages in the MIDI file
for i, track in enumerate(mid.tracks):
    time_since_last_note = 0
    for msg in track:
        # Increment time_since_last_note by the event's delta time
        time_since_last_note += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            # If there's a gap since the last note, insert a rest note before this note
            if time_since_last_note > 0:
                rest_duration = ticks_to_sixteenth_notes(time_since_last_note, mid.ticks_per_beat)
                if rest_duration > 0:
                    melody.append(['R', round(rest_duration)])  # 'R' denotes a rest
                time_since_last_note = 0
            # Proceed as normal for a note_on event
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            # Handle the note_off event as before
            duration = ticks_to_sixteenth_notes(time_since_last_note, mid.ticks_per_beat)
            if duration > 0:
                note_name = note_number_to_name(msg.note)
                melody.append([note_name, round(duration)])
            time_since_last_note = 0

# Preparing JSON output
melody_json = {"twinkle_twinkle_melody": melody}

# Serialize JSON to a string first
json_str = json.dumps(melody_json, ensure_ascii=False)

# Replace patterns to format it as desired
formatted_str = json_str.replace("], [", "],\n[")  # Add new line between items

# Save the formatted string to a JSON file
with open('melody_corrected.json', 'w', encoding='utf-8') as f:
    f.write(formatted_str)
