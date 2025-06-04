import numpy as np
import miditoolkit
import copy

DEFAULT_VELOCITY_BINS = np.array([ 0, 32, 48, 64, 80, 96, 128])
# read notes and tempo changes from midi (assume there is only one track)
def read_items(file_path, sanity_check=False, task='pretrain'):
    try:
        midi_obj = miditoolkit.midi.parser.MidiFile(file_path)
    except:
        return None, None
        
    if sanity_check:
        if (len(midi_obj.time_signature_changes) != 1 or midi_obj.time_signature_changes[0].numerator != 4
            or midi_obj.time_signature_changes[0].denominator != 4):
            return None, None
    # note
    note_items = []
    num_of_instr = len(midi_obj.instruments) 
    
    for i in range(num_of_instr):
        notes = midi_obj.instruments[i].notes
        notes.sort(key=lambda x: (x.start, x.pitch))

        for note in notes:
            # Onset, offset, pitch
            velocity_index = np.searchsorted(DEFAULT_VELOCITY_BINS, note.velocity, side='right') - 1
            if (note.end - note.start) / midi_obj.ticks_per_beat >= (1.0 / 16.0) - 0.0001:
                note_items.append([note.start / midi_obj.ticks_per_beat,
                                    note.end / midi_obj.ticks_per_beat,
                                    note.pitch, i, velocity_index])
                
    note_items.sort(key=lambda x: (x[0], x[2]))

    track_id = [note[3] for note in note_items]
    velocity = [note[4] for note in note_items]

    sorted_note_items = []
    for note in note_items:
        sorted_note_items.append([note[0], note[1], note[2]])

    if task == 'melody':
        return sorted_note_items, track_id
    elif task == 'velocity':
        return sorted_note_items, velocity
    else:
        return sorted_note_items

import statistics
def convert_pm2s_data(note_sequence):
    estimated_sec_per_beat = statistics.median(note_sequence[:,2])
    # 40~200 (1.5sec ~ 0.3sec)
    while estimated_sec_per_beat < 0.3:
        estimated_sec_per_beat = estimated_sec_per_beat * 2.0

    while estimated_sec_per_beat > 1.5:
        estimated_sec_per_beat = estimated_sec_per_beat / 2.0

    # Convert second to pseudo-bar position
    note_sequence[:,1] = note_sequence[:,1] / estimated_sec_per_beat
    note_sequence[:,2] = note_sequence[:,2] / estimated_sec_per_beat

    # note
    note_items = []
    # pitch, start, duration, velocity
    # print (note_sequence)
    note_sequence = list(note_sequence)
    note_sequence.sort(key=lambda x: (x[1], x[0]))

    for note in note_sequence:
        note_items.append([note[1], note[1] + note[2], int(note[0]), int(round(note[4])) + 1])

    note_items.sort(key=lambda x: (x[0], x[2]))
    groundtruth = [note[3] for note in note_items]

    sorted_note_items = []
    for note in note_items:
        sorted_note_items.append([note[0], note[1], note[2]])
    # print (sorted_note_items)
    # print (groundtruth)
    return sorted_note_items, groundtruth