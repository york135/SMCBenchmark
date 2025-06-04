import numpy as np
import pickle
from tqdm import tqdm
import utils_noquantize as utils
import miditoolkit
import csv, math
import xlrd
from utils_noquantize import *

from music21 import converter
import music21


Composer = {
    "Bethel": 0,
    "Clayderman": 1,
    "Einaudi": 2,
    "Hancock": 3,
    "Hillsong": 4,
    "Hisaishi": 5,
    "Ryuichi": 6,
    "Yiruma": 7,
    "Padding": 8
}

Genre = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
}

Emotion = {
    "Q1": 0,
    "Q2": 1,
    "Q3": 2,
    "Q4": 3,
}

Texture = {'pad': 0, 'mel': 1, 'rhythm': 2, 'harm': 3,
                'mel+rhythm': 4, 'mel+harm': 5, 
                'rhythm+harm': 6, 'mel+rhythm+harm': 7, 'rhythn': 2, 'melody': 1}

Chord_root = {'pad': 0, 
                'C': 1, 'B#': 13, 'D--': 25,
                'C#': 2, 'D-': 14, 'B##': 26,
                'D': 3, 'E--': 15, 'C##': 27,
                'D#': 4, 'E-': 16, 'F--': 28,
                'E': 5, 'F-': 17,'D##': 29,
                'F': 6, 'E#': 18, 'G--': 30,
                'F#': 7, 'G-': 19, 'E##': 31,
                'G': 8, 'A--': 20, 'F##': 32,
                'G#': 9, 'A-': 21,
                'A': 10, 'B--': 22, 'G##': 33,
                'A#': 11, 'B-': 23, 'C--': 34,
                'B': 12, 'C-': 24, 'A##': 35,}

Localkey = {'pad': 0, 'b': 1, 'A-': 2, 'f': 3, 'c': 4, 'D-': 5, 'd': 6, 'F': 7, 
            'A': 8, 'E': 9, 'C': 10, 'B-': 11, 'a': 12, 'c#': 13, 'D': 14, 'G-': 15, 
            'G': 16, 'e-': 17, 'E-': 18, 'g': 19, 'b-': 20, 'F#': 21, 'd#': 22, 'B': 23, 
            'f#': 24, 'C-': 25, 'e': 26, 'a-': 27, 'g#': 28, 'd-': 29, 'D#': 30, 'C#': 31, 
            'a#': 32, 'b#': 33, 'e#': 34, 'g-': 35, 'G#': 36, 'F-': 37}


class CP(object):
    def __init__(self):
        return

    def extract_events(self, input_path, task='pretrain'):
        if task == 'pretrain':
            note_items = utils.read_items(input_path, sanity_check=True)

            if note_items is None or len(note_items) == 0:   # if the midi contains nothing
                return None
            return note_items
        else:
            note_items, groundtruth = utils.read_items(input_path
                , sanity_check=False
                , task=task)

            if note_items is None or len(note_items) == 0:   # if the midi contains nothing
                return None, None
            return note_items, groundtruth

    def prepare_pretrain_data(self, midi_paths):
        all_note_list = []
        for path in tqdm(midi_paths):
            # extract events
            note_items = self.extract_events(path)
            if not note_items:  # if midi contains nothing
                print(f'skip {path} because it is empty')
                continue
            all_note_list.append(note_items)
        return all_note_list

    def prepare_finetune_seq_data(self, midi_paths, task):
        all_note_list, all_groundtruth = [], []
        for path in tqdm(midi_paths):
            # extract events
            note_items, _ = self.extract_events(path, task)
            if not note_items:  # if midi contains nothing
                print(f'skip {path} because it is empty')
                continue
            if task == "composer":
                name = path.split('/')[-2]
                y = Composer[name]
            elif task == "emotion":
                name = path.split('/')[-1].split('_')[0]
                y = Emotion[name]
            elif task == "genre":
                name = path.split('/')[-2]
                y = Genre[name]

            all_note_list.append(note_items)
            all_groundtruth.append(y)
        return all_note_list, all_groundtruth


    def prepare_finetune_pop909_data(self, midi_paths, task):
        all_note_list, all_groundtruth = [], []
        for path in tqdm(midi_paths):
            note_items, groundtruth = self.extract_events(path, task)
            groundtruth = np.array(groundtruth) + 1
            # print (note_items, groundtruth)
            if not note_items:  # if midi contains nothing
                print(f'skip {path} because it is empty')
                continue

            all_note_list.append(note_items)
            all_groundtruth.append(groundtruth)
        return all_note_list, all_groundtruth

    def extract_events_from_csv(self, csv_path, dbeat_path, task):
        # Get notes. For MNID, automatically "resize" the time signature to 4/4
        notes = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if row[0] != 'onset':
                    onset = float(row[0])
                    duration = float(row[3])
                    # 1: non-motif note; 2: motif note
                    if row[-1] == '':
                        note = [onset, int(row[1]), duration, 1]
                    else:
                        note = [onset, int(row[1]), duration, 2]
                    if duration > 0:
                        notes.append(note)

        if dbeat_path[-4:] == 'xlsx':
            workbook = xlrd.open_workbook(dbeat_path)
            sheet = workbook.sheet_by_index(0)
            dbeats = [0.0]
            for rowx in range(sheet.nrows):
                cols = sheet.row_values(rowx)
                dbeats.append(cols[0])
        else:
            dbeats = [0.0]
            with open(dbeat_path, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                for row in reader:
                    dbeats.append(float(row[0]))

        dbeats[0] = dbeats[1] - (dbeats[2] - dbeats[1])
        dbeats.append(dbeats[-1] + (dbeats[-1] - dbeats[-2]))

        resized_notes = []
        groundtruth = []
        cur_downbeat_timing = 0
        for db1, db2 in zip(dbeats[:-1], dbeats[1:]):
            for i in range(len(notes)):
                if (notes[i][0] >= db1) and (notes[i][0] < db2):                    
                    start_interp_in_bar = max((notes[i][0] - db1) / (db2 - db1), 0)
                    end_interp_in_bar = (notes[i][0] + notes[i][2] - db1) / (db2 - db1)
                    resized_notes.append([cur_downbeat_timing + start_interp_in_bar * 4.0,
                        cur_downbeat_timing + end_interp_in_bar * 4.0,
                        notes[i][1]])
                    groundtruth.append(notes[i][3])
            cur_downbeat_timing = cur_downbeat_timing + 4.0
        return resized_notes, groundtruth


    def prepare_finetune_bpsmotif_data(self, csv_paths, dbeat_paths, task):
        all_note_list, all_groundtruth = [], []
        for j in tqdm(range(len(csv_paths))):
            csv_path = csv_paths[j]
            dbeat_path = dbeat_paths[j]
            # extract events
            note_items, groundtruth = self.extract_events_from_csv(csv_path, dbeat_path, task)
            # print (note_items)
            # print (groundtruth)
            if not note_items:  # if midi contains nothing
                print(f'skip {path} because it is empty')
                continue

            all_note_list.append(note_items)
            all_groundtruth.append(groundtruth)
        return all_note_list, all_groundtruth

    def extract_orch_events(self, midi_path, annotation_path, dbeat_path, task):
        all_note_list, all_groundtruth = [], []
        # Get texture annotation
        all_texture_annotation = np.load(annotation_path)
        # print (all_texture_annotation[0:5,0])
        dbeats = []
        with open(dbeat_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if row[0] != 'onset':
                    dbeats.append(float(row[0]))
        dbeats.append(dbeats[-1] + (dbeats[-1] - dbeats[-2]))
        # print (len(dbeats))

        resized_notes = []
        groundtruth = []

        midi_obj = miditoolkit.midi.parser.MidiFile(midi_path)
        for j in range(all_texture_annotation.shape[1]):
            cur_texture_annotation = []
            for i in range(len(all_texture_annotation)):
                # 7-classes
                cur_texture_annotation.append([dbeats[i], dbeats[i+1], (all_texture_annotation[i][j][0]*4 
                    + all_texture_annotation[i][j][1]*2 + all_texture_annotation[i][j][2])])
            # print (cur_texture_annotation)

            midi_notes = midi_obj.instruments[j].notes
            midi_notes.sort(key=lambda x: (x.start, x.pitch))

            notes = []
            cur_annotation_id = 0
            for note in midi_notes:
                onset = note.start / midi_obj.ticks_per_beat
                duration = (note.end - note.start) / midi_obj.ticks_per_beat
                pitch = note.pitch

                while (cur_annotation_id < len(cur_texture_annotation) - 1 
                    and onset >= cur_texture_annotation[cur_annotation_id+1][0]):
                    cur_annotation_id += 1

                if duration > 0:
                    notes.append([onset, pitch, duration
                        , cur_texture_annotation[cur_annotation_id][2]])

            notes = sorted(notes, key=lambda x: (x[0], x[2]))

            cur_resized_notes = []
            cur_groundtruth = []
            cur_downbeat_timing = 0
            for db1, db2 in zip(dbeats[:-1], dbeats[1:]):
                for i in range(len(notes)):
                    if (notes[i][0] >= db1) and (notes[i][0] < db2):                    
                        start_interp_in_bar = max((notes[i][0] - db1) / (db2 - db1), 0)
                        end_interp_in_bar = (notes[i][0] + notes[i][2] - db1) / (db2 - db1)
                        cur_resized_notes.append([cur_downbeat_timing + start_interp_in_bar * 4.0,
                            cur_downbeat_timing + end_interp_in_bar * 4.0,
                            notes[i][1]])
                        cur_groundtruth.append(notes[i][3])
                cur_downbeat_timing = cur_downbeat_timing + 4.0
            resized_notes.append(cur_resized_notes)
            groundtruth.append(cur_groundtruth)
            # print (cur_resized_notes)
            # print (notes)
        return resized_notes, groundtruth

    def prepare_finetune_orch_data(self, midi_paths, annotation_paths, dbeat_paths, task):
        all_note_list, all_groundtruth = [], []
        for j in tqdm(range(len(midi_paths))):
            midi_path = midi_paths[j]
            annotation_path = annotation_paths[j]
            dbeat_path = dbeat_paths[j]
            # extract events
            note_items_list, groundtruth_list = self.extract_orch_events(midi_path, annotation_path, dbeat_path, task)
            for k in range(len(note_items_list)):
                all_note_list.append(note_items_list[k])
                all_groundtruth.append(groundtruth_list[k])

        return all_note_list, all_groundtruth

    def note2event(self, notes, dbeats, task):
        resized_notes = []
        groundtruth = []
        cur_downbeat_timing = 0
        for db1, db2 in zip(dbeats[:-1], dbeats[1:]):
            for i in range(len(notes)):
                if (notes[i][0] >= db1) and (notes[i][0] < db2):                    
                    start_interp_in_bar = max((notes[i][0] - db1) / (db2 - db1), 0)
                    end_interp_in_bar = (notes[i][0] + notes[i][2] - db1) / (db2 - db1)
                    resized_notes.append([cur_downbeat_timing + start_interp_in_bar * 4.0,
                        cur_downbeat_timing + end_interp_in_bar * 4.0,
                        notes[i][1]])
                    groundtruth.append(notes[i][3])
            cur_downbeat_timing = cur_downbeat_timing + 4.0
        return resized_notes, groundtruth

    def extract_augnet_events(self, mxl_path, annotation_path, task):
        # Get downbeat and anontation
        dbeats = [0,]
        root = []
        with open(annotation_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            previous_gt_measure = 0
            for row in reader:
                if row[0] != 'j_offset':
                    if task == 'chordroot':
                        root.append([float(row[0]), float(row[0]) + 0.125, Chord_root[str(row[16])]])
                    elif task == 'localkey':
                        root.append([float(row[0]), float(row[0]) + 0.125, Localkey[str(row[20])]])
                    
                    if float(row[2]) > previous_gt_measure:
                        # Add a barline
                        dbeats.append(float(row[0]))
                        previous_gt_measure = float(row[2])

        if len(dbeats) > 2:
            dbeats[0] = dbeats[1] - (dbeats[2] - dbeats[1])
        else:
            # If no downbeat, then assume a 4/4 timesig 
            dbeats[0] = dbeats[1] - 4.0
        dbeats.append(dbeats[-1] + (dbeats[-1] - dbeats[-2]))

        # Parse mxl file
        note_file = converter.parse(mxl_path)
        note_list = []
        for note in note_file.flat.notes:
            # print (note)
            if isinstance(note, music21.note.Note):
                # onset, pitch, duration
                note_list.append([note.offset, note.pitch.midi, note.duration.quarterLength])
            elif isinstance(note, music21.chord.Chord):
                for j in range(len(note.pitches)):
                    note_list.append([note.offset, note.pitches[j].midi, note.duration.quarterLength])

        notes = []
        cur_annotation_id = 0
        for note in note_list:
            # onset, pitch, duration
            onset = note[0]
            pitch = note[1]
            duration = note[2]

            while (cur_annotation_id < len(root) - 1 
                and onset >= root[cur_annotation_id+1][0]):
                cur_annotation_id += 1

            if duration > 0:
                notes.append([onset, pitch, duration, root[cur_annotation_id][2]])

        notes = sorted(notes, key=lambda x: (x[0], x[1], x[2]))
        note_items, groundtruth = self.note2event(notes, dbeats, task)
        return note_items, groundtruth

    
    def prepare_finetune_augnet_data(self, mxl_paths, annotation_paths, task):
        all_note_list, all_groundtruth = [], []
        for j in tqdm(range(len(mxl_paths))):
            mxl_path = mxl_paths[j]
            annotation_path = annotation_paths[j]
            # extract events
            note_items, groundtruth = self.extract_augnet_events(mxl_path, annotation_path, task)
            if not note_items:  # if midi contains nothing
                print(f'skip {path} because it is empty')
                continue
            all_note_list.append(note_items)
            all_groundtruth.append(groundtruth)
        return all_note_list, all_groundtruth

    def extract_pm2s_data(self, note_sequence, annotations, task):
        # 1 is always negative; 2 is always positive
        resolution = 0.01
        tolerance = 0.07
        beats = annotations['beats']
        downbeats = annotations['downbeats']

        # time to beat/downbeat/inter-beat-interval dictionaries
        end_time = max(beats[-1], note_sequence[-1][1] + note_sequence[-1][2]) + 1.0
        time2beat = np.zeros(int(np.ceil(end_time / resolution)))
        time2downbeat = np.zeros(int(np.ceil(end_time / resolution)))
        time2ibi = np.zeros(int(np.ceil(end_time / resolution)))
        for idx, beat in enumerate(beats):
            l = np.round((beat - tolerance) / resolution).astype(int)
            r = np.round((beat + tolerance) / resolution).astype(int)
            time2beat[l:r+1] = 1.0

            ibi = beats[idx+1] - beats[idx] if idx+1 < len(beats) else beats[-1] - beats[-2]
            l = np.round((beat - tolerance) / resolution).astype(int) if idx > 0 else 0
            r = np.round((beat + ibi) / resolution).astype(int) if idx+1 < len(beats) else len(time2ibi)
            if ibi > 4:
                # reset ibi to 0 if it's too long, index 0 will be ignored during training
                ibi = np.array(0)
            time2ibi[l:r+1] = np.round(ibi / resolution)
        
        for downbeat in downbeats:
            l = np.round((downbeat - tolerance) / resolution).astype(int)
            r = np.round((downbeat + tolerance) / resolution).astype(int)
            time2downbeat[l:r+1] = 1.0
        
        # get beat probabilities at note onsets
        beat_probs = np.zeros(len(note_sequence), dtype=np.float32)
        downbeat_probs = np.zeros(len(note_sequence), dtype=np.float32)
        ibis = np.zeros(len(note_sequence), dtype=np.float32)
        for i in range(len(note_sequence)):
            onset = note_sequence[i][1]
            beat_probs[i] = time2beat[np.round(onset / resolution).astype(int)]
            downbeat_probs[i] = time2downbeat[np.round(onset / resolution).astype(int)]
            ibis[i] = time2ibi[np.round(onset / resolution).astype(int)]

        # print (note_sequence.shape, beat_probs.shape)
        if task == 'beat':
            note_sequence = np.concatenate((note_sequence, np.expand_dims(beat_probs, axis=1)), axis=1)
        elif task == 'downbeat':
            note_sequence = np.concatenate((note_sequence, np.expand_dims(downbeat_probs, axis=1)), axis=1)

        # print (note_sequence.shape)
        sorted_note_items, groundtruth = utils.convert_pm2s_data(note_sequence)
        return sorted_note_items, groundtruth 

    def prepare_finetune_pm2s_data(self, file_paths, task):
        all_note_list, all_groundtruth = [], []
        for file_path in tqdm(file_paths):
            note_sequence, annotations = pickle.load(open(file_path, 'rb'))
            note_items, groundtruth = self.extract_pm2s_data(note_sequence, annotations, task)
            if not note_items:  # if midi contains nothing
                print(f'skip {path} because it is empty')
                continue
            all_note_list.append(note_items)
            all_groundtruth.append(groundtruth)
        return all_note_list, all_groundtruth

    def extract_tnua_data(self, file_path, task):
        # Get notes
        notes = []
        groundtruth = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if row[0] != 'pitch':
                    localbeat_onset = float(row[1])
                    duration = float(row[2])
                    pitch = int(row[0])
                    string = int(row[4])
                    position = int(row[5])
                    finger = int(row[6]) + 1
                    # String: 1-4
                    # position: 1-12
                    # finger: 0-4
                    # class = String*60+(pos-1)*5+finger+1
                    # Adjust cases where pos=0 (set pos to at least 1) 
                    # or string=0 (directly set to class 0, i.e., ignoring this annotation)
                    annotation = max((string - 1) * 60 + (max(position, 1) - 1) * 5 + finger + 1, 0)
                    note = [localbeat_onset, localbeat_onset + duration, pitch]
                    notes.append(note)
                    groundtruth.append(annotation)

        # Since there is no explicit time signature/barline annotation,
        # we assume a 4/4 timesig for all songs (which is obviously not always correct)
        # In other words, we did not resize the bars to make them 4/4
        # Furthermore, we **retain*( the localbeat annotations here and do not try to
        # convert them to global beat (which is different from all 11 other tasks)
        return notes, groundtruth

    def prepare_finetune_tnua_data(self, file_paths, task):
        all_note_list, all_groundtruth = [], []
        for file_path in tqdm(file_paths):
            note_items, groundtruth = self.extract_tnua_data(file_path, task)
            all_note_list.append(note_items)
            all_groundtruth.append(groundtruth)
            # print (note_items)
            # print (groundtruth)
        return all_note_list, all_groundtruth