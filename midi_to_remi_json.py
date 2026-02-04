#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import miditoolkit
import utils

NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PC_TO_NAME = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
NAME_TO_PC = {
    "C": 0, "B#": 0,
    "C#": 1, "Db": 1,
    "D": 2,
    "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4,
    "F": 5, "E#": 5,
    "F#": 6, "Gb": 6,
    "G": 7,
    "G#": 8, "Ab": 8,
    "A": 9,
    "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11,
}
MAJOR_DEGREE_LABEL = {
    0: "1", 1: "b2", 2: "2", 3: "b3", 4: "3", 5: "4",
    6: "#4", 7: "5", 8: "b6", 9: "6", 10: "b7", 11: "7",
}
MINOR_DEGREE_LABEL = {
    0: "1", 1: "b2", 2: "2", 3: "b3", 4: "3", 5: "4",
    6: "#4", 7: "5", 8: "b6", 9: "6", 10: "b7", 11: "7",
}


def remi_events_for_midi(midi_path, with_chords=True):
    note_items, tempo_items = utils.read_items(midi_path)
    note_items = utils.quantize_items(note_items)
    max_time = note_items[-1].end
    if with_chords:
        chord_items = utils.extract_chords(note_items)
        items = chord_items + tempo_items + note_items
    else:
        items = tempo_items + note_items
    groups = utils.group_items(items, max_time)
    events = utils.item2event(groups)
    return events


def _normalize_json_value(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _pc_to_name(pc):
    return PC_TO_NAME[int(pc) % 12]


def _name_to_pc(name):
    if name is None:
        return None
    key = name.strip().replace("-", "b")
    key = key.replace("major", "").replace("minor", "").strip()
    key = key.replace(" ", "")
    return NAME_TO_PC.get(key)


def _parse_key_name(key_name):
    if key_name is None:
        return None
    name = key_name.strip()
    mode = "major"
    if "minor" in name:
        mode = "minor"
    elif "major" in name:
        mode = "major"
    root = name.replace("minor", "").replace("major", "").strip()
    pc = _name_to_pc(root)
    if pc is None:
        return None
    return pc, mode


def _degree_label(delta_pc, mode):
    if mode == "minor":
        return MINOR_DEGREE_LABEL.get(delta_pc)
    return MAJOR_DEGREE_LABEL.get(delta_pc)


def _resolve_mcr_root(mcr_path=None):
    if mcr_path is not None:
        root = Path(mcr_path).expanduser()
    else:
        root = Path(__file__).resolve().parents[1] / "midi-chord-recognition"
    if not root.exists():
        raise FileNotFoundError(f"midi-chord-recognition not found at {root}")
    return root


def _load_mcr_modules(mcr_path=None):
    root = _resolve_mcr_root(mcr_path)
    sys.path.insert(0, str(root))
    from mir import DataEntry
    from mir import io as mir_io
    from extractors.midi_utilities import MidiBeatExtractor
    from extractors.rule_based_channel_reweight import midi_to_thickness_and_bass_weights
    from midi_chord import ChordRecognition
    from chord_class import ChordClass
    return DataEntry, mir_io, MidiBeatExtractor, midi_to_thickness_and_bass_weights, ChordRecognition, ChordClass


def extract_mcr_chords_with_obs(midi_path, mcr_path=None, use_transition=True):
    (
        DataEntry,
        mir_io,
        MidiBeatExtractor,
        midi_to_thickness_and_bass_weights,
        ChordRecognition,
        ChordClass,
    ) = _load_mcr_modules(mcr_path)
    entry = DataEntry()
    entry.append_file(midi_path, mir_io.MidiIO, "midi")
    entry.append_extractor(MidiBeatExtractor, "beat")
    rec = ChordRecognition(entry, ChordClass(), single_beat_switch=True)
    weights = midi_to_thickness_and_bass_weights(entry.midi)
    rec.process_feature(weights)
    _, _, segments = rec.decode(use_transition=use_transition, return_obs=True)
    return segments


def _build_chord_segments(midi_path, include_tensor=False, mcr_path=None, use_transition=True):
    midi = miditoolkit.MidiFile(midi_path)
    tick_to_time = midi.get_tick_to_time_mapping()
    segments = extract_mcr_chords_with_obs(
        midi_path,
        mcr_path=mcr_path,
        use_transition=use_transition,
    )
    for seg in segments:
        start_sec = float(seg["start"])
        end_sec = float(seg["end"])
        start_tick = int(np.searchsorted(tick_to_time, start_sec, side="left"))
        end_tick = int(np.searchsorted(tick_to_time, end_sec, side="left"))
        seg["start_sec"] = start_sec
        seg["end_sec"] = end_sec
        seg["start_tick"] = start_tick
        seg["end_tick"] = end_tick
        if include_tensor:
            seg["chord_tensor"] = seg["obs"].astype(float).tolist()
    return tick_to_time, segments


def _get_key_signature_segments(midi, tick_to_time):
    keys = sorted(midi.key_signature_changes, key=lambda k: k.time)
    if not keys:
        return []
    segments = []
    max_tick = midi.max_tick
    for i, ks in enumerate(keys):
        parsed = _parse_key_name(ks.key_name)
        if parsed is None:
            continue
        pc, mode = parsed
        start_tick = int(ks.time)
        end_tick = int(keys[i + 1].time) if i + 1 < len(keys) else int(max_tick)
        start_sec = float(tick_to_time[start_tick]) if start_tick < len(tick_to_time) else float(tick_to_time[-1])
        end_sec = float(tick_to_time[end_tick]) if end_tick < len(tick_to_time) else float(tick_to_time[-1])
        segments.append({
            "root_pc": int(pc),
            "root": _pc_to_name(pc),
            "mode": mode,
            "start_tick": start_tick,
            "end_tick": end_tick,
            "start_sec": start_sec,
            "end_sec": end_sec,
        })
    return segments


def _get_key_windows_music21(midi_path, window_quarters):
    try:
        from music21 import converter, analysis
    except Exception as exc:
        raise RuntimeError("music21 is required for key analysis without key_signature") from exc
    stream = converter.parse(midi_path)
    processor = analysis.discrete.KrumhanslSchmuckler()
    wa = analysis.windowed.WindowedAnalysis(stream, processor)
    results, _ = wa.analyze(window_quarters)
    windows = []
    for pitch, mode, corr in results:
        pc = int(pitch.pitchClass)
        windows.append({
            "root_pc": pc,
            "root": _pc_to_name(pc),
            "mode": mode,
            "corr": float(corr),
        })
    return windows


def _compress_key_windows(windows, ticks_per_beat):
    if not windows:
        return []
    segments = []
    start_q = 0
    acc_corr = windows[0].get("corr", 0.0)
    count = 1
    prev = windows[0]
    for i in range(1, len(windows)):
        w = windows[i]
        if w["root_pc"] != prev["root_pc"] or w["mode"] != prev["mode"]:
            segments.append({
                "root_pc": prev["root_pc"],
                "root": prev["root"],
                "mode": prev["mode"],
                "start_tick": start_q * ticks_per_beat,
                "end_tick": i * ticks_per_beat,
                "avg_corr": acc_corr / max(1, count),
            })
            start_q = i
            prev = w
            acc_corr = w.get("corr", 0.0)
            count = 1
        else:
            acc_corr += w.get("corr", 0.0)
            count += 1
    segments.append({
        "root_pc": prev["root_pc"],
        "root": prev["root"],
        "mode": prev["mode"],
        "start_tick": start_q * ticks_per_beat,
        "end_tick": len(windows) * ticks_per_beat,
        "avg_corr": acc_corr / max(1, count),
    })
    return segments


def _build_key_info(midi_path, tick_to_time, window_quarters=8):
    midi = miditoolkit.MidiFile(midi_path)
    ticks_per_beat = int(midi.ticks_per_beat)
    segments = _get_key_signature_segments(midi, tick_to_time)
    if segments:
        return {
            "source": "key_signature",
            "segments": segments,
            "ticks_per_beat": ticks_per_beat,
            "tick_to_time": tick_to_time,
        }
    windows = _get_key_windows_music21(midi_path, window_quarters)
    segments = _compress_key_windows(windows, ticks_per_beat)
    for seg in segments:
        start_tick = seg["start_tick"]
        end_tick = seg["end_tick"]
        seg["start_sec"] = float(tick_to_time[start_tick]) if start_tick < len(tick_to_time) else float(tick_to_time[-1])
        seg["end_sec"] = float(tick_to_time[end_tick]) if end_tick < len(tick_to_time) else float(tick_to_time[-1])
    return {
        "source": "music21",
        "segments": segments,
        "windows": windows,
        "ticks_per_beat": ticks_per_beat,
        "window_quarters": window_quarters,
        "tick_to_time": tick_to_time,
    }


def _key_for_tick_from_segments(segments, tick):
    if not segments:
        return None
    idx = 0
    for i, seg in enumerate(segments):
        if tick < seg["end_tick"]:
            idx = i
            break
    seg = segments[idx]
    return seg


def _key_for_tick_from_windows(windows, tick, ticks_per_beat):
    if not windows:
        return None
    q_index = int(tick // ticks_per_beat)
    if q_index >= len(windows):
        q_index = len(windows) - 1
    return windows[q_index]


def iter_dense_entries_with_key(iterator, key_info, include_scales=True):
    source = key_info.get("source")
    segments = key_info.get("segments", [])
    windows = key_info.get("windows", [])
    ticks_per_beat = key_info.get("ticks_per_beat", 480)
    tick_to_time = key_info.get("tick_to_time")
    seg_idx = 0
    last_key = None
    for entry in iterator:
        tick = int(entry["time_start"])
        key_seg = None
        if source == "key_signature":
            while seg_idx + 1 < len(segments) and tick >= segments[seg_idx]["end_tick"]:
                seg_idx += 1
            if segments:
                key_seg = segments[seg_idx]
        else:
            key_seg = _key_for_tick_from_windows(windows, tick, ticks_per_beat)
        if key_seg is None:
            yield entry
            continue
        root_pc = int(key_seg["root_pc"])
        mode = key_seg["mode"]
        key_tuple = (root_pc, mode)
        if key_tuple != last_key:
            comment = {
                "_type": "key_change",
                "index": entry.get("index"),
                "key": "{} {}".format(_pc_to_name(root_pc), mode),
                "comment": "Key: {} {}".format(_pc_to_name(root_pc), mode),
                "key_root": _pc_to_name(root_pc),
                "key_mode": mode,
                "source": source,
                "start_tick": _normalize_json_value(tick),
            }
            if tick_to_time is not None:
                if tick < len(tick_to_time):
                    comment["start_sec"] = _normalize_json_value(float(tick_to_time[tick]))
                else:
                    comment["start_sec"] = _normalize_json_value(float(tick_to_time[-1]))
            if "corr" in key_seg:
                comment["corr"] = _normalize_json_value(key_seg["corr"])
            if "avg_corr" in key_seg:
                comment["avg_corr"] = _normalize_json_value(key_seg["avg_corr"])
            yield comment
            last_key = key_tuple
        if include_scales:
            notes = entry.get("notes") or []
            if notes:
                highest = int(max(notes))
                delta = (highest % 12 - root_pc) % 12
                entry["treble_scale"] = _degree_label(delta, mode)
            if "chord" in entry and entry["chord"] not in (None, "N"):
                chord_root = entry["chord"].split(":")[0]
                chord_pc = _name_to_pc(chord_root)
                if chord_pc is not None:
                    delta = (chord_pc - root_pc) % 12
                    entry["chord_scale"] = _degree_label(delta, mode)
        yield entry


def iter_dense_entries_with_chords(iterator, tick_to_time, segments, include_tensor=False):
    seg_idx = 0
    seg_count = len(segments)
    for entry in iterator:
        entry_time_start = entry["time_start"]
        if entry_time_start >= len(tick_to_time):
            entry_time_start = len(tick_to_time) - 1
        entry_start_sec = float(tick_to_time[entry_time_start])
        while seg_idx + 1 < seg_count and entry_start_sec >= segments[seg_idx]["end_sec"]:
            seg_idx += 1
        if seg_count == 0:
            yield entry
            continue
        seg = segments[seg_idx]
        if entry_start_sec < seg["start_sec"] or entry_start_sec >= seg["end_sec"]:
            yield entry
            continue
        entry["chord"] = seg["chord"]
        entry["chord_start_sec"] = _normalize_json_value(seg["start_sec"])
        entry["chord_end_sec"] = _normalize_json_value(seg["end_sec"])
        entry["chord_start_tick"] = _normalize_json_value(seg["start_tick"])
        entry["chord_end_tick"] = _normalize_json_value(seg["end_tick"])
        if include_tensor:
            entry["chord_tensor"] = seg["chord_tensor"]
        yield entry


def event_to_dict(event):
    value = _normalize_json_value(event.value)
    time = _normalize_json_value(event.time)
    value_str = "None" if value is None else str(value)
    return {
        "event": f"{event.name}_{value_str}",
        "name": event.name,
        "value": value,
        "time": time,
        "text": event.text,
    }


def pitch_to_note_name(pitch):
    octave = pitch // 12 - 1
    name = NOTE_NAMES_SHARP[pitch % 12]
    return f"{name}{octave}"


def _position_index(position_value):
    return int(position_value.split("/")[0])


def remi_bars_for_midi(midi_path, with_chords=True):
    note_items, tempo_items = utils.read_items(midi_path)
    note_items = utils.quantize_items(note_items)
    max_time = note_items[-1].end
    if with_chords:
        chord_items = utils.extract_chords(note_items)
        items = chord_items + tempo_items + note_items
    else:
        items = tempo_items + note_items
    groups = utils.group_items(items, max_time)
    return groups_to_bar_items(groups)


def groups_to_bar_items(groups):
    bars = []
    n_downbeat = 0
    for group in groups:
        if 'Note' not in [item.name for item in group[1:-1]]:
            continue
        bar_st, bar_et = group[0], group[-1]
        n_downbeat += 1
        flags = np.linspace(bar_st, bar_et, utils.DEFAULT_FRACTION, endpoint=False)
        bar = {
            "bar": n_downbeat,
            "items": [],
        }
        for item in group[1:-1]:
            index = np.argmin(abs(flags - item.start))
            position_value = '{}/{}'.format(index + 1, utils.DEFAULT_FRACTION)
            position_event = utils.Event(
                name='Position',
                time=item.start,
                value=position_value,
                text='{}'.format(item.start),
            )
            events = [position_event]
            if item.name == 'Note':
                velocity_index = np.searchsorted(
                    utils.DEFAULT_VELOCITY_BINS,
                    item.velocity,
                    side='right',
                ) - 1
                events.append(utils.Event(
                    name='Note Velocity',
                    time=item.start,
                    value=velocity_index,
                    text='{}/{}'.format(
                        item.velocity,
                        utils.DEFAULT_VELOCITY_BINS[velocity_index],
                    ),
                ))
                events.append(utils.Event(
                    name='Note On',
                    time=item.start,
                    value=item.pitch,
                    text='{}'.format(item.pitch),
                ))
                duration = item.end - item.start
                duration_index = np.argmin(abs(utils.DEFAULT_DURATION_BINS - duration))
                events.append(utils.Event(
                    name='Note Duration',
                    time=item.start,
                    value=duration_index,
                    text='{}/{}'.format(duration, utils.DEFAULT_DURATION_BINS[duration_index]),
                ))
            elif item.name == 'Chord':
                events.append(utils.Event(
                    name='Chord',
                    time=item.start,
                    value=item.pitch,
                    text='{}'.format(item.pitch),
                ))
            elif item.name == 'Tempo':
                tempo = item.pitch
                if tempo in utils.DEFAULT_TEMPO_INTERVALS[0]:
                    tempo_style = utils.Event('Tempo Class', item.start, 'slow', None)
                    tempo_value = utils.Event(
                        'Tempo Value',
                        item.start,
                        tempo - utils.DEFAULT_TEMPO_INTERVALS[0].start,
                        None,
                    )
                elif tempo in utils.DEFAULT_TEMPO_INTERVALS[1]:
                    tempo_style = utils.Event('Tempo Class', item.start, 'mid', None)
                    tempo_value = utils.Event(
                        'Tempo Value',
                        item.start,
                        tempo - utils.DEFAULT_TEMPO_INTERVALS[1].start,
                        None,
                    )
                elif tempo in utils.DEFAULT_TEMPO_INTERVALS[2]:
                    tempo_style = utils.Event('Tempo Class', item.start, 'fast', None)
                    tempo_value = utils.Event(
                        'Tempo Value',
                        item.start,
                        tempo - utils.DEFAULT_TEMPO_INTERVALS[2].start,
                        None,
                    )
                elif tempo < utils.DEFAULT_TEMPO_INTERVALS[0].start:
                    tempo_style = utils.Event('Tempo Class', item.start, 'slow', None)
                    tempo_value = utils.Event('Tempo Value', item.start, 0, None)
                elif tempo > utils.DEFAULT_TEMPO_INTERVALS[2].stop:
                    tempo_style = utils.Event('Tempo Class', item.start, 'fast', None)
                    tempo_value = utils.Event('Tempo Value', item.start, 59, None)
                events.append(tempo_style)
                events.append(tempo_value)
            bar["items"].append({
                "kind": item.name,
                "time": _normalize_json_value(item.start),
                "position": position_value,
                "events": [event_to_dict(e) for e in events],
            })
        bars.append(bar)
    return bars


def iter_dense_steps_for_midi(midi_path, step_ticks=None, note_names=False):
    note_items, _ = utils.read_items(midi_path)
    note_items = utils.quantize_items(note_items)
    note_items.sort(key=lambda x: (x.start, x.pitch))
    max_time = note_items[-1].end
    ticks_per_bar = utils.DEFAULT_RESOLUTION * 4
    if step_ticks is None:
        step_ticks = ticks_per_bar // utils.DEFAULT_FRACTION
    steps = np.arange(0, max_time, step_ticks, dtype=int)
    active = []
    idx = 0
    for step_index, t in enumerate(steps):
        while idx < len(note_items) and note_items[idx].start <= t:
            active.append(note_items[idx])
            idx += 1
        if active:
            active = [note for note in active if note.end > t]
        notes = sorted({note.pitch for note in active})
        entry = {
            "index": step_index,
            "time": _normalize_json_value(t),
            "bar": int(t // ticks_per_bar) + 1,
            "position": "{}/{}".format(
                int((t % ticks_per_bar) // step_ticks) + 1,
                utils.DEFAULT_FRACTION,
            ),
            "notes": notes,
        }
        if note_names:
            entry["note_names"] = [pitch_to_note_name(p) for p in notes]
        yield entry


def iter_dense_merged_steps_for_midi(midi_path, step_ticks=None, note_names=False):
    ticks_per_bar = utils.DEFAULT_RESOLUTION * 4
    if step_ticks is None:
        step_ticks = ticks_per_bar // utils.DEFAULT_FRACTION
    prev = None
    prev_key = None
    prev_last_index = None
    segment_index = 0
    emitted_any = False
    for entry in iter_dense_steps_for_midi(
        midi_path,
        step_ticks=step_ticks,
        note_names=note_names,
    ):
        key = tuple(entry["notes"])
        if prev is None:
            prev = {
                "time_start": entry["time"],
                "time_end": entry["time"],
                "time_sum": step_ticks,
                "bar_start": entry["bar"],
                "bar_end": entry["bar"],
                "position_start": entry["position"],
                "position_end": entry["position"],
                "notes": entry["notes"],
            }
            if note_names:
                prev["note_names"] = entry["note_names"]
            prev_key = key
            prev_last_index = entry["index"]
            continue
        if (
            key == prev_key
            and entry["index"] == prev_last_index + 1
        ):
            prev["time_end"] = entry["time"]
            prev["bar_end"] = entry["bar"]
            prev["position_end"] = entry["position"]
            prev["time_sum"] += step_ticks
            prev_last_index = entry["index"]
            continue
        out = {
            "time_start": prev["time_start"],
            "time_end": prev["time_end"],
            "time_sum": prev["time_sum"],
            "position_start": prev["position_start"],
            "position_end": prev["position_end"],
            "notes": prev["notes"],
        }
        pos_start = _position_index(prev["position_start"])
        pos_end = _position_index(prev["position_end"])
        bar_span = prev["bar_end"] - prev["bar_start"]
        out["position_sum"] = bar_span * utils.DEFAULT_FRACTION + (pos_end - pos_start) + 1
        if note_names:
            out["note_names"] = prev["note_names"]
        if out["notes"] or emitted_any:
            segment_index += 1
            out["index"] = segment_index
            emitted_any = True
            yield out
        prev = {
            "time_start": entry["time"],
            "time_end": entry["time"],
            "time_sum": step_ticks,
            "bar_start": entry["bar"],
            "bar_end": entry["bar"],
            "position_start": entry["position"],
            "position_end": entry["position"],
            "notes": entry["notes"],
        }
        if note_names:
            prev["note_names"] = entry["note_names"]
        prev_key = key
        prev_last_index = entry["index"]
    if prev is not None:
        out = {
            "time_start": prev["time_start"],
            "time_end": prev["time_end"],
            "time_sum": prev["time_sum"],
            "position_start": prev["position_start"],
            "position_end": prev["position_end"],
            "notes": prev["notes"],
        }
        pos_start = _position_index(prev["position_start"])
        pos_end = _position_index(prev["position_end"])
        bar_span = prev["bar_end"] - prev["bar_start"]
        out["position_sum"] = bar_span * utils.DEFAULT_FRACTION + (pos_end - pos_start) + 1
        if note_names:
            out["note_names"] = prev["note_names"]
        if out["notes"] or emitted_any:
            segment_index += 1
            out["index"] = segment_index
            emitted_any = True
            yield out


def main():
    parser = argparse.ArgumentParser(
        description="Convert a MIDI file to REMI events JSON using midi2remi.ipynb logic."
    )
    parser.add_argument("midi_path", help="Input MIDI file path.")
    parser.add_argument("output_json", help="Output JSON file path.")
    parser.add_argument(
        "--no-chord",
        action="store_true",
        help="Skip chord extraction (use only tempo + notes).",
    )
    parser.add_argument(
        "--dense-jsonl",
        action="store_true",
        help="Output per-step JSONL with active notes for each time step.",
    )
    parser.add_argument(
        "--dense-merge",
        action="store_true",
        help="Merge consecutive dense steps with identical notes (JSONL output).",
    )
    parser.add_argument(
        "--add-chord",
        action="store_true",
        help="Add chord annotations from midi-chord-recognition to dense JSONL output.",
    )
    parser.add_argument(
        "--add-chord-tensor",
        action="store_true",
        help="Include chord_tensor (obs) from midi-chord-recognition in dense JSONL output.",
    )
    parser.add_argument(
        "--add-key",
        action="store_true",
        help="Add key, treble_scale, and chord_scale to dense JSONL output.",
    )
    parser.add_argument(
        "--key-window",
        type=int,
        default=8,
        help="Window size in quarter notes for music21 key analysis (default: 8).",
    )
    parser.add_argument(
        "--mcr-path",
        type=str,
        default=None,
        help="Path to midi-chord-recognition repo (auto-detected if omitted).",
    )
    parser.add_argument(
        "--step-ticks",
        type=int,
        default=None,
        help="Step size in ticks for dense output (default: 120).",
    )
    parser.add_argument(
        "--note-names",
        action="store_true",
        help="Include note names in dense output (uses sharp names).",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Output a flat event list instead of bar->items structure.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level (default: 2).",
    )
    args = parser.parse_args()

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if (args.add_chord or args.add_chord_tensor) and not (args.dense_jsonl or args.dense_merge):
        raise ValueError("--add-chord/--add-chord-tensor requires --dense-jsonl or --dense-merge")
    if args.add_key and not (args.dense_jsonl or args.dense_merge):
        raise ValueError("--add-key requires --dense-jsonl or --dense-merge")
    if args.dense_jsonl or args.dense_merge:
        with output_path.open("w", encoding="utf-8") as f:
            chord_segments = None
            tick_to_time = None
            key_info = None
            if args.add_chord or args.add_chord_tensor:
                tick_to_time, chord_segments = _build_chord_segments(
                    args.midi_path,
                    include_tensor=args.add_chord_tensor,
                    mcr_path=args.mcr_path,
                    use_transition=True,
                )
            if args.add_key:
                if tick_to_time is None:
                    midi_for_time = miditoolkit.MidiFile(args.midi_path)
                    tick_to_time = midi_for_time.get_tick_to_time_mapping()
                key_info = _build_key_info(
                    args.midi_path,
                    tick_to_time=tick_to_time,
                    window_quarters=args.key_window,
                )
            if args.dense_merge:
                iterator = iter_dense_merged_steps_for_midi(
                    args.midi_path,
                    step_ticks=args.step_ticks,
                    note_names=args.note_names,
                )
            else:
                iterator = iter_dense_steps_for_midi(
                    args.midi_path,
                    step_ticks=args.step_ticks,
                    note_names=args.note_names,
                )
            if args.add_chord or args.add_chord_tensor:
                iterator = iter_dense_entries_with_chords(
                    iterator,
                    tick_to_time=tick_to_time,
                    segments=chord_segments,
                    include_tensor=args.add_chord_tensor,
                )
            if args.add_key:
                iterator = iter_dense_entries_with_key(
                    iterator,
                    key_info=key_info,
                    include_scales=True,
                )
            for entry in iterator:
                json.dump(entry, f, ensure_ascii=True)
                f.write("\n")
        return

    if args.flat:
        events = remi_events_for_midi(args.midi_path, with_chords=not args.no_chord)
        data = [event_to_dict(e) for e in events]
    else:
        data = remi_bars_for_midi(args.midi_path, with_chords=not args.no_chord)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=args.indent)


if __name__ == "__main__":
    main()
