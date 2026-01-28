#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import utils

NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


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
    if args.dense_jsonl or args.dense_merge:
        with output_path.open("w", encoding="utf-8") as f:
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
