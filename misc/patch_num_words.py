"""
Patch num_words into existing NDA submission CSVs.

Reads a CSV with a transcript_file column, finds the matching transcript files
in a given directory, runs Stanza tokenization to count words per speaker role,
and inserts a num_words column between num_sent and word_freq.

Usage:
    python misc/patch_num_words.py \
        --input-csv nda4_redo/journals/part-00000-....csv \
        --transcripts /path/to/transcripts \
        --o nda4_redo/journals/part-00000-..._patched.csv \
        --gpu 0
"""

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import csv
import argparse
import stanza
from pathlib import Path
from collections import defaultdict

from utils.transcripts import Transcript
from features.grammar import detect_language_for_transcript, LANG_TO_STANZA
from data.langs import Language


LANG_VALUE_TO_ENUM: dict[str, Language] = {
    lang.value.lower(): lang
    for lang in Language
    if lang not in (Language.UNKNOWN, Language.cn)
}


def map_language_string(lang_str: str) -> Language:
    return LANG_VALUE_TO_ENUM.get(lang_str.strip().lower(), Language.UNKNOWN)


def count_words_by_role(transcript: Transcript, nlp) -> dict[str, int]:
    is_diary = any("diary" in p.lower() for p in transcript.filename.parts)
    roles = [('participant', transcript.participant_lines, 'Participant')]
    if not is_diary:
        roles.append(('interviewer', transcript.interviewer_lines, 'Interviewer'))

    results = {}
    for _, lines, label in roles:
        n = 0
        for line in lines:
            if not line.text.strip():
                continue
            for sent in nlp(line.text).sentences:
                n += len(sent.words)
        if n > 0:
            results[label] = n
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch num_words column into an existing NDA submission CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input-csv", type=str, required=True,
                        help="Input CSV file to patch")
    parser.add_argument("--transcripts", type=str, required=True,
                        help="Root directory to search for transcript .txt files")
    parser.add_argument("--o", type=str, required=True,
                        help="Output CSV file path")
    parser.add_argument("--gpu", type=int, required=True,
                        help="GPU device ID to use")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    input_csv_path = Path(args.input_csv)
    transcripts_dir = Path(args.transcripts)
    output_path = Path(args.o)

    # Read input CSV
    with open(input_csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print("Error: CSV file is empty or has no header.")
            return

        required_cols = {'transcript_file', 'speaker_role', 'num_sent', 'word_freq', 'chrspeech_other_lang'}
        missing = required_cols - set(reader.fieldnames)
        if missing:
            print(f"Error: Missing required columns: {missing}")
            return

        rows = list(reader)
        original_fieldnames = list(reader.fieldnames)

    print(f"Loaded {len(rows)} rows from {input_csv_path.name}")

    # Build output fieldnames: insert num_words after num_sent
    if 'num_words' in original_fieldnames:
        print("'num_words' column already present — values will be overwritten.")
        fieldnames = original_fieldnames
    else:
        idx = original_fieldnames.index('num_sent')
        fieldnames = original_fieldnames[:idx + 1] + ['num_words'] + original_fieldnames[idx + 1:]

    # Map each unique filename to its Language enum via chrspeech_other_lang
    filename_to_lang: dict[str, Language] = {}
    for row in rows:
        fname = Path(row['transcript_file']).name
        if fname not in filename_to_lang:
            filename_to_lang[fname] = map_language_string(row['chrspeech_other_lang'])

    print(f"Found {len(filename_to_lang)} unique transcript filenames in CSV.")

    # Find transcript files on disk
    print(f"\nSearching for transcripts in {transcripts_dir} ...")
    disk_files: dict[str, Path] = {p.name: p for p in transcripts_dir.rglob('*.txt')}

    found: dict[str, Path] = {
        fname: disk_files[fname]
        for fname in filename_to_lang
        if fname in disk_files
    }
    not_found = [fname for fname in filename_to_lang if fname not in disk_files]

    print(f"Matched {len(found)} / {len(filename_to_lang)} files on disk.")
    for fname in not_found:
        print(f"  WARNING: not found on disk: {fname}")

    if not found:
        print("No transcript files found. Exiting.")
        return

    # Group matched files by Stanza language code
    transcripts_by_stanza: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    cn_files: list[tuple[str, Path]] = []

    for fname, fpath in found.items():
        lang = filename_to_lang[fname]
        if lang == Language.cn:
            cn_files.append((fname, fpath))
        elif lang in LANG_TO_STANZA:
            transcripts_by_stanza[LANG_TO_STANZA[lang]].append((fname, fpath))
        else:
            print(f"  WARNING: unsupported/unknown language '{lang}' for {fname}, skipping.")

    # Handle Chinese transcripts that need language detection
    if cn_files:
        print(f"\nDetecting script variant for {len(cn_files)} Chinese transcripts...")
        Transcript.set_directory_path(transcripts_dir)
        langid_pipeline = stanza.Pipeline(lang='multilingual', processors='langid', use_gpu=True)
        for fname, fpath in cn_files:
            transcript = Transcript(fpath)
            detected = detect_language_for_transcript(transcript, langid_pipeline)
            print(f"  {fname} -> '{detected}'")
            transcripts_by_stanza[detected].append((fname, fpath))
        del langid_pipeline

    # Count words per transcript per speaker role
    results: dict[tuple[str, str], int] = {}  # (filename, speaker_role_label) -> num_words

    Transcript.set_directory_path(transcripts_dir)

    for stanza_code in sorted(transcripts_by_stanza.keys()):
        file_pairs = transcripts_by_stanza[stanza_code]
        print(f"\n{'=' * 55}")
        print(f"Language: {stanza_code}  ({len(file_pairs)} transcripts)")
        print(f"{'=' * 55}")

        nlp = stanza.Pipeline(stanza_code, processors='tokenize,mwt', use_gpu=True)

        for i, (fname, fpath) in enumerate(file_pairs):
            print(f"[{i + 1}/{len(file_pairs)}] {fname}")
            try:
                transcript = Transcript(fpath)
                counts = count_words_by_role(transcript, nlp)
                for role_label, n in counts.items():
                    results[(fname, role_label)] = n
                    print(f"  {role_label}: {n} words")
                if not counts:
                    print("  WARNING: no words extracted.")
            except Exception as e:
                print(f"  ERROR: {e}")

        del nlp

    # Patch rows in place
    rows_updated = 0
    for row in rows:
        fname = Path(row['transcript_file']).name
        role = row['speaker_role']
        count = results.get((fname, role))
        if count is not None:
            row['num_words'] = str(count)
            rows_updated += 1
        else:
            row.setdefault('num_words', '')

    # Write output CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {rows_updated} / {len(rows)} rows updated with num_words.")
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    main()
