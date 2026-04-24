"""
Patch num_words into existing NDA submission CSVs.

Reads all CSV part files in an input directory, finds the matching transcript
files in a transcripts directory (by basename from the transcript_file column),
and inserts a num_words column between num_sent and word_freq in each CSV.

Strict validation: if any transcript file referenced across ALL input CSVs
cannot be matched on disk, the script exits immediately without processing
anything.

Usage:
    python misc/patch_num_words.py \
        --input-dir  nda4_redo/journals/ \
        --transcripts /path/to/transcripts \
        --output-dir nda4_redo/journals_patched/ \
        --gpu 0
"""

import os
import sys

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import csv
import re
import argparse
import stanza
from pathlib import Path
from collections import defaultdict

from utils.transcripts import Transcript
from features.grammar import detect_language_for_transcript, LANG_TO_STANZA
from data.langs import Language


REQUIRED_COLS = {'transcript_file', 'speaker_role', 'num_sent', 'word_freq', 'chrspeech_other_lang'}

_SUBMISSION_RE = re.compile(r'(submission|session)(\d+)', re.IGNORECASE)


def normalize_fname(fname: str) -> str:
    """Strip leading zeros from submission/session numbers in a filename."""
    return _SUBMISSION_RE.sub(lambda m: m.group(1) + str(int(m.group(2))), fname)

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
        description="Patch num_words column into a directory of NDA submission CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing CSV part files to patch")
    parser.add_argument("--transcripts", type=str, required=True,
                        help="Root directory to search for transcript .txt files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to write patched CSVs (same filenames as input)")
    parser.add_argument("--gpu", type=int, required=True,
                        help="GPU device ID to use")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    input_dir = Path(args.input_dir)
    transcripts_dir = Path(args.transcripts)
    output_dir = Path(args.output_dir)

    # ------------------------------------------------------------------ #
    # Phase 1: Read and validate — no Stanza, no output until this passes #
    # ------------------------------------------------------------------ #

    csv_files = sorted(input_dir.rglob("*.csv"))
    if not csv_files:
        print(f"Error: no CSV files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV file(s) in {input_dir}")

    # Read all CSVs and validate columns
    # csv_data: {csv_path: (fieldnames, rows)}
    csv_data: dict[Path, tuple[list[str], list[dict]]] = {}

    for csv_path in csv_files:
        with open(csv_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                print(f"Error: {csv_path.name} is empty or has no header.")
                sys.exit(1)
            missing_cols = REQUIRED_COLS - set(reader.fieldnames)
            if missing_cols:
                print(f"Error: {csv_path.name} is missing required columns: {missing_cols}")
                sys.exit(1)
            rows = list(reader)
            csv_data[csv_path] = (list(reader.fieldnames), rows)

    print("All CSV files passed column validation.")

    # Collect all unique transcript basenames (exact) and their language across all CSVs
    filename_to_lang: dict[str, Language] = {}
    for _, (_, rows) in csv_data.items():
        for row in rows:
            fname = Path(row['transcript_file']).name
            if fname not in filename_to_lang:
                filename_to_lang[fname] = map_language_string(row['chrspeech_other_lang'])

    print(f"Found {len(filename_to_lang)} unique transcript filenames across all CSVs.")

    # Build disk indices: exact and normalized (for fallback)
    print(f"\nIndexing transcript files in {transcripts_dir} ...")
    disk_files_exact: dict[str, Path] = {}
    disk_files_normalized: dict[str, Path] = {}
    for p in transcripts_dir.rglob('*.txt'):
        disk_files_exact[p.name] = p
        disk_files_normalized[normalize_fname(p.name)] = p
    print(f"Indexed {len(disk_files_exact)} .txt files on disk.")

    # Two-pass resolution: exact match first, normalized fallback second
    resolved: dict[str, Path] = {}  # csv_basename -> disk path
    truly_missing: list[str] = []

    for csv_fname in filename_to_lang:
        if csv_fname in disk_files_exact:
            resolved[csv_fname] = disk_files_exact[csv_fname]
        else:
            normalized = normalize_fname(csv_fname)
            if normalized in disk_files_normalized:
                resolved[csv_fname] = disk_files_normalized[normalized]
                print(f"  Matched via normalization: {csv_fname}")
            else:
                truly_missing.append(csv_fname)

    if truly_missing:
        print(f"\nError: {len(truly_missing)} transcript file(s) not found in {transcripts_dir}:")
        for fname in sorted(truly_missing):
            print(f"  {fname}")
        print("\nAborting — no output written.")
        sys.exit(1)

    print("All transcript files matched on disk. Proceeding with processing.")

    # ------------------------------------------------------------------ #
    # Phase 2: Count words                                                #
    # ------------------------------------------------------------------ #

    # Group by Stanza language code
    transcripts_by_stanza: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    cn_files: list[tuple[str, Path]] = []

    for fname in filename_to_lang:
        lang = filename_to_lang[fname]
        fpath = resolved[fname]
        if lang == Language.cn:
            cn_files.append((fname, fpath))
        elif lang in LANG_TO_STANZA:
            transcripts_by_stanza[LANG_TO_STANZA[lang]].append((fname, fpath))
        else:
            print(f"  WARNING: unsupported/unknown language '{lang}' for {fname}, skipping.")

    # Language detection for Chinese transcripts
    if cn_files:
        print(f"\nDetecting script variant for {len(cn_files)} Chinese transcript(s)...")
        Transcript.set_directory_path(transcripts_dir)
        langid_pipeline = stanza.Pipeline(lang='multilingual', processors='langid', use_gpu=True)
        for fname, fpath in cn_files:
            transcript = Transcript(fpath)
            detected = detect_language_for_transcript(transcript, langid_pipeline)
            print(f"  {fname} -> '{detected}'")
            transcripts_by_stanza[detected].append((fname, fpath))
        del langid_pipeline

    results: dict[tuple[str, str], int] = {}  # (basename, speaker_role_label) -> num_words

    Transcript.set_directory_path(transcripts_dir)

    for stanza_code in sorted(transcripts_by_stanza.keys()):
        file_pairs = transcripts_by_stanza[stanza_code]
        print(f"\n{'=' * 55}")
        print(f"Language: {stanza_code}  ({len(file_pairs)} transcript(s))")
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

    # ------------------------------------------------------------------ #
    # Phase 3: Write output CSVs                                          #
    # ------------------------------------------------------------------ #

    total_rows = 0
    total_updated = 0

    for csv_path, (original_fieldnames, rows) in csv_data.items():
        if 'num_words' in original_fieldnames:
            fieldnames = original_fieldnames
        else:
            idx = original_fieldnames.index('num_sent')
            fieldnames = original_fieldnames[:idx + 1] + ['num_words'] + original_fieldnames[idx + 1:]

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

        out_path = output_dir / csv_path.relative_to(input_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)

        print(f"  {csv_path.name}: {rows_updated}/{len(rows)} rows updated -> {out_path}")
        total_rows += len(rows)
        total_updated += rows_updated

    print(f"\nDone. {total_updated}/{total_rows} total rows updated with num_words.")
    print(f"Output written to {output_dir}")


if __name__ == "__main__":
    main()
