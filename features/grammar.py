"""
For every participant, tallies total feature count derived from Stanza
and creates an output CSV where:

            clinical_status feat1 feat2 ...
patient1    CHR             119   16
patient2    CHR             38    9
patient3    HC              13    NaN
...

Refactored to use Transcript class abstraction.
"""

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import csv
import math
import argparse
import numpy as np
import stanza
from pathlib import Path
from typing import Optional
from collections import defaultdict

from utils.transcripts import Transcript, ClinicalGroup
from data.langs import Language

# Languages with Stanza support
SUPPORTED_STANZA_LANGUAGES = {'zh', 'es', 'en', 'ko', 'it', 'ja', 'da', 'de', 'fr', 'yue'}

# Mapping from Language enum to Stanza language code
LANG_TO_STANZA = {
    Language.zh: 'zh',
    Language.es: 'es',
    Language.en: 'en',
    Language.ko: 'ko',
    Language.it: 'it',
    Language.ja: 'ja',
    Language.da: 'da',
    Language.de: 'de',
    Language.fr: 'fr',
    Language.yue: 'yue',
    Language.cn: 'zh',  # Default for cn, but will be overridden by language detection
}


def save_failed_files_log(failed_files: list[dict], output_path: Path) -> None:
    """Save failed files log to CSV."""
    if not failed_files:
        print("No failed files to log.")
        return

    fieldnames = ['filename', 'filepath', 'language', 'reason', 'error_message']

    # Ensure parent directories exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failed_files)

    print(f"Saved failed files log ({len(failed_files)} files) to {output_path}")


def extract_feature(featstr: Optional[str], feat_type: str) -> str:
    """Extract specific morphological feature from Stanza output."""
    feature = ''
    if featstr is not None:
        feature_list = featstr.split('|')
        for f in feature_list:
            if feat_type in f:
                if feat_type != 'Mood':
                    feature = f[len(feat_type) + 1:]
                else:
                    feature = f[len(feat_type) + 1:] + '_mood'
    return feature


def detect_language_for_transcript(transcript: Transcript, langid_pipeline) -> str:
    """
    Detect language for a transcript using Stanza's langid.
    Returns a Stanza language code.
    """
    # Get sample text from transcript
    sample_lines = transcript.participant_lines[:10] if transcript.participant_lines else transcript.lines[:100]
    sample_text = ' '.join([line.text for line in sample_lines if line.text.strip()])

    if not sample_text.strip():
        print(f"  No text found for language detection, defaulting to 'zh'")
        return 'zh'

    doc = langid_pipeline(sample_text)
    detected_lang = doc.lang

    # Map detected language to supported Stanza language
    if detected_lang in SUPPORTED_STANZA_LANGUAGES:
        return detected_lang
    else:
        print(f"  Detected language '{detected_lang}' not supported, defaulting to 'zh'")
        return 'zh'


def build_tag_feat_dict(tags_inputfile: str) -> dict[str, int]:
    """Build dictionary of feature tags initialized to zero."""
    tag_feat_dict = {}
    with open(tags_inputfile, 'r') as tags_infile:
        for tag in tags_infile:
            tag = tag.rstrip('\n')
            tag_feat_dict[tag] = 0
    return tag_feat_dict


def fill_tag_feat_slots(
        tag_feat_dict: dict[str, int],
        tags: list[list[str]],
        freq_statistics: dict
) -> dict:
    """Count occurrences of each linguistic feature."""
    tally_dict = {key: 0 for key in tag_feat_dict}

    # Count tags (skip lemma at index 0)
    if tags:
        for i in range(1, len(tags[0])):
            for tlist in tags:
                if tlist[i] != '' and tlist[i] in tally_dict:
                    tally_dict[tlist[i]] += 1

    # Add aggregate statistics
    for stat_name, stat_value in freq_statistics.items():
        tally_dict[stat_name] = stat_value

    return tally_dict


def save_tags(
        tally_tags_feat_dict: dict,
        speaker_role: str,
        output_file: Path
) -> None:
    """Save feature counts to TSV file."""
    if not tally_tags_feat_dict:
        print("No data to save.")
        return

    # Map speaker role to output format
    speaker_role_map = {
        'participant': 'Participant',
        'interviewer': 'Interviewer'
    }
    speaker_role_output = speaker_role_map.get(speaker_role, speaker_role)

    keys = list(tally_tags_feat_dict.keys())
    first_key = keys[0]

    # Get feature labels, renaming problematic ones
    labels_original = list(tally_tags_feat_dict[first_key].keys())
    labels_renamed = []
    for label in labels_original:
        if label == '1':
            label = 'p1'
        elif label == '2':
            label = 'p2'
        elif label == '3':
            label = 'p3'
        elif label == 'Yes':
            label = 'pronoun_possession'
        labels_renamed.append(label)

    # Exclude: num_sent, num_words, word_freq, file_name
    non_feature_labels = {'num_sent', 'num_words', 'word_freq', 'file_name'}

    header = [
                 'network', 'language', 'src_subject_id', 'interview_type',
                 'day', 'interview_number', 'transcript_speaker_label', 'speaker_role'
             ] + labels_renamed[:-4] + ['num_sent', 'num_words', 'word_freq', 'file_name.txt']

    # Ensure parent directories exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as outfile:
        outfile.write('\t'.join(header) + '\n')

        for key, features in tally_tags_feat_dict.items():
            # Parse key: site_patient_id_language_transcript_type_day_session
            parts = key.split('_')
            site = parts[0]
            patient_id = parts[1]
            language_code = parts[2]
            transcript_type = parts[3]
            day = parts[4]
            session = parts[5]

            # Convert language code to full name
            try:
                language_name = Language[language_code].value
            except KeyError:
                language_name = language_code

            # Build row values (excluding the last 4: num_sent, num_words, word_freq, file_name)
            row_values = [str(features[label]) for label in labels_original[:-4]]

            # Get the trailing statistics
            num_sent = str(features['num_sent'])
            num_words = str(features['num_words'])
            word_freq = str(features['word_freq'])
            file_name = str(features['file_name'])

            row = [
                      site, language_name, patient_id, transcript_type,
                      day, session, '', speaker_role_output  # Empty string for transcript_speaker_label placeholder
                  ] + row_values + [num_sent, num_words, word_freq, file_name]

            outfile.write('\t'.join(row) + '\n')

    print(f"Saved output to {output_file}")


def process_transcript_lines(
        transcript: Transcript,
        nlp,
        tag_feat_dict: dict[str, int],
        speaker_role: str,
        lang_code: str,
        word_freq: Optional[float] = None
) -> tuple[Optional[dict], Optional[dict]]:
    """
    Process transcript lines for a specific speaker role.

    Args:
        transcript: Transcript object
        nlp: Stanza NLP pipeline
        tag_feat_dict: Feature dictionary
        speaker_role: 'participant' or 'interviewer'
        lang_code: Language code for error reporting
        word_freq: Pre-calculated word frequency (optional)

    Returns:
        Tuple of (tally_dict, error_dict) where one is None
    """
    try:
        # Check if this is a diary (no participant/interviewer labels)
        is_diary = any("diary" in part.lower() for part in transcript.filename.parts)

        # Get lines based on speaker role
        if is_diary:
            lines = transcript.lines
        elif speaker_role == "participant":
            lines = transcript.participant_lines
        elif speaker_role == "interviewer":
            lines = transcript.interviewer_lines
        else:
            raise ValueError(f"Invalid speaker role: {speaker_role}")

        if not lines:
            return None, {
                'filename': str(transcript.filename),
                'filepath': str(transcript.full_path),
                'language': lang_code,
                'reason': 'no_lines',
                'error_message': f"No {speaker_role} lines found"
            }

        tags = []
        num_words = 0
        num_sentences = 0

        # Process each line
        for transcript_line in lines:
            sentence_text = transcript_line.text
            if not sentence_text.strip():
                continue

            doc = nlp(sentence_text)
            for sent in doc.sentences:  # type: ignore
                num_sentences += 1
                for word in sent.words:
                    case = extract_feature(word.feats, 'Case')
                    number = extract_feature(word.feats, 'Number')
                    person = extract_feature(word.feats, 'Person')
                    gender = extract_feature(word.feats, 'Gender')
                    prontype = extract_feature(word.feats, 'PronType')
                    definite = extract_feature(word.feats, 'Definite')
                    mood = extract_feature(word.feats, 'Mood')
                    tense = extract_feature(word.feats, 'Tense')
                    verbform = extract_feature(word.feats, 'VerbForm')
                    poss = extract_feature(word.feats, 'Poss')
                    ntype = extract_feature(word.feats, 'NumType')

                    tags.append([
                        word.lemma, word.upos, word.xpos, word.deprel,
                        case, number, person, gender, prontype, definite,
                        mood, tense, verbform, poss, ntype
                    ])

                    num_words += 1

        # Check if processing returned nothing
        if num_words == 0 or num_sentences == 0:
            return None, {
                'filename': str(transcript.filename),
                'filepath': str(transcript.full_path),
                'language': lang_code,
                'reason': 'empty_output',
                'error_message': f"Extracted 0 sentences and 0 words"
            }

        # Use provided word frequency or set to NaN
        mean_word_freq = word_freq if word_freq is not None else np.nan

        # Build unique key for this transcript
        key = '_'.join([
            transcript.site or 'UNKNOWN',
            transcript.patient_id or 'UNKNOWN',
            transcript.language.name if transcript.language else 'UNKNOWN',
            transcript.transcript_type or 'UNKNOWN',
            transcript.day or 'UNKNOWN',
            transcript.session or 'UNKNOWN'
        ])

        freq_statistics = {
            'num_sent': num_sentences,
            'num_words': num_words,
            'word_freq': mean_word_freq,
            'file_name': str(transcript.filename)
        }

        tally_dict = fill_tag_feat_slots(tag_feat_dict, tags, freq_statistics)

        return tally_dict, None

    except Exception as e:
        return None, {
            'filename': str(transcript.filename),
            'filepath': str(transcript.full_path),
            'language': lang_code,
            'reason': 'processing_error',
            'error_message': str(e)
        }


def save_tags_combined(
        tally_tags_feat_dict_by_speaker: dict[str, dict],
        output_file: Path
) -> None:
    """
    Save feature counts for multiple speaker roles to a single TSV file.

    Args:
        tally_tags_feat_dict_by_speaker: Dict mapping speaker role to tally dict
            e.g., {'participant': {...}, 'interviewer': {...}}
        output_file: Output TSV file path
    """
    if not tally_tags_feat_dict_by_speaker:
        print("No data to save.")
        return

    # Map speaker role to output format
    speaker_role_map = {
        'participant': 'Participant',
        'interviewer': 'Interviewer'
    }

    # Get the first available speaker's first key to determine structure
    first_speaker_role = list(tally_tags_feat_dict_by_speaker.keys())[0]
    first_tally_dict = tally_tags_feat_dict_by_speaker[first_speaker_role]

    if not first_tally_dict:
        print("No data to save.")
        return

    first_key = list(first_tally_dict.keys())[0]

    # Get feature labels, renaming problematic ones
    labels_original = list(first_tally_dict[first_key].keys())
    labels_renamed = []
    for label in labels_original:
        if label == '1':
            label = 'p1'
        elif label == '2':
            label = 'p2'
        elif label == '3':
            label = 'p3'
        elif label == 'Yes':
            label = 'pronoun_possession'
        labels_renamed.append(label)

    header = [
                 'network', 'language', 'src_subject_id', 'interview_type',
                 'day', 'interview_number', 'transcript_speaker_label', 'speaker_role'
             ] + labels_renamed[:-4] + ['num_sent', 'num_words', 'word_freq', 'file_name.txt']

    # Ensure parent directories exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as outfile:
        outfile.write('\t'.join(header) + '\n')

        # Iterate through each speaker role
        for speaker_role, tally_dict in tally_tags_feat_dict_by_speaker.items():
            speaker_role_output = speaker_role_map.get(speaker_role, speaker_role)

            for key, features in tally_dict.items():
                # Parse key: site_patient_id_language_transcript_type_day_session
                parts = key.split('_')
                site = parts[0]
                patient_id = parts[1]
                language_code = parts[2]
                transcript_type = parts[3]
                day = parts[4]
                session = parts[5]

                # Convert language code to full name
                try:
                    language_name = Language[language_code].value
                except KeyError:
                    language_name = language_code

                # Build row values
                row_values = [str(features[label]) for label in labels_original[:-4]]

                # Get trailing statistics
                num_sent = str(features['num_sent'])
                num_words = str(features['num_words'])
                word_freq = str(features['word_freq'])
                file_name = str(features['file_name'])

                row = [
                          site, language_name, patient_id, transcript_type,
                          day, session, '', speaker_role_output
                      ] + row_values + [num_sent, num_words, word_freq, file_name]

                outfile.write('\t'.join(row) + '\n')

    print(f"Saved combined output to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tally morphosyntactic features from transcripts using Stanza.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--i", type=str, required=True,
                        help="Input directory containing transcript files")
    parser.add_argument("--o", type=str, required=True,
                        help="Output TSV file path")
    parser.add_argument("--failed_log", type=str, required=False, default=None,
                        help="Output CSV file path for failed files log (optional)")
    parser.add_argument("--feats", type=str, required=True,
                        help="Path to feature list file (tags_upos_xpos.txt)")
    parser.add_argument("--speaker", type=str, default="participant",
                        choices=['participant', 'interviewer'],
                        help="Speaker role to analyze")
    parser.add_argument("--gpu", type=int, required=True,
                        help="GPU device ID to use")
    parser.add_argument("--batch_size", type=int, default=400,
                        help="Batch size for Stanza dependency parsing")
    parser.add_argument("--slice", type=int, default=None,
                        help="Slice size for testing small batches of transcripts (per language)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    input_dir = Path(args.i)
    output_file = Path(args.o)
    failed_log_path = Path(args.failed_log) if args.failed_log else None
    feature_list_path = Path(args.feats)

    # Track failed files
    failed_files: list[dict] = []

    # Build feature dictionary
    tag_feat_dict = build_tag_feat_dict(feature_list_path)
    print(f"Loaded {len(tag_feat_dict)} feature tags")

    # Set transcript directory and collect transcripts
    Transcript.set_directory_path(input_dir)
    all_transcripts = Transcript.list_transcripts()
    print(f"Found {len(all_transcripts)} total transcripts")

    # Group transcripts by language (Stanza code)
    transcripts_by_lang: dict[str, list] = defaultdict(list)
    cn_transcripts: list = []

    for t in all_transcripts:
        if t.language is None:
            print(f"  Skipping transcript with unknown language: {t.filename}")
            failed_files.append({
                'filename': str(t.filename),
                'filepath': str(t.full_path),
                'language': 'UNKNOWN',
                'reason': 'unknown_language',
                'error_message': 'Transcript language could not be determined'
            })
            continue
        if t.language == Language.cn:
            cn_transcripts.append(t)
        elif t.language.name in SUPPORTED_STANZA_LANGUAGES:
            stanza_code = LANG_TO_STANZA.get(t.language)
            if stanza_code:
                transcripts_by_lang[stanza_code].append(t)
        else:
            print(f"  Skipping unsupported language {t.language.name}: {t.filename}")
            failed_files.append({
                'filename': str(t.filename),
                'filepath': str(t.full_path),
                'language': t.language.name,
                'reason': 'unsupported_language',
                'error_message': f"Language '{t.language.name}' is not supported by Stanza"
            })

    # Handle cn transcripts with language detection
    if cn_transcripts:
        print(f"\nDetecting languages for {len(cn_transcripts)} 'cn' transcripts...")
        langid_pipeline = stanza.Pipeline(lang='multilingual', processors='langid', use_gpu=True)

        for t in cn_transcripts:
            detected_lang = detect_language_for_transcript(t, langid_pipeline)
            print(f"  {t.filename} -> detected '{detected_lang}'")
            transcripts_by_lang[detected_lang].append(t)

        del langid_pipeline  # Free memory

    # Print summary
    print("\nTranscripts by language:")
    for lang_code, trans_list in sorted(transcripts_by_lang.items()):
        print(f"  {lang_code}: {len(trans_list)} transcripts")

    tally_tags_feat_dict = {}

    # Process each language group
    for lang_code in sorted(transcripts_by_lang.keys()):
        transcripts = transcripts_by_lang[lang_code]
        if args.slice:
            transcripts = transcripts[:args.slice]

        print(f"\n{'=' * 60}")
        print(f"Processing {len(transcripts)} transcripts for language: {lang_code}")
        print(f"{'=' * 60}")

        # Initialize Stanza pipeline for this language
        nlp = stanza.Pipeline(lang_code, depparse_batch_size=args.batch_size, use_gpu=True)

        for i, transcript in enumerate(transcripts):
            print(f"[{i + 1}/{len(transcripts)}] Processing: {transcript.filename}")

            try:
                lines = []
                # Check if this is a diary (no participant/interviewer labels)
                is_diary = False
                for part in transcript.filename.parts:
                    if "diary" in part.lower():
                        is_diary = True
                        break

                if is_diary:
                    lines = transcript.lines
                elif args.speaker == "participant":
                    lines = transcript.participant_lines
                else:
                    # lines = transcript.interviewer_lines if hasattr(transcript, 'interviewer_lines') else []
                    print("Interviewer lines not yet tested - exiting")
                    exit(1)

                if not lines:
                    print(f"  No {args.speaker} lines found, skipping.")
                    failed_files.append({
                        'filename': str(transcript.filename),
                        'filepath': str(transcript.full_path),
                        'language': lang_code,
                        'reason': 'no_lines',
                        'error_message': f"No {args.speaker} lines found"
                    })
                    continue

                tags = []
                num_words = 0
                num_sentences = 0
                word_freq_list = []

                # Process each line
                for transcript_line in lines:
                    sentence_text = transcript_line.text
                    if not sentence_text.strip():
                        continue

                    doc = nlp(sentence_text)
                    for sent in doc.sentences:  # type: ignore
                        num_sentences += 1
                        for word in sent.words:
                            case = extract_feature(word.feats, 'Case')
                            number = extract_feature(word.feats, 'Number')
                            person = extract_feature(word.feats, 'Person')
                            gender = extract_feature(word.feats, 'Gender')
                            prontype = extract_feature(word.feats, 'PronType')
                            definite = extract_feature(word.feats, 'Definite')
                            mood = extract_feature(word.feats, 'Mood')
                            tense = extract_feature(word.feats, 'Tense')
                            verbform = extract_feature(word.feats, 'VerbForm')
                            poss = extract_feature(word.feats, 'Poss')
                            ntype = extract_feature(word.feats, 'NumType')

                            tags.append([
                                word.lemma, word.upos, word.xpos, word.deprel,
                                case, number, person, gender, prontype, definite,
                                mood, tense, verbform, poss, ntype
                            ])

                            word_freq_list = determine_freqs(word, wordfreqs, word_freq_list)
                            num_words += 1

                # Check if processing returned nothing
                if num_words == 0 or num_sentences == 0:
                    print(f"  No words/sentences extracted, skipping.")
                    failed_files.append({
                        'filename': str(transcript.filename),
                        'filepath': str(transcript.full_path),
                        'language': lang_code,
                        'reason': 'empty_output',
                        'error_message': f"Extracted 0 sentences and 0 words"
                    })
                    continue

                # Calculate mean word frequency
                if word_freq_list:
                    mean_word_freq = np.array(word_freq_list).mean()
                else:
                    mean_word_freq = np.nan

                # Build unique key for this transcript
                key = '_'.join([
                    transcript.site or 'UNKNOWN',
                    transcript.patient_id or 'UNKNOWN',
                    transcript.language.name if transcript.language else 'UNKNOWN',
                    transcript.transcript_type or 'UNKNOWN',
                    transcript.day or 'UNKNOWN',
                    transcript.session or 'UNKNOWN'
                ])

                freq_statistics = {
                    'num_sent': num_sentences,
                    'num_words': num_words,
                    'word_freq': mean_word_freq,
                    'file_name': str(transcript.filename)
                }

                tally_tags_feat_dict[key] = fill_tag_feat_slots(
                    tag_feat_dict, tags, freq_statistics
                )

                print(f"  Processed {num_sentences} sentences, {num_words} words")

            except Exception as e:
                print(f"  ERROR: {str(e)}")
                failed_files.append({
                    'filename': str(transcript.filename),
                    'filepath': str(transcript.full_path),
                    'language': lang_code,
                    'reason': 'processing_error',
                    'error_message': str(e)
                })

        # Free memory before loading next language pipeline
        del nlp

    # Save results
    save_tags(tally_tags_feat_dict, args.speaker, output_file)

    # Save failed files log
    if failed_log_path:
        save_failed_files_log(failed_files, failed_log_path)
    elif failed_files:
        print(f"\nWarning: {len(failed_files)} files failed but no --failed_log path specified.")


if __name__ == "__main__":
    main()