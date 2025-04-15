import json
import random
import numpy as np
from pydub import AudioSegment

LANGUAGES = ["DE", "EN"]

def pickSpeakers(dict, num_speakers):
    available_keys = []
    for language in LANGUAGES:
        for key in dict[language].keys():
            available_keys.append({
                'language': language,
                'key': key,
                'counter': 0
            })
    selected_segments = random.sample(available_keys, num_speakers)

    return selected_segments

def arrangeSegments(speakers, dict, num_segments):
    segments = [] # {"language": str, "id": str, "segment": str, "start": float, "end": float}
    for i in range(num_segments):
        # Pick random speaker with available segments
        # Try to find valid speaker
        valid_speaker = False
        for speaker in speakers:
            if speaker['counter'] < len(dict[speaker['language']][speaker['key']]):
                valid_speaker = True
                break
        
        # Return segments if no valid speaker found
        if not valid_speaker:
            return segments
            
        # Pick random valid speaker
        valid_speakers = [s for s in speakers if s['counter'] < len(dict[s['language']][s['key']])]
        speaker = random.choice(valid_speakers)
        # Take segment from speaker
        segment = dict[speaker['language']][speaker['key']][speaker['counter']]
        segments.append({
            "language": speaker['language'],
            "id": speaker['key'],
            "segment": segment['segment'],
            "start": 0 if i == 0 else segments[i-1]["end"],
            "end": segment['duration'] if i == 0 else segments[i-1]["end"] + segment['duration']
        })
        speaker['counter'] += 1

    return segments

# Apply Gap between segments using Gaussian Distribution
def applyGaussianGap(segments, mean, std):
    for i in range(1, len(segments)):
        gap = np.random.normal(mean, std)
        # If same speaker, ensure gap is positive
        if segments[i]["id"] == segments[i-1]["id"]:
            gap = abs(gap)
        if gap < 0:
            print("Gap is negative: ", gap)
        segments[i]["start"] = segments[i-1]["end"] + gap
        segments[i]["end"] = segments[i]["start"] + (segments[i]["end"] - segments[i]["start"])
    return segments

def createAudio(segments, output_path):
    # Create empty audio segment
    final_audio = AudioSegment.empty()
    
    # Add each segment
    for segment in segments:
        # Load audio file
        audio_path = f"audio/{segment['language']}/{segment['id']}.wav"
        audio = AudioSegment.from_wav(audio_path)
        
        # Extract segment portion
        start_ms = segment['start'] * 1000  # Convert to milliseconds
        end_ms = segment['end'] * 1000
        segment_audio = audio[start_ms:end_ms]
        
        # Add silence before segment if needed
        if len(final_audio) < start_ms:
            silence_duration = start_ms - len(final_audio)
            final_audio += AudioSegment.silent(duration=silence_duration)
            
        # Append segment
        final_audio += segment_audio
        
    # Export final audio
    final_audio.export(output_path, format="wav")

    

if __name__ == "__main__":
    with open('dictionary.json', 'r', encoding='utf-8') as f:
        fileDict = json.load(f)
    
    # Get all languages
    languages = list(fileDict.keys())
    print("All Language: ", languages)
    print("Selected Languages: ", LANGUAGES)

    # Pick Speakers
    selected_segments = pickSpeakers(fileDict, 2)
    print("Selected Segments: ", selected_segments)

    # Arrange Segments
    segments = arrangeSegments(selected_segments, fileDict, 10)
    print("Segments: ", segments)

    # Apply Gaussian Gap
    segments = applyGaussianGap(segments, 0, 0.75) # Mean, Std
    print("Segments with Gap: ", segments)

    # Write segments to file
    with open('segments_gapped.json', 'w', encoding='utf-8') as f:
        json.dump(segments, f, ensure_ascii=False, indent=4)