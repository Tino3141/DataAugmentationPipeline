import json
import random
import numpy as np
from pydub import AudioSegment
import os
from typing import List


class AudioConversation:
    def __init__(self, sample_rate: int = 48000, languages: List[str] = ["DE", "EN"]):
        self.LANGUAGES = languages
        self.SAMPLE_RATE = sample_rate

    def pickSpeakers(self,dict, num_speakers):
        available_keys = []
        for language in self.LANGUAGES:
            for key in dict[language].keys():
                available_keys.append({
                    'language': language,
                    'key': key,
                    'counter': 0
                })
        selected_segments = random.sample(available_keys, num_speakers)

        return selected_segments

    def arrangeSegments(self, speakers, dict, num_segments):
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
                "file_name": segment['file_name'],
                "start": 0 if i == 0 else segments[i-1]["end"],
                "end": segment['duration'] if i == 0 else segments[i-1]["end"] + segment['duration']
            })
            speaker['counter'] += 1

        return segments

    # Apply Gap between segments using Gaussian Distribution
    def applyGaussianGap(self, segments, mean, std):
        for i in range(1, len(segments)):
            gap = np.random.normal(mean, std)
            # If same speaker, ensure gap is positive
            if segments[i]["id"] == segments[i-1]["id"]:
                gap = abs(gap)
            if gap < 0:
                print("Gap is negative: ", gap)
            segments[i]["start"] = 0 + gap if i == 0 else segments[i-1]["end"] + gap
            segments[i]["end"] = segments[i]["end"] - segments[i]["start"] + gap if i == 0 else segments[i-1]["end"] + gap + segments[i]["end"] - segments[i]["start"]
        return segments

    # Create Audio from segments
    def createAudio(self, segments, audio_root, output_path):
        # Create empty audio segment
        # Get total duration from last segment's end time
        total_duration_ms = segments[-1]['end'] * 1000  # Convert to milliseconds
        final_audio = AudioSegment.silent(duration=total_duration_ms)
        
        # Add each segment
        for segment in segments:
            # Load audio file
            audio_path = f"{audio_root}/{segment['language']}/{segment['file_name']}"
            audio = AudioSegment.from_file(audio_path)
            
            # Extract segment portion
            start_ms = segment['start'] * 1000  # Convert to milliseconds
            end_ms = segment['end'] * 1000
            segment_audio = audio
            
            # Add silence before segment if needed
            if len(final_audio) < start_ms:
                silence_duration = start_ms - len(final_audio)
                final_audio = final_audio.overlay(AudioSegment.silent(duration=silence_duration), position=len(final_audio))
                
            # Overlay segment at the correct position
            final_audio = final_audio.overlay(segment_audio, position=start_ms)
        # Export final audio
        final_audio.export(output_path, format="wav")


    def augmentAudio(self, segmentDict, num_speakers, num_segments, output_dir, audio_root, mean=0, std=0.75):
        selected_segments = self.pickSpeakers(segmentDict, num_speakers)
        # Arrange Segments
        segments = self.arrangeSegments(selected_segments, segmentDict, num_segments)

        # Apply Gaussian Gap
        segments = self.applyGaussianGap(segments, mean, std) # Mean, Std

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create output path
        output_path = os.path.join(output_dir, "output.wav")
        self.createAudio(segments, audio_root, output_path)

        # Write segments to file
        with open(os.path.join(output_dir, "segments.json"), 'w', encoding='utf-8') as f:
            json.dump(segments, f, ensure_ascii=False, indent=4)

        return segments
