import json
import random
import numpy as np
from pydub import AudioSegment
import os
from typing import List
from components.Dataloaders import Dataloader


class AudioConversation:
    def __init__(self,  data: Dataloader, sample_rate: int = 48000, languages: List[str] = ["DE", "EN"]):
        self.LANGUAGES = languages
        self.SAMPLE_RATE = sample_rate
        self.data = data

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
    
    def getSegmentsForSpeaker(self, speaker: str):
        return self.data.get_segementsForSpeaker(speaker)

    # def arrangeSegments(self, speakers, num_segments):
    #     segments = [] # {"language": str, "id": str, "segment": str, "start": float, "end": float}
    #     for i in range(num_segments):
    #         # Pick random speaker with available segments
    #         # Try to find valid speaker
    #         valid_speaker = False
    #         for speaker in speakers:
    #             if speaker['counter'] < len(self.getSegmentsForSpeaker(speaker['key'])):
    #                 valid_speaker = True
    #                 break
            
    #         # Return segments if no valid speaker found
    #         if not valid_speaker:
    #             return segments
                
    #         # Pick random valid speaker
    #         valid_speakers = [s for s in speakers if s['counter'] < len(self.getSegmentsForSpeaker(s['key']))]
    #         speaker = random.choice(valid_speakers)
    #         # Take segment from speaker
    #         speaker_segments = self.getSegmentsForSpeaker(speaker['key'])
            
    #         segment = speaker_segments[speaker['counter']]
            
    #         segments.append({
    #             "language": speaker['language'],
    #             "id": speaker['key'],
    #             "segment": segment['json']['text'],
    #             "file_name": segment['mp3']['path'],
    #             "audio": segment["mp3"]["array"],
    #             "sampling_rate": segment["mp3"]["sampling_rate"],
    #             "start": 0 if i == 0 else segments[i-1]["end"],
    #             "end": segment['json']['duration'] if i == 0 else segments[i-1]["end"] + segment['json']['duration']
    #         })
    #         speaker['counter'] += 1

    #     return segments
    def arrangeSegments(self, speakers, num_segments):
        # 1) Build a dict of full segment lists once
        segments_by_key = {
            sp['key']: self.getSegmentsForSpeaker(sp['key'])
            for sp in speakers
        }
        # 2) Initialize counters in a separate dict
        counters = {sp['key']: 0 for sp in speakers}

        result = []
        for _ in range(num_segments):
            # 3) Filter speakers who still have segments left
            avail = [
                sp for sp in speakers
                if counters[sp['key']] < len(segments_by_key[sp['key']])
            ]
            if not avail:
                break

            # 4) Pick one at random
            speaker = random.choice(avail)
            key     = speaker['key']
            seglist = segments_by_key[key]
            idx     = counters[key]

            seg = seglist[idx]
            result.append({
                "language": speaker['language'],
                "id":       key,
                "segment":  seg['json']['text'],
                "file_name":seg['mp3']['path'],
                "audio":     seg['mp3']['array'],         # avoid copying if possible
                "sampling_rate": seg['mp3']['sampling_rate'],
                "start": result[-1]["end"] if result else 0,
                "end":   (result[-1]["end"] if result else 0) + seg['json']['duration']
            })
            counters[key] += 1

        return result

    # Apply Gap between segments using Gaussian Distribution
    def applyGaussianGap(self, segments, mean, std):
        for i in range(1, len(segments)):
            gap = np.random.normal(mean, std)
            # If same speaker, ensure gap is positive
            if segments[i]["id"] == segments[i-1]["id"]:
                gap = abs(gap)
            # if gap < 0:
            #     print("Gap is negative: ", gap)
            segments[i]["start"] = 0 + gap if i == 0 else segments[i-1]["end"] + gap
            segments[i]["end"] = segments[i]["end"] - segments[i]["start"] + gap if i == 0 else segments[i-1]["end"] + gap + segments[i]["end"] - segments[i]["start"]
        return segments

    # Create Audio from segments
    def createAudio(self, segments, output_path):
        # Create empty audio segment
        # Get total duration from last segment's end time
        total_duration_ms = segments[-1]['end'] * 1000  # Convert to milliseconds
        final_audio = AudioSegment.silent(duration=total_duration_ms)
        
        # Add each segment
        for segment in segments:
            # Use the audio data directly from the segment
            # The audio data is already loaded in the 'audio' field
            audio_data = np.int16(segment['audio'] * 32767)
            segment_audio = AudioSegment(
                data=audio_data.tobytes(),
                sample_width=2,  # Assuming 16-bit audio
                frame_rate=segment['sampling_rate'],  # Assuming 16kHz sample rate
                channels=1  # Assuming mono audio
            )

            # Calculate position in milliseconds
            start_ms = segment['start'] * 1000  # Convert to milliseconds
            # Overlay segment at the correct position
            final_audio = final_audio.overlay(segment_audio, position=start_ms)
        
        # Export final audio
        final_audio.export(output_path, format="wav")


    def augmentAudio(self, segmentDict, num_speakers, num_segments, output_dir, audio_root, mean=0, std=0.75):
        selected_segments = self.data.get_random_speakers(num_speakers)
        # Arrange Segments
        segments = self.arrangeSegments(selected_segments, segmentDict, num_segments)

        # Apply Gaussian Gap
        segments = self.applyGaussianGap(segments, mean, std) # Mean, Std

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create output path
        output_path = os.path.join(output_dir, "output.wav")
        self.createAudio(segments, output_path)

        serializable_segments = []
        for segment in segments:
            # Create a copy of the segment without the audio field
            serializable_segment = segment.copy()
            if 'audio' in serializable_segment:
                del serializable_segment['audio']
            serializable_segments.append(serializable_segment)

        # Write segments to file
        with open(os.path.join(output_dir, "segments.json"), 'w', encoding='utf-8') as f:
            json.dump(serializable_segments, f, ensure_ascii=False, indent=4)

        return segments
