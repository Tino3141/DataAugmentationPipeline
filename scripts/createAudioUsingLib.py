import json
import random
import numpy as np
from pydub import AudioSegment
import os
from components.AudioConversation import AudioConversation
from components.AudioEffects import AudioEffects
from components.Dataloaders import Dataloader
from components.MusicHandler import MusicHandler
import librosa
import soundfile as sf

if __name__ == "__main__":
    dataloader = Dataloader(
        sound_effects_path="/Users/constantinpinkl/Downloads/Emilia/sfxSound",
        background_music_path="/Users/constantinpinkl/Downloads/Emilia/mtgJamendo",
        speech_samples_path="/Users/constantinpinkl/Downloads/Emilia/Emilia"
    )

    SAMPLE_RATE = 48000
    MAX_SOUND_EFFECT_LENGTH = 2.0 
    COVERAGE = 0.3  # 30% chance of adding sound effect at each position
    MIN_GAP = 1.0 
    print("########################")
    print("Init Dataloader")
    audio_conversation = AudioConversation(dataloader)
    print("########################")
    speakers = dataloader.get_random_speakers(2)
    print(speakers)

    # Get segments for speaker
    segments = audio_conversation.getSegmentsForSpeaker(speakers[0]['key'])
    print("Segements for Speaker: ",len(segments))
    print(segments[0])

    # Arrage Segments
    segments = audio_conversation.arrangeSegments(speakers, 10)
    # Apply Gaussian Gap
    segments = audio_conversation.applyGaussianGap(segments, 0, 0.75)  # Mean, Std
    
    # Create output directory if it doesn't exist
    output_dir = "output_segments"
    os.makedirs(output_dir, exist_ok=True)
    print(segments)
    # Write segments to file
    
    audio_conversation.createAudio(segments, "output_segments/output.wav")

    # Write segments to file - exclude audio parameter as it can't be serialized
    serializable_segments = []
    for segment in segments:
        # Create a copy of the segment without the audio field
        serializable_segment = segment.copy()
        if 'audio' in serializable_segment:
            del serializable_segment['audio']
        serializable_segments.append(serializable_segment)
    
    # Write the serializable segments to a JSON file
    with open(os.path.join(output_dir, "segments.json"), 'w', encoding='utf-8') as f:
        json.dump(serializable_segments, f, ensure_ascii=False, indent=4)
    
    print(f"Segments saved to {os.path.join(output_dir, 'segments.json')}")

    # Add background music
    music_handler = MusicHandler(dataloader)
    # load the audio
    audio, sr = librosa.load("output_segments/output.wav", sr=48000)
    audio = music_handler.add_background_music(audio)
    # print(len(audio))
    # Save the audio 
    sf.write("output_segments/output_with_music.wav", audio, sr)
    print("Audio with background music saved to output_segments/output_with_music.wav")

    # Add Sound Effects
    audio_effects = AudioEffects(
        dataloader=dataloader,
        sample_rate=SAMPLE_RATE,
        max_sound_effect_length=MAX_SOUND_EFFECT_LENGTH
    )
    print("Applying sound effects...")
    processed_audio = audio_effects.apply_sound_effects(
        audio=audio,
        coverage=COVERAGE,
        min_gap=MIN_GAP
    )
    # Save processed audio
    output_path = os.path.join("output_segments", "output_with_music_effects.wav")
    sf.write(output_path, processed_audio, sr)
    print(f"Audio with sound effects saved to {output_path}")
    print("Done!")