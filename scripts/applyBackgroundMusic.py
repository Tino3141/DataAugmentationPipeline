import os
import librosa
import soundfile as sf
import numpy as np
import librosa
import random
from typing import Tuple, List, Optional, Dict
import os
from components.MusicHandler import MusicHandler

def main():
    # Configuration
    SAMPLE_RATE = 48000
    CROSSFADE_DURATION = 2.0  # seconds
    MUSIC_VOLUME = 0.8  # 80% of original volume
    
    # Paths
    input_audio_path = "samples/output/output_with_effects.wav"
    music_dir = "music"
    output_dir = "samples/output"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize music handler
    music_handler = MusicHandler(
        sample_rate=SAMPLE_RATE,
        crossfade_duration=CROSSFADE_DURATION
    )
    
    # Load input audio
    print("Loading input audio...")
    audio, sr = librosa.load(input_audio_path, sr=SAMPLE_RATE)
    
    # Add background music
    print("Adding background music...")
    processed_audio = music_handler.add_background_music(
        audio=audio,
        music_dir=music_dir,
        music_volume=MUSIC_VOLUME,
        loop_music=True
    )
    
    # Save processed audio
    output_path = os.path.join(output_dir, "output_with_music.wav")
    print(f"Saving processed audio to {output_path}...")
    sf.write(output_path, processed_audio, SAMPLE_RATE)
    
    print("Done!")

if __name__ == "__main__":
    main() 