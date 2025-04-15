import os
import librosa
import soundfile as sf
import numpy as np
import librosa
import random
from typing import Tuple, List, Optional
import os
from components.AudioEffects import AudioEffects

def main():
    # Configuration
    SAMPLE_RATE = 48000
    MAX_SOUND_EFFECT_LENGTH = 2.0  # seconds
    COVERAGE = 0.3  # 30% chance of adding sound effect at each position
    MIN_GAP = 1.0  # minimum gap between sound effects in seconds
    
    # Paths
    input_audio_path = "samples/output/output.wav"
    sound_effects_dir = "soundEffects"
    output_dir = "samples/output"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize audio effects processor
    audio_effects = AudioEffects(
        sample_rate=SAMPLE_RATE,
        max_sound_effect_length=MAX_SOUND_EFFECT_LENGTH
    )
    
    # Load input audio
    print("Loading input audio...")
    audio, sr = librosa.load(input_audio_path, sr=SAMPLE_RATE)
    
    # Apply sound effects
    print("Applying sound effects...")
    processed_audio = audio_effects.apply_sound_effects(
        audio=audio,
        sound_effects_dir=sound_effects_dir,
        coverage=COVERAGE,
        min_gap=MIN_GAP
    )
    
    # Save processed audio
    output_path = os.path.join(output_dir, "output_with_effects.wav")
    print(f"Saving processed audio to {output_path}...")
    sf.write(output_path, processed_audio, SAMPLE_RATE)
    
    print("Done!")

if __name__ == "__main__":
    main() 