import os
import librosa
import soundfile as sf
import numpy as np
import librosa
import random
from typing import Tuple, List, Optional
import os

class AudioEffects:
    def __init__(self, sample_rate: int = 48000, max_sound_effect_length: float = 2.0):
        """
        Initialize AudioEffects with configuration parameters.
        
        Args:
            sample_rate (int): Target sample rate for audio processing
            max_sound_effect_length (float): Maximum length of sound effects in seconds
        """
        self.sample_rate = sample_rate
        self.max_sound_effect_length = max_sound_effect_length
        
    def apply_fade(self, audio: np.ndarray, fade_duration: float = 0.1) -> np.ndarray:
        """
        Apply fade-in and fade-out to an audio segment.
        
        Args:
            audio (np.ndarray): Audio signal to process
            fade_duration (float): Duration of fade in seconds
            
        Returns:
            np.ndarray: Processed audio with fades applied
        """
        fade_length = int(fade_duration * self.sample_rate)
        
        # Create fade-in and fade-out curves
        fade_in = np.linspace(0, 1, fade_length)
        fade_out = np.linspace(1, 0, fade_length)
        
        # Apply fades
        audio[:fade_length] *= fade_in
        audio[-fade_length:] *= fade_out
        
        return audio
    
    def load_sound_effect(self, effect_path: str) -> Tuple[np.ndarray, int]:
        """
        Load and preprocess a sound effect.
        
        Args:
            effect_path (str): Path to the sound effect file
            
        Returns:
            Tuple[np.ndarray, int]: Processed sound effect and its sample rate
        """
        # Load sound effect
        effect, sr = librosa.load(effect_path, sr=self.sample_rate)
        
        # Resample if necessary
        if sr != self.sample_rate:
            effect = librosa.resample(effect, orig_sr=sr, target_sr=self.sample_rate)
        
        # Trim to maximum length
        max_samples = int(self.max_sound_effect_length * self.sample_rate)
        if len(effect) > max_samples:
            effect = effect[:max_samples]
            
        # Apply fades
        effect = self.apply_fade(effect)
        
        return effect, self.sample_rate
    
    def overlay_audio(self, base_audio: np.ndarray, effect: np.ndarray, position: int) -> np.ndarray:
        """
        Overlay a sound effect on the base audio at a specific position.
        
        Args:
            base_audio (np.ndarray): Base audio signal
            effect (np.ndarray): Sound effect to overlay
            position (int): Position in samples where to place the effect
            
        Returns:
            np.ndarray: Combined audio signal
        """
        # Ensure position is within bounds
        if position < 0:
            position = 0
        elif position + len(effect) > len(base_audio):
            # Trim effect if it would exceed audio length
            effect = effect[:len(base_audio) - position]
            
        # Create a copy of base audio
        result = base_audio.copy()
        
        # Overlay effect with crossfade
        result[position:position + len(effect)] += effect
        
        return result
    
    def apply_sound_effects(
        self,
        audio: np.ndarray,
        sound_effects_dir: str,
        coverage: float = 0.3,
        min_gap: float = 1.0
    ) -> np.ndarray:
        """
        Apply random sound effects to the audio with specified coverage.
        
        Args:
            audio (np.ndarray): Base audio signal
            sound_effects_dir (str): Directory containing sound effect files
            coverage (float): Probability of adding a sound effect at each position
            min_gap (float): Minimum gap between sound effects in seconds
            
        Returns:
            np.ndarray: Audio with sound effects applied
        """
        # Get list of sound effect files
        effect_files = [f for f in os.listdir(sound_effects_dir) 
                       if f.endswith(('.wav', '.mp3', '.ogg'))]
        
        if not effect_files:
            return audio
            
        # Calculate minimum gap in samples
        min_gap_samples = int(min_gap * self.sample_rate)
        
        # Initialize position
        position = 0
        
        # Process audio in chunks
        while position < len(audio):
            # Decide whether to add sound effect
            if random.random() < coverage:
                # Select random sound effect
                effect_file = random.choice(effect_files)
                effect_path = os.path.join(sound_effects_dir, effect_file)
                
                # Load and process effect
                effect, _ = self.load_sound_effect(effect_path)
                
                # Apply effect if there's enough space
                if position + len(effect) <= len(audio):
                    audio = self.overlay_audio(audio, effect, position)
                    
                # Move position forward
                position += len(effect) + min_gap_samples
            else:
                # Move to next potential position
                position += min_gap_samples
                
        return audio 
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