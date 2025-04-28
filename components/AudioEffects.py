import numpy as np
import librosa
import random
from typing import Tuple, List, Optional
import os
from components.Dataloaders import Dataloader

class AudioEffects:
    def __init__(self, dataloader: Dataloader, sample_rate: int = 48000, max_sound_effect_length: float = 2.0, effect_gain: float = 0.5):
        """
        Initialize AudioEffects with configuration parameters.
        
        Args:
            sample_rate (int): Target sample rate for audio processing
            max_sound_effect_length (float): Maximum length of sound effects in seconds
        """
        self.dataloader = dataloader
        self.sample_rate = sample_rate
        self.max_sound_effect_length = max_sound_effect_length
        # Gain factor for sound effects (linear scale)
        self.effect_gain = effect_gain
        
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
        # Limit fade_length to at most half the audio length to avoid broadcast errors
        fade_length = min(fade_length, len(audio) // 2)
        
        # Create fade-in and fade-out curves
        fade_in = np.linspace(0, 1, fade_length)
        fade_out = np.linspace(1, 0, fade_length)
        
        # Apply fades
        audio[:fade_length] *= fade_in
        audio[-fade_length:] *= fade_out
        
        return audio
    
    def load_sound_effect(self) -> Tuple[np.ndarray, int]:
        """
        Load and preprocess a sound effect.
        
        Args:
            effect_path (str): Path to the sound effect file
            
        Returns:
            Tuple[np.ndarray, int]: Processed sound effect and its sample rate
        """
        # Load sound effect
        effect = self.dataloader.get_random_sound_effect()
        # Resample if necessary
        # if sr != self.sample_rate:
        #     effect = librosa.resample(effect, orig_sr=sr, target_sr=self.sample_rate)
        
        # Trim to maximum length
        max_samples = int(self.max_sound_effect_length * self.sample_rate)
        if len(effect) > max_samples:
            effect = effect[:max_samples]
            
        # Apply fades
        effect = self.apply_fade(effect)
        # Adjust effect loudness
        effect = effect * self.effect_gain
        
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
        # Calculate minimum gap in samples
        min_gap_samples = int(min_gap * self.sample_rate)
        
        # Initialize position
        position = 0
        
        # Process audio in chunks
        while position < len(audio):
            # Decide whether to add sound effect
            if random.random() < coverage:
                # Load and process effect
                effect, _ = self.load_sound_effect()
                
                # Apply effect if there's enough space
                if position + len(effect) <= len(audio):
                    audio = self.overlay_audio(audio, effect, position)
                    
                # Move position forward
                position += len(effect) + min_gap_samples
            else:
                # Move to next potential position
                position += min_gap_samples
                
        return audio 