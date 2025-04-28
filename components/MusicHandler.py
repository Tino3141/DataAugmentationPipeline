import numpy as np
import librosa
import random
from typing import Tuple, List, Optional, Dict
import os
from components.Dataloaders import Dataloader

class MusicHandler:
    def __init__(self, dataloader: Dataloader, sample_rate: int = 48000, crossfade_duration: float = 2.0):
        """
        Initialize MusicHandler with configuration parameters.
        
        Args:
            sample_rate (int): Target sample rate for audio processing
            crossfade_duration (float): Duration of crossfade between music segments in seconds
        """
        self.sample_rate = sample_rate
        self.crossfade_duration = crossfade_duration
        self.music_cache = {}  # Cache for loaded music files
        self.dataloader = dataloader

    def load_music(self) -> Tuple[np.ndarray, int]:
        """
        Load and preprocess a music file.
        
        Args:
            music_path (str): Path to the music file
            
        Returns:
            Tuple[np.ndarray, int]: Processed music and its sample rate
        """
        # Get random music file from dataloade
        music = self.dataloader.get_random_music()
        return music["opus"]["array"], music["opus"]["sampling_rate"]
    
    def apply_crossfade(self, audio1: np.ndarray, audio2: np.ndarray, 
                        position: int, fade_length: int) -> np.ndarray:
        """
        Apply crossfade between two audio segments.
        
        Args:
            audio1 (np.ndarray): First audio segment
            audio2 (np.ndarray): Second audio segment
            position (int): Position where crossfade starts
            fade_length (int): Length of crossfade in samples
            
        Returns:
            np.ndarray: Combined audio with crossfade
        """
        # Create fade curves
        fade_out = np.linspace(1, 0, fade_length)
        fade_in = np.linspace(0, 1, fade_length)
        
        # Apply crossfade
        result = audio1.copy()
        result[position:position + fade_length] = (
            audio1[position:position + fade_length] * fade_out +
            audio2[:fade_length] * fade_in
        )
        
        # Add the rest of the second audio
        if len(audio2) > fade_length:
            result = np.concatenate([result, audio2[fade_length:]])
            
        return result
    
    def loop_music(self, music: np.ndarray, target_length: int) -> np.ndarray:
        """
        Loop music to reach target length with crossfades.
        
        Args:
            music (np.ndarray): Music to loop
            target_length (int): Target length in samples
            
        Returns:
            np.ndarray: Looped music
        """
        if len(music) >= target_length:
            return music[:target_length]
            
        # Calculate number of loops needed
        num_loops = int(np.ceil(target_length / len(music)))
        
        # Create array of appropriate size
        result = np.zeros(target_length)
        
        # Fill with music loops
        position = 0
        fade_length = int(self.crossfade_duration * self.sample_rate)
        
        for i in range(num_loops):
            # Calculate how much of this loop to use
            remaining = target_length - position
            if remaining <= 0:
                break
                
            # If this is the first loop, just copy
            if i == 0:
                copy_length = min(len(music), remaining)
                result[position:position + copy_length] = music[:copy_length]
            else:
                # Apply crossfade with previous loop
                copy_length = min(len(music), remaining)
                result = self.apply_crossfade(
                    result, music[:copy_length], position, fade_length
                )
                
            position += copy_length
            
        return result
    
    def mix_music_with_audio(
        self,
        audio: np.ndarray,
        music_volume: float = 0.2,
        loop_music: bool = True
    ) -> np.ndarray:
        """
        Mix background music with the main audio.
        
        Args:
            audio (np.ndarray): Main audio signal
            music_path (str): Path to the music file
            music_volume (float): Volume level for music (0.0 to 1.0)
            loop_music (bool): Whether to loop music if shorter than audio
            
        Returns:
            np.ndarray: Combined audio with background music
        """
        # Load music
        music, _ = self.load_music()
        
        # Adjust music length to match audio
        if len(music) < len(audio):
            if loop_music:
                music = self.loop_music(music, len(audio))
            else:
                pad_length = len(audio) - len(music)
                music = np.pad(music, (0, pad_length), mode='constant')
        else:
            music = music[:len(audio)]
            
        # Apply volume adjustment
        music = music * music_volume
        
        # Mix with main audio
        result = audio + music
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(result))
        if max_val > 1.0:
            result = result / max_val
            
        return result
    
    def add_background_music(
        self,
        audio: np.ndarray,
        music_volume: float = 0.2,
        loop_music: bool = True
    ) -> np.ndarray:
        """
        Add background music from a directory to the audio.
        
        Args:
            audio (np.ndarray): Main audio signal
            music_dir (str): Directory containing music files
            music_volume (float): Volume level for music (0.0 to 1.0)
            loop_music (bool): Whether to loop music if shorter than audio
            
        Returns:
            np.ndarray: Audio with background music
        """        
        
        # Mix music with audio
        return self.mix_music_with_audio(
            audio=audio,
            music_volume=music_volume,
            loop_music=loop_music
        ) 