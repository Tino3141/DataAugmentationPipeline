import numpy as np
import librosa
import random
from typing import Tuple, List, Optional, Dict
import os

class MusicHandler:
    def __init__(self, sample_rate: int = 48000, crossfade_duration: float = 2.0):
        """
        Initialize MusicHandler with configuration parameters.
        
        Args:
            sample_rate (int): Target sample rate for audio processing
            crossfade_duration (float): Duration of crossfade between music segments in seconds
        """
        self.sample_rate = sample_rate
        self.crossfade_duration = crossfade_duration
        self.music_cache = {}  # Cache for loaded music files
        
    def load_music(self, music_path: str) -> Tuple[np.ndarray, int]:
        """
        Load and preprocess a music file.
        
        Args:
            music_path (str): Path to the music file
            
        Returns:
            Tuple[np.ndarray, int]: Processed music and its sample rate
        """
        # Check if already in cache
        if music_path in self.music_cache:
            return self.music_cache[music_path]
            
        # Load music
        music, sr = librosa.load(music_path, sr=self.sample_rate)
        
        # Resample if necessary
        if sr != self.sample_rate:
            music = librosa.resample(music, orig_sr=sr, target_sr=self.sample_rate)
            
        # Cache the result
        self.music_cache[music_path] = (music, self.sample_rate)
        
        return music, self.sample_rate
    
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
        music_path: str,
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
        music, _ = self.load_music(music_path)
        
        # Loop music if needed
        if loop_music and len(music) < len(audio):
            music = self.loop_music(music, len(audio))
        elif len(music) > len(audio):
            # Trim music if longer than audio
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
        music_dir: str,
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
        # Get list of music files
        music_files = [f for f in os.listdir(music_dir) 
                      if f.endswith(('.wav', '.mp3', '.ogg', '.flac'))]
        
        if not music_files:
            return audio
            
        # Select random music file
        music_file = random.choice(music_files)
        music_path = os.path.join(music_dir, music_file)
        
        # Mix music with audio
        return self.mix_music_with_audio(
            audio=audio,
            music_path=music_path,
            music_volume=music_volume,
            loop_music=loop_music
        ) 