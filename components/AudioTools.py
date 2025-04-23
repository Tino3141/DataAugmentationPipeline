# Author: Constantin Pinkl
# Date: 28/03/2025
# Description: AudioTools

import librosa as lr
import numpy as np
from typing import Optional, List
import pyloudnorm as pyln

class AudioTools:
    """
    AudioTools class for loading and processing audio files.
    """
    def __init__(self):
        pass

    def load_audio(self, file_path: str, sr: int = 16000, mono: bool = True) -> tuple:
        """
        Load an audio file and resample it to the specified sample rate.
        
        Args:
            file_path (str): Path to the audio file
            sr (int): Sample rate for loading the audio
        
        Returns:
            tuple: Tuple containing the audio signal and its sample rate
        """
        audio, sr = lr.load(file_path, sr=sr, mono=mono)
        return audio, sr
    
    def save_audio(self, file_path: str, audio: np.ndarray, sr: int = 16000) -> None:
        """
        Save an audio signal to a file.
        
        Args:
            file_path (str): Path to save the audio file
            audio (np.ndarray): Audio signal to save
            sr (int): Sample rate for saving the audio
        """
        lr.output.write_wav(file_path, audio, sr)

    def resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample an audio signal to a different sample rate.
        
        Args:
            audio (np.ndarray): Audio signal to resample
            orig_sr (int): Original sample rate of the audio
            target_sr (int): Target sample rate for resampling
        
        Returns:
            np.ndarray: Resampled audio signal
        """
        return lr.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    
    def trim_audio(self, audio: np.ndarray, top_db: int = 20) -> np.ndarray:
        """
        Trim silence from the beginning and end of an audio signal.
        
        Args:
            audio (np.ndarray): Audio signal to trim
            top_db (int): Threshold for trimming silence
        Returns:
            np.ndarray: Trimmed audio signal
        """ 
        return lr.effects.trim(audio, top_db=top_db)[0]

    def adjust_loudness(self, audio: np.ndarray, sr: int, target_loudness: float, allow_variance: bool = False) -> np.ndarray:
        """
        Adjust the loudness of an audio signal to a target level in decibels (dB).
        
        Args:
            audio (np.ndarray): Audio signal to adjust.
            sr (int): Sample rate of the audio signal.
            target_loudness (float): Desired loudness in dB.
            allow_variance (bool): If True, allows variance in loudness adjustment.
        
        Returns:
            np.ndarray: Loudness-adjusted audio signal.
        """

        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio)
        if allow_variance:
            # Calculate the variance in loudness
            loudness_variance = np.std(audio)
            # Adjust the target loudness based on variance
            target_loudness += loudness_variance
        return pyln.normalize.loudness(audio, loudness, target_loudness)
       
    