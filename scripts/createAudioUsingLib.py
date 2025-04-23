import json
import random
import numpy as np
from pydub import AudioSegment
import os
from components.AudioConversation import AudioConversation
from components.Dataloaders import Dataloader

if __name__ == "__main__":
    dataloader = Dataloader(
        sound_effects_path="/Users/constantinpinkl/Downloads/Emilia/sfxSound",
        background_music_path="/Users/constantinpinkl/Downloads/Emilia/mtgJamendo",
        speech_samples_path="/Users/constantinpinkl/Downloads/Emilia/Emilia"
    )
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
