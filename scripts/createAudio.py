import json
import random
import numpy as np
from pydub import AudioSegment
import os
from components.AudioConversation import AudioConversation

if __name__ == "__main__":
    # Load dictionary
    with open('/Users/constantinpinkl/University/Research/SpeakerDiarization/Datasets/DataAugmentationPipeline/scripts/dictionary.json', 'r', encoding='utf-8') as f:
        segmentDict = json.load(f)
        
    # Get all languages
    languages = list(segmentDict.keys())
    print("All Language: ", languages)
    conversation = AudioConversation()

    print("Selected Languages: ", conversation.LANGUAGES)

    # Extract inputs to variables
    num_speakers = 3
    num_segments = 10
    output_dir = "/Users/constantinpinkl/University/Research/SpeakerDiarization/Datasets/DataAugmentationPipeline/samples/output2"
    audio_root = "/Users/constantinpinkl/University/Research/SpeakerDiarization/Datasets/DataAugmentationPipeline/speech"
    gap_mean = 0
    gap_std = 0.75

    conversation.augmentAudio(segmentDict, num_speakers, num_segments, output_dir, audio_root, gap_mean, gap_std)