# Author: Constantin Pinkl
# Date: 23/05/2025
# Description: Dataloaders

import librosa as lr
import numpy as np
from datasets import load_dataset

class Dataloader:
    def __init__(self, sound_effects_path: str, background_music_path: str, speech_samples_path: str):
        self._load_sound_effects(sound_effects_path)
        self._load_background_music(background_music_path)
        self._load_speech_samples(speech_samples_path)

        # Convert to list and print the result
        unique_speakers = set()
        for item in self.speech_samples["train"]["json"]:
            unique_speakers.add(item["speaker"])
        self.unique_speakers_list = list(unique_speakers)

        self.length_music = len(self.background_music["train"])
        self.length_sfx = len(self.sound_effects["train"])
        self.length_speech = len(self.speech_samples["train"])

    def get_segementsForSpeaker(self, speaker: str):
        filtered_items = [(idx, item) for idx, item in enumerate(self.speech_samples["train"]["json"]) if item["speaker"] == speaker]
        filtered_items = [self.speech_samples["train"][item[0]] for item in filtered_items]
        return filtered_items

    def get_random_speakers(self, num_speakers: int):
        selected_speakers = np.random.choice(self.unique_speakers_list, size=min(num_speakers, len(self.unique_speakers_list)), replace=False)
        return [{'language': speaker.split('_')[0], 'key': speaker, 'counter': 0} for speaker in selected_speakers]

    def get_random_music(self):
        random_index = np.random.randint(0, self.length_music)
        return self.background_music["train"][random_index]
    
    def get_random_sound_effect(self):
        random_index = np.random.randint(0, self.length_sfx)
        return self.sound_effects["train"][random_index][list(self.sound_effects["train"][random_index].keys())[0]]["array"]
    
    def _load_sound_effects(self, dataset_path: str):
        self.sound_effects = load_dataset(dataset_path)

    def _load_background_music(self, dataset_path: str):
        self.background_music = load_dataset(dataset_path)

    def _load_speech_samples(self, dataset_path: str):
        self.speech_samples = load_dataset(dataset_path)

class ConversationDataloader:
    def __init__(self, path:str):
        pass

    

