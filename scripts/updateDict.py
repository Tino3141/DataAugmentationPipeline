import librosa as lr
import os
import numpy as np
from IPython.display import Audio
import librosa
import json
from tqdm import tqdm
def load_audioFilesFromDirBasic(basePath):
    files = os.listdir(basePath)
    files = [f for f in files if (f.endswith('.wav') or f.endswith('.mp3') or f.endswith('.ogg'))]
    return files

def load_audioFilesFromDir(basePath):
    files = os.listdir(basePath)
    files = [f for f in files if (f.endswith('.wav') or f.endswith('.mp3') or f.endswith('.ogg'))]
    dictionary = {}
    for f in tqdm(files):
        # path format Lang_ID_SEGMENT.mp3
        subpath = f.split("_")
        lang = subpath[0]
        segment = subpath[-1].split(".")[0]
        id_ = "_".join(subpath[1:-1])

        # Get audio length
        audio_path = os.path.join(basePath, f)
        duration = librosa.get_duration(path=audio_path)

        if lang not in dictionary:
            dictionary[lang] = {}
        if id_ not in dictionary[lang]:
            dictionary[lang][id_] = []
        if segment not in dictionary[lang][id_]:
            dictionary[lang][id_].append({"segment": segment, "duration": duration})
    
    # Sort segments in dictionary by segment name
    for lang in dictionary:
        for id_ in dictionary[lang]:
            dictionary[lang][id_] = sorted(dictionary[lang][id_], key=lambda x: x["segment"])
    return files, dictionary

def mergeDicts(dict1, dict2):
    finalDict = {}
    dict1Keys = list(dict1.keys())
    dict2Keys = list(dict2.keys())
    for key in dict1Keys:
        if key not in finalDict:
            finalDict[key] = {}
        for subkey in dict1[key]:
            if subkey not in finalDict[key]:
                finalDict[key][subkey] = []
            finalDict[key][subkey].extend(dict1[key][subkey])
    for key in dict2Keys:
        if key not in finalDict:
            finalDict[key] = {}
        for subkey in dict2[key]:
            if subkey not in finalDict[key]:
                finalDict[key][subkey] = []
            finalDict[key][subkey].extend(dict2[key][subkey])
    # Sort segements in dictionary
    for lang in finalDict:
        for id_ in finalDict[lang]:
            finalDict[lang][id_] = sorted(finalDict[lang][id_], key=lambda x: x["segment"])
    return finalDict

if __name__ == "__main__":
    # German Speech Files
    speechPath = '../speech/DE/'
    speechFiles, fileDict = load_audioFilesFromDir(speechPath)

    # English Speech Files
    speechPath2 = '../speech/EN/'
    speechFiles2, fileDict2 = load_audioFilesFromDir(speechPath2)

    # Merge dictionaries
    fileDictMerged = mergeDicts(fileDict, fileDict2)

    # Write merged dictionary to file
    with open('dictionary.json', 'w', encoding='utf-8') as f:
        json.dump(fileDictMerged, f, ensure_ascii=False, indent=4)