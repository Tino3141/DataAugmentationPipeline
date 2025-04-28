from typing import Optional
from components.Dataloaders import Dataloader
import os
import tempfile
import json
import io
import librosa
import soundfile as sf
import tarfile
import numpy as np
from pydub import AudioSegment
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import time

from components.AudioConversation import AudioConversation
from components.MusicHandler import MusicHandler
from components.AudioEffects import AudioEffects

class DataGen:
    def __init__(
        self,
        dataloader: Dataloader,
        n_samples: int,
        files_per_tar: int,
        output_dir: str = "output_data",
        sample_rate: int = 48000,
        max_sound_effect_length: float = 2.0,
        coverage: float = 0.3,
        min_gap: float = 1.0,
        effect_gain: float = 0.5,
        speakers: Optional[int] = 2,
        num_segments: Optional[int] = 10,
        num_processors: Optional[int] = 12,
    ):
        """
        Initialize DataGen with pipeline components and generation parameters.
        """
        self.dataloader = dataloader
        self.n_samples = n_samples
        self.files_per_tar = files_per_tar
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.max_sound_effect_length = max_sound_effect_length
        self.coverage = coverage
        self.min_gap = min_gap
        self.speakers = speakers
        self.num_segments = num_segments
        self.num_processors = num_processors
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize pipeline components
        self.audio_conversation = AudioConversation(self.dataloader)
        self.music_handler = MusicHandler(self.dataloader)
        self.audio_effects = AudioEffects(
            dataloader=self.dataloader,
            sample_rate=self.sample_rate,
            max_sound_effect_length=self.max_sound_effect_length,
            effect_gain=effect_gain
        )

    def _generate_sample(self, i):
        # Generate segments and apply gaps
        speakers = self.dataloader.get_random_speakers(self.speakers)
        segments = self.audio_conversation.arrangeSegments(speakers, self.num_segments)
        segments = self.audio_conversation.applyGaussianGap(segments, 0, self.min_gap)
        # Render to a temporary WAV and process
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_wav = os.path.join(tmpdir, f"sample_{i}.wav")
            self.audio_conversation.createAudio(segments, raw_wav)
            audio, sr = librosa.load(raw_wav, sr=self.sample_rate)
            audio = self.music_handler.add_background_music(audio)
            processed_audio = self.audio_effects.apply_sound_effects(
                audio=audio,
                coverage=self.coverage,
                min_gap=self.min_gap
            )
        # Define a zero-padded key
        key = f"{i-1:06d}"
        # Prepare metadata JSON bytes
        meta = []
        for idx, seg in enumerate(segments):
            entry = seg.copy()
            entry.pop('audio', None)
            entry['stem_path'] = f"{key}_stem_{idx}.mp3"
            meta.append(entry)
        meta_bytes = json.dumps({"segments": meta}, ensure_ascii=False).encode("utf-8")
        # Prepare stems bytes
        stem_bytes_list = []
        for idx, seg in enumerate(segments):
            pcm = (seg['audio'] * np.iinfo(np.int16).max).astype(np.int16).tobytes()
            stem_seg = AudioSegment(pcm, frame_rate=seg["sampling_rate"], sample_width=2, channels=1)
            buf = io.BytesIO()
            stem_seg.export(buf, format="mp3")
            stem_bytes_list.append(buf.getvalue())
        # Prepare final mix bytes
        final_pcm = (processed_audio * np.iinfo(np.int16).max).astype(np.int16).tobytes()
        final_seg = AudioSegment(final_pcm, frame_rate=self.sample_rate, sample_width=2, channels=1)
        final_buf = io.BytesIO()
        final_seg.export(final_buf, format="mp3")
        # Assemble stems list
        stems = [(f"{key}_stem_{idx}.mp3", data) for idx, data in enumerate(stem_bytes_list)]
        return key, meta_bytes, stems, final_buf.getvalue()

    def generate_data(self):
        """
        Generate synthetic audio samples and save them as WebDataset shards (tar files).
        """
        os.makedirs(self.output_dir, exist_ok=True)
        # Parallel generation of samples with progress bar
        sample_results = process_map(
            self._generate_sample,
            range(1, self.n_samples + 1),
            max_workers=self.num_processors,
            desc="Generating samples"
        )
        # Sequentially write shards
        shard_idx = 0
        sample_count = 0
        tar = None
        for key, meta_bytes, stems, final_bytes in sample_results:
            if sample_count % self.files_per_tar == 0:
                if tar is not None:
                    tar.close()
                shard_path = os.path.join(self.output_dir, f"shard_{shard_idx:05d}.tar")
                tar = tarfile.open(shard_path, "w")
                shard_idx += 1
                sample_count = 0
            # add metadata JSON
            meta_buf = io.BytesIO(meta_bytes)
            meta_info = tarfile.TarInfo(f"{key}.json")
            meta_info.size = len(meta_bytes)
            tar.addfile(meta_info, meta_buf)
            # add stems
            for stem_filename, stem_data in stems:
                stem_buf = io.BytesIO(stem_data)
                stem_info = tarfile.TarInfo(stem_filename)
                stem_info.size = len(stem_data)
                tar.addfile(stem_info, stem_buf)
            # add final mix
            final_buf = io.BytesIO(final_bytes)
            final_info = tarfile.TarInfo(f"{key}.mp3")
            final_info.size = len(final_bytes)
            tar.addfile(final_info, final_buf)
            sample_count += 1
        if tar is not None:
            tar.close()