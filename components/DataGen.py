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

    def generate_data(self):
        """
        Generate synthetic audio samples and save them as WebDataset shards (tar files).
        """
        os.makedirs(self.output_dir, exist_ok=True)
        shard_idx = 0
        sample_count = 0
        tar = None

        for i in tqdm(range(1, self.n_samples + 1)):
            # Open a new shard file if needed
            if sample_count % self.files_per_tar == 0:
                if tar is not None:
                    tar.close()
                shard_path = os.path.join(self.output_dir, f"shard_{shard_idx:05d}.tar")
                tar = tarfile.open(shard_path, "w")
                shard_idx += 1
                sample_count = 0

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

            # Define a zero-padded key for this sample
            key = f"{i-1:06d}"

            # 1) Add metadata JSON
            meta = []
            for idx, seg in enumerate(segments):
                entry = seg.copy()
                if 'audio' in entry:
                    del entry['audio']
                entry['stem_path'] = f"{key}_stem_{idx}.mp3"
                meta.append(entry)
            meta_buf = io.BytesIO(json.dumps({"segments": meta}, ensure_ascii=False).encode("utf-8"))
            meta_info = tarfile.TarInfo(f"{key}.json")
            meta_info.size = meta_buf.getbuffer().nbytes
            tar.addfile(meta_info, meta_buf)

            # 2) Add each stem as MP3
            for idx, seg in enumerate(segments):
                pcm = (seg['audio'] * np.iinfo(np.int16).max).astype(np.int16).tobytes()
                stem_seg = AudioSegment(pcm, frame_rate=self.sample_rate, sample_width=2, channels=1)
                stem_buf = io.BytesIO()
                stem_seg.export(stem_buf, format="mp3")
                stem_buf.seek(0)
                stem_info = tarfile.TarInfo(f"{key}_stem_{idx}.mp3")
                stem_info.size = stem_buf.getbuffer().nbytes
                tar.addfile(stem_info, stem_buf)

            # 3) Add final mix as MP3
            final_pcm = (processed_audio * np.iinfo(np.int16).max).astype(np.int16).tobytes()
            final_seg = AudioSegment(final_pcm, frame_rate=self.sample_rate, sample_width=2, channels=1)
            final_buf = io.BytesIO()
            final_seg.export(final_buf, format="mp3")
            final_buf.seek(0)
            final_info = tarfile.TarInfo(f"{key}.mp3")
            final_info.size = final_buf.getbuffer().nbytes
            tar.addfile(final_info, final_buf)

            sample_count += 1

        if tar is not None:
            tar.close()