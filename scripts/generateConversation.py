import os
import librosa
import soundfile as sf
import argparse
import json
from components.AudioEffects import AudioEffects
from components.MusicHandler import MusicHandler
from components.AudioConversation import AudioConversation
import numpy as np

from components.Segments2Image import SpeakerVisualization

def main():
    # Parse command line arguments

    parser = argparse.ArgumentParser(description='Apply audio enhancements to a file')
    
    parser.add_argument('--sample-rate', type=int, default=48000,
                        help='Target sample rate')
    parser.add_argument('--num-speakers', type=int, default=3,
                        help='Number of speakers in the conversation')
    parser.add_argument('--num-segments', type=int, default=10,
                        help='Number of segments in the conversation')
    parser.add_argument('--gap-mean', type=float, default=0.0,
                        help='Mean duration of gaps between segments in seconds')
    parser.add_argument('--gap-std', type=float, default=0.75,
                        help='Standard deviation of gaps between segments in seconds')
    parser.add_argument('--output-dir', type=str, default='samples/output',
                        help='Directory to save output audio files')
    parser.add_argument('--audio-root', type=str, default='speech',
                        help='Directory containing audio files')
    parser.add_argument('--sound-effects', action='store_true', default=True,
                        help='Apply sound effects')
    parser.add_argument('--sound-effects-dir', type=str, default='soundEffects',
                        help='Directory containing sound effects')
    parser.add_argument('--sound-effects-coverage', type=float, default=0.3,
                        help='Coverage probability for sound effects')
    parser.add_argument('--sound-effects-min-gap', type=float, default=1.0,
                        help='Minimum gap between sound effects in seconds')
    parser.add_argument('--max-sound-effect-length', type=float, default=2.0,
                        help='Maximum length of sound effects in seconds')
    parser.add_argument('--background-music', action='store_true', default=True,
                        help='Add background music')
    parser.add_argument('--music-dir', type=str, default='music',
                        help='Directory containing music files')
    parser.add_argument('--music-volume', type=float, default=0.2,
                        help='Volume level for background music')
    parser.add_argument('--crossfade-duration', type=float, default=2.0,
                        help='Crossfade duration for music in seconds')
    parser.add_argument('--loop-music', action='store_true', default=True,
                        help='Whether to loop the background music')
    parser.add_argument('--data-path', type=str, default='/Users/constantinpinkl/University/Research/SpeakerDiarization/Datasets/DataAugmentationPipeline/scripts/dictionary.json',
                        help='Path to the data')
    
    args = parser.parse_args()
    print(os.path.dirname(args.audio_root))
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.data_path, 'r', encoding='utf-8') as f:
        segmentDict = json.load(f)
    
    conversation = AudioConversation()
    segments = conversation.augmentAudio(segmentDict, args.num_speakers, args.num_segments, args.output_dir, args.audio_root, args.gap_mean, args.gap_std)

    # Load input audio
    print(f"Loading input audio from {args.output_dir}/output.wav...")
    audio, sr = librosa.load(args.output_dir + "/output.wav", sr=args.sample_rate)
    
    # Apply sound effects if requested
    if args.sound_effects:
        print("Applying sound effects...")
        audio_effects = AudioEffects(
            sample_rate=args.sample_rate,
            max_sound_effect_length=args.max_sound_effect_length
        )
        audio = audio_effects.apply_sound_effects(
            audio=audio,
            sound_effects_dir=args.sound_effects_dir,
            coverage=args.sound_effects_coverage,
            min_gap=args.sound_effects_min_gap
        )
    
    # Add background music if requested
    if args.background_music:
        print("Adding background music...")
        music_handler = MusicHandler(
            sample_rate=args.sample_rate,
            crossfade_duration=args.crossfade_duration
        )
        audio = music_handler.add_background_music(
            audio=audio,
            music_dir=args.music_dir,
            music_volume=args.music_volume,
            loop_music=args.loop_music
        )
    
    # Save processed audio
    print(f"Saving processed audio to {args.output_dir}/output_enhanced.wav...")
    sf.write(args.output_dir + "/output_enhanced.wav", audio, args.sample_rate)

    # Save segments as visualization
    visualizer = SpeakerVisualization(segments)
    visualizer.create_visualization(
        output_path=os.path.join(args.output_dir, "speaker_diarization.png"),
        figsize=(12, 4),
        dpi=300
    )    
    print(f"Speaker diarization visualization saved to {args.output_dir}/speaker_diarization.png")
    print("Done!")

if __name__ == "__main__":
    main() 