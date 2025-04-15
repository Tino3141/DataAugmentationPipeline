import os
import librosa
import soundfile as sf
import argparse
import json
from components.AudioEffects import AudioEffects
from components.MusicHandler import MusicHandler
from components.AudioConversation import AudioConversation

def main():
    # Parse command line arguments

    parser = argparse.ArgumentParser(description='Apply audio enhancements to a file')
    # parser.add_argument('--input', type=str, default='samples/output/output.wav',
    #                    help='Path to input audio file')
    #  parser.add_argument('--output', type=str, default='samples/output/enhanced.wav',
    #                     help='Path to output audio file')
    parser.add_argument('--sample-rate', type=int, default=48000,
                        help='Target sample rate')
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
    parser.add_argument('--background-music', action='store_true', default=True,
                        help='Add background music')
    parser.add_argument('--music-dir', type=str, default='music',
                        help='Directory containing music files')
    parser.add_argument('--music-volume', type=float, default=0.2,
                        help='Volume level for background music')
    parser.add_argument('--crossfade-duration', type=float, default=2.0,
                        help='Crossfade duration for music in seconds')
    
    args = parser.parse_args()
    print(os.path.dirname(args.audio_root))
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    with open('/Users/constantinpinkl/University/Research/SpeakerDiarization/Datasets/DataAugmentationPipeline/scripts/dictionary.json', 'r', encoding='utf-8') as f:
        segmentDict = json.load(f)
    
    conversation = AudioConversation()
    conversation.augmentAudio(segmentDict, 3, 10, args.output_dir, args.audio_root, 0, 0.75)

    # Load input audio
    print(f"Loading input audio from {args.output_dir}/output.wav...")
    audio, sr = librosa.load(args.output_dir + "/output.wav", sr=args.sample_rate)
    
    # Apply sound effects if requested
    if args.sound_effects:
        print("Applying sound effects...")
        audio_effects = AudioEffects(
            sample_rate=args.sample_rate,
            max_sound_effect_length=2.0
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
            loop_music=True
        )
    
    # Save processed audio
    print(f"Saving processed audio to {args.output_dir}/output_enhanced.wav...")
    sf.write(args.output_dir + "/output_enhanced.wav", audio, args.sample_rate)
    
    print("Done!")

if __name__ == "__main__":
    main() 