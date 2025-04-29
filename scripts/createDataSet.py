import argparse
from components.DataGen import DataGen
from components.Dataloaders import Dataloader

def main():
    parser = argparse.ArgumentParser(
        description="Generate augmented data for speaker diarization."
    )
    parser.add_argument(
        "--sound_effects_path",
        default="/Users/constantinpinkl/Downloads/Emilia/sfxSound",
        help="Path to sound effects directory"
    )
    parser.add_argument(
        "--background_music_path",
        default="/Users/constantinpinkl/Downloads/Emilia/mtgJamendo",
        help="Path to background music directory"
    )
    parser.add_argument(
        "--speech_samples_path",
        default="/Users/constantinpinkl/Downloads/Emilia/Emilia",
        help="Path to speech samples directory"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--files_per_tar",
        type=int,
        default=200,
        help="Number of files per tar archive"
    )

    parser.add_argument(
        "--num_processors",
        type=int,
        default=12,
        help="Number of processors to use for parallel processing"
    )
    parser.add_argument(
        "--output_dir",
        default="output_data",
        help="Directory to save generated data"
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=48000,
        help="Sample rate for generated audio"
    )
    parser.add_argument(
        "--max_sound_effect_length",
        type=float,
        default=2.0,
        help="Maximum length of sound effect in seconds"
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.3,
        help="Coverage ratio of sound effects"
    )
    parser.add_argument(
        "--min_gap",
        type=float,
        default=1.0,
        help="Minimum gap between sound effects in seconds"
    )
    parser.add_argument(
        "--effect_gain",
        type=float,
        default=0.5,
        help="Gain applied to sound effects"
    )

    parser.add_argument(
        "--num_speakers",
        type=int,
        default=3,
        help="Number of speakers to use in the conversation"
    )

    parser.add_argument(
        "--num_segments",
        type=int,
        default=10,
        help="Number of segments to generate for each speaker"
    )

    args = parser.parse_args()

    print("########################")
    print("Init Dataloader")
    print("########################")
    dataloader = Dataloader(
        sound_effects_path=args.sound_effects_path,
        background_music_path=args.background_music_path,
        speech_samples_path=args.speech_samples_path
    )

    print("Init DataGenerator")
    print("########################")
    generator = DataGen(
        dataloader=dataloader,
        n_samples=args.n_samples,
        files_per_tar=args.files_per_tar,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        max_sound_effect_length=args.max_sound_effect_length,
        coverage=args.coverage,
        min_gap=args.min_gap,
        effect_gain=args.effect_gain,
        speakers=args.num_speakers,
        num_processors=args.num_processors,
        num_segments=args.num_segments,
    )
    print("Starting data generation...")
    generator.generate_data()
    print("#########################")
    print("Data generation completed!")
    print("#########################")

if __name__ == "__main__":
    main()