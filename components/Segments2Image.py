import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

class SpeakerVisualization:
    def __init__(self, segments_input):
        """
        Initialize the visualization class.
        
        Args:
            segments_input: Either a string path to JSON file or a list/dict of segments
        """
        if isinstance(segments_input, str):
            self.segments = self._load_segments(segments_input)
        else:
            self.segments = segments_input
    
        self.speakers = self._get_unique_speakers()
        self.speaker_colors = self._generate_speaker_colors()
    def _load_segments(self, file_path: str) -> list:
        """Load segments from JSON file."""
        with open(file_path, 'r') as f:
            return json.load(f)

    def _get_unique_speakers(self) -> list:
        """Get unique speaker IDs from segments."""
        speakers = set()
        for segment in self.segments:
            speakers.add(f"{segment['language']}_{segment['id']}")
        return sorted(list(speakers))

    def _generate_speaker_colors(self) -> dict:
        """Generate color mapping for speakers."""
        colors = plt.cm.get_cmap('Set3')(np.linspace(0, 1, len(self.speakers)))
        return dict(zip(self.speakers, colors))

    def create_visualization(self, output_path: str, figsize=(12, 4), dpi=300):
        """
        Create and save the speaker diarization visualization.
        
        Args:
            output_path (str): Path where to save the output image
            figsize (tuple): Figure size (width, height)
            dpi (int): DPI for output image
        """
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Plot each speaker's segments on their own row
        for i, speaker in enumerate(self.speakers):
            lang = speaker.split('_', 1)[0]
            spk_id = speaker.split('_', 1)[1]
            # Get segments for this speaker
            segments = [(seg['start'], seg['end']) for seg in self.segments 
                       if seg['language'] == lang and seg['id'] == spk_id]
            
            # Plot segments as horizontal bars
            for start, end in segments:
                ax.barh(y=i, width=end-start, left=start, 
                       color=self.speaker_colors[speaker], alpha=0.6,
                       height=0.6)
            
            # Add speaker label on y-axis
            ax.text(-0.5, i, f'Speaker {speaker}', ha='right', va='center')

        # Customize the plot
        ax.set_ylim(-0.5, len(self.speakers)-0.5)
        ax.set_yticks([])
        ax.set_xlabel('Time (seconds)')
        ax.set_title('Speaker Diarization Timeline')
        ax.grid(True, axis='x', alpha=0.3)

        # Add padding for speaker labels
        ax.set_position([0.1, 0.1, 0.85, 0.8])

        # Save the plot
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()

# Example usage
# visualizer = SpeakerVisualization('segments.json')
# visualizer.create_visualization('output/speaker_visualization.png')