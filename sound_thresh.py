import numpy as np
import matplotlib.pyplot as plt

from moviepy import VideoFileClip
from config import VIDEO_PATH, FILE_PATH, THRESHOLD, WINDOW_DURATION

def compute_rms(signal_segment):
    return np.sqrt(np.mean(signal_segment ** 2))

def compute_rms_db(signal_segment):
    return 20 * np.log10(np.sqrt(np.mean(signal_segment ** 2)))

class AudioProcessor:
    """Handles audio extraction and RMS computation from a video file."""
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.sample_rate = None
        self.audio_signal = None
    
    def load_audio(self):
        """Extract audio track from the video and convert to mono."""
        clip = VideoFileClip(VIDEO_PATH)
        # Extract audio track 
        audio = clip.audio
        audio_array = audio.to_soundarray()  # Shape: [n_samples, n_channels]
        # Convert stereo to mono by averaging channels 
        self.audio_signal = audio_array.mean(axis=1)
        # Get sampling rate (samples per second) 
        self.sample_rate = audio.fps
        print(f"[INFO] Audio sample rate: {self.sample_rate} Hz")

    def compute_rms_over_time(self, window_duration : float) -> tuple[np.ndarray, np.ndarray]:
        """Compute RMS (in dB) values over time windows."""
        if self.audio_signal is None:
            raise ValueError("Audio not loaded. Call load_audio() first.")
        # Split audio into time windows 
        window_size = int(self.sample_rate * window_duration)  # Samples per window
        num_windows = len(self.audio_signal) // window_size    # Number of full windows
        rms_values = []  # Store RMS values for each window

        for i in range(num_windows):
            start_index = i * window_size
            end_index = start_index + window_size
            window = self.audio_signal[start_index:end_index]
            rms = compute_rms(window)
            rms_values.append(rms)
        
        times = np.arange(len(rms_values)) * window_duration
        return times, np.array(rms_values)
    

class SoundDetector():
    """Handles threshold-based detection and file export."""

    def __init__(self, threshold : int):
        self.threshold = threshold
    
    def detect(self, times : np.array, rms_values : np.array):
        """Return timestamps where RMS exceeds the threshold."""

        detections = times[rms_values > self.threshold]
        print(f"[INFO] {len(detections)} detections found.")
        return detections
    
    def save_detections(self, detections : np.ndarray, file_path : str):
        with open(file_path, "w") as f :
            for d in detections:
                f.write(f"{int(d)}\n")
        print(f"[INFO] Detections saved to {file_path}")


def plot_rms(times, rms_values, threshold):
    times = np.array(times)
    rms_values = np.array(rms_values)
    above_threshold = rms_values > threshold
    plt.plot(times, rms_values, label = "RMS", color = "blue")
    plt.plot(times[above_threshold], rms_values[above_threshold], label = "Detection", color = "red", linestyle = "", marker = "o")
    plt.xlabel("Time (s)")
    plt.ylabel("RMS amplitude")
    plt.title("Audio Energy over Time")
    plt.show()


def main():
    processor = AudioProcessor(VIDEO_PATH)
    processor.load_audio()
    times, rms_values = processor.compute_rms_over_time(WINDOW_DURATION)

    detector = SoundDetector(THRESHOLD)
    detections = detector.detect(times, rms_values)
    # detector.save_detections(detections, FILE_PATH)

    plot_rms(times, rms_values, THRESHOLD)


if __name__ == "__main__":
    main()
