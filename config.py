import os

BASE_DIR = r"/home/esteban-dreau-darizcuren/Doctorat/"

VIDEO_NAME = "Raie"

VIDEO_DIR = os.path.join(BASE_DIR, "Dataset", "Raw")
VIDEO_PATH = os.path.join(VIDEO_DIR, f"{VIDEO_NAME}.MP4")

RESULTS_DIR = os.path.join(BASE_DIR, "Code", "SoundDetector", "Results")
FILE_PATH = os.path.join(RESULTS_DIR, f"{VIDEO_NAME}_results.txt")

WINDOW_DURATION = 0.1  # Window length in seconds (e.g., 0.1 s = 100 ms)
THRESHOLD = -64