# 🌊 video-waves

**video-waves** is a high-performance, minimalist audio waveform renderer. Optimized for macOS (M1/M2/M3) and designed with an industrial, cold-aesthetic in mind. It transforms your audio files into stunning 1080p videos with reactive lighting, grain textures, and fluid motion.

## ⚡ Highlights

- **M1/M2/M3 Native**: Uses `h264_videotoolbox` for hardware-accelerated encoding.
- **Industrial Aesthetic**: Cold-white waveforms, mirrored shadows, and CRT-style scanlines.
- **Reactive Glow**: Dynamic lighting that pulses with the audio's intensity (RMS).
- **Vectorized Performance**: Frame rendering is fully vectorized using NumPy and OpenCV for maximum speed.
- **Batch Processing**: Point it at a folder and let it render your entire album or library.
- **Smart Progress**: Real-time progress bar with ETA and FPS tracking.

## 🛠️ Requirements

- **Python 3.8+**
- **FFmpeg** (installed via Homebrew: `brew install ffmpeg`)
- **Python Libraries**:
  ```bash
  pip install librosa numpy moviepy opencv-python
  ```

## 🚀 Usage

Navigate to the project directory and run:

### Process a single file
```bash
python3 visualizer.py "/path/to/your/audio.aif"
```

### Batch process a folder
```bash
python3 visualizer.py "/path/to/your/music_folder"
```

### Options
- `-d`, `--duration`: Limit the render (e.g., `-d 15` for a 15-second preview).
- `--fps`: Set output framerate (default: 30).
- `--width` / `--height`: Custom resolution (default: 1920x1080).
- `--out-dir`: Specify where to save the videos.

## 🎨 Visual Configuration
You can tweak the look and feel by editing the `CFG` dictionary at the top of `visualizer.py`. Change colors, grain intensity, sphere distortion, and more.

## 🙏 Credits
Developed as a high-fidelity visual tool for musicians and producers.
