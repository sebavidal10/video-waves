# video-waves

![demo](wave.png)

A Python CLI that turns audio files into 1080p waveform videos. Built for Apple Silicon with hardware-accelerated encoding and a cold, industrial aesthetic — white waveforms, mirrored shadows, grain and scanlines.

## Requirements

- Python 3.8+
- FFmpeg — `brew install ffmpeg`
```bash
pip install librosa numpy moviepy opencv-python
```

## Usage
```bash
# Single file
python3 visualizer.py "/path/to/audio.aif"

# Batch
python3 visualizer.py "/path/to/music_folder"
```

**Options**

| Flag | Description | Default |
|------|-------------|---------|
| `-d` | Duration limit in seconds | full |
| `--fps` | Framerate | 30 |
| `--width` / `--height` | Resolution | 1920x1080 |
| `--out-dir` | Output directory | same as input |

## Configuration

Edit the `CFG` dict at the top of `visualizer.py` to adjust colors, grain intensity, glow and distortion.
