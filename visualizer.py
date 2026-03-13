import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import time

import cv2
import librosa
import numpy as np
from moviepy.editor import VideoClip

# ─────────────────────────────────────────────
#  VISUAL CONFIG  — industrial / minimal
# ─────────────────────────────────────────────
CFG = {
    # Background
    "bg_base": np.array([10, 10, 10], dtype=np.uint8),
    # Waveform colours
    "wave_top_color": (230, 230, 230),  # cold white — primary
    "wave_bot_color": (55, 55, 55),  # dark grey  — mirror
    "fill_color": (18, 18, 18),  # almost-black fill
    # Grain
    "grain_intensity": 8,
    # ── True 2D barrel / fisheye ──────────────────────────────────────
    # Model:  r' = r · (1 + k · r²)
    # k > 0 → barrel distortion (wide-angle look).
    # Both X and Y warp together so peaks curve outward like a real lens.
    # Tweak to taste; 0 = no distortion.
    "fisheye_k": 0.0,  # disabled
    # ── Gaussian zoom — amplitude magnification at "now" (screen centre) ──
    # A bell curve is applied to y_scale so the centre of the frame has
    # the highest amplitude and it tapers smoothly toward the edges.
    # zoom_peak   : how many times larger the centre is vs the edges (1 = off)
    # zoom_sigma  : width of the bell — smaller = tighter zoom (0.25–0.6 range)
    "zoom_peak": 1.8,  # centre amplitude multiplier
    "zoom_sigma": 0.30,  # bell width in normalised [-1,1] units
    # Waveform time window (seconds visible on screen)
    "window_sec": 1.6,
    # Number of polyline points
    "num_points": 1200,
    # Amplitude scale
    "v_scale_base": 4.0,
    "v_scale_edge_boost": 0.18,
    # Glow
    "glow_rms_thresh": 0.03,
    "glow_kernel": 21,
    "glow_weight": 1.2,
    # Vignette
    "vignette_power": 0.42,
    # Scanlines
    "scanlines": True,
    "scanline_alpha": 0.07,
}


# ─────────────────────────────────────────────
#  STATIC LAYER PRE-COMPUTATION
# ─────────────────────────────────────────────
def build_static_layers(width: int, height: int, cfg: dict) -> dict:
    """Pre-compute everything that is constant across frames."""

    # Vignette
    X, Y = np.meshgrid(
        np.linspace(-1, 1, width, dtype=np.float32),
        np.linspace(-1, 1, height, dtype=np.float32),
    )
    vig = np.clip(1.0 - (X**2 + Y**2) * cfg["vignette_power"], 0.0, 1.0)
    vignette = np.stack([(vig * 255).astype(np.uint8)] * 3, axis=-1)

    # Grain buffers (cycle at runtime)
    n_bufs = 16
    grain_bufs = [
        np.random.randint(0, cfg["grain_intensity"], (height, width, 3), dtype=np.uint8)
        for _ in range(n_bufs)
    ]

    # Scanline mask
    if cfg["scanlines"]:
        scanline_mask = np.ones((height, width, 3), dtype=np.float32)
        scanline_mask[::2, :, :] *= 1.0 - cfg["scanline_alpha"]
    else:
        scanline_mask = None

    # Normalised X positions for the waveform polyline  [-1 … +1]
    p_lin = np.linspace(-1.0, 1.0, cfg["num_points"], dtype=np.float32)

    # Gaussian zoom bell — peaks at the centre (the "now" position) and
    # tapers smoothly to ~1.0 at the edges.
    # Formula:  zoom(x) = 1 + (peak-1) * exp(-x^2 / (2*sigma^2))
    # At x=0 (centre) -> zoom_peak.  At x=+/-1 (edges) -> ~1.0.
    sigma = cfg["zoom_sigma"]
    zoom_bell = 1.0 + (cfg["zoom_peak"] - 1.0) * np.exp(-(p_lin**2) / (2.0 * sigma**2))

    # Base scale x gaussian zoom x subtle edge taper
    y_scale = (
        (height / cfg["v_scale_base"])
        * zoom_bell
        * (1.0 + cfg["v_scale_edge_boost"] * (1.0 - np.abs(p_lin)))
    )

    return {
        "vignette": vignette,
        "grain_bufs": grain_bufs,
        "n_bufs": n_bufs,
        "scanline_mask": scanline_mask,
        "p_lin": p_lin,
        "y_scale": y_scale,
    }


def barrel_distort(nx: np.ndarray, ny: np.ndarray, k: float):
    """
    Apply barrel (fisheye) distortion to arrays of normalised coords.

    Model:  r' = r · (1 + k · r²)

    Both X and Y are distorted by the same radial factor, so peaks and
    horizontal positions warp together — exactly like a wide-angle lens.
    Returns (nx', ny') still in normalised [-1, 1] space.
    """
    r2 = nx**2 + ny**2
    factor = 1.0 + k * r2
    return nx * factor, ny * factor


# ─────────────────────────────────────────────
#  VIDEO RENDER  (no audio)
# ─────────────────────────────────────────────
def render_video_only(
    audio_path: str,
    video_path: str,
    fps: int,
    width: int,
    height: int,
    max_duration: float,
) -> float:
    """Render the waveform video track (silent) and return the actual duration."""

    print("  LOADING    audio for waveform analysis...")
    audio_mono, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = librosa.get_duration(y=audio_mono, sr=sr)
    if max_duration:
        duration = min(duration, max_duration)
        audio_mono = audio_mono[: int(duration * sr)]

    # RMS envelope — drives background pulse and glow
    hop = 512
    rms = librosa.feature.rms(y=audio_mono, hop_length=hop)[0]

    def rms_at(t: float) -> float:
        idx = int((t * sr) / hop)
        return float(rms[idx]) if idx < len(rms) else 0.0

    # ── Static layers ──────────────────────────────────────────────────
    S = build_static_layers(width, height, CFG)
    bg_base = CFG["bg_base"]
    half_h = height // 2
    num_pts = CFG["num_points"]
    p_lin = S["p_lin"]
    y_scale = S["y_scale"]
    k_lens = CFG["fisheye_k"]

    # ── Window: current playhead sits at the exact screen centre ───────
    # p_lin=0 maps to px=width/2.  We take equal samples before and after
    # the playhead so "now" is always centred horizontally — no more lag.
    win_total = int(sr * CFG["window_sec"])
    win_before = win_total // 2  # samples to the left  of "now"
    win_after = win_total // 2  # samples to the right of "now"

    total_frames = int(duration * fps)
    render_start = time.time()
    last_frame = [0]

    # ── Per-frame renderer ─────────────────────────────────────────────
    def make_frame(t: float) -> np.ndarray:
        rms_val = rms_at(t)

        # Background pulse
        pulse = int(rms_val * 12)
        frame = np.full((height, width, 3), bg_base + pulse, dtype=np.uint8)

        # Grain
        buf_idx = int(t * fps) % S["n_bufs"]
        cv2.add(frame, S["grain_bufs"][buf_idx], frame)

        # ── Audio chunk centred on "now" ───────────────────────────────
        center = int(t * sr)
        start = center - win_before
        end = center + win_after

        # Zero-pad at track boundaries so chunk is always win_total long
        if start < 0:
            chunk = np.concatenate(
                [
                    np.zeros(-start, dtype=np.float32),
                    audio_mono[0:end],
                ]
            )
        elif end > len(audio_mono):
            chunk = np.concatenate(
                [
                    audio_mono[start:],
                    np.zeros(end - len(audio_mono), dtype=np.float32),
                ]
            )
        else:
            chunk = audio_mono[start:end]

        if len(chunk) == 0:
            return frame

        # Resample chunk to num_pts
        chunk_r = np.interp(
            np.linspace(0, len(chunk) - 1, num_pts),
            np.arange(len(chunk)),
            chunk,
        ).astype(np.float32)

        # ── True 2D barrel distortion ──────────────────────────────────
        # Normalise amplitude to [-1,1] so the lens is resolution-independent
        ny_norm = chunk_r * y_scale / (height * 0.5)

        # Distort X and Y together through the barrel lens
        px_d, py_d = barrel_distort(p_lin, ny_norm, k_lens)

        # Map back to pixel coords
        px_d = np.clip(px_d, -1.0, 1.0)
        py_d = np.clip(py_d, -1.0, 1.0)

        x_px = np.clip(((px_d + 1.0) * 0.5 * width).astype(np.int32), 0, width - 1)
        # Positive amplitude → above centreline (subtract from half_h)
        y_top_px = np.clip(
            (half_h - py_d * (height * 0.5)).astype(np.int32), 0, height - 1
        )
        y_bot_px = height - y_top_px  # mirrored bottom

        pts_top = np.stack([x_px, y_top_px], axis=1)[:, np.newaxis, :]
        pts_bot = np.stack([x_px, y_bot_px], axis=1)[:, np.newaxis, :]

        # Fill polygon
        poly = np.vstack([pts_top[:, 0, :], pts_bot[::-1, 0, :]])
        cv2.fillPoly(frame, [poly], CFG["fill_color"])

        # Waveform lines
        cv2.polylines(frame, [pts_bot], False, CFG["wave_bot_color"], 1, cv2.LINE_AA)
        cv2.polylines(frame, [pts_top], False, CFG["wave_top_color"], 1, cv2.LINE_AA)

        # Glow on loud transients
        if rms_val > CFG["glow_rms_thresh"]:
            gk = CFG["glow_kernel"]
            glow = cv2.GaussianBlur(frame, (gk, gk), 0)
            cv2.addWeighted(frame, 1.0, glow, CFG["glow_weight"], 0, frame)

        # Vignette
        frame = cv2.multiply(frame, S["vignette"], scale=1.0 / 255.0).astype(np.uint8)

        # Scanlines
        if S["scanline_mask"] is not None:
            frame = (frame * S["scanline_mask"]).astype(np.uint8)

        # Progress bar
        frame_n = int(t * fps)
        if frame_n > last_frame[0] or frame_n == 0:
            last_frame[0] = frame_n
            elapsed = time.time() - render_start
            if elapsed > 0 and frame_n > 0:
                fps_actual = frame_n / elapsed
                remaining = (total_frames - frame_n) / fps_actual
                pct = frame_n / total_frames * 100
                bar_len = 28
                filled = int(bar_len * frame_n / total_frames)
                bar = "█" * filled + "░" * (bar_len - filled)
                m, s = divmod(int(remaining), 60)
                print(
                    f"\r  [{bar}] {pct:5.1f}%  "
                    f"{frame_n:>5}/{total_frames}  "
                    f"{fps_actual:4.1f}fps  ETA {m:02d}:{s:02d}",
                    end="",
                    flush=True,
                )

        return frame

    # ── Encode video-only track ────────────────────────────────────────
    clip = VideoClip(make_frame, duration=duration)

    print(f"\n  ENCODING   video track (no audio)...")
    clip.write_videofile(
        video_path,
        fps=fps,
        codec="h264_videotoolbox",
        audio=False,
        bitrate="20000k",
        ffmpeg_params=[
            "-profile:v",
            "high",
            "-level",
            "5.1",
            "-pix_fmt",
            "yuv420p",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-colorspace",
            "bt709",
        ],
        threads=10,
        logger=None,
    )

    elapsed = time.time() - render_start
    m, s = divmod(int(elapsed), 60)
    print(
        f"\r  {'█' * 28}  100.0%  {total_frames}/{total_frames}  DONE in {m:02d}:{s:02d}\n"
    )

    return duration


# ─────────────────────────────────────────────
#  AUDIO MUX  (ffmpeg — original source quality)
# ─────────────────────────────────────────────
def mux_audio(
    audio_path: str,
    video_path: str,
    output_path: str,
    max_duration: float,
) -> None:
    """
    Combine the rendered video with the original audio using ffmpeg.
    Audio never passes through librosa — it goes straight from the source
    file into the output container as AAC-LC 320k 48 kHz stereo.
    """
    print(f"  MUXING     audio + video → {os.path.basename(output_path)}")

    duration_args = ["-t", str(max_duration)] if max_duration else []

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-i",
        audio_path,
        *duration_args,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "320k",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        "-metadata",
        f"title={os.path.splitext(os.path.basename(audio_path))[0]}",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n  ERROR in ffmpeg mux:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  OUTPUT     {os.path.basename(output_path)}  ({size_mb:.1f} MB)\n")


# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────
def create_waveform_video(
    audio_path: str,
    output_path: str,
    fps: int = 60,
    width: int = 2560,
    height: int = 1440,
    max_duration: float = None,
) -> None:
    print(f"\n  RENDERING  {os.path.basename(audio_path)}")
    print(f"  ──────────────────────────────────────")

    tmp_dir = tempfile.mkdtemp(prefix="waveform_")
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    tmp_video = os.path.join(tmp_dir, f"{base_name}_video_only.mp4")

    try:
        render_video_only(audio_path, tmp_video, fps, width, height, max_duration)
        mux_audio(audio_path, tmp_video, output_path, max_duration)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Waveform renderer — industrial minimal / M1 Pro / max quality"
    )
    parser.add_argument("path", help="Audio file or directory")
    parser.add_argument("-d", "--duration", type=float, help="Max duration (seconds)")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=2560)
    parser.add_argument("--height", type=int, default=1440)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: same as input)",
    )
    args = parser.parse_args()

    exts = ("*.aif", "*.aiff", "*.wav", "*.mp3", "*.flac")
    files = []

    if os.path.isdir(args.path):
        for e in exts:
            files.extend(glob.glob(os.path.join(args.path, e)))
        files.sort()
    elif os.path.isfile(args.path):
        files.append(args.path)
    else:
        print(f"ERROR: path not found — {args.path}", file=sys.stderr)
        sys.exit(1)

    if not files:
        print("No audio files found.", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_dir or (
        args.path if os.path.isdir(args.path) else os.path.dirname(args.path)
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n  WAVEFORM RENDERER  [1440p / 60fps / gaussian zoom]")
    print(f"  {len(files)} file(s) → {out_dir}\n")

    for i, audio_file in enumerate(files, 1):
        base = os.path.splitext(os.path.basename(audio_file))[0]
        out = os.path.join(out_dir, f"{base}.mp4")
        print(f"  [{i}/{len(files)}]", end=" ")
        create_waveform_video(
            audio_file,
            out,
            fps=args.fps,
            width=args.width,
            height=args.height,
            max_duration=args.duration,
        )
