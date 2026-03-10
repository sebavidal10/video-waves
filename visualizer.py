import librosa
import numpy as np
import cv2
import argparse
import os
import glob
import sys
import time
from moviepy.editor import VideoFileClip, VideoClip, AudioFileClip

# ─────────────────────────────────────────────
#  VISUAL CONFIG  — industrial / minimal
# ─────────────────────────────────────────────
CFG = {
    # Background: near-black, not pure black (industrial feel)
    "bg_base":          np.array([10, 10, 10],  dtype=np.uint8),

    # Waveform lines
    "wave_top_color":   (230, 230, 230),   # cold white — primary waveform
    "wave_bot_color":   (55,  55,  55),    # dark grey  — mirror / shadow
    "fill_color":       (18,  18,  18),    # almost-black fill between waves

    # Glow / pulse tint (very subtle warm grey — not blue, not purple)
    "glow_tint":        np.array([18, 16, 14], dtype=np.float32),

    # Noise grain intensity (0–12 → industrial texture, not filmgrain)
    "grain_intensity":  8,

    # Sphere / fisheye distortion
    "sphere_strength":  2.0,

    # Waveform time window (seconds visible)
    "window_sec":       1.6,

    # Number of x-points for waveform polyline
    "num_points":       1200,

    # Vertical scale multiplier
    "v_scale_base":     2.7,
    "v_scale_edge_boost": 0.18,

    # Glow threshold & kernel
    "glow_rms_thresh":  0.03,
    "glow_kernel":      21,
    "glow_weight":      1.2,

    # Vignette falloff (higher = tighter)
    "vignette_power":   0.42,

    # Scanlines — subtle CRT/industrial texture (0 = off, 1 = on)
    "scanlines":        True,
    "scanline_alpha":   0.07,      # very subtle
}


def build_static_layers(width: int, height: int, fps: int, cfg: dict):
    """Pre-compute everything that doesn't change per-frame."""
    X, Y = np.meshgrid(
        np.linspace(-1, 1, width, dtype=np.float32),
        np.linspace(-1, 1, height, dtype=np.float32),
    )

    # Vignette — elliptical, tighter than original
    vig = np.clip(1.0 - (X**2 + Y**2) * cfg["vignette_power"], 0.0, 1.0)
    vignette = np.stack([(vig * 255).astype(np.uint8)] * 3, axis=-1)  # (H, W, 3)

    # Grain buffers — pre-generate N frames, cycle at runtime
    n_bufs = 16
    grain_bufs = [
        np.random.randint(0, cfg["grain_intensity"], (height, width, 3), dtype=np.uint8)
        for _ in range(n_bufs)
    ]

    # Scanline mask (horizontal lines every 2px)
    if cfg["scanlines"]:
        scanline_mask = np.ones((height, width, 3), dtype=np.float32)
        scanline_mask[::2, :, :] *= (1.0 - cfg["scanline_alpha"])
    else:
        scanline_mask = None

    # Fisheye X-mapping (vectorized, computed once)
    p_lin = np.linspace(-1, 1, cfg["num_points"], dtype=np.float32)
    p_fish = p_lin * (1.0 + cfg["sphere_strength"] * (p_lin ** 2))
    p_fish = np.clip(p_fish, -1.0, 1.0)
    x_coords = ((p_fish + 1.0) / 2.0 * width).astype(np.int32)
    x_coords = np.clip(x_coords, 0, width - 1)

    # Y scale per point (wider at center, tapers at edges)
    y_scale_per_point = (height / cfg["v_scale_base"]) * (
        1.0 + cfg["v_scale_edge_boost"] * (1.0 - np.abs(p_fish))
    )

    return {
        "vignette":        vignette,
        "grain_bufs":      grain_bufs,
        "n_bufs":          n_bufs,
        "scanline_mask":   scanline_mask,
        "x_coords":        x_coords,
        "y_scale_per_point": y_scale_per_point,
    }


def create_waveform_video(
    audio_path: str,
    output_path: str,
    fps: int   = 30,
    width: int = 1920,
    height: int = 1080,
    max_duration: float = None,
):
    print(f"\n  RENDERING  {os.path.basename(audio_path)}")
    print(f"  ──────────────────────────────────────")

    # ── Load audio ──────────────────────────────
    audio, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=audio, sr=sr)
    if max_duration:
        duration = min(duration, max_duration)
        audio = audio[: int(duration * sr)]

    # ── RMS envelope ────────────────────────────
    hop = 512
    rms = librosa.feature.rms(y=audio, hop_length=hop)[0]

    def rms_at(t: float) -> float:
        idx = int((t * sr) / hop)
        return float(rms[idx]) if idx < len(rms) else 0.0

    # ── Pre-compute static layers ────────────────
    S = build_static_layers(width, height, fps, CFG)
    bg_base   = CFG["bg_base"]
    half_h    = height // 2
    win_half  = int(sr * CFG["window_sec"] / 2)
    num_pts   = CFG["num_points"]
    x_coords  = S["x_coords"]
    y_scale   = S["y_scale_per_point"]

    # ── Frame maker (fully vectorized inner loop) ──
    def make_frame(t: float) -> np.ndarray:
        rms_val = rms_at(t)

        # — Background with RMS pulse (subtle, not distracting) —
        pulse = int(rms_val * 12)
        frame = np.full((height, width, 3), bg_base + pulse, dtype=np.uint8)

        # — Grain texture (cycle through pre-built buffers) —
        buf_idx = int(t * fps) % S["n_bufs"]
        cv2.add(frame, S["grain_bufs"][buf_idx], frame)

        # — Extract & resample audio chunk —
        center = int(t * sr)
        start  = max(0, center - win_half)
        end    = min(len(audio), center + win_half)
        chunk  = audio[start:end]
        if len(chunk) == 0:
            return frame

        # Resample to num_points — fully vectorized
        chunk_resampled = np.interp(
            np.linspace(0, len(chunk) - 1, num_pts),
            np.arange(len(chunk)),
            chunk,
        ).astype(np.float32)

        # — Compute Y positions (vectorized) —
        y_top = np.clip(
            half_h + (chunk_resampled * y_scale),
            0, height - 1,
        ).astype(np.int32)
        y_bot = height - y_top  # mirror

        # Build polyline arrays (shape: [N, 1, 2])
        pts_top = np.stack([x_coords, y_top], axis=1)[:, np.newaxis, :]
        pts_bot = np.stack([x_coords, y_bot], axis=1)[:, np.newaxis, :]

        # — Fill polygon between top and mirrored bottom —
        poly = np.vstack([
            pts_top[:, 0, :],
            pts_bot[::-1, 0, :],
        ])
        cv2.fillPoly(frame, [poly], CFG["fill_color"])

        # — Draw lines —
        cv2.polylines(frame, [pts_bot], False, CFG["wave_bot_color"], 1, cv2.LINE_AA)
        cv2.polylines(frame, [pts_top], False, CFG["wave_top_color"], 1, cv2.LINE_AA)

        # — Glow on loud transients —
        if rms_val > CFG["glow_rms_thresh"]:
            k = CFG["glow_kernel"]
            glow = cv2.GaussianBlur(frame, (k, k), 0)
            cv2.addWeighted(frame, 1.0, glow, CFG["glow_weight"], 0, frame)

        # — Vignette —
        frame = cv2.multiply(
            frame, S["vignette"], scale=1.0 / 255.0
        ).astype(np.uint8)

        # — Scanlines —
        if S["scanline_mask"] is not None:
            frame = (frame * S["scanline_mask"]).astype(np.uint8)

        return frame

    # ── Assemble & export ────────────────────────
    total_frames = int(duration * fps)
    render_start = [0.0]
    last_frame   = [0]

    # Wrap make_frame to print progress
    def make_frame_tracked(t: float) -> np.ndarray:
        frame_n = int(t * fps)
        if frame_n > last_frame[0] or frame_n == 0:
            last_frame[0] = frame_n
            elapsed = time.time() - render_start[0]
            if elapsed > 0 and frame_n > 0:
                fps_actual = frame_n / elapsed
                remaining  = (total_frames - frame_n) / fps_actual
                pct        = frame_n / total_frames * 100
                bar_len    = 28
                filled     = int(bar_len * frame_n / total_frames)
                bar        = "█" * filled + "░" * (bar_len - filled)
                mins, secs = divmod(int(remaining), 60)
                print(
                    f"\r  [{bar}] {pct:5.1f}%  "
                    f"{frame_n:>5}/{total_frames}  "
                    f"{fps_actual:4.1f}fps  "
                    f"ETA {mins:02d}:{secs:02d}",
                    end="", flush=True
                )
        return make_frame(t)

    clip       = VideoClip(make_frame_tracked, duration=duration)
    audio_clip = AudioFileClip(audio_path).subclip(0, duration)
    clip       = clip.set_audio(audio_clip)

    print(f"\n  ENCODING   {os.path.basename(output_path)}")
    render_start[0] = time.time()
    clip.write_videofile(
        output_path,
        fps=fps,
        codec="h264_videotoolbox",
        audio_codec="aac",
        bitrate="12000k",
        threads=10,
        logger=None,
    )
    elapsed_total = time.time() - render_start[0]
    mins, secs = divmod(int(elapsed_total), 60)
    print(f"\r  {'█' * 28}  100.0%  {total_frames}/{total_frames}  DONE in {mins:02d}:{secs:02d}\n")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Waveform renderer — industrial minimal / M1 Pro optimized"
    )
    parser.add_argument("path",       help="Audio file or directory")
    parser.add_argument("-d", "--duration", type=float, help="Max duration (seconds)")
    parser.add_argument("--fps",      type=int,   default=30)
    parser.add_argument("--width",    type=int,   default=1920)
    parser.add_argument("--height",   type=int,   default=1080)
    parser.add_argument("--out-dir",  type=str,   default=None,
                        help="Output directory (default: same as input)")
    args = parser.parse_args()

    exts  = ("*.aif", "*.aiff", "*.wav", "*.mp3", "*.flac")
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

    out_dir = args.out_dir or (args.path if os.path.isdir(args.path) else os.path.dirname(args.path))
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n  WAVEFORM RENDERER")
    print(f"  {len(files)} file(s) → {out_dir}\n")

    for i, audio_file in enumerate(files, 1):
        base = os.path.splitext(os.path.basename(audio_file))[0]
        out  = os.path.join(out_dir, f"{base}.mp4")
        print(f"  [{i}/{len(files)}]", end=" ")
        create_waveform_video(
            audio_file, out,
            fps=args.fps,
            width=args.width,
            height=args.height,
            max_duration=args.duration,
        )
