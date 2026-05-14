#!/usr/bin/env python
# Streaming counterpart of infer_audio.py — exercises the InferEngine streaming API
# (start_stream / push_audio / finalize_stream) end-to-end against an audio file
# (or pair of files), then renders the same dual-speaker video.

import argparse
import os
import warnings

import torch
import torch.nn.functional as F

from core.libs.utils_videos import read_audio, write_video
from core.trainer.inferencer import InferEngine


@torch.inference_mode()
def infer_audio_streaming(
    resume_path,
    audio_path,
    audio_path_2=None,
    dump_dir="./render_results",
    tau=1.0,
    cfg=1.5,
    chunk_ms=400,
    seed=0,
):
    if seed is not None:
        torch.manual_seed(seed)

    infer_engine = InferEngine(resume_path)
    print(f"Inference start, loading model from {resume_path}")

    fps = int(infer_engine.meta_cfg.DATASET.MOTION_FPS)
    sample_rate = int(infer_engine.meta_cfg.DATASET.AUDIO_SAMPLE_RATE)
    device = infer_engine.device
    two_speaker = audio_path_2 is not None

    # Load audio (already resampled to target_sr, 1D mono).
    audio_0, _ = read_audio(audio_path, target_sr=sample_rate)
    if two_speaker:
        audio_1, _ = read_audio(audio_path_2, target_sr=sample_rate)
    else:
        audio_1 = torch.zeros_like(audio_0)

    # Align lengths so push_audio chunks stay in lockstep.
    n = max(audio_0.shape[0], audio_1.shape[0])
    audio_0 = F.pad(audio_0, (0, n - audio_0.shape[0]))
    audio_1 = F.pad(audio_1, (0, n - audio_1.shape[0]))

    # Drive the streaming API.
    infer_engine.start_stream()
    chunk_samples = max(1, int(chunk_ms / 1000.0 * sample_rate))
    chunks_0 = list(audio_0.split(chunk_samples))
    chunks_1 = list(audio_1.split(chunk_samples))
    print(
        f"Streaming {n / sample_rate:.2f}s of audio in {len(chunks_0)} chunks "
        f"(~{chunk_ms}ms each); patch={infer_engine._stream_patch_audio_length} samples "
        f"({max(infer_engine.model.patch_nums)} frames @ {fps}fps)"
    )

    motions_0, motions_1 = [], []
    for c0, c1 in zip(chunks_0, chunks_1):
        m0, m1 = infer_engine.push_audio(
            c0.to(device), c1.to(device), sample_rate=sample_rate, tau=tau, cfg=cfg
        )
        if m0.numel():
            motions_0.append(m0)
        if m1.numel():
            motions_1.append(m1)

    m0, m1 = infer_engine.finalize_stream(tau=tau, cfg=cfg)
    if m0.numel():
        motions_0.append(m0)
    if m1.numel():
        motions_1.append(m1)

    assert motions_0, "Streaming produced no motion frames — audio shorter than one patch?"
    pred_code_0 = infer_engine.smooth_motion_savgol(torch.cat(motions_0, dim=0))
    pred_code_1 = infer_engine.smooth_motion_savgol(torch.cat(motions_1, dim=0))

    # Render — mirror infer_audio.py:58-90.
    if not hasattr(infer_engine, "face_decoder"):
        infer_engine._init_face_decoder()
    colors = infer_engine.face_decoder.get_colors()

    verts_0 = infer_engine.face_decoder.get_flame_verts(pred_code_0[None])[0]
    frames_0 = torch.cat(
        [infer_engine.face_renderer(v[None], colors=colors)[0] for v in verts_0], dim=0
    ).cpu()

    if two_speaker:
        verts_1 = infer_engine.face_decoder.get_flame_verts(pred_code_1[None])[0]
        frames_1 = torch.cat(
            [infer_engine.face_renderer(v[None], colors=colors)[0] for v in verts_1], dim=0
        ).cpu()
        vis_frames = torch.cat([frames_0, frames_1], dim=-1)
    else:
        vis_frames = frames_0

    audio_len = vis_frames.shape[0] * sample_rate // fps
    if two_speaker:
        vis_audio = (audio_0[:audio_len] + audio_1[:audio_len]) / 2
    else:
        vis_audio = audio_0[:audio_len]
    peak = vis_audio.abs().max()
    if peak > 1e-6:
        vis_audio = vis_audio / peak * 0.9

    os.makedirs(dump_dir, exist_ok=True)
    save_name = os.path.splitext(os.path.basename(audio_path))[0]
    if two_speaker:
        save_name += f"_x_{os.path.splitext(os.path.basename(audio_path_2))[0]}"
    dump_path = os.path.join(
        dump_dir, f"{save_name}_tau{tau}_cfg{cfg}_chunk{chunk_ms}ms_stream.mp4"
    )

    write_video(vis_frames, dump_path, fps, vis_audio, sample_rate, "aac")
    print(f"Streaming inference done. Saved to {dump_path} ({vis_frames.shape[0]} frames)")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*The `srun` command is available.*")
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_path", "-r", type=str, required=True)
    parser.add_argument("--audio", "-a", type=str, required=True, help="Path to speaker 0 audio file")
    parser.add_argument("--audio2", type=str, default=None, help="Path to speaker 1 audio file (optional)")
    parser.add_argument("--dump_dir", "-d", type=str, default="./render_results")
    parser.add_argument("--tau", default=1.0, type=float)
    parser.add_argument("--cfg", default=1.5, type=float)
    parser.add_argument(
        "--chunk_ms",
        default=400,
        type=int,
        help="Caller-side audio chunk size in milliseconds. The engine internally "
        "buffers until a full 4-second patch accumulates before emitting motion.",
    )
    parser.add_argument("--seed", default=0, type=int, help="torch.manual_seed; pass -1 to skip seeding.")
    args = parser.parse_args()
    print("Command Line Args: {}".format(args))

    torch.set_float32_matmul_precision("high")
    infer_audio_streaming(
        args.resume_path,
        args.audio,
        args.audio2,
        args.dump_dir,
        args.tau,
        args.cfg,
        args.chunk_ms,
        seed=None if args.seed < 0 else args.seed,
    )
