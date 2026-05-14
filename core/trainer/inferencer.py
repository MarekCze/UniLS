#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from scipy.signal import savgol_filter

from core.libs.flame_model import FLAMEModel, RenderMesh
from core.libs.utils import ConfigDictWrapper, find_latest_model
from core.libs.utils_videos import write_video
from core.models import build_model
from core.models.modules import calc_val_metrics


class InferEngine:
    def __init__(self, resume_path, device="cuda:0", meta_cfg=None):
        # build config
        if os.path.isdir(resume_path):
            resume_path = find_latest_model(os.path.join(resume_path, "checkpoints"))
        print(f"Loading model from {resume_path}...")
        self.device = device

        # load checkpoint
        full_checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
        if meta_cfg is None:
            meta_cfg = self._extract_meta_cfg(full_checkpoint)
            if meta_cfg is None:
                raise KeyError(
                    "Cannot find model config in checkpoint. "
                    "Please pass `meta_cfg` to InferEngine or save `meta_cfg` in checkpoint."
                )
        self.meta_cfg = meta_cfg

        # build model
        model = build_model(meta_cfg.MODEL, init_submodule=False).to(device)
        model_state = self._extract_model_state(full_checkpoint)
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        if len(unexpected) > 0:
            print(f"Unexpected keys: {unexpected}")
        if len(missing) > 0:
            missing = list(set([m.split(".")[0] for m in missing]))
            print(f"Missing keys: {missing}")
        model.eval()
        self.model = model

        # Streaming state (initialised by start_stream).
        self._is_audio_driven = hasattr(self.model, "audio_encoder")
        self._stream_state_0 = None
        self._stream_state_1 = None
        self._stream_buf_0 = None
        self._stream_buf_1 = None
        self._stream_sample_rate = None
        self._stream_patch_audio_length = None

    @staticmethod
    def _extract_meta_cfg(full_checkpoint):
        if not isinstance(full_checkpoint, dict):
            return None
        _meta_cfg = full_checkpoint.get("meta_cfg", None)
        _meta_cfg["MODEL"]["PRETRAIN_PATH"] = None
        _meta_cfg["MODEL"]["VAE_CONFIG"]["STATS_PATH"] = "./assets/talk_motion_stats.json"
        return ConfigDictWrapper(_meta_cfg)

    @staticmethod
    def _extract_model_state(full_checkpoint):
        if not isinstance(full_checkpoint, dict):
            raise TypeError("Checkpoint must be a dict-like object.")

        state = None
        for key in ("model", "state_dict", "model_state_dict", "net"):
            if key in full_checkpoint and isinstance(full_checkpoint[key], dict):
                state = full_checkpoint[key]
                break
        if state is None and all(isinstance(v, torch.Tensor) for v in full_checkpoint.values()):
            state = full_checkpoint
        if state is None:
            raise KeyError(
                "Cannot find model weights in checkpoint. Expected keys: model/state_dict/model_state_dict/net."
            )

        # Try to normalize common training wrappers: module.*, model.*, net.*
        strip_prefixes = ("module.", "model.", "net.")
        normalized_state = {}
        for key, value in state.items():
            new_key = key
            for prefix in strip_prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
            normalized_state[new_key] = value
        return normalized_state

    def _init_face_decoder(self):
        # build face decoder
        self.face_decoder = FLAMEModel(n_shape=300, n_exp=100)
        self.face_decoder.eval()
        self.face_decoder.to(self.device)
        self.face_renderer = RenderMesh(image_size=512, faces=self.face_decoder.get_faces())

    @torch.inference_mode()
    def inference(self, batch_data, dump_path=None, clip_length=20, tau=1.0, cfg=2.0):
        for key in batch_data:
            if isinstance(batch_data[key], torch.Tensor):
                batch_data[key] = batch_data[key].to(self.device)
        # inference
        infer_results = self.model.inference(**batch_data, tau=tau, cfg=cfg)
        # print("pred motion code shape: ", infer_results["pred_motion_code"][0].shape)
        if dump_path is not None:
            pred_motion_code = infer_results["pred_motion_code"][0]
            pred_motion_code = self.smooth_motion_savgol(pred_motion_code)
            if "gt_motion_code" in infer_results.keys():
                gt_motion_code = infer_results["gt_motion_code"][0]
            else:
                gt_motion_code = None
            vis_audios = infer_results["audio"][0]
            self.visualize(
                pred_motion_code,
                dump_path=dump_path,
                vis_audios=vis_audios,
                gt_motion_code=gt_motion_code,
                render_length=clip_length,
                fps=25,
                sample_rate=16000,
            )
        return infer_results

    # ------------------------------------------------------------------ #
    # Streaming API                                                       #
    # ------------------------------------------------------------------ #

    def start_stream(self, style_motion_0=None, style_motion_1=None):
        """Initialise per-speaker streaming state.

        For audio-driven models (UniLSGen), style may be None (→ zeros).
        For audio-free models (UniLSFreeGen), style is required.
        """
        self._stream_state_0 = self.model.init_stream_state(style_motion_0)
        self._stream_state_1 = self.model.init_stream_state(style_motion_1)
        self._stream_sample_rate = int(self.meta_cfg.DATASET.AUDIO_SAMPLE_RATE)
        self._stream_patch_audio_length = int(
            max(self.model.patch_nums)
            * self._stream_sample_rate
            / self.meta_cfg.DATASET.MOTION_FPS
        )
        self._stream_buf_0 = torch.zeros(0, device=self.device)
        self._stream_buf_1 = torch.zeros(0, device=self.device)

    @torch.inference_mode()
    def push_audio(self, audio_0, audio_1, sample_rate=16000, tau=1.0, cfg=1.5):
        """Push paired audio chunks; emit (motion_0, motion_1) for any completed patches.

        audio_0 / audio_1 may be torch tensors (1D or 2D mono), numpy arrays, or None.
        Returns two [frames, motion_dim] tensors (possibly empty).
        """
        if not self._is_audio_driven:
            raise RuntimeError(
                "push_audio is only valid for audio-driven models. "
                "For UniLSFreeGen use pull_motion()."
            )
        if self._stream_state_0 is None:
            raise RuntimeError("Stream not initialised. Call start_stream() first.")

        a0 = self._coerce_audio(audio_0, sample_rate)
        a1 = self._coerce_audio(audio_1, sample_rate)

        # Zero-pad the shorter side so the two buffers always grow in lockstep.
        if a0.numel() != a1.numel():
            n = max(a0.numel(), a1.numel())
            a0 = F.pad(a0, (0, n - a0.numel()))
            a1 = F.pad(a1, (0, n - a1.numel()))

        self._stream_buf_0 = torch.cat([self._stream_buf_0, a0])
        self._stream_buf_1 = torch.cat([self._stream_buf_1, a1])

        out_0, out_1 = [], []
        plen = self._stream_patch_audio_length
        while self._stream_buf_0.numel() >= plen:
            patch_0 = self._stream_buf_0[:plen]
            patch_1 = self._stream_buf_1[:plen]
            self._stream_buf_0 = self._stream_buf_0[plen:]
            self._stream_buf_1 = self._stream_buf_1[plen:]

            # Speaker 0: self=audio_0, other=audio_1.
            paired_0 = torch.stack([patch_0, patch_1], dim=0).unsqueeze(0)
            m0, self._stream_state_0 = self.model.stream_step(
                paired_0, self._stream_state_0, tau=tau, cfg=cfg
            )
            # Speaker 1: self=audio_1, other=audio_0 (swap channels).
            paired_1 = torch.stack([patch_1, patch_0], dim=0).unsqueeze(0)
            m1, self._stream_state_1 = self.model.stream_step(
                paired_1, self._stream_state_1, tau=tau, cfg=cfg
            )
            out_0.append(m0[0])
            out_1.append(m1[0])

        if not out_0:
            empty = torch.empty((0, self.model.motion_dim), device=self.device)
            return empty, empty
        return torch.cat(out_0, dim=0), torch.cat(out_1, dim=0)

    @torch.inference_mode()
    def pull_motion(self, num_chunks=1, tau=1.0, cfg=1.5):
        """Generate the next `num_chunks * patch_len` motion frames for both speakers.

        Only valid for audio-free models (UniLSFreeGen).
        """
        if self._is_audio_driven:
            raise RuntimeError(
                "pull_motion is only valid for audio-free models. "
                "For UniLSGen use push_audio()."
            )
        if self._stream_state_0 is None:
            raise RuntimeError("Stream not initialised. Call start_stream() first.")
        out_0, out_1 = [], []
        for _ in range(num_chunks):
            m0, self._stream_state_0 = self.model.stream_step(
                self._stream_state_0, tau=tau, cfg=cfg
            )
            m1, self._stream_state_1 = self.model.stream_step(
                self._stream_state_1, tau=tau, cfg=cfg
            )
            out_0.append(m0[0])
            out_1.append(m1[0])
        return torch.cat(out_0, dim=0), torch.cat(out_1, dim=0)

    @torch.inference_mode()
    def finalize_stream(self, tau=1.0, cfg=1.5):
        """Drain the audio buffer with zero-padding; trim output to the actual audio length."""
        if not self._is_audio_driven:
            empty = torch.empty((0, self.model.motion_dim), device=self.device)
            return empty, empty
        if self._stream_state_0 is None:
            raise RuntimeError("Stream not initialised. Call start_stream() first.")

        rem = self._stream_buf_0.numel()
        if rem == 0:
            empty = torch.empty((0, self.model.motion_dim), device=self.device)
            return empty, empty

        plen = self._stream_patch_audio_length
        pad = plen - rem
        p0 = F.pad(self._stream_buf_0, (0, pad))
        p1 = F.pad(self._stream_buf_1, (0, pad))

        paired_0 = torch.stack([p0, p1], dim=0).unsqueeze(0)
        paired_1 = torch.stack([p1, p0], dim=0).unsqueeze(0)
        m0, self._stream_state_0 = self.model.stream_step(
            paired_0, self._stream_state_0, tau=tau, cfg=cfg
        )
        m1, self._stream_state_1 = self.model.stream_step(
            paired_1, self._stream_state_1, tau=tau, cfg=cfg
        )

        keep = math.ceil(
            rem / self._stream_sample_rate * self.meta_cfg.DATASET.MOTION_FPS
        )
        self._stream_buf_0 = self._stream_buf_0.new_zeros(0)
        self._stream_buf_1 = self._stream_buf_1.new_zeros(0)
        return m0[0][:keep], m1[0][:keep]

    def _coerce_audio(self, audio, sample_rate):
        """Convert input audio to a 1D float tensor on device, resampled to the stream rate."""
        if audio is None:
            return torch.zeros(0, device=self.device)
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        if not isinstance(audio, torch.Tensor):
            audio = torch.as_tensor(audio)
        if audio.dim() == 2:
            audio = audio.mean(dim=0)
        assert audio.dim() == 1, f"Expected 1D mono audio, got shape {tuple(audio.shape)}"
        audio = audio.to(self.device).float()
        target_sr = self._stream_sample_rate
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr).to(self.device)
            audio = resampler(audio[None])[0]
        return audio

    @torch.inference_mode()
    def visualize(
        self,
        pred_motion_code,
        dump_path,
        vis_audios=None,
        gt_motion_code=None,
        render_length=20,
        fps=30,
        sample_rate=16000,
    ):
        if not hasattr(self, "face_decoder"):
            self._init_face_decoder()
        assert pred_motion_code.dim() == 2, "pred_motion_code should be 2D"
        frame_length = int(render_length * fps)
        audio_length = int(render_length * sample_rate)
        pred_motion_code = pred_motion_code[:frame_length][None]
        pred_verts = self.face_decoder.get_flame_verts(pred_motion_code)[0]
        vis_frames = []
        for pidx, pred_vert in enumerate(pred_verts):
            vis_frame, _ = self.face_renderer(pred_vert[None], colors=self.face_decoder.get_colors())
            vis_frames.append(vis_frame)
        vis_frames = torch.cat(vis_frames, dim=0).cpu()
        if gt_motion_code is not None:
            gt_verts = self.face_decoder.get_flame_verts(gt_motion_code[None])[0]
            gt_frames = []
            for gidx, gt_vert in enumerate(gt_verts):
                gt_frame, _ = self.face_renderer(gt_vert[None], colors=self.face_decoder.get_colors())
                gt_frames.append(gt_frame)
            gt_frames = torch.cat(gt_frames, dim=0).cpu()
            vis_frames = torch.cat([gt_frames, vis_frames], dim=-1)
        if vis_audios is not None:
            assert vis_audios.dim() == 1, "vis_audios should be 1D"
            vis_audios = vis_audios[:audio_length]
            write_video(vis_frames, dump_path, fps, vis_audios, sample_rate, "aac")
        else:
            write_video(vis_frames, dump_path, fps)

    @staticmethod
    def smooth_motion_savgol(motion_code, window_length=9, polyorder=3):
        motion_np = motion_code.clone().detach().cpu().numpy()
        motion_smoothed = savgol_filter(motion_np, window_length=window_length, polyorder=polyorder, axis=0)
        motion_smoothed = torch.from_numpy(motion_smoothed).type_as(motion_code)
        return motion_smoothed

    @torch.inference_mode()
    def _calc_metrics(self, infer_results, batch_data):
        if not hasattr(self, "face_decoder"):
            self._init_face_decoder()
        gt_motion_code = batch_data["motion_code"]
        pred_motion_code = infer_results["pred_motion_code"]
        speech_mask = batch_data["speech_mask"]
        assert gt_motion_code.shape[0] == 1 and gt_motion_code.shape[1] == 2, "gt_motion_code [1, 2, len, 108]"
        assert pred_motion_code.shape[0] == 1, "pred_motion_code [1, len, 108]"
        assert speech_mask.shape[0] == 1, "speech_mask [1, 2, len]"
        gt_motion_code = gt_motion_code[:, 0]
        speaking_mask = speech_mask[0, 0].bool()  # [len]
        listening_mask = ~speaking_mask

        # generate flame verts once for full sequence
        gt_verts = self.face_decoder.get_flame_verts(gt_motion_code, with_headpose=False)
        pred_verts = self.face_decoder.get_flame_verts(pred_motion_code, with_headpose=False)
        _, gt_gpose_code, gt_jaw_code, _ = gt_motion_code.split([100, 3, 1, 4], dim=-1)
        _, pred_gpose_code, pred_jaw_code, _ = pred_motion_code.split([100, 3, 1, 4], dim=-1)

        # always return all keys (NaN for invalid) to avoid gather deadlock in multi-GPU
        _nan = torch.tensor(float("nan"), device=pred_motion_code.device)
        metrics = {
            "S_LVE": _nan,
            "S_MHD": _nan,
            "S_FDD": _nan,
            "S_PDD": _nan,
            "S_JDD": _nan,
            "L_FDD": _nan,
            "L_PDD": _nan,
            "L_JDD": _nan,
        }

        # Speaking metrics: LVE, MHD, FDD, PDD, JDD (std needs >= 2 frames)
        if speaking_mask.sum() > 1:
            s_lve, s_mhd, s_fdd = calc_val_metrics(pred_verts[:, speaking_mask], gt_verts[:, speaking_mask])
            s_pdd = self._calc_code_dispersion_delta(pred_gpose_code[:, speaking_mask], gt_gpose_code[:, speaking_mask])
            s_jdd = self._calc_code_dispersion_delta(pred_jaw_code[:, speaking_mask], gt_jaw_code[:, speaking_mask])
            metrics.update({"S_LVE": s_lve, "S_MHD": s_mhd, "S_FDD": s_fdd, "S_PDD": s_pdd, "S_JDD": s_jdd})

        # Listening metrics: FDD, PDD, JDD only (std needs >= 2 frames)
        if listening_mask.sum() > 1:
            _, _, l_fdd = calc_val_metrics(pred_verts[:, listening_mask], gt_verts[:, listening_mask])
            l_pdd = self._calc_code_dispersion_delta(
                pred_gpose_code[:, listening_mask], gt_gpose_code[:, listening_mask]
            )
            l_jdd = self._calc_code_dispersion_delta(pred_jaw_code[:, listening_mask], gt_jaw_code[:, listening_mask])
            metrics.update({"L_FDD": l_fdd, "L_PDD": l_pdd, "L_JDD": l_jdd})

        return metrics

    @staticmethod
    def _calc_code_dispersion_delta(pred_code, gt_code):
        # Use the same formula as FDD, applied on selected motion-code channels.
        std_pred = pred_code.std(dim=1)
        std_gt = gt_code.std(dim=1)
        return (std_pred - std_gt).abs().mean().detach() * 100
