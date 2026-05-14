# UniLS Streaming Architecture Plan

## Context

UniLS is the two-speaker evolution of ARTalk. Its current inference path
(`infer_audio.py` → `InferEngine.inference()` → `UniLSGen.inference()` /
`UniLSFreeGen.inference()`) is **file-based**: a complete audio waveform is
loaded, internally split into 4-second patches, then fed sequentially to the
model. There is no way for an external caller (e.g. a websocket / RTC frontend,
or a live mic) to push audio incrementally and receive motion as it arrives.

ARTalk's `feature/streaming/chunks` branch already solved this for the
single-speaker case with a minimal-surgery design:
- Add `init_stream_state(batch)` + `stream_step(audio_chunk, state)` to the
  model (state is just a dict of tensors carried across calls).
- Add `start_stream / push_audio / finalize_stream` to the engine
  (`inference.py`), which buffers raw audio until a full 4-second patch
  accumulates, then calls `stream_step` and emits the new motion frames.
- The offline `inference()` is rewritten as `init_stream_state` + a `stream_step`
  loop (single source of truth).

We will port the same pattern to UniLS, with two adaptations forced by the
codebase differences confirmed during exploration:

1. **Two-speaker coupling.** UniLS's `UniLSGen.inference()` generates motion
   for ONE speaker per call, but conditions the model on BOTH speakers' audio
   (cross-attention over `audio_feat_0` = self and `audio_feat_1` = other —
   see `core/models/unils_gen/models.py:111-118, 196-207`). `infer_audio.py`
   calls `inference()` twice with audio channels swapped
   (`infer_audio.py:45-52`). The streaming engine must therefore maintain
   **two model-level states** (one per speaker) and run `stream_step` twice
   per accumulated patch — once with `(audio_0, audio_1)`, once with
   `(audio_1, audio_0)`.
2. **CFG batch-doubling stays inside `stream_step`.** UniLS's inference loop
   doubles the batch with an unconditional copy for classifier-free guidance
   (`unils_gen/models.py:185-192`). The state's `prev_motion_code` is also
   doubled (`prev_motion_code = torch.cat([pred_motion_code, pred_motion_code],
   dim=0)` at `unils_gen/models.py:233`). We keep that pattern: `stream_step`
   accepts a single (un-doubled) `prev_motion_code` in state and does the
   doubling internally each step, the way the current loop does today.

**User-confirmed decisions:**
- API shape: **paired push** — `push_audio(audio_0, audio_1)` returns
  `(motion_0, motion_1)` for any completed patches.
- Scope: **`UniLSGen` and `UniLSFreeGen`**. Skip `UniLSCodec` (VAE submodule,
  not used standalone).
- Chunk size: **4 seconds** (`patch_nums[-1] = 100` frames × 640 samples/frame
  at 16 kHz). No retraining.
- Offline path: **keep current `inference()` methods untouched.** Add
  `init_stream_state` / `stream_step` alongside. This trades a bit of code
  duplication for zero risk of regressing the existing offline numerics and
  CLI behaviour.

**Outcome:** A new engine API
(`InferEngine.start_stream / push_audio / pull_motion / finalize_stream`) that
accepts paired audio chunks of arbitrary length/sample-rate and emits
`(motion_0, motion_1)` blocks of up to 100 frames each, with autoregressive
state carried across chunks. The existing `infer_audio.py` continues to work
unchanged.

---

## Files to modify / create

### 1. `core/models/unils_gen/models.py` — add streaming methods

Add two new methods to `UniLSGen` (alongside the existing `inference()` at
line 153). The structure mirrors ARTalk's `init_stream_state` + `stream_step`
but adapted for UniLS's dual-audio cross-attention and CFG.

**`init_stream_state(style_motion_code=None, device=None) → dict`**

Responsibility: compute everything that is constant across an utterance for a
single speaker, plus the initial mutable state.

Behaviour, taken directly from `unils_gen/models.py:159-193`:
- If `style_motion_code is None`: build zeros of shape
  `[1, patch_len, motion_dim]` (matches `inference()` line 162).
- Otherwise: take `style_motion_code.unbind(dim=1)[0]` to drop the dyad axis
  (matches line 160).
- Build the CFG-doubled style: `style_motion_code_cfg = cat([style, zeros])`
  → `[2, patch_len, motion_dim]` (matches line 192).
- Pre-compute `style_feat = code_token_embed(get_motion_feat(style_cfg))`
  once and stash it (matches line 193). This is the only constant feature.
- Initialise `prev_motion_code` as zeros `[1, patch_len, motion_dim]`
  (matches lines 164-165). The CFG-doubling for `prev_motion_code` happens
  inside `stream_step` each call (matches line 189).
- Return:
  ```python
  {
      "style_feat": style_feat,          # [2, sum(patch_nums), attn_dim], constant
      "prev_motion_code": prev_motion_code,  # [1, patch_len, motion_dim], mutable
      "sos_token": self.sos_embed.expand(2, 1, -1),  # constant
  }
  ```

**`stream_step(audio_chunk, state, tau=1.0, cfg=2.0) → (pred_motion_code, state)`**

Responsibility: process exactly one 4-second patch of paired audio for ONE
speaker, with audio channel 0 = self, channel 1 = other.

Signature:
```python
@torch.inference_mode()
def stream_step(self, audio_chunk, state, tau=1.0, cfg=2.0):
    # audio_chunk: [1, 2, patch_audio_length]  (channel 0 = self, 1 = other)
    # state: dict from init_stream_state
    # returns: pred_motion_code [1, patch_len, motion_dim], updated state
```

Body — port of the inner block of `inference()` at lines 196-233:

```python
patch_len = max(self.patch_nums)
patch_audio_length = int(patch_len * self._sample_rate / self._motion_fps)
assert audio_chunk.shape == (1, 2, patch_audio_length), \
    f"stream_step expects [1, 2, {patch_audio_length}], got {tuple(audio_chunk.shape)}"

# Encode this patch's audio (mirror lines 174-182).
audio_0, audio_1 = audio_chunk.unbind(dim=1)        # each [1, T]
af_0 = self.audio_encoder(audio_0)                  # [1, L, audio_dim]
af_1 = self.audio_encoder(audio_1)
audio_uncond = af_0.new_zeros(af_0.shape)
curr_audio_feat_0 = torch.cat([af_0, audio_uncond], dim=0)  # CFG, [2, L, audio_dim]
curr_audio_feat_1 = torch.cat([af_1, audio_uncond], dim=0)

# CFG-double the carried prev_motion_code (mirror line 189).
prev_motion_code_single = state["prev_motion_code"]               # [1, patch_len, motion_dim]
prev_uncond = prev_motion_code_single.new_zeros(prev_motion_code_single.shape)
prev_motion_code = torch.cat([prev_motion_code_single, prev_uncond], dim=0)
prev_feat = self.code_token_embed(self.get_motion_feat(prev_motion_code))

# Hierarchical patch decoding (verbatim port of lines 198-230).
sos_token = state["sos_token"]
style_feat = state["style_feat"]
next_ar_vqfeat = sos_token
patch_motion_bits = []
for pidx, pn in enumerate(self.patch_nums):
    attn_feat = self.attn_blocks(
        next_ar_vqfeat, curr_audio_feat_0, curr_audio_feat_1, prev_feat, style_feat
    )
    motion_logits = self.logits_head(attn_feat)
    motion_logits = motion_logits[:, sum(self.patch_nums[:pidx]):]
    motion_logits = motion_logits.mul(1 / tau)
    motion_logits = motion_logits.view(motion_logits.shape[0], motion_logits.shape[1], -1, 2)
    if cfg > 1.0:
        motion_logits = cfg * motion_logits[:1] + (1 - cfg) * motion_logits[1:]
    else:
        motion_logits = motion_logits[:1]
    motion_bits = sample_idx_with_top_p_(motion_logits)
    patch_motion_bits.append(motion_bits)
    if pidx < len(self.patch_nums) - 1:
        nxt = self.base_codec.vqidx_to_next_feat(
            torch.cat(patch_motion_bits, dim=1), pidx, "accum_next"
        )
        nxt = self.code_token_embed(nxt)
        nxt = torch.cat([nxt, nxt], dim=0)            # CFG
        next_ar_vqfeat = torch.cat([sos_token, nxt], dim=1)

patch_motion_bits = torch.cat(patch_motion_bits, dim=1)
pred_motion_code = self.base_codec.vqidx_to_motion(patch_motion_bits)  # [1, patch_len, motion_dim]

# Update state — single (un-doubled) form, redoubling happens next call.
state["prev_motion_code"] = pred_motion_code
return pred_motion_code, state
```

Reuse: `audio_encoder`, `code_token_embed`, `get_motion_feat`, `attn_blocks`,
`logits_head`, `base_codec.vqidx_to_next_feat`, `base_codec.vqidx_to_motion`,
`sample_idx_with_top_p_` — all already on `self`. No new submodules.

**Do not modify `UniLSGen.inference()`.** The new methods are additive.

### 2. `core/models/unils_freegen/models.py` — add streaming methods

UniLSFreeGen has no audio input; its "streaming" is really *pull-style*: each
call produces the next 4 s of motion for one speaker conditioned on
`style_motion_code` and the carried `prev_motion_code`. The current
`inference()` (lines 102-166) already runs that loop `frame_chunk_length` times
to produce a fixed 20 s output.

**`init_stream_state(style_motion_code) → dict`**

- Assert `style_motion_code.shape[0] == 1` (matches line 105).
- CFG-double: `style_motion_code_cfg = cat([style, zeros])` (lines 116-117).
- Pre-compute `style_feat` once (line 118).
- Initial `prev_motion_code = style_motion_code` (matches line 115 — note:
  freegen seeds `prev_motion_code` with the *style*, unlike `unils_gen` which
  seeds with zeros; preserve this behaviour).
- Return:
  ```python
  {
      "style_feat": style_feat,
      "prev_motion_code": style_motion_code,   # [1, patch_len, motion_dim]
      "sos_token": self.sos_embed.expand(2, 1, -1),
  }
  ```

**`stream_step(state, tau=1.0, cfg=2.0) → (pred_motion_code, state)`**

No audio input. Body is a verbatim port of the *inner* `for _ in range(...)`
body of `inference()` at lines 122-153:
- Recompute `prev_feat` from `state["prev_motion_code"]` (CFG-doubled).
- Run the `pidx, pn` patch loop with `attn_blocks(next_ar_vqfeat, prev_feat,
  style_feat)` (no audio arguments — UniLSFreeGen's `MixedARTalkDecoder` takes
  3 args, not 5; see line 127).
- Return `pred_motion_code` and updated state (with new `prev_motion_code`).

Keep `inference()` untouched.

### 3. `core/trainer/inferencer.py` — add streaming API to `InferEngine`

Add the following methods to `InferEngine` (`core/trainer/inferencer.py:16`).
The engine owns audio buffering, sample-rate coercion, two per-speaker
`stream_state` dicts, and the post-processing pipeline.

**Detection of model kind.** At engine construction, after `self.model` is set,
record whether the loaded model is audio-driven or free-gen:

```python
self._is_audio_driven = hasattr(self.model, "audio_encoder")
```

(UniLSGen has `audio_encoder`; UniLSFreeGen does not.)

**`start_stream(style_motion_0=None, style_motion_1=None)`**

```python
def start_stream(self, style_motion_0=None, style_motion_1=None):
    """Initialise per-speaker streaming state.

    style_motion_X: optional [1, patch_len, motion_dim] tensor. None → zeros
    (UniLSGen) or required (UniLSFreeGen, will raise).
    """
    self._stream_state_0 = self.model.init_stream_state(style_motion_0)
    self._stream_state_1 = self.model.init_stream_state(style_motion_1)
    self._stream_sample_rate = int(self.meta_cfg.DATASET.AUDIO_SAMPLE_RATE)  # 16000
    self._stream_patch_audio_length = int(
        max(self.model.patch_nums) * self._stream_sample_rate
        / self.meta_cfg.DATASET.MOTION_FPS
    )  # = 64000
    self._stream_buf_0 = torch.zeros(0, device=self.device)
    self._stream_buf_1 = torch.zeros(0, device=self.device)
```

**`push_audio(audio_0, audio_1, sample_rate=16000) → (motion_0, motion_1)`** —
audio-driven only

```python
@torch.inference_mode()
def push_audio(self, audio_0, audio_1, sample_rate=16000, tau=1.0, cfg=1.5):
    if not self._is_audio_driven:
        raise RuntimeError("push_audio is only valid for audio-driven models. "
                           "For UniLSFreeGen use pull_motion().")
    if self._stream_state_0 is None:
        raise RuntimeError("Stream not initialised. Call start_stream() first.")

    a0 = self._coerce_audio(audio_0, sample_rate)   # → [T], 16 kHz, mono, on device
    a1 = self._coerce_audio(audio_1, sample_rate)

    # Zero-pad the shorter one so we always emit synchronised chunks.
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

        # Speaker 0: self=audio_0, other=audio_1
        paired_0 = torch.stack([patch_0, patch_1], dim=0).unsqueeze(0)
        m0, self._stream_state_0 = self.model.stream_step(
            paired_0, self._stream_state_0, tau=tau, cfg=cfg
        )
        # Speaker 1: self=audio_1, other=audio_0 (swap channels, matches infer_audio.py:51)
        paired_1 = torch.stack([patch_1, patch_0], dim=0).unsqueeze(0)
        m1, self._stream_state_1 = self.model.stream_step(
            paired_1, self._stream_state_1, tau=tau, cfg=cfg
        )
        out_0.append(m0[0])  # drop batch dim → [patch_len, motion_dim]
        out_1.append(m1[0])

    if not out_0:
        empty = torch.empty((0, self.model.motion_dim), device=self.device)
        return empty, empty
    return torch.cat(out_0, dim=0), torch.cat(out_1, dim=0)
```

**`pull_motion(num_chunks=1, tau=1.0, cfg=1.5) → (motion_0, motion_1)`** —
audio-free only

```python
@torch.inference_mode()
def pull_motion(self, num_chunks=1, tau=1.0, cfg=1.5):
    if self._is_audio_driven:
        raise RuntimeError("pull_motion is only valid for audio-free models. "
                           "For UniLSGen use push_audio().")
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
```

**`finalize_stream(tau=1.0, cfg=1.5) → (motion_0, motion_1)`** — flushes the tail

```python
@torch.inference_mode()
def finalize_stream(self, tau=1.0, cfg=1.5):
    if not self._is_audio_driven:
        # Nothing to drain for audio-free streams.
        return torch.empty((0, self.model.motion_dim), device=self.device), \
               torch.empty((0, self.model.motion_dim), device=self.device)

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
    m0, self._stream_state_0 = self.model.stream_step(paired_0, self._stream_state_0, tau=tau, cfg=cfg)
    m1, self._stream_state_1 = self.model.stream_step(paired_1, self._stream_state_1, tau=tau, cfg=cfg)

    keep = math.ceil(rem / self._stream_sample_rate * self.meta_cfg.DATASET.MOTION_FPS)
    self._stream_buf_0 = self._stream_buf_0.new_zeros(0)
    self._stream_buf_1 = self._stream_buf_1.new_zeros(0)
    return m0[0][:keep], m1[0][:keep]
```

**`_coerce_audio(audio, sample_rate) → 1D float tensor on device, 16 kHz mono`**

Helper that mirrors `read_audio` semantics but for arbitrary in-memory input:

```python
def _coerce_audio(self, audio, sample_rate):
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)
    if audio is None:
        # caller signals silence — same length is enforced by push_audio padding.
        return torch.zeros(0, device=self.device)
    if audio.dim() == 2:
        audio = audio.mean(dim=0)
    assert audio.dim() == 1, f"Expected 1D mono audio, got shape {tuple(audio.shape)}"
    audio = audio.to(self.device).float()
    target_sr = self._stream_sample_rate
    if sample_rate != target_sr:
        audio = torchaudio.transforms.Resample(sample_rate, target_sr).to(self.device)(audio[None])[0]
    return audio
```

Imports to add at the top of `inferencer.py`: `math`, `numpy as np`,
`torch.nn.functional as F`, `torchaudio`.

### 4. `infer_audio_streaming.py` — new validation entry point

Create a new top-level script that mirrors `infer_audio.py` but exercises the
streaming API end-to-end. Purpose: parity test (offline vs streamed motion
should be approximately equal for the same audio, modulo non-determinism from
multinomial sampling — fix the seed) and a runnable example for downstream
users.

Structure:

```python
@torch.inference_mode()
def infer_audio_streaming(resume_path, audio_path, audio_path_2=None,
                          dump_dir="./render_results", tau=1.0, cfg=1.5,
                          chunk_ms=400, seed=0):
    torch.manual_seed(seed)
    engine = InferEngine(resume_path)
    fps = int(engine.meta_cfg.DATASET.MOTION_FPS)
    sr  = int(engine.meta_cfg.DATASET.AUDIO_SAMPLE_RATE)

    audio_0, _ = read_audio(audio_path, target_sr=sr)
    if audio_path_2 is not None:
        audio_1, _ = read_audio(audio_path_2, target_sr=sr)
    else:
        audio_1 = torch.zeros_like(audio_0)

    # Pad to common length so finalize math is clean.
    n = max(audio_0.shape[0], audio_1.shape[0])
    audio_0 = F.pad(audio_0, (0, n - audio_0.shape[0]))
    audio_1 = F.pad(audio_1, (0, n - audio_1.shape[0]))

    engine.start_stream()  # zero style for both speakers
    chunk_samples = int(chunk_ms / 1000.0 * sr)
    chunks_0 = audio_0.split(chunk_samples)
    chunks_1 = audio_1.split(chunk_samples)
    motions_0, motions_1 = [], []
    for c0, c1 in zip(chunks_0, chunks_1):
        m0, m1 = engine.push_audio(c0, c1, sample_rate=sr, tau=tau, cfg=cfg)
        if m0.numel(): motions_0.append(m0)
        if m1.numel(): motions_1.append(m1)
    m0, m1 = engine.finalize_stream(tau=tau, cfg=cfg)
    if m0.numel(): motions_0.append(m0)
    if m1.numel(): motions_1.append(m1)

    pred_code_0 = engine.smooth_motion_savgol(torch.cat(motions_0, dim=0))
    pred_code_1 = engine.smooth_motion_savgol(torch.cat(motions_1, dim=0))

    # Render exactly like infer_audio.py — reuse the same rendering block.
    ...
```

The rendering tail is identical to `infer_audio.py:58-90` — copy it (or
factor into a shared helper if doing so is trivial; otherwise just copy to
avoid touching the existing script).

### Files NOT modified

- `infer_audio.py` — unchanged; offline path is preserved.
- `core/models/unils_codec/*` — VAE submodule, out of scope.
- Configs (`configs/unils_*.yaml`) — `patch_nums[-1]=100` is fine; no config
  changes needed.

---

## Reused existing utilities (no duplication)

| Need | Existing utility | Location |
|---|---|---|
| Audio load + resample (offline) | `read_audio` | `core/libs/utils_videos.py` |
| Top-p sampling | `sample_idx_with_top_p_` | `core/models/unils_gen/models.py:341` (and a sibling copy in `unils_freegen/models.py:266`) |
| VQ ↔ motion conversion | `base_codec.vqidx_to_motion`, `vqidx_to_next_feat`, `quant_to_sum_feat` | `core/models/unils_codec` |
| Audio encoder | `self.audio_encoder` (Mimi or Wav2Vec2) | `core/models/modules/{mimi.py,…}` |
| Savitzky-Golay post-smoothing | `InferEngine.smooth_motion_savgol` | `core/trainer/inferencer.py:156` |
| Face mesh render | `FLAMEModel`, `RenderMesh` | `core/libs/flame_model/` |

---

## Verification

End-to-end, the streaming path should produce motion that closely matches the
offline path for the same audio. The sampling is non-deterministic
(`multinomial` in `sample_idx_with_top_p_`), so absolute equality is not
expected — fix `torch.manual_seed` and compare distributional metrics.

1. **Smoke test — audio-driven, 1 speaker**:
   ```bash
   python infer_audio_streaming.py -r <ckpt> -a path/to/speaker0.wav --chunk_ms 400
   ```
   Confirm the resulting video has the same duration as `infer_audio.py` on
   the same input and looks visually similar.

2. **Smoke test — audio-driven, 2 speakers**:
   ```bash
   python infer_audio_streaming.py -r <ckpt> -a spk0.wav --audio2 spk1.wav --chunk_ms 800
   ```
   Confirm two motion tracks render side-by-side (same dual-render layout as
   `infer_audio.py`).

3. **Parity sanity check**: with `seed=0` and `chunk_ms` chosen so each push
   triggers exactly one `stream_step` (e.g. `chunk_ms = 4000`), the streaming
   output and the offline output should differ only at the RNG level. Run
   both on the same input and assert
   `torch.allclose(stream_motion, offline_motion, atol=1e-3)` or at minimum
   confirm per-channel std/mean are within 5%.

4. **Free-gen smoke test** (audio-free):
   ```python
   engine = InferEngine(freegen_ckpt)
   engine.start_stream(style_motion_0=style_a, style_motion_1=style_b)
   motion_0, motion_1 = engine.pull_motion(num_chunks=5)  # 5 × 4 s = 20 s
   # Compare to engine.model.inference(style_a).pred_motion_code shape / range.
   ```

5. **Boundary correctness**: push `N` chunks of `100 ms` each, where total
   audio = 4.3 s. Confirm:
   - `push_audio` returns 100 frames of motion exactly once (after second
     1–4 of audio accumulates).
   - `finalize_stream` returns the remaining ~7-8 frames (math: 0.3 s × 25 fps).
   - No assertion errors from the `stream_step` shape check.

6. **No regression in offline path**: run the existing `infer_audio.py`
   command from the README and confirm bit-identical output (with seed) to
   pre-change. The offline `inference()` was not touched, so this is a
   structural check that nothing else in the import graph drifted.
