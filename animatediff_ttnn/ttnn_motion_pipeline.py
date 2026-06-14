# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Phase 3: staged TTNN UNet forward pass with MotionAdapter temporal attention.

Replicates the TTNN UNet __call__ orchestration from:
  ~/tt-metal/models/demos/vision/generative/stable_diffusion/wormhole/tt/
  ttnn_functional_unet_2d_condition_model_new_conv.py

without modifying that file. Calls the same block objects in the same order,
inserting _apply_temporal() at 7 injection points between blocks.

The TTNN UNet is a monolithic __call__ — we cannot inject mid-call, so we
replicate the orchestration here and call each block object directly.
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

# Import ttnn at module level so tests can patch
# 'animatediff_ttnn.ttnn_motion_pipeline.ttnn'. In environments where the
# ttnn wheel is not installed (CI, unit-test runners) the import will fail;
# we fall back to None so the module still loads. _apply_temporal guards
# against this via the mock in tests.
try:
    import ttnn
except ModuleNotFoundError:
    ttnn = None  # type: ignore[assignment]

# Import to_device at module level for the same patchability reason.
try:
    from animatediff_ttnn.ttnn_pipeline import to_device
except Exception:
    to_device = None  # type: ignore[assignment]


def _apply_temporal(
    samples: list,
    module_list: list,
    device,
    num_frames: int,
    C: int,
    spatial_h: int,
    spatial_w: int,
    injection_alpha: float = 1.0,
) -> list:
    """Bridge between TTNN device tensors and diffusers AnimateDiffTransformer3D.

    Pulls N TTNN hidden states to CPU, reshapes to [B*N, C, H, W] (the format
    AnimateDiffTransformer3D.forward expects), runs the full diffusers temporal
    attention module (LayerNorm, QKV, positional embedding, feedforward), then
    pushes back to device with the same dtype/layout.

    Using the full diffusers module rather than a partial QKV-only kernel ensures
    all components (norm, proj_in/out, bias, positional embedding) are applied.

    Args:
        samples:     List of N TTNN tensors, each [1, 1, 2*S, C] where S=H*W.
                     (TTNN pre_process_input folds CFG batch=2 into spatial dim.)
        module_list: List of AnimateDiffTransformer3D modules (on CPU, eval mode).
                     Typically 2-3 for down/up blocks, 1 for mid_block.
        device:      TTNN device (MeshDevice from setup_blackhole).
        num_frames:  N (length of samples).
        C:           Channel dimension C.
        spatial_h:   Spatial height H at this UNet stage.
        spatial_w:   Spatial width  W at this UNet stage.
        injection_alpha: 0.0 = bypass (no-op for debugging), 1.0 = full injection.

    Returns:
        List of N TTNN tensors, same shape as input, with temporal attention applied.
        Original input tensors are deallocated.
    """
    # Step 1: pull all N frames to CPU as float32.
    # TTNN layout: [1, 1, 2*H*W, C] (batch=2 folded into spatial).
    # We need [2, C, H, W] per frame (CFG pair per-frame NCHW).
    orig_dtype = samples[0].dtype
    orig_memory_configs = [ttnn.get_memory_config(s) for s in samples]

    dram_samples = [
        ttnn.to_memory_config(s, ttnn.DRAM_MEMORY_CONFIG) if ttnn.get_memory_config(s).is_sharded() else s
        for s in samples
    ]
    raw_tensors = [ttnn.to_torch(s).float() for s in dram_samples]
    orig_shape = raw_tensors[0].shape  # [1, 1, 2*S, C] — save for reconstruction

    # Reshape [1, 1, 2*S, C] → [2, C, H, W] per frame.
    # S = H * W; CFG batch=2 was folded into spatial by pre_process_input.
    # raw[0,0, 0:S, :] = uncond frame i, raw[0,0, S:2S, :] = cond frame i.
    frame_tensors = []
    for raw in raw_tensors:
        flat = raw.reshape(2, spatial_h * spatial_w, C)           # [2, S, C]
        nchw = flat.permute(0, 2, 1).reshape(2, C, spatial_h, spatial_w)  # [2, C, H, W]
        frame_tensors.append(nchw)

    # Step 2: apply each AnimateDiffTransformer3D module sequentially.
    # Module expects [B*N, C, H, W]; B=2 (uncond+cond), N frames.
    # Stack N frames along batch → [2*N, C, H, W], run module, unstack.
    pre_energy = sum(t.norm().item() for t in frame_tensors)
    for module in module_list:
        # Stack: [2*N, C, H, W]
        stacked = torch.stack(frame_tensors, dim=1).reshape(2 * num_frames, C, spatial_h, spatial_w)

        with torch.no_grad():
            attended = module.forward(stacked, num_frames=num_frames)  # [2*N, C, H, W]

        # Unstack back to N tensors of [2, C, H, W]
        attended_frames = attended.reshape(2, num_frames, C, spatial_h, spatial_w).permute(1, 0, 2, 3, 4)
        attended_list = list(attended_frames)  # N tensors of [2, C, H, W]

        if injection_alpha >= 1.0:
            frame_tensors = attended_list
        else:
            frame_tensors = [
                (1.0 - injection_alpha) * frame_tensors[i] + injection_alpha * attended_list[i]
                for i in range(num_frames)
            ]

    post_energy = sum(t.norm().item() for t in frame_tensors)
    print(f"  [_apply_temporal C={C} {spatial_h}x{spatial_w}] "
          f"energy: {pre_energy:.1f} → {post_energy:.1f}"
          f"  ratio={post_energy/max(pre_energy,1e-6):.3f}  alpha={injection_alpha}")

    # Step 3: reshape back to [1, 1, 2*S, C] and push to device.
    out = []
    for i in range(num_frames):
        # [2, C, H, W] → [1, 1, 2*S, C]
        nchw = frame_tensors[i]                                        # [2, C, H, W]
        sc = nchw.reshape(2, C, -1).permute(0, 2, 1).reshape(orig_shape)  # [1,1,2*S,C]
        t_device = to_device(
            sc,
            device,
            dtype=orig_dtype,
            layout=ttnn.TILE_LAYOUT,
        )
        mc = orig_memory_configs[i]
        if mc.is_sharded():
            t_device = ttnn.to_memory_config(t_device, mc)
        out.append(t_device)
        samples[i].deallocate(True)

    return out


def forward_unet_staged(
    ttnn_model,
    frame_samples: list,
    timestep,
    encoder_hidden_states,
    config,
    temporal_kernels: dict,
    device,
    num_frames: int,
    *,
    attention_mask=None,
    cross_attention_kwargs=None,
    injection_alpha: float = 1.0,
) -> list:
    """Staged TTNN UNet forward pass with MotionAdapter temporal attention injection.

    Replicates the TTNN UNet __call__ orchestration (SD 1.4 architecture) but processes
    N frames and inserts _apply_temporal() at 7 injection points (3 down-blocks, 1
    mid-block, 3 up-blocks).  Each frame travels through the UNet independently for
    spatial operations; the temporal attention bridge cross-attends between all N frames
    at each injection point.

    This function NEVER modifies the TTNN UNet source file.  Instead it calls the same
    block objects (ttnn_model.down_blocks, ttnn_model.mid_block, ttnn_model.up_blocks)
    directly — they are plain Python callables.

    Args:
        ttnn_model:            Instantiated TTNN UNet2DConditionModel (the __call__-able
                               object from ttnn_functional_unet_2d_condition_model_new_conv).
        frame_samples:         List of N TTNN tensors, one per frame, shape [B, C, H, W]
                               (NCHW, same format as the single-frame UNet input).
        timestep:              Shared timestep embedding tensor (already passed through
                               time_proj / time_embedding — same object for all frames).
        encoder_hidden_states: Text conditioning tensor, shared across all frames.
        config:                SD config object (center_input_sample, etc.).
        temporal_kernels:      Dict mapping injection-point keys to lists of
                               TemporalAttentionKernel.  Expected keys:
                                 "down0", "down1", "down2"  — CrossAttn down blocks
                                 "mid"                       — mid block
                                 "up0", "up1", "up2"         — CrossAttn up blocks
                               A missing key means no injection at that point (e.g.,
                               "down3" / "up3" for DownBlock2D / UpBlock2D are omitted).
        device:                TTNN device (MeshDevice from setup_blackhole).
        num_frames:            N, the length of frame_samples.
        attention_mask:        Optional attention mask (always None in SD 1.4 usage).
        cross_attention_kwargs: Optional cross-attention kwargs dict.

    Returns:
        List of N TTNN tensors, each [B, 4, H, W] (predicted noise, NCHW), ready for
        the scheduler step (scheduler.step) and VAE decode.

    Architecture constants (SD 1.4 defaults matching the TTNN UNet weights):
        block_out_channels   = (320, 640, 1280, 1280)
        layers_per_block     = 2
        down_block_types     = ("CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
                                "CrossAttnDownBlock2D", "DownBlock2D")
        up_block_types       = ("CrossAttnUpBlock2D", "CrossAttnUpBlock2D",
                                "CrossAttnUpBlock2D", "UpBlock2D")
        time_embed_dim       = block_out_channels[0] * 4 = 1280
        attention_head_dim   = 8  (scalar → broadcasted to tuple)
        norm_num_groups      = 32
        norm_eps             = 1e-5
        cross_attention_dim  = 1280
        act_fn               = "silu"
        downsample_padding   = 1
    """
    # ------------------------------------------------------------------ imports
    # Deferred inside the function body to avoid ImportError when ttnn is absent
    # (CI unit-test runners, the functional simulator, etc.).
    from models.demos.vision.generative.stable_diffusion.wormhole.tt.ttnn_functional_utility_functions import (  # noqa: E501
        get_default_compute_config,
        pre_process_input,
    )
    from models.demos.vision.generative.stable_diffusion.wormhole.sd_helper_funcs import (
        reshard_for_output_channels_divisibility,
    )

    # ------------------------------------------------------------------ SD 1.4 constants
    # These mirror the default arguments in __call__ of
    # ttnn_functional_unet_2d_condition_model_new_conv.py.
    block_out_channels = (320, 640, 1280, 1280)
    layers_per_block = 2
    downsample_padding = 1
    mid_block_scale_factor = 1  # unused in forward path but kept for clarity
    act_fn = "silu"
    norm_num_groups = 32
    norm_eps = 1e-5
    cross_attention_dim = 1280
    attention_head_dim = 8  # scalar; broadcast to per-block tuple below
    only_cross_attention = False  # scalar; broadcast below
    dual_cross_attention = False
    use_linear_projection = False
    upcast_attention = False
    resnet_time_scale_shift = "default"
    time_embed_dim = block_out_channels[0] * 4  # 1280

    # Broadcast scalars to per-block tuples (matches __call__ lines 382-386).
    if isinstance(only_cross_attention, bool):
        only_cross_attention = [only_cross_attention] * len(ttnn_model.down_block_types)
    if isinstance(attention_head_dim, int):
        attention_head_dim = (attention_head_dim,) * len(ttnn_model.down_block_types)

    # ------------------------------------------------------------------ 1. Pre-process: pad → permute → reshape
    # Replicates __call__ lines 331-333 for each frame.
    # Input frame: [B, C, H, W] (NCHW).  After pad + permute: NHWC.
    # After reshape: [1, 1, B*H*W, C] (tiled, interleaved).
    processed_samples = []
    for frame in frame_samples:
        s = ttnn.pad(frame, padding=((0, 0), (0, 28), (0, 0), (0, 0)), value=0)
        s = ttnn.permute(s, (0, 2, 3, 1))  # NCHW → NHWC
        s = ttnn.reshape(s, (1, 1, s.shape[0] * s.shape[1] * s.shape[2], s.shape[3]))
        processed_samples.append(s)

    # ------------------------------------------------------------------ 2. conv_in for each frame
    # Replicates __call__ lines 338-378 (conv_in setup + per-frame call).
    # ttnn_model.conv_in_weights / conv_in_bias are updated in-place via
    # return_weights_and_bias=True (same pattern as __call__).
    out_channels = ttnn_model.parameters.conv_in.weight.shape[0]
    in_channels_conv = ttnn_model.parameters.conv_in.weight.shape[1]
    shard_layout = (
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        if in_channels_conv < 320
        else ttnn.TensorMemoryLayout.BLOCK_SHARDED
    )
    conv_in_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat8_b,
        shard_layout=shard_layout,
        reshard_if_not_optimal=True,
        enable_act_double_buffer=True,
        enable_weights_double_buffer=True,
    )
    compute_config = get_default_compute_config(ttnn_model.device)
    conv_in_kwargs = {
        "in_channels": in_channels_conv,
        "out_channels": out_channels,
        "batch_size": ttnn_model.batch_size,
        "input_height": ttnn_model.input_height,
        "input_width": ttnn_model.input_width,
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "padding": (1, 1),
        "dilation": (1, 1),
        "groups": 1,
        "device": ttnn_model.device,
        "conv_config": conv_in_config,
        "slice_config": ttnn.Conv2dL1FullSliceConfig,
    }

    hidden_samples = []
    for s in processed_samples:
        s, [ttnn_model.conv_in_weights, ttnn_model.conv_in_bias] = ttnn.conv2d(
            input_tensor=s,
            weight_tensor=ttnn_model.conv_in_weights,
            bias_tensor=ttnn_model.conv_in_bias,
            **conv_in_kwargs,
            compute_config=compute_config,
            dtype=ttnn.bfloat8_b,
            return_weights_and_bias=True,
        )
        s = reshard_for_output_channels_divisibility(s, out_channels)
        s = ttnn.reallocate(s)
        # Evict to DRAM so subsequent frames' conv_in L1 allocations don't clash.
        s = ttnn.to_memory_config(s, ttnn.DRAM_MEMORY_CONFIG)
        hidden_samples.append(s)

    # ------------------------------------------------------------------ 3. Time embedding (shared)
    # __call__ line 305: emb = self.emb(t_emb)
    # timestep is already the preprocessed t_emb tensor — shared across all frames.
    emb = ttnn_model.emb(timestep)

    # ------------------------------------------------------------------ 4. Down blocks
    # Replicates __call__ lines 389-445.
    # down_res_per_frame[i] accumulates the residual samples for frame i across all
    # down blocks (equivalent to down_block_res_samples in the original).
    output_channel = block_out_channels[0]
    down_res_per_frame = [[] for _ in range(num_frames)]

    # Prime residuals with the conv_in output (same as original line 390).
    # The original does: down_block_res_samples = (sample_copied_to_dram,)
    for i in range(num_frames):
        sample_dram = ttnn.to_memory_config(hidden_samples[i], ttnn.DRAM_MEMORY_CONFIG)
        down_res_per_frame[i].append(sample_dram)

    for block_idx, (down_block_type, down_block) in enumerate(
        zip(ttnn_model.down_block_types, ttnn_model.down_blocks)
    ):
        input_channel = output_channel
        output_channel = block_out_channels[block_idx]
        is_final_block = block_idx == len(block_out_channels) - 1

        if down_block_type == "CrossAttnDownBlock2D":
            # Process every frame through this block.
            # IMPORTANT: after each frame, evict the output to DRAM before running
            # the next frame.  The TTNN cross-attention kernel uses statically-allocated
            # circular buffers (L1 CBs).  If a previous frame's output tensor still sits
            # in L1 when the *same* program runs for the next frame, the static CBs clash
            # with the live L1 buffer → TT_THROW at program.cpp:1476.
            new_hidden_dram = []
            for i in range(num_frames):
                # Ensure input is in L1 (block's resnet entry point expects sharded L1).
                hs_in = hidden_samples[i]
                if ttnn.get_memory_config(hs_in).memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED:
                    # DRAM-interleaved → let the block re-shard internally via reshard_if_not_optimal
                    pass
                s, res_samples = down_block(
                    hidden_states=hs_in,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    num_layers=layers_per_block,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    add_downsample=not is_final_block,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    config=config,
                    resnet_groups=norm_num_groups,
                    downsample_padding=downsample_padding,
                    cross_attention_dim=cross_attention_dim,
                    attn_num_head_channels=attention_head_dim[block_idx],
                    dual_cross_attention=dual_cross_attention,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention[block_idx],
                    upcast_attention=upcast_attention,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                )
                down_res_per_frame[i].extend(list(res_samples))
                # Evict output to DRAM so next frame's L1 allocation doesn't conflict.
                s_dram = ttnn.to_memory_config(s, ttnn.DRAM_MEMORY_CONFIG)
                s.deallocate(True)
                new_hidden_dram.append(s_dram)

            hidden_samples = new_hidden_dram

            # Temporal attention injection at this CrossAttnDownBlock2D.
            key = f"down{block_idx}"
            if key in temporal_kernels and temporal_kernels[key]:
                from animatediff_ttnn.motion_weights import get_injection_point_info
                ip = get_injection_point_info(key)
                hidden_samples = _apply_temporal(
                    hidden_samples,
                    temporal_kernels[key],
                    device,
                    num_frames,
                    output_channel,
                    ip.spatial_h,
                    ip.spatial_w,
                    injection_alpha=injection_alpha,
                )

        elif down_block_type == "DownBlock2D":
            # No temporal injection on plain DownBlock2D (no motion module here
            # in the AnimateDiff MotionAdapter checkpoint).
            new_hidden_dram = []
            for i in range(num_frames):
                s, res_samples = down_block(
                    hidden_states=hidden_samples[i],
                    temb=emb,
                    num_layers=layers_per_block,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    add_downsample=not is_final_block,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    downsample_padding=downsample_padding,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    dtype=None,
                    compute_kernel_config=None,
                )
                down_res_per_frame[i].extend(list(res_samples))
                s_dram = ttnn.to_memory_config(s, ttnn.DRAM_MEMORY_CONFIG)
                s.deallocate(True)
                new_hidden_dram.append(s_dram)
            hidden_samples = new_hidden_dram

        else:
            raise AssertionError(
                f"Unexpected down_block_type: {down_block_type}.  "
                "Only CrossAttnDownBlock2D and DownBlock2D are supported."
            )

    # ------------------------------------------------------------------ 5. Mid block
    # Replicates __call__ lines 447-468.
    new_hidden_dram = []
    for i in range(num_frames):
        s = ttnn_model.mid_block(
            hidden_states=hidden_samples[i],
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift=resnet_time_scale_shift,
            cross_attention_dim=cross_attention_dim,
            config=config,
            attn_num_head_channels=attention_head_dim[-1],
            resnet_groups=norm_num_groups,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            upcast_attention=upcast_attention,
        )
        s_dram = ttnn.to_memory_config(s, ttnn.DRAM_MEMORY_CONFIG)
        s.deallocate(True)
        new_hidden_dram.append(s_dram)

    hidden_samples = new_hidden_dram

    # Temporal injection at the mid block (1 motion module in the checkpoint).
    if "mid" in temporal_kernels and temporal_kernels["mid"]:
        from animatediff_ttnn.motion_weights import get_injection_point_info
        ip_mid = get_injection_point_info("mid")
        hidden_samples = _apply_temporal(
            hidden_samples,
            temporal_kernels["mid"],
            device,
            num_frames,
            block_out_channels[-1],  # 1280
            ip_mid.spatial_h,
            ip_mid.spatial_w,
            injection_alpha=injection_alpha,
        )

    # ------------------------------------------------------------------ 6. Up blocks
    # Replicates __call__ lines 470-550.
    # resnets = layers_per_block + 1 = 3 for both CrossAttn and plain up blocks.
    reversed_block_out_channels = list(reversed(block_out_channels))
    reversed_attention_head_dim = list(reversed(attention_head_dim))
    only_cross_attention_up = list(reversed(only_cross_attention))
    output_channel = reversed_block_out_channels[0]

    for block_idx, (up_block_type, up_block) in enumerate(
        zip(ttnn_model.up_block_types, ttnn_model.up_blocks)
    ):
        is_final_block = block_idx == len(block_out_channels) - 1
        prev_output_channel = output_channel
        output_channel = reversed_block_out_channels[block_idx]
        input_channel = reversed_block_out_channels[min(block_idx + 1, len(block_out_channels) - 1)]
        add_upsample = not is_final_block

        # Number of residuals consumed by this up block (SD 1.4: always 3).
        resnets = layers_per_block + 1

        # Consume residuals from each frame's accumulator.
        res_tuples = []
        for i in range(num_frames):
            res_tuple = tuple(down_res_per_frame[i][-resnets:])
            down_res_per_frame[i] = down_res_per_frame[i][:-resnets]
            res_tuples.append(res_tuple)

        # upsample_size is only set when forward_upsample_size is True (non-standard
        # spatial dims).  SD 1.4 always uses 64×64 inputs so this stays None.
        upsample_size = None

        if up_block_type == "CrossAttnUpBlock2D":
            new_hidden_dram = []
            for i in range(num_frames):
                s = up_block(
                    hidden_states=hidden_samples[i],
                    temb=emb,
                    res_hidden_states_tuple=res_tuples[i],
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    num_layers=layers_per_block + 1,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    temb_channels=time_embed_dim,
                    add_upsample=add_upsample,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    config=config,
                    cross_attention_dim=cross_attention_dim,
                    attn_num_head_channels=reversed_attention_head_dim[block_idx],
                    dual_cross_attention=dual_cross_attention,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention_up[block_idx],
                    upcast_attention=upcast_attention,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    index=block_idx,
                )
                s_dram = ttnn.to_memory_config(s, ttnn.DRAM_MEMORY_CONFIG)
                s.deallocate(True)
                new_hidden_dram.append(s_dram)

            hidden_samples = new_hidden_dram

            # Temporal injection at this CrossAttnUpBlock2D.
            key = f"up{block_idx}"
            if key in temporal_kernels and temporal_kernels[key]:
                from animatediff_ttnn.motion_weights import get_injection_point_info
                ip = get_injection_point_info(key)
                hidden_samples = _apply_temporal(
                    hidden_samples,
                    temporal_kernels[key],
                    device,
                    num_frames,
                    output_channel,
                    ip.spatial_h,
                    ip.spatial_w,
                    injection_alpha=injection_alpha,
                )

        elif up_block_type == "UpBlock2D":
            # No temporal injection on plain UpBlock2D.
            new_hidden_dram = []
            for i in range(num_frames):
                s = up_block(
                    hidden_states=hidden_samples[i],
                    temb=emb,
                    res_hidden_states_tuple=res_tuples[i],
                    upsample_size=upsample_size,
                    num_layers=layers_per_block + 1,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    temb_channels=time_embed_dim,
                    add_upsample=add_upsample,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                )
                s_dram = ttnn.to_memory_config(s, ttnn.DRAM_MEMORY_CONFIG)
                s.deallocate(True)
                new_hidden_dram.append(s_dram)
            hidden_samples = new_hidden_dram

        else:
            raise AssertionError(
                f"Unexpected up_block_type: {up_block_type}.  "
                "Only CrossAttnUpBlock2D and UpBlock2D are supported."
            )

    # ------------------------------------------------------------------ 7. Post-process
    # Exact replica of __call__ lines 553-660, applied per frame.
    # ttnn_model.conv_out_weights / conv_out_bias are updated in-place via
    # return_weights_and_bias=True (matches the original).
    output_samples = []
    for sample in hidden_samples:
        # --- group_norm ---
        sample = ttnn.to_layout(sample, ttnn.ROW_MAJOR_LAYOUT)
        if ttnn_model.fallback_on_groupnorm:
            sample = ttnn.reshape(
                sample,
                (
                    ttnn_model.batch_size,
                    ttnn_model.conv_out_input_height,
                    ttnn_model.conv_out_input_width,
                    ttnn_model.conv_out_in_channels,
                ),
            )
            sample = ttnn.permute(sample, (0, 3, 1, 2))
            sample = ttnn.operations.normalization._fallback_group_norm(
                sample,
                num_groups=norm_num_groups,
                weight=ttnn_model.parameters.conv_norm_out.weight,
                bias=ttnn_model.parameters.conv_norm_out.bias,
                epsilon=norm_eps,
            )
            sample = pre_process_input(ttnn_model.device, sample)
        else:
            sample = ttnn.to_memory_config(sample, ttnn_model.gn_expected_input_sharded_memory_config)
            sample = ttnn.reshape(
                sample,
                (
                    ttnn_model.batch_size,
                    1,
                    ttnn_model.conv_out_input_height * ttnn_model.conv_out_input_width,
                    ttnn_model.conv_out_in_channels,
                ),
            )
            sample = ttnn.group_norm(
                sample,
                num_groups=norm_num_groups,
                epsilon=norm_eps,
                input_mask=ttnn_model.norm_input_mask,
                weight=ttnn_model.parameters.conv_norm_out.weight,
                bias=ttnn_model.parameters.conv_norm_out.bias,
                memory_config=ttnn_model.gn_expected_input_sharded_memory_config,
                core_grid=ttnn_model.group_norm_core_grid,
            )

        sample = ttnn.reshape(
            sample,
            (
                1,
                1,
                ttnn_model.batch_size * ttnn_model.conv_out_input_height * ttnn_model.conv_out_input_width,
                ttnn_model.conv_out_in_channels,
            ),
        )

        # --- SiLU + interleave ---
        sample = ttnn.silu(sample, memory_config=ttnn.get_memory_config(sample))
        sample = ttnn.sharded_to_interleaved(sample, ttnn.L1_MEMORY_CONFIG, sample.dtype)

        # --- conv_out ---
        conv_out_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat8_b,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            act_block_h_override=64,
            reshard_if_not_optimal=True,
            enable_act_double_buffer=True,
        )
        compute_config_out = get_default_compute_config(ttnn_model.device)
        conv_out_kwargs = {
            "in_channels": ttnn_model.conv_out_in_channels,
            "out_channels": ttnn_model.conv_out_out_channels,
            "batch_size": ttnn_model.batch_size,
            "input_height": ttnn_model.conv_out_input_height,
            "input_width": ttnn_model.conv_out_input_width,
            "kernel_size": (3, 3),
            "stride": (1, 1),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "device": ttnn_model.device,
            "conv_config": conv_out_config,
            "slice_config": ttnn.Conv2dL1FullSliceConfig,
        }
        sample, [ttnn_model.conv_out_weights, ttnn_model.conv_out_bias] = ttnn.conv2d(
            input_tensor=sample,
            **conv_out_kwargs,
            weight_tensor=ttnn_model.conv_out_weights,
            bias_tensor=ttnn_model.conv_out_bias,
            compute_config=compute_config_out,
            dtype=ttnn.bfloat8_b,
            return_weights_and_bias=True,
        )
        sample = reshard_for_output_channels_divisibility(sample, ttnn_model.conv_out_out_channels)

        # --- final reshape + permute → NCHW → slice first 4 channels ---
        sample = ttnn.to_memory_config(sample, ttnn.L1_MEMORY_CONFIG)
        sample = ttnn.clone(sample, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        sample = ttnn.reshape(
            sample,
            (
                ttnn_model.batch_size,
                ttnn_model.conv_out_input_height,
                ttnn_model.conv_out_input_width,
                -1,
            ),
        )
        sample = ttnn.permute(sample, (0, 3, 1, 2))  # NHWC → NCHW
        sample = sample[:, :4, :, :]  # keep only the 4 latent channels

        output_samples.append(sample)

    return output_samples
