# Copyright 2025 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Changes from Qualcomm Technologies, Inc. are provided under the following license:
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

# Based on: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py
"""
Flux Artist pipeline with **conditional-image masking** support.

Extends the original Flux pipeline by allowing **lists** of conditional
images (canvases) and **lists** of masks. Each canvas is **preprocessed** and
**VAE-encoded individually**; its mask is applied to its conditional token
stream; finally, all (masked) conditional
streams are **concatenated** before being appended to the denoising stream.

Singleton inputs still behave exactly like before.
"""

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
    T5TokenizerFast,
)
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import (
    FluxIPAdapterMixin,
    FluxLoraLoaderMixin,
    FromSingleFileMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models import AutoencoderKL, FluxTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_torch_xla_available,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)

__all__ = ["FluxArtistPipeline"]

# Default noise-schedule shift bounds used in calculate_shift and __call__.
_DEFAULT_BASE_SHIFT: float = 0.5
_DEFAULT_MAX_SHIFT: float = 1.15

PREFERRED_ARTIST_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = _DEFAULT_BASE_SHIFT,
    max_shift: float = _DEFAULT_MAX_SHIFT,
) -> float:
    """Compute the noise-schedule shift mu via linear interpolation over
    sequence length."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers Stable Diffusion pipelines
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, int]:
    """Configure the scheduler's timestep schedule and return the timesteps
    tensor plus step count."""
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. "
            "Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s"
                f" `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are"
                f" using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s"
                f" `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are"
                f" using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    sample_mode: str = "sample",
) -> torch.Tensor:
    """Extract the latent tensor from a VAE encoder output using the
    requested sampling mode."""
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class FluxArtistPipeline(
    DiffusionPipeline,
    FluxLoraLoaderMixin,
    FromSingleFileMixin,
    TextualInversionLoaderMixin,
    FluxIPAdapterMixin,
):
    """Flux Artist pipeline w/ conditional-image masking (+ list support)."""

    model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->transformer->vae"
    _optional_components = ["image_encoder", "feature_extractor"]
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
    ):
        """Initialise the FluxArtistPipeline.

        Args:
            scheduler: Flow-matching Euler scheduler driving the diffusion
                process.
            vae: VAE auto-encoder for latent encode/decode.
            text_encoder: CLIP text encoder (pooled embeddings).
            tokenizer: CLIP tokenizer.
            text_encoder_2: T5 text encoder (sequence embeddings).
            tokenizer_2: T5 tokenizer.
            transformer: Flux transformer backbone.
            image_encoder: Optional CLIP vision encoder for IP-Adapter.
            feature_extractor: Optional CLIP feature extractor for IP-Adapter.
        """
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        )
        # Flux latents are turned into 2x2 patches and packed. So the VAE
        # scale factor is multiplied by the patch size
        self.latent_channels = self.vae.config.latent_channels if getattr(self, "vae", None) else 16
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length
            if hasattr(self, "tokenizer") and self.tokenizer is not None
            else 77
        )
        self.default_sample_size = 128

    # ---- Prompt encoders (unchanged) ----
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Encode prompts with the T5 text encoder and return batched
        sequence embeddings."""
        device = device or self._execution_device
        dtype = dtype or self.text_encoder_2.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer_2)

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer_2.batch_decode(
                untruncated_ids[:, self.tokenizer_max_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because "
                "`max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[
            0
        ]
        dtype = self.text_encoder_2.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Encode prompts with the CLIP text encoder and return pooled
        embeddings."""
        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer_max_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because "
                "CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False)
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)
        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        lora_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode text prompts into T5 sequence embeddings, CLIP pooled
        embeddings, and text token ids."""
        device = device or self._execution_device

        if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
            self._lora_scale = lora_scale
        if self.text_encoder is not None and USE_PEFT_BACKEND:
            scale_lora_layers(self.text_encoder, lora_scale)
        if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
            scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        if self.text_encoder is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                unscale_lora_layers(self.text_encoder, lora_scale)
        if self.text_encoder_2 is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
        return prompt_embeds, pooled_prompt_embeds, text_ids

    # ---- Image encoders (unchanged) ----
    def encode_image(
        self,
        image: Union[PipelineImageInput, torch.Tensor],
        device: torch.device,
        num_images_per_prompt: int,
    ) -> torch.Tensor:
        """Encode an image with the CLIP vision encoder and return repeated
        image embeddings."""
        dtype = next(self.image_encoder.parameters()).dtype
        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values
            image = image.to(device=device, dtype=dtype)
        image_embeds = self.image_encoder(image).image_embeds
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        return image_embeds

    def prepare_ip_adapter_image_embeds(
        self,
        ip_adapter_image,
        ip_adapter_image_embeds,
        device,
        num_images_per_prompt,
    ) -> List[torch.Tensor]:
        """Build and replicate per-adapter image embeddings for
        IP-Adapter conditioning."""
        image_embeds = []
        if ip_adapter_image_embeds is None:
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]
            if len(ip_adapter_image) != (self.transformer.encoder_hid_proj.num_ip_adapters):
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the "
                    f"number of IP Adapters. Got "
                    f"{len(ip_adapter_image)} images and "
                    f"{self.transformer.encoder_hid_proj.num_ip_adapters}"
                    f" IP Adapters."
                )
            for single_ip_adapter_image in ip_adapter_image:
                single_image_embeds = self.encode_image(single_ip_adapter_image, device, 1)
                image_embeds.append(single_image_embeds[None, :])
        else:
            if not isinstance(ip_adapter_image_embeds, list):
                ip_adapter_image_embeds = [ip_adapter_image_embeds]
            if len(ip_adapter_image_embeds) != (self.transformer.encoder_hid_proj.num_ip_adapters):
                raise ValueError(
                    f"`ip_adapter_image_embeds` must have same length as "
                    f"the number of IP Adapters. Got "
                    f"{len(ip_adapter_image_embeds)} image embeds and "
                    f"{self.transformer.encoder_hid_proj.num_ip_adapters}"
                    f" IP Adapters."
                )
            for single_image_embeds in ip_adapter_image_embeds:
                image_embeds.append(single_image_embeds)

        ip_adapter_image_embeds = []
        for single_image_embeds in image_embeds:
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            single_image_embeds = single_image_embeds.to(device=device)
            ip_adapter_image_embeds.append(single_image_embeds)
        return ip_adapter_image_embeds

    # ---- Input checks (unchanged) ----
    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ) -> None:
        """Validate all pipeline inputs and raise descriptive errors for
        any incompatible combinations."""
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by "
                f"{self.vae_scale_factor * 2} but are {height} and "
                f"{width}. Dimensions will be resized accordingly"
            )
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in "
                f"{self._callback_tensor_inputs}, but found "
                + str(
                    [
                        k
                        for k in callback_on_step_end_tensor_inputs
                        if k not in self._callback_tensor_inputs
                    ]
                )
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and "
                f"`prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and "
                f"`prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot "
                "leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is " f"{type(prompt)}"
            )
        elif prompt_2 is not None and (
            not isinstance(prompt_2, str) and not isinstance(prompt_2, list)
        ):
            raise ValueError(
                f"`prompt_2` has to be of type `str` or `list` but is " f"{type(prompt_2)}"
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: "
                f"{negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only "
                f"forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: "
                f"{negative_prompt_2} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only "
                f"forward one of the two."
            )

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds`"
                " also have to be passed. Make sure to generate "
                "`pooled_prompt_embeds` from the same text encoder that "
                "was used to generate `prompt_embeds`."
            )
        if negative_prompt_embeds is not None and (negative_pooled_prompt_embeds is None):
            raise ValueError(
                "If `negative_prompt_embeds` are provided, "
                "`negative_pooled_prompt_embeds` also have to be passed. "
                "Make sure to generate `negative_pooled_prompt_embeds` "
                "from the same text encoder that was used to generate "
                "`negative_prompt_embeds`."
            )
        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(
                f"`max_sequence_length` cannot be greater than 512 but " f"is {max_sequence_length}"
            )

    # ---- Latent/grid helpers (unchanged) ----
    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype) -> torch.Tensor:
        """Build a (height*width, 3) grid of RoPE position ids for the
        latent spatial tokens."""
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]
        (
            latent_image_id_height,
            latent_image_id_width,
            latent_image_id_channels,
        ) = latent_image_ids.shape
        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width,
            latent_image_id_channels,
        )
        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width) -> torch.Tensor:
        """Rearrange spatial latents into packed 2x2 patch tokens of
        shape (batch, patches, channels*4)."""
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(
            batch_size,
            (height // 2) * (width // 2),
            num_channels_latents * 4,
        )
        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor) -> torch.Tensor:
        """Unpack packed patch tokens back into spatial latents of shape
        (batch, channels, height, width)."""
        batch_size, num_patches, channels = latents.shape
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))
        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)
        return latents

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """VAE-encode an image tensor and return scaled latents."""
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(
                    self.vae.encode(image[i : i + 1]),
                    generator=generator[i],
                    sample_mode="argmax",
                )
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(
                self.vae.encode(image),
                generator=generator,
                sample_mode="argmax",
            )
        image_latents = (
            image_latents - self.vae.config.shift_factor
        ) * self.vae.config.scaling_factor
        return image_latents

    # ---- VAE helpers (unchanged) ----
    def enable_vae_slicing(self) -> None:
        """Enable sliced VAE decoding to reduce memory usage
        (deprecated wrapper)."""
        deprecate("enable_vae_slicing", "0.40.0", "Use `pipe.vae.enable_slicing()`.")
        self.vae.enable_slicing()

    def disable_vae_slicing(self) -> None:
        """Disable sliced VAE decoding (deprecated wrapper)."""
        deprecate(
            "disable_vae_slicing",
            "0.40.0",
            "Use `pipe.vae.disable_slicing()`.",
        )
        self.vae.disable_slicing()

    def enable_vae_tiling(self) -> None:
        """Enable tiled VAE decoding to reduce memory usage
        (deprecated wrapper)."""
        deprecate("enable_vae_tiling", "0.40.0", "Use `pipe.vae.enable_tiling()`.")
        self.vae.enable_tiling()

    def disable_vae_tiling(self) -> None:
        """Disable tiled VAE decoding (deprecated wrapper)."""
        deprecate("disable_vae_tiling", "0.40.0", "Use `pipe.vae.disable_tiling()`.")
        self.vae.disable_tiling()

    # ---- Latent preparation (unchanged) ----
    def prepare_latents(
        self,
        image: Optional[torch.Tensor],
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        """Prepare packed denoising latents and optional conditional image
        latents with their positional ids."""
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length "
                f"{len(generator)}, but requested an effective batch "
                f"size of {batch_size}. Make sure the batch size matches "
                f"the length of the generators."
            )

        # align with VAE + 2x2 packing
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (batch_size, num_channels_latents, height, width)

        image_latents = image_ids = None
        if image is not None:
            image = image.to(device=device, dtype=dtype)
            if image.shape[1] != self.latent_channels:
                image_latents = self._encode_vae_image(image=image, generator=generator)
            else:
                image_latents = image
            if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
                additional = batch_size // image_latents.shape[0]
                image_latents = torch.cat([image_latents] * additional, dim=0)
            elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size "
                    f"{image_latents.shape[0]} to {batch_size} text "
                    f"prompts."
                )
            else:
                image_latents = torch.cat([image_latents], dim=0)

            img_lat_h, img_lat_w = image_latents.shape[2:]
            image_latents = self._pack_latents(
                image_latents,
                batch_size,
                num_channels_latents,
                img_lat_h,
                img_lat_w,
            )
            image_ids = self._prepare_latent_image_ids(
                batch_size, img_lat_h // 2, img_lat_w // 2, device, dtype
            )
            image_ids[..., 0] = 1  # mark conditional-stream tokens

        latent_ids = self._prepare_latent_image_ids(
            batch_size, height // 2, width // 2, device, dtype
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents, image_latents, latent_ids, image_ids

    # ---- Mask helpers (unchanged) ----
    def _packed_hw(self, height: int, width: int) -> Tuple[int, int]:
        h_lat = int(height) // self.vae_scale_factor
        w_lat = int(width) // self.vae_scale_factor
        return (h_lat // 2, w_lat // 2)

    def _resolve_dummy_mask_to_keep_idx(
        self,
        dummy_mask: Union[np.ndarray, torch.Tensor],
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.LongTensor:
        """Convert a denoising-stream mask into a 1-D tensor of kept
        packed-token indices."""
        Hp, Wp = self._packed_hw(height, width)

        def to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x
            return torch.from_numpy(np.array(x))

        m = to_tensor(dummy_mask).float()
        if m.ndim == 1:
            if m.numel() != Hp * Wp:
                raise ValueError(f"dummy_mask length {m.numel()} != " f"packed area {Hp*Wp}")
            keep = (m > 0.5).to(device=device)
            return torch.nonzero(keep, as_tuple=False).squeeze(1)

        if m.ndim == 2:
            Hm, Wm = m.shape
            m = m.unsqueeze(0).unsqueeze(0)  # NCHW
            if (Hm, Wm) == (height, width):
                m = F.interpolate(m, size=(Hp, Wp), mode="nearest")
            elif (Hm, Wm) == (
                height // self.vae_scale_factor,
                width // self.vae_scale_factor,
            ):
                m = F.avg_pool2d(m, kernel_size=2, stride=2)
            elif (Hm, Wm) != (Hp, Wp):
                m = F.interpolate(m, size=(Hp, Wp), mode="nearest")
            keep = (m[0, 0] > 0.5).to(device=device)
            return torch.nonzero(keep.reshape(-1), as_tuple=False).squeeze(1)

        raise ValueError(f"Unsupported dummy_mask shape {tuple(m.shape)}")

    def _resolve_cond_mask_to_keep_idx_img(
        self,
        cond_mask: Union[np.ndarray, torch.Tensor],
        img_height_px: int,
        img_width_px: int,
        Hp_img: int,
        Wp_img: int,
        device: torch.device,
    ) -> torch.LongTensor:
        """Normalize a *conditional-image* mask into KEEP indices on
        packed conditional grid (Hp_img, Wp_img).
        Accepts: (img_height_px,img_width_px) pixels,
        (Hp_img,Wp_img) packed, or (Hp_img*Wp_img,) sequence."""
        if not isinstance(cond_mask, torch.Tensor):
            m = torch.from_numpy(np.array(cond_mask))
        else:
            m = cond_mask
        m = m.float()

        if m.ndim == 1:
            if m.numel() != Hp_img * Wp_img:
                raise ValueError(
                    f"cond_image_mask length {m.numel()} != " f"packed area {Hp_img*Wp_img}"
                )
            keep = (m > 0.5).to(device=device)
            return torch.nonzero(keep, as_tuple=False).squeeze(1)

        if m.ndim == 2:
            Hm, Wm = m.shape
            m = m.unsqueeze(0).unsqueeze(0)
            if (Hm, Wm) == (img_height_px, img_width_px):
                m = F.interpolate(m, size=(Hp_img, Wp_img), mode="nearest")
            elif (Hm, Wm) == (Hp_img * 2, Wp_img * 2):  # latent grid pre-pack
                m = F.avg_pool2d(m, kernel_size=2, stride=2)
            elif (Hm, Wm) != (Hp_img, Wp_img):
                m = F.interpolate(m, size=(Hp_img, Wp_img), mode="nearest")
            keep = (m[0, 0] > 0.5).to(device=device)
            return torch.nonzero(keep.reshape(-1), as_tuple=False).squeeze(1)

        raise ValueError(f"Unsupported cond_image_mask shape {tuple(m.shape)}")

    # ---- (NEW) single-image preprocessing helper for list inputs ----
    def _preprocess_single_canvas(
        self, img: PipelineImageInput, image_height, image_width
    ) -> Tuple[int, int, torch.Tensor]:
        """Return (H, W, tensor[1,3,H,W]) for a single PIL/np image."""
        resized = self.image_processor.resize(img, image_height, image_width)
        tensor = self.image_processor.preprocess(resized, image_height, image_width)  # 1x3xH x W
        return image_height, image_width, tensor

    # ---- Properties (unchanged) ----
    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    # ---- Main call (modified to accept lists for conditional image &
    # mask) ----
    @torch.no_grad()
    def __call__(
        self,
        image: Optional[PipelineImageInput] = None,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[PipelineImageInput] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        # -- existing: mask the denoising-latent stream (optional) --
        dummy_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
        # -- existing (now list-aware): mask the *conditional image* stream --
        cond_image_mask: Optional[
            Union[
                np.ndarray,
                torch.Tensor,
                List[Union[np.ndarray, torch.Tensor]],
            ]
        ] = None,
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: Optional[List[str]] = None,
        max_sequence_length: int = 512,
        max_area: int = 1024**2,
        _auto_resize: bool = True,
    ) -> Union["FluxPipelineOutput", Tuple]:
        """Run the full Flux Artist denoising pipeline with optional
        conditional-image and mask support."""
        if callback_on_step_end_tensor_inputs is None:
            callback_on_step_end_tensor_inputs = ["latents"]
        # 0) normalize generation size to multiples
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        original_height, original_width = height, width

        aspect_ratio = width / height
        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        # 1) input checks
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=(callback_on_step_end_tensor_inputs),
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2) call params / device
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        lora_scale = (
            self.joint_attention_kwargs.get("scale", None)
            if self.joint_attention_kwargs is not None
            else None
        )
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                negative_text_ids,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

        # 3) Preprocess conditional image(s) (now list-aware). We do *not*
        #    feed them to prepare_latents; we VAE-encode and pack each one
        #    separately, then concatenate masked tokens.
        preprocessed_canvases: List[torch.Tensor] = []
        per_img_hw: List[Tuple[int, int]] = []  # (H, W) after resize
        if image is not None and not (
            isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels
        ):
            if isinstance(image, list):
                for img in image:
                    H_i, W_i, tensor_i = self._preprocess_single_canvas(img, height, width)
                    per_img_hw.append((H_i, W_i))
                    preprocessed_canvases.append(tensor_i)
            else:
                H_i, W_i, tensor_i = self._preprocess_single_canvas(image, height, width)
                per_img_hw.append((H_i, W_i))
                preprocessed_canvases.append(tensor_i)

        # 4) Prepare *denoising* latents (image=None here; conditional stream
        # handled separately)
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, _, latent_ids, _ = self.prepare_latents(
            image=None,
            batch_size=batch_size * num_images_per_prompt,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )
        img_latent_ids = latent_ids  # (S_lat, 3)

        # 4.b) Build conditional stream from one or more canvases
        cond_latents_list: List[torch.Tensor] = []
        cond_ids_list: List[torch.Tensor] = []

        if len(preprocessed_canvases) > 0:
            for i, canvas_tensor in enumerate(preprocessed_canvases):
                # ensure dtype/device
                canvas_tensor = canvas_tensor.to(
                    device=device, dtype=prompt_embeds.dtype
                )  # 1x3xH x W

                # expand to batch_size if needed
                if batch_size > canvas_tensor.shape[0]:
                    canvas_tensor = canvas_tensor.repeat(batch_size, 1, 1, 1)
                elif batch_size < canvas_tensor.shape[0]:
                    # If someone gave us a batched conditional tensor,
                    # trim/align to batch_size
                    canvas_tensor = canvas_tensor[:batch_size]

                # VAE encode -> latents (B,C,H_lat,W_lat) -> pack to (B,S_i,C4)
                img_latents = self._encode_vae_image(image=canvas_tensor, generator=generator)
                img_lat_h, img_lat_w = img_latents.shape[2:]
                packed = self._pack_latents(
                    img_latents,
                    batch_size,
                    num_channels_latents,
                    img_lat_h,
                    img_lat_w,
                )

                # token ids for this conditional stream (S_i, 3); mark as
                # conditional tokens
                Hp_i, Wp_i = img_lat_h // 2, img_lat_w // 2
                ids_i = self._prepare_latent_image_ids(
                    batch_size, Hp_i, Wp_i, device, prompt_embeds.dtype
                )
                ids_i[..., 0] = 1

                # apply corresponding cond mask (if provided)
                if cond_image_mask is not None:
                    if isinstance(cond_image_mask, list):
                        if i >= len(cond_image_mask):
                            raise ValueError(
                                f"cond_image_mask list length "
                                f"{len(cond_image_mask)} is smaller than "
                                f"image list length "
                                f"{len(preprocessed_canvases)}."
                            )
                        mask_i = cond_image_mask[i]
                    else:
                        mask_i = cond_image_mask

                    H_px, W_px = per_img_hw[i]

                    keep_idx_img = self._resolve_cond_mask_to_keep_idx_img(
                        mask_i, H_px, W_px, Hp_i, Wp_i, device
                    )

                    if keep_idx_img.numel() == 0:
                        # skip this conditional stream entirely if nothing
                        # is kept
                        continue
                    packed = packed[:, keep_idx_img, :]
                    ids_i = ids_i[keep_idx_img, :]

                cond_latents_list.append(packed)
                cond_ids_list.append(ids_i)

        # aggregate conditional streams
        if len(cond_latents_list) > 0:
            image_latents = torch.cat(cond_latents_list, dim=1)  # (B, sum S_i_kept, C4)
            img_image_ids = torch.cat(cond_ids_list, dim=0)  # (sum S_i_kept, 3)
        else:
            image_latents = None
            img_image_ids = None

        # 5) Timesteps
        sigmas = (
            np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            if sigmas is None
            else sigmas
        )
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", _DEFAULT_BASE_SHIFT),
            self.scheduler.config.get("max_shift", _DEFAULT_MAX_SHIFT),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # IP-Adapter setup (unchanged)
        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
            negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
        ):
            negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            negative_ip_adapter_image = [
                negative_ip_adapter_image
            ] * self.transformer.encoder_hid_proj.num_ip_adapters
        elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
            negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
        ):
            ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            ip_adapter_image = [
                ip_adapter_image
            ] * self.transformer.encoder_hid_proj.num_ip_adapters

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        image_embeds = None
        negative_image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )
        if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
            negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )

        # 6.a) Optional: mask *denoising* latents (dummy_mask)
        use_mask = dummy_mask is not None
        if use_mask:
            keep_idx = self._resolve_dummy_mask_to_keep_idx(dummy_mask, height, width, device)
            if keep_idx.numel() == 0:
                raise ValueError("`dummy_mask` selected zero tokens; nothing to denoise.")
            full_latents = latents
            roi_latents = latents[:, keep_idx, :]
            roi_latent_ids = img_latent_ids[keep_idx, :]
        else:
            keep_idx = None
            full_latents = None
            roi_latents = None
            roi_latent_ids = None

        # 6.b) Masked *conditional* stream is already prepared above
        # (image_latents/img_image_ids).

        # 7) Denoising loop
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                self._current_timestep = t

                if image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds

                # Build model input
                if use_mask:
                    latent_model_input = roi_latents
                    img_ids_for_step = roi_latent_ids
                else:
                    latent_model_input = latents
                    img_ids_for_step = img_latent_ids

                # Append conditional stream (possibly empty)
                if image_latents is not None and img_image_ids is not None:
                    latent_model_input = torch.cat([latent_model_input, image_latents], dim=1)
                    img_ids_for_step = torch.cat([img_ids_for_step, img_image_ids], dim=0)

                curr_latent_len = roi_latents.size(1) if use_mask else latents.size(1)

                # Light debug on step 0
                if i == 0:
                    S_lat = latents.shape[1]
                    S_img_now = 0 if (image_latents is None) else image_latents.shape[1]
                    logger.info(
                        "[Artist] step=%02d use_mask=%s " "S_lat=%d S_img=%d S_total=%d",
                        i,
                        use_mask,
                        S_lat,
                        S_img_now,
                        latent_model_input.shape[1],
                    )

                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=img_ids_for_step,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred[:, :curr_latent_len]

                if do_true_cfg:
                    if negative_image_embeds is not None:
                        self._joint_attention_kwargs["ip_adapter_image_embeds"] = (
                            negative_image_embeds
                        )
                    neg_noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=img_ids_for_step,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    neg_noise_pred = neg_noise_pred[:, :curr_latent_len]
                    noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                # scheduler step
                latents_dtype = latents.dtype
                if use_mask:
                    roi_latents = self.scheduler.step(
                        noise_pred, t, roi_latents, return_dict=False
                    )[0]
                    full_latents[:, keep_idx, :] = roi_latents
                    latents = full_latents
                else:
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)
