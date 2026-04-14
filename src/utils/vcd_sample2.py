import copy
from dataclasses import dataclass
import inspect
from typing import Optional, Union, Callable
import warnings
from dataclasses import dataclass
import os
import torch
import torch.nn as nn
import transformers
from transformers.masking_utils import create_masks_for_generate
from transformers.cache_utils import Cache
from transformers.generation import (
    GenerationMixin,
    GenerateBeamDecoderOnlyOutput,
    GenerateBeamEncoderDecoderOutput
)
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.utils import ModelOutput, logging
from transformers import GenerationConfig

from utils.utils import scaled_normalized_entropy, sigmoid_decayed_entropy


logger = logging.get_logger(__name__)


@dataclass
class GenerateDecoderOnlyOutput(ModelOutput):
    sequences: torch.LongTensor
    scores: Optional[tuple[torch.FloatTensor]] = None
    logits: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[tuple[tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Cache] = None


@dataclass
class GenerateEncoderDecoderOutput(ModelOutput):
    sequences: torch.LongTensor
    scores: Optional[tuple[torch.FloatTensor]] = None
    logits: Optional[tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[tuple[tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[tuple[tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[tuple[tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Cache] = None


@dataclass
class CustomGenerateDecoderOnlyOutput(GenerateDecoderOnlyOutput):
    thresholds: Optional[float] = None


@dataclass
class CustomGenerateEncoderDecoderOutput(GenerateEncoderDecoderOutput):
    thresholds: Optional[float] = None

GenerateNonBeamOutput = Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput]
GenerateBeamOutput = Union[GenerateBeamDecoderOnlyOutput, GenerateBeamEncoderDecoderOutput]
GenerateOutput = Union[GenerateNonBeamOutput, GenerateBeamOutput]


def prepare_inputs_for_generation2(
    self,
    input_ids: torch.LongTensor,
    past_key_values: Optional[Cache] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    cd_input = None,
    **kwargs,
):
    """
    Prepare the model inputs for generation. Notable steps include selecting the correct input key and cloning when appropriate,
    creating position_ids from the attention_mask when missing, slicing inputs and converting 2D attention masks to 4D for
    compilable caches, and finally forwarding all additional keyword arguments unchanged to the model's forward pass.

    See the forward pass in the model documentation for expected arguments (different models might have different
    requirements for e.g. `past_key_values`). This function should work as is for most LLMs.
    """
    kwargs_copy = copy.deepcopy(kwargs)
    if "cd_tensor" in kwargs_copy:
        if cd_input:
            kwargs_copy["pixel_values"] = kwargs_copy["cd_tensor"]
        del kwargs_copy["cd_tensor"]

    # 1. Handle BC:
    model_inputs = {}
    model_inputs["cache_position"] = cache_position

    # 2. Generic cache-dependent input preparation
    if past_key_values is not None:
        model_inputs["past_key_values"] = past_key_values
        # TODO (joao): handle the case where cache length == input_ids length. The function below results in an
        # exception because we get empty input_ids after slicing. In essence, we need to roll back the cache 1
        # token to recompute the logits for the first token to be generated (but not all caches support roll backs)
        inputs_embeds, input_ids = self._cache_dependant_input_preparation(
            input_ids, inputs_embeds, cache_position
        )

    # 3. Prepare base model inputs
    input_ids_key = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step for every prompt.
    if not self.config.is_encoder_decoder:
        if inputs_embeds is not None and len(cache_position) == inputs_embeds.shape[1]:
            model_inputs[input_ids_key] = None
            model_inputs["inputs_embeds"] = inputs_embeds
        else:
            # `clone` calls in this function ensure a consistent stride. See #32227
            model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)
            model_inputs["inputs_embeds"] = None
    else:
        model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)

    # 4. Create missing `position_ids` on the fly
    encoder_attention_mask = attention_mask if self.config.is_encoder_decoder else None
    attention_mask = (
        kwargs_copy.pop("decoder_attention_mask", None) if self.config.is_encoder_decoder else attention_mask
    )
    attention_mask_key = "decoder_attention_mask" if self.config.is_encoder_decoder else "attention_mask"
    position_ids_key = "decoder_position_ids" if self.config.is_encoder_decoder else "position_ids"
    if (
        attention_mask is not None
        and kwargs_copy.get(position_ids_key) is None
        and position_ids_key in set(inspect.signature(self.forward).parameters.keys())
    ):
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        kwargs_copy[position_ids_key] = position_ids  # placed in kwargs for further processing (see below)

    # 5. Slice model inputs if it's an input that should have the same length as `input_ids`
    for model_input_name in ["position_ids", "token_type_ids", "decoder_position_ids"]:
        model_input = kwargs_copy.get(model_input_name)
        if model_input is not None:
            if past_key_values is not None:
                current_input_length = (
                    model_inputs["inputs_embeds"].shape[1]
                    if model_inputs.get("inputs_embeds") is not None
                    else model_inputs[input_ids_key].shape[1]
                )
                model_input = model_input[:, -current_input_length:]
                model_input = model_input.clone(memory_format=torch.contiguous_format)
            model_inputs[model_input_name] = model_input

    # 6. Create 4D attention mask is we are using a compilable cache (important for performant compiled forward
    # pass)
    if (
        isinstance(past_key_values, Cache)
        and past_key_values.is_compileable
        and attention_mask is not None
        and attention_mask.ndim == 2
    ):
        if not self.config.is_encoder_decoder and model_inputs["inputs_embeds"] is not None:
            batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
        else:
            batch_size, sequence_length = model_inputs[input_ids_key].shape[:2]

        # Create the causal mask with fixed shape in advance, to reduce recompilations. If the function to create
        # the 4D causal mask exists, it should be present in the base model (XXXModel class) or in its decoder.
        base_model = getattr(self, self.base_model_prefix, self)
        decoder = base_model.get_decoder() if hasattr(base_model, "get_decoder") else None
        causal_mask_creation_function = getattr(
            base_model, "_prepare_4d_causal_attention_mask_with_cache_position", None
        )
        if causal_mask_creation_function is None and decoder is not None:  # it may be in the decoder
            causal_mask_creation_function = getattr(
                decoder, "_prepare_4d_causal_attention_mask_with_cache_position", None
            )

        # If it's not defined, it means the model uses the new general mask API
        if causal_mask_creation_function is None:  # can't be found
            token_type_ids = model_inputs.get("token_type_ids")
            position_ids = model_inputs.get(position_ids_key)
            # Some models may overwrite the general one
            causal_mask_creation_function = getattr(self, "create_masks_for_generate", create_masks_for_generate)
            attention_mask = causal_mask_creation_function(
                config=self.config,
                # we only need batch size, seq_length and dtype here - we don't care about the values of the embeddings
                input_embeds=torch.empty((batch_size, sequence_length), dtype=self.dtype),
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
            )
        else:
            attention_mask = causal_mask_creation_function(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_cache_shape(),
                dtype=self.dtype,
                cache_position=cache_position,
                batch_size=batch_size,
                config=self.config,
                past_key_values=past_key_values,
            )
    if attention_mask is not None:
        model_inputs[attention_mask_key] = attention_mask

    if encoder_attention_mask is not None:
        model_inputs["attention_mask"] = encoder_attention_mask

    # 7. Forward ALL kwargs that are uninitialized (e.g. `use_cache`).
    for key, value in kwargs_copy.items():
        if key not in model_inputs:
            model_inputs[key] = value

    # 8. Remove unexpected `generate` inputs (TODO @joao: fix trainer and examples)
    model_inputs.pop("labels", None)
    return model_inputs


@torch.no_grad()
def generate2(
    self,
    inputs: Optional[torch.Tensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], list[int]]] = None,
    synced_gpus: Optional[bool] = None,
    assistant_model: Optional["PreTrainedModel"] = None,
    streamer: Optional["BaseStreamer"] = None,
    negative_prompt_ids: Optional[torch.Tensor] = None,
    negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    use_model_defaults: Optional[bool] = None,
    custom_generate: Optional[Union[str, Callable]] = None,
    **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:
    # 0. If requested, load an arbitrary generation recipe from the Hub and run it instead
    trust_remote_code = kwargs.pop("trust_remote_code", None)

    if custom_generate is not None and isinstance(custom_generate, str):
        # Get all `generate` arguments in a single variable. Custom functions are responsible for handling them:
        # they receive the same inputs as `generate`, with `model` instead of `self` and excluding the arguments to
        # trigger the custom generation. They can access to methods from `GenerationMixin` through `model`.
        global_keys_to_exclude = {
            "self",
            "kwargs",
            "global_keys_to_exclude",
            "trust_remote_code",
            "custom_generate",
        }
        generate_arguments = {key: value for key, value in locals().items() if key not in global_keys_to_exclude}
        generate_arguments.update(kwargs)

        custom_generate_function = self.load_custom_generate(
            custom_generate, trust_remote_code=trust_remote_code, **kwargs
        )
        return custom_generate_function(model=self, **generate_arguments)

    # 1. Handle kwargs, `generation_config`, validate them and obtain generation mode
    generation_mode_kwargs = self._extract_generation_mode_kwargs(
        custom_generate,
        kwargs,
        synced_gpus,
        assistant_model,
        streamer,
    )
    generation_config, model_kwargs = self._prepare_generation_config(
        generation_config, use_model_defaults, **kwargs
    )

    generation_mode = generation_config.get_generation_mode(assistant_model)
    if isinstance(custom_generate, Callable):
        decoding_method = custom_generate
    else:
        # type() required to access the unbound class-level method
        decoding_method = getattr(type(self), "_sample")


    # Deprecation-related step: set Hub repo for deprecated strategies.
    # NOTE: This must come after initializing generation_config, since we need it to determine if this is a deprecated mode.
    # It must also be before any preparation steps, since Hub repos expect to be loaded before preparation steps.
    # TODO joao, manuel: remove this in v4.62.0
    if deprecated_mode_repo := self._get_deprecated_gen_repo(generation_mode, trust_remote_code, custom_generate):
        return GenerationMixin.generate(
            self,
            inputs=inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            assistant_model=assistant_model,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            use_model_defaults=use_model_defaults,
            custom_generate=deprecated_mode_repo,
            trust_remote_code=trust_remote_code,
            **generation_mode_kwargs,
            **kwargs,
        )

    # 2. Set generation parameters if not already defined
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList() # []
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList() # []

    accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys()) # True
    requires_attention_mask = "encoder_outputs" not in model_kwargs # True
    kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None # True

    # 3. Define model inputs
    inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )

    # Some generation modes (e.g. assisted) need `inputs_tensor` to rerun encoder.forward()
    if "inputs_tensor" in inspect.signature(decoding_method).parameters.keys():
        generation_mode_kwargs["inputs_tensor"] = inputs_tensor
    batch_size = inputs_tensor.shape[0]

    device = inputs_tensor.device
    self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

    # decoder-only models must use left-padding for batched generation.
    if not self.config.is_encoder_decoder:
        # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
        # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
        if (
            generation_config._pad_token_tensor is not None # tensor([151643])
            and batch_size > 1
            and len(inputs_tensor.shape) == 2 # inpupt_tensor is text token tensors
            and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
        ):
            logger.warning(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `padding_side='left'` when initializing the tokenizer."
            )

    # 4. Define other model kwargs
    # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
    # generating the first new token or not, and we only want to use the embeddings for the first new token)
    if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds": # satisfies this.
        generation_config.use_cache = True

    if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask: # satisfy
        model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
            inputs_tensor, generation_config, model_kwargs
        )
    elif kwargs_has_attention_mask:
        # TODO (joao): generalize this check with other types of inputs
        if model_input_name == "input_ids" and len(model_kwargs["attention_mask"].shape) > 2:
            raise ValueError("`attention_mask` passed to `generate` must be 2D.")

    if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
        model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name, generation_config
        )

    # 5. Prepare `input_ids` which will be used for auto-regressive generation
    if self.config.is_encoder_decoder:
        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config._decoder_start_token_tensor,
            device=inputs_tensor.device,
        )
    else:
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

    # Expand inputs depending on the generation mode
    input_ids, model_kwargs = self._expand_inputs_for_generation(
        input_ids=input_ids,
        expand_size=max(generation_config.num_beams, generation_config.num_return_sequences),
        is_encoder_decoder=self.config.is_encoder_decoder,
        **model_kwargs,
    )

    if generation_config.token_healing:
        input_ids = self.heal_tokens(input_ids, generation_mode_kwargs.get("tokenizer"))

    if streamer is not None:
        streamer.put(input_ids.cpu())

    # 6. Prepare `max_length` depending on other stopping criteria.
    input_ids_length = input_ids.shape[1]
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
    generation_config = self._prepare_generated_length(
        generation_config=generation_config,
        has_default_max_length=has_default_max_length,
        has_default_min_length=has_default_min_length,
        model_input_name=model_input_name,
        inputs_tensor=inputs_tensor,
        input_ids_length=input_ids_length,
    )

    # If the model supports `logits_to_keep` in forward(), set it to 1 to avoid computing the whole
    # logit matrix. This can save a lot of memory during the first forward pass. Note that assisted decoding
    # dynamically overrides this value as it can need more than the last token logits
    if self._supports_logits_to_keep() and "logits_to_keep" not in model_kwargs:
        model_kwargs["logits_to_keep"] = 1

    self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

    # 7. Prepare the cache.
    # - `model_kwargs` may be updated in place with a cache as defined by the parameters in `generation_config`.
    # - different models have a different cache name expected by the model (default = "past_key_values")
    # - `max_length`, prepared above, is used to determine the maximum cache length
    max_cache_length = generation_config.max_length - 1
    if (
        inputs_tensor.shape[1] != input_ids_length
        and model_input_name == "inputs_embeds"
        and not self.config.is_encoder_decoder
    ):
        max_cache_length += inputs_tensor.shape[1]
    self._prepare_cache_for_generation(
        generation_config, model_kwargs, generation_mode, batch_size, max_cache_length
    )

    if self.device.type != input_ids.device.type:
        warnings.warn(
            "You are calling .generate() with the `input_ids` being on a device type different"
            f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
            f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
            " Please make sure that you have put `input_ids` to the"
            f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
            " running `.generate()`.",
            UserWarning,
        )

    # 8. prepare logits processors and stopping criteria
    prepared_logits_processor = self._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
        device=inputs_tensor.device,
        model_kwargs=model_kwargs,
        negative_prompt_ids=negative_prompt_ids,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
    )

    prepared_stopping_criteria = self._get_stopping_criteria(
        generation_config=generation_config,
        stopping_criteria=stopping_criteria,
        tokenizer=generation_mode_kwargs.get("tokenizer"),
    )

    # Set model_kwargs `use_cache` so we can use it later in forward runs
    model_kwargs["use_cache"] = generation_config.use_cache

    # 9. Call generation mode
    result = decoding_method(
        self,
        input_ids,
        logits_processor=prepared_logits_processor,
        stopping_criteria=prepared_stopping_criteria,
        generation_config=generation_config,
        **generation_mode_kwargs,
        **model_kwargs,
    )

    # Convert to legacy cache format if requested
    if (
        generation_config.return_legacy_cache is True
        and hasattr(result, "past_key_values")
        and getattr(result.past_key_values, "to_legacy_cache") is not None
    ):
        result.past_key_values = result.past_key_values.to_legacy_cache()
    return result


def sample2(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    # init CD values
    cd_tensor = model_kwargs.get('cd_tensor', None)
    cd_config = model_kwargs.pop('cd_config', None)
    use_cd = False if cd_tensor is None else True

    # init values
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape[:2]
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

    model_forward = self.__call__
    compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
    if compile_forward:
        os.environ["TOKENIZERS_PARALLELISM"] = "0"
        # If we use FA2 and a static cache, we cannot compile with fullgraph
        if self.config._attn_implementation == "flash_attention_2":
            # only raise warning if the user passed an explicit compile-config
            if generation_config.compile_config is not None and generation_config.compile_config.fullgraph:
                logger.warning_once(
                    "When using Flash Attention 2 and a static cache, you cannot use the option `CompileConfig(fullgraph=True)` as "
                    "FA2 introduces graph breaks. We overrode the option with `fullgraph=False`."
                )
                generation_config.compile_config.fullgraph = False
        model_forward = self.get_compiled_call(generation_config.compile_config)

    if generation_config.prefill_chunk_size is not None:
        model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
        is_prefill = False
    else:
        is_prefill = True

    model_kwargs_cd = copy.deepcopy(model_kwargs)
    thresholds = ()

    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, cd_input=False, **model_kwargs)

        if is_prefill:
            outputs = self(**model_inputs, return_dict=True)
            if not use_cd:
                is_prefill = False
        else:
            outputs = model_forward(**model_inputs, return_dict=True)

        # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )
        if synced_gpus and this_peer_finished:
            continue

        # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

        #######################################################################
        if use_cd:
            output_attentions_wo_img = (
                output_attentions if output_attentions is not None else self.generation_config.output_attentions
            )
            output_hidden_states_wo_img = (
                output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
            )

            model_inputs_cd = self.prepare_inputs_for_generation(input_ids, cd_input=True, **model_kwargs_cd)
            if is_prefill:
                cd_outputs = self(
                    **model_inputs_cd,
                    return_dict=True,
                    output_attentions=output_attentions_wo_img,
                    output_hidden_states=output_hidden_states_wo_img
                )
                is_prefill = False
            else:
                cd_outputs = model_forward(
                    **model_inputs_cd,
                    return_dict=True,
                    output_attentions=output_attentions_wo_img,
                    output_hidden_states=output_hidden_states_wo_img
                )

            model_kwargs_cd = self._update_model_kwargs_for_generation(
                cd_outputs,
                model_kwargs_cd,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            next_token_logits_cd = cd_outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

            cd_beta = cd_config.cd_beta
            num_filtered_tokens = None
            if cd_config.cd_tau is not None:
                if cd_config.cd_tau > 0:
                    cd_beta = sigmoid_decayed_entropy(next_token_logits, cd_config.cd_tau)
                else:
                    cd_beta = scaled_normalized_entropy(next_token_logits, -cd_config.cd_tau)
            threshold = torch.log(torch.tensor(cd_config.cd_beta)) + next_token_logits.max(dim=-1, keepdim=True).values

            cd_logits = (1 + cd_config.cd_alpha) * next_token_logits - cd_config.cd_alpha * next_token_logits_cd
            mask = next_token_logits < threshold
            cd_logits = cd_logits.masked_fill(mask, -float('inf'))
            num_filtered_tokens = 151935 - torch.sum(mask).item()
            next_token_scores = logits_processor(input_ids, cd_logits)
        else:
            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            cd_beta = None
            num_filtered_tokens = None
        thresholds += (cd_beta,)
        #######################################################################

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # token selection
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return CustomGenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
                thresholds=thresholds,
            )
        else:
            return CustomGenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
                thresholds=thresholds,
            )
    else:
        return input_ids

def evolve_vcd_sampling2():
    print('Monkey Patching sampling method')
    transformers.generation.utils.GenerationMixin.generate = generate2
    transformers.generation.utils.GenerationMixin.prepare_inputs_for_generation = prepare_inputs_for_generation2
    transformers.generation.utils.GenerationMixin.sample = sample2
    # sample is now a protected function in the latest Transformers library
    transformers.generation.utils.GenerationMixin._sample = sample2
