from transformers.models.t5.modeling_t5 import T5Stack, T5PreTrainedModel
from transformers.models.mt5.modeling_mt5 import MT5Stack, MT5PreTrainedModel
from transformers.modeling_outputs import (BaseModelOutput,
                                           Seq2SeqLMOutput)
from transformers.utils.model_parallel_utils import get_device_map, assert_device_map
from pytorch_lightning.core.module import LightningModule
import warnings
import copy
import torch
import torch.nn as nn
import numpy as np

from source.utils import apply_noise
from source.loss import BarlowTwinsLoss
from source.paths import get_repo_dir
from source.helpers import write_lines, get_temp_filepath
from easse.cli import report, get_orig_and_refs_sents, evaluate_system_output
import wandb
from source.feature_extraction import (
    get_lexical_complexity_score,
    get_dependency_tree_depth,
)
from source.helpers import tokenize

__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""

#def Linear(in_features, out_features, bias=True, uniform=True):
#    m = nn.Linear(in_features, out_features, bias)
#    if uniform:
#        nn.init.xavier_uniform_(m.weight)
#    else:
#        nn.init.xavier_normal_(m.weight)
#    if bias:
#        nn.init.constant_(m.bias, 0.)
#    return m

def LayerNorm(embedding_dim, eps=1e-6):
    m = nn.LayerNorm(embedding_dim, eps)
    return m

class FeaturesExtractorFeedForward(nn.Module):
    def __init__(self, d_model):
        super(FeaturesExtractorFeedForward, self).__init__()
        self.mlp = nn.Sequential(
            LayerNorm(4),
            nn.Linear(4, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            LayerNorm(d_model)
        )
        
    def forward(self, x):
        return self.mlp(x)
    
class MT5ForConditionalGenerationWithExtractor(MT5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]
    
    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model
        self.lambda_factor = 1
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = MT5Stack(encoder_config, self.shared)

        extractor_config = copy.deepcopy(config)
        extractor_config.is_decoder = False
        extractor_config.use_cache = False
        extractor_config.is_encoder_decoder = False
        self.extractor = MT5Stack(extractor_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = MT5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block),
                           range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.extractor.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.encoder.deparallelize()
        self.extractor.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.extractor = self.extractor.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()
        
    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.extractor.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_extractor(self):
        return self.extractor

    def get_decoder(self):
        return self.decoder
    
    def get_extractor_output(self,
                             input_ids=None,
                             # use cache is simply to a trick to use the generator mixin
                             use_cache_context_ids=None,
                             use_cache_target_examplars_ids=None,
                             use_cache_origin_examplars_ids=None,
                             attention_mask=None,
                             decoder_input_ids=None,
                             decoder_attention_mask=None,
                             head_mask=None,
                             decoder_head_mask=None,
                             cross_attn_head_mask=None,
                             encoder_outputs=None,
                             extractor_outputs=None,
                             past_key_values=None,
                             inputs_embeds=None,
                             context_embeds=None,
                             decoder_inputs_embeds=None,
                             labels=None,
                             use_cache=None,
                             output_attentions=None,
                             output_hidden_states=None,
                             return_dict=None,):
        extractor_hidden = None
        if use_cache_context_ids is None:
            target_styles = ()
            for target_ids in use_cache_target_examplars_ids:
                extractor_hidden = self.extractor(
                    input_ids=target_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=context_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )[0]
                target_styles += (extractor_hidden,)

            original_styles = ()
            for origin_ids in use_cache_origin_examplars_ids:
                extractor_hidden = self.extractor(
                    input_ids=origin_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=context_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )[0]
                original_styles += (extractor_hidden,)

            input_style = self.extractor(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=context_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )[0]
            extractor_hidden = self.lambda_factor * (torch.mean(torch.vstack(
                target_styles), 0) - (torch.mean(torch.vstack(original_styles), 0))) + input_style

        else:
            #for i, vector in enumerate(use_cache_context_ids):
            #    print(vector)
            if extractor_outputs is None:
                if attention_mask is None:
                    batch_size, seq_length = use_cache_context_ids.size()
                    attention_mask = torch.ones(batch_size, seq_length, device=use_cache_context_ids.device)
                    attention_mask[use_cache_context_ids == 0] = 0
                    
                extractor_outputs = self.extractor(
                    input_ids=use_cache_context_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=context_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
                extractor_outputs = BaseModelOutput(
                    last_hidden_state=extractor_outputs[0],
                    hidden_states=extractor_outputs[1] if len(
                        extractor_outputs) > 1 else None,
                    attentions=extractor_outputs[2] if len(extractor_outputs) > 2 else None,)
            extractor_hidden = extractor_outputs[0]
        return extractor_hidden

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        use_cache_extractor_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        context_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`
        Returns:
        Examples:
        ```python
        >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
        >>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")
        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        >>> ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(
                    encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(
                    encoder_outputs) > 2 else None,
            )

        #if torch.is_tensor(use_cache_extractor_outputs):
        #    use_cache_extractor_outputs = torch.mean(use_cache_extractor_outputs, 1).unsqueeze(1)
        hidden_states = encoder_outputs[0] + use_cache_extractor_outputs

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(
                    self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(
                    self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            # loss_extractor_fct = BarlowTwinsLoss(batch_size=64)
            loss_output_fct = torch.nn.CrossEntropyLoss(ignore_index=0)
            # loss_extractor = loss_extractor_fct()
            loss = loss_output_fct(
                lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        use_cache_extractor_outputs=None,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            # "input_ids": input_ids,
            # "use_cache_context_ids": use_cache_context_ids,
            # "use_cache_target_examplars_ids": use_cache_target_examplars_ids,
            # "use_cache_origin_examplars_ids": use_cache_origin_examplars_ids,
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "use_cache_extractor_outputs": use_cache_extractor_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            warnings.warning(
                "You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(
                        0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + \
                (reordered_layer_past_states,)
        return reordered_decoder_past

class T5ForConditionalGenerationWithExtractor(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model
        self.lambda_factor = 1
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        extractor_config = copy.deepcopy(config)
        extractor_config.is_decoder = False
        extractor_config.use_cache = False
        extractor_config.is_encoder_decoder = False
        self.extractor = T5Stack(extractor_config, self.shared)
        self.feature_extractor = FeaturesExtractorFeedForward(config.d_model)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.)
    
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block),
                           range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.extractor.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.encoder.deparallelize()
        self.extractor.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.extractor = self.extractor.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.extractor.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_extractor(self):
        return self.extractor

    def get_decoder(self):
        return self.decoder

    def get_extractor_output(self,
                             input_ids=None,
                             # use cache is simply to a trick to use the generator mixin
                             use_cache_context_ids=None,
                             use_cache_target_examplars_ids=None,
                             use_cache_origin_examplars_ids=None,
                             attention_mask=None,
                             decoder_input_ids=None,
                             decoder_attention_mask=None,
                             head_mask=None,
                             decoder_head_mask=None,
                             cross_attn_head_mask=None,
                             encoder_outputs=None,
                             extractor_outputs=None,
                             past_key_values=None,
                             inputs_embeds=None,
                             context_embeds=None,
                             decoder_inputs_embeds=None,
                             labels=None,
                             use_cache=None,
                             output_attentions=None,
                             output_hidden_states=None,
                             return_dict=None,):
        extractor_hidden = None
        if use_cache_context_ids is None:
            target_styles = ()
            for target_ids in use_cache_target_examplars_ids:
                extractor_hidden = self.extractor(
                    input_ids=target_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=context_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )[0]
                target_styles += (extractor_hidden,)

            original_styles = ()
            for origin_ids in use_cache_origin_examplars_ids:
                extractor_hidden = self.extractor(
                    input_ids=origin_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=context_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )[0]
                original_styles += (extractor_hidden,)

            input_style = self.extractor(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=context_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )[0]
            extractor_hidden = self.lambda_factor * (torch.mean(torch.vstack(
                target_styles), 0) - (torch.mean(torch.vstack(original_styles), 0))) + input_style

        else:
            #for i, vector in enumerate(use_cache_context_ids):
            #    print(vector)
            if extractor_outputs is None:
                if attention_mask is None:
                    batch_size, seq_length = use_cache_context_ids.size()
                    attention_mask = torch.ones(batch_size, seq_length, device=use_cache_context_ids.device)
                    attention_mask[use_cache_context_ids == 0] = 0
                    
                extractor_outputs = self.extractor(
                    input_ids=use_cache_context_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=context_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
                extractor_outputs = BaseModelOutput(
                    last_hidden_state=extractor_outputs[0],
                    hidden_states=extractor_outputs[1] if len(
                        extractor_outputs) > 1 else None,
                    attentions=extractor_outputs[2] if len(extractor_outputs) > 2 else None,)
            extractor_hidden = extractor_outputs[0]
        return extractor_hidden

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        use_cache_extractor_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        context_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`
        Returns:
        Examples:
        ```python
        >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
        >>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")
        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        >>> ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(
                    encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(
                    encoder_outputs) > 2 else None,
            )

        #if torch.is_tensor(use_cache_extractor_outputs):
        #    use_cache_extractor_outputs = torch.mean(use_cache_extractor_outputs, 1).unsqueeze(1)
        hidden_states = encoder_outputs[0] + use_cache_extractor_outputs

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(
                    self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(
                    self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            # loss_extractor_fct = BarlowTwinsLoss(batch_size=64)
            loss_output_fct = torch.nn.CrossEntropyLoss(ignore_index=0)
            # loss_extractor = loss_extractor_fct()
            loss = loss_output_fct(
                lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        use_cache_extractor_outputs=None,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            # "input_ids": input_ids,
            # "use_cache_context_ids": use_cache_context_ids,
            # "use_cache_target_examplars_ids": use_cache_target_examplars_ids,
            # "use_cache_origin_examplars_ids": use_cache_origin_examplars_ids,
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "use_cache_extractor_outputs": use_cache_extractor_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            warnings.warning(
                "You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(
                        0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + \
                (reordered_layer_past_states,)
        return reordered_decoder_past


class TextSettrModel(LightningModule):
    def __init__(self, sent_length, batch_size, delta_val, lambda_val, rec_val, lr, evaluate_kwargs, model_version, load_ckpt, tokenizer):
        super().__init__()
        if 'ptt5' in model_version:
            self.net = T5ForConditionalGenerationWithExtractor.from_pretrained(model_version)
        else:
            self.net = MT5ForConditionalGenerationWithExtractor.from_pretrained(model_version)
        self.net.extractor = copy.deepcopy(self.net.encoder)
        #print(self.net.encoder.state_dict()['block.1.layer.0.SelfAttention.q.weight'])
        #print(self.net.extractor.state_dict()['block.1.layer.0.SelfAttention.q.weight'])
        self.lambda_val = lambda_val
        self.sent_length = sent_length
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.delta_val = delta_val
        self.rec_val = rec_val
        self.lr = lr
        self.evaluate_kwargs = evaluate_kwargs
        self.val_simplified_sentences = []
        self.read_exemplars()

    def read_exemplars(self):
        with open(self.evaluate_kwargs['src_exemplars'], 'r') as f1, open(self.evaluate_kwargs['tgt_exemplars'], 'r') as f2:
            src_seq = f1.readlines()
            tgt_seq = f2.readlines()
            self.src_ids = [self.tokenizer.encode(sentence, max_length=self.sent_length,
                                                  truncation=True, padding="max_length",
                                                  return_tensors="pt")[0] for sentence in src_seq]
            self.src_ids = self.src_ids[:100]
            self.src_ids = torch.stack(self.src_ids, 0)
            self.tgt_ids = [self.tokenizer.encode(sentence, max_length=self.sent_length,
                                                  truncation=True, padding="max_length",
                                                  return_tensors="pt")[0] for sentence in tgt_seq]
            self.tgt_ids = self.tgt_ids[:100]
            self.tgt_ids = torch.stack(self.tgt_ids, 0)
        
        self.src_ids_features = [self.extract_features(sentence) for sentence in src_seq]
        self.src_ids_features = self.src_ids_features[:100]
        self.src_ids_features = torch.stack(self.src_ids_features, 0)
        self.tgt_ids_features = [self.extract_features(sentence) for sentence in tgt_seq]
        self.tgt_ids_features = self.tgt_ids_features[:100]
        self.tgt_ids_features = torch.stack(self.tgt_ids_features, 0)
    
    def training_step(self, batch):
        context_ids, labels_ids, decoder_attention_mask, input_ids, attention_mask, features = batch[0], batch[2], batch[3], batch[4], batch[5], batch[6]
        # print('input ids size', input_ids.size())
        #noisy_input_ids = apply_noise(
        #    input_ids, self.tokenizer, self.sent_length)
        #if np.random.choice([False, True]):
        #    # Noisy back translation
        #    noisy_input_ids = self.net.generate(input_ids=noisy_input_ids, use_cache_extractor_outputs=0,
        #                                        max_length=self.sent_length)
        extractor_output = self.net.get_extractor_output(
            use_cache_context_ids=context_ids)

        feature_extractor_output = self.net.feature_extractor(features)
        #extractor_output_input = self.net.get_extractor_output(
        #    use_cache_context_ids=labels_ids)
        
        #extractor_output = torch.mean(extractor_output, 1).unsqueeze(1)
        #extractor_output_input = torch.mean(extractor_output_input, 1).unsqueeze(1)
        if self.lambda_val > 0:
            barlow_twins_loss_func = BarlowTwinsLoss(batch_size=64, lambda_coeff = self.delta_val)
            barlow_twins_loss = barlow_twins_loss_func(
                extractor_output_input, extractor_output)
            self.log('train_bt_loss', barlow_twins_loss, on_epoch = True)
        else:
            barlow_twins_loss = 0
        #input_ids[input_ids == self.tokenizer.pad_token_id] = -100
        extractor_output = torch.mean(extractor_output, 1).unsqueeze(1)
        if self.rec_val > 0:            
            output_loss = self.net(input_ids=noisy_input_ids, labels=input_ids,
                                   use_cache_extractor_outputs=extractor_output).loss
            self.log('train_reconstruction_loss', output_loss, on_epoch = True)
        else:
            output_loss = 0
        
        para_loss = self.net(input_ids=input_ids, labels=labels_ids,
                             attention_mask=attention_mask, decoder_attention_mask=decoder_attention_mask,
                               use_cache_extractor_outputs=extractor_output+feature_extractor_output.unsqueeze(1)).loss
        self.log('train_para_loss', para_loss, on_epoch = True)
        
        if output_loss is not None:
            output_loss = self.rec_val * output_loss + para_loss + self.lambda_val * barlow_twins_loss
            self.log('train_loss', output_loss, on_epoch = True)
            return output_loss
        else:
            self.log('train_loss', 0)
            return None
        # return self.net(input_ids=noisy_input_ids, labels = input_ids, use_cache_extractor_outputs=extractor_output).loss + barlow_twins_loss

    def validation_step(self, batch, batch_idx):
        #source_ids, _ = batch[0], batch[1]
        source_ids,  attention_masks, features = batch[0], batch[1], batch[4]
        style = self.net.get_extractor_output(
            use_cache_context_ids=source_ids)
        feature_extractor_output = self.net.feature_extractor(features)
        feature_extractor_output = feature_extractor_output.unsqueeze(1)
        style += feature_extractor_output
        
        style_src = self.net.get_extractor_output(
            use_cache_context_ids=self.src_ids.to(device='cuda'))
        style_tgt = self.net.get_extractor_output(
            use_cache_context_ids=self.tgt_ids.to(device='cuda'))
        style_src = torch.mean(style_src, dim = 0).unsqueeze(0)
        style_tgt = torch.mean(style_tgt, dim = 0).unsqueeze(0)
        style_src = torch.mean(style_src, dim = 1).unsqueeze(1)
        style_tgt = torch.mean(style_tgt, dim = 1).unsqueeze(1)
        #print(style_src.size())
        
        style_src_features = self.net.feature_extractor(self.src_ids_features.to(device='cuda'))
        style_tgt_features = self.net.feature_extractor(self.tgt_ids_features.to(device='cuda'))
        style_src_features = torch.mean(style_src_features, dim = 0).unsqueeze(0).unsqueeze(1)
        style_tgt_features = torch.mean(style_tgt_features, dim = 0).unsqueeze(0).unsqueeze(1)
        #print(style_src_features.size())
        
        style += (style_tgt + style_tgt_features - style_src - style_src_features) * self.evaluate_kwargs['beta']
        style = torch.mean(style, 1).unsqueeze(1)
        outputs = self.net.generate(input_ids=source_ids, use_cache_extractor_outputs=style,
                                    #do_sample=False, num_beams = 8,
                                    max_length=self.sent_length, attention_mask=attention_masks)
        simplified_sentences = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        self.val_simplified_sentences.extend(simplified_sentences)
    
    def on_validation_epoch_end(self):
        sys_sents_path = get_repo_dir() / str(wandb.run.project) / str(wandb.run.name) / f'{self.current_epoch}_{self.global_step}'
        write_lines(self.val_simplified_sentences, sys_sents_path)
        scores =  evaluate_system_output(
            self.evaluate_kwargs['test_set'],
            sys_sents_path=sys_sents_path,
            orig_sents_path=self.evaluate_kwargs['orig_sents_path'],
            refs_sents_paths=self.evaluate_kwargs['refs_sents_paths'],
            metrics=['sari', 'bleu', 'fkgl'],
            quality_estimation=False,
        )
        
        self.log_dict(scores)
        self.val_simplified_sentences.clear()
        #noisy_input_ids = apply_noise(
        #    input_ids, self.tokenizer, self.sent_length)
        #noisy_input_ids = self.net.generate(input_ids=noisy_input_ids, use_cache_extractor_outputs=0,
        #                                    do_sample=True, max_length=self.sent_length, min_length=self.sent_length)
        #extractor_output = self.net.get_extractor_output(
        #    use_cache_context_ids=context_ids)
#
        #extractor_output_input = self.net.get_extractor_output(
        #    use_cache_context_ids=input_ids)
        #barlow_twins_loss_func = BarlowTwinsLoss(batch_size=64, lambda_coeff = self.delta_val)
        #barlow_twins_loss = barlow_twins_loss_func(
        #    extractor_output_input, extractor_output)
        #self.log('val_bt_loss', barlow_twins_loss)
#
        #output_loss = self.net(input_ids=noisy_input_ids, labels=input_ids,
        #                       use_cache_extractor_outputs=extractor_output).loss
        #self.log('val_reconstruction_loss', output_loss)
#
        #if output_loss is not None:
        #    self.log("val_loss", output_loss +
        #             self.lambda_val * barlow_twins_loss)
        #else:
        #    self.log("val_loss", 0)
            
        #scores = self.evaluate_and_save(batch)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.net.parameters(), self.lr)
    
    def extract_features(self, sentence):
        features = torch.FloatTensor([len(tokenize(sentence)), len(sentence),
                              get_dependency_tree_depth(sentence, language='pt'), 
                              get_lexical_complexity_score(sentence, language='pt')])
        return features