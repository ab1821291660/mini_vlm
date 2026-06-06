import os
import torch
import warnings
from .model_minimind import *
from typing import Optional, Tuple, List, Union
from torch import nn
from transformers import SiglipImageProcessor, SiglipVisionModel
from transformers.modeling_outputs import MoeCausalLMOutputWithPast##===================================

warnings.filterwarnings('ignore')

class VLMConfig(MiniMindConfig):
    model_type = "minimind-v"
    def __init__(self, image_special_token='<|image_pad|>', image_ids=[12], **kwargs):
        self.image_special_token = image_special_token
        self.image_ids = image_ids
        self.image_hidden_size = kwargs.get("image_hidden_size", 768)
        self.image_token_len = kwargs.get("image_token_len", 64)
        super().__init__(**kwargs)
def ________all():pass


                # class MMVisionProjector(nn.Module):
                #     def __init__(self, in_dim, out_dim, source_tokens=256, target_tokens=64):
                #         super().__init__()
                #         self.target_tokens = target_tokens
                #         self.merge = source_tokens // target_tokens
                #         self.mlp = nn.Sequential(
                #             nn.Linear(in_dim * self.merge, out_dim),
                #             nn.GELU(),
                #             nn.Linear(out_dim, out_dim),
                #         )
                #     def forward(self, x):
                #         b, n, d = x.shape
                #         x = x.reshape(b, self.target_tokens, d * self.merge)
                #         return self.mlp(x)
class MMVisionProjector(nn.Module):
    def __init__(self, in_dim, out_dim, source_tokens=64, target_tokens=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),#768-768
            nn.GELU(),
            nn.Linear(out_dim, out_dim),#768-768
        )
    def forward(self, x):#1-64-768
        return self.mlp(x)#1-64-768
# 继承自语言模型
class MiniMindVLM(MiniMindForCausalLM):
    config_class = VLMConfig

    def __init__(self, config: VLMConfig = None, vision_model_path="./model/siglip2-base-p32-256-ve"):
        self.config = config or VLMConfig()##===================================
        super().__init__(self.config)
        self.vision_encoder, self.processor = self.__class__.get_vision_model(vision_model_path)##===================================
        self.vision_proj = MMVisionProjector(self.config.image_hidden_size, self.config.hidden_size, target_tokens=self.config.image_token_len)
    @staticmethod
    def get_vision_model(model_path: str):
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        if not os.path.exists(model_path):
            return None, None
        try:
            model = SiglipVisionModel.from_pretrained(model_path)##===================================
        except (RuntimeError, ValueError):
            return None, None
        processor = SiglipImageProcessor.from_pretrained(model_path)##===================================
        # 冻结 vision_encoder 的所有参数
        for param in model.parameters():
            param.requires_grad = False##===================================
        return model.eval(), processor
    def ________________allmodel(self):
        pass




                    # @staticmethod
                    # def image2tensor(image, processor):
                    #     if image.mode in ['RGBA', 'LA']: image = image.convert('RGB')
                    #     inputs = processor(images=image, return_tensors="pt")['pixel_values']
                    #     return inputs
                    # @staticmethod
                    # def get_image_embeddings(image_tensors, vision_model):
                    #     with torch.no_grad():
                    #         outputs = vision_model.vision_model(pixel_values=image_tensors)
                    #     img_embedding = outputs.last_hidden_state[:, 1:, :].squeeze()
                    #     return img_embedding
    @staticmethod
    def image2tensor(image, processor):##在lm_dataset.py中调用了，已处理
        if image.mode in ['RGBA', 'LA']: image = image.convert('RGB')
        inputs = processor(images=image, return_tensors="pt")##===================================
        return inputs
    ##
    ##
    @staticmethod
    def get_image_embeddings(image_inputs, vision_model):#1-3-256-256
        if hasattr(image_inputs, 'keys'):
            image_inputs = {k: v.squeeze(1) if v.ndim > 2 and v.shape[1] == 1    else v    for k, v in image_inputs.items()}#1-3-256-256
        with torch.no_grad():#1-64-768----#1-3-256-256
            outputs = vision_model(**image_inputs)##===================================
        return outputs.last_hidden_state
    def _______vit(self):
        pass


    def _______proj(self):
        pass
    @torch.compiler.disable
    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512):
        if vision_tensors is None or not self.config.image_ids:
            return h
        marker, vf = self.config.image_ids[0], vision_tensors
        if vf.dim() == 3:
            vf = vf.unsqueeze(1)
        out = []
        for b in range(h.size(0)):
            hb, seq, k, i = h[b], tokens[b].tolist(), 0, 0
            while i < len(seq):
                if seq[i] == marker:
                    start = i
                    while i < len(seq) and seq[i] == marker:
                        i += 1
                    if k < vf.size(1):
                        hb = torch.cat((hb[:start], vf[b][k][:i - start], hb[i:]), dim=0)[:seqlen]
                        k += 1
                else:
                    i += 1
            out.append(hb)
        return torch.stack(out)

    def _______llm(self):
        pass
    def ________________all(self):
        pass
    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,##========
                labels: Optional[torch.Tensor] = None,
                pixel_values: Optional[torch.FloatTensor] = None,##===================================
                **args):
        batch_size, seq_length = input_ids.shape#1 #166
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.model.layers)#[None, None, None, None,    None, None, None, None]
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0##0



        #b1-166-768     ----#b1-166
        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))##===================================




        if pixel_values is not None and start_pos == 0:
            if hasattr(pixel_values, 'keys'):
                sample_val = next(iter(pixel_values.values()))
                if sample_val.ndim == 5:
                    bs, num = sample_val.shape[:2]
                    vision_tensors = self.vision_proj(
                                       MiniMindVLM.get_image_embeddings({k: v.flatten(0, 1) for k, v in pixel_values.items()}, self.vision_encoder)).view(bs, num, self.config.image_token_len, -1)
                else:##1-64-768##========
                    vision_tensors = self.vision_proj(##1-64-768----##1-64-768##===================================
                                       MiniMindVLM.get_image_embeddings(pixel_values, self.vision_encoder))###1-64-768----#1-3-256-256##===================================
            else:
                if len(pixel_values.shape) == 6:
                    pixel_values = pixel_values.squeeze(2)
                bs, num, c, im_h, im_w = pixel_values.shape
                vision_tensors = torch.stack([self.vision_proj(
                                                MiniMindVLM.get_image_embeddings(pixel_values[:, i, :, :, :], self.vision_encoder)) for i in range(num)], dim=1)



            ##1-166-768
            hidden_states = self.count_vision_proj(tokens=input_ids, h=hidden_states,##1-166  ##1-166-768##===================================
                                                   vision_tensors=vision_tensors, seqlen=input_ids.shape[1])##1-64-768





        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.model.freqs_cos[0, 0] == 0:#less##===================================##===================================##===================================##===================================
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.model.freqs_cos, self.model.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        position_embeddings = (##===================================##===================================##===================================##===================================
            self.model.freqs_cos[start_pos:start_pos + seq_length],#32768-96  #0-166==166-96
            self.model.freqs_sin[start_pos:start_pos + seq_length]#32768-96  #0-166==166-96
        )
        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.model.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,#1-166-768
                position_embeddings,#166-96 #166-96##===================================##===================================##===================================##===================================
                past_key_value=past_key_value,#none
                use_cache=use_cache,#true
                attention_mask=attention_mask#1-166
            )
            presents.append(present)


        #1-166-768
        hidden_states = self.model.norm(hidden_states)##===================================
        aux_loss = sum([l.mlp.aux_loss for l in self.model.layers if isinstance(l.mlp, MOEFeedForward)],  hidden_states.new_zeros(1).squeeze())
        aux_loss = aux_loss + sum(p.sum() for p in self.vision_proj.parameters()) * 0#tensor(0., device='cuda:0', dtype=torch.float16)  # dummy gradient for DDP ##===================================
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int)    else logits_to_keep##slice(0, None, None)
        logits = self.lm_head(hidden_states[:, slice_indices, :])#1-166-6400##===================================



        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
        output = MoeCausalLMOutputWithPast(loss=loss,#none
                                           aux_loss=aux_loss,#tensor(0., device='cuda:0', dtype=torch.float16)
                                           logits=logits,#1-166-6400
                                           past_key_values=presents,#8个的【（1-166-4-96），（1-166-4-96）】
                                           hidden_states=hidden_states)#1-166-768
        return output


    def generate(self, *args, num_return_sequences=1, **kwargs):
        if num_return_sequences > 1 and 'pixel_values' in kwargs:
            pv = kwargs['pixel_values']##===================================
            if hasattr(pv, 'keys'):
                kwargs['pixel_values'] = {k: v.repeat(num_return_sequences, *([1] * (v.ndim - 1))) for k, v in pv.items()}
            else:
                kwargs['pixel_values'] = pv.repeat(num_return_sequences, *([1] * (pv.ndim - 1)))
        return super().generate(*args,
                                num_return_sequences=num_return_sequences,#1
                                **kwargs)
