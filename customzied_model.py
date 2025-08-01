from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
import bitsandbytes as bnb
import torch
from transformers.models.t5.modeling_t5 import T5Attention , T5LayerNorm, T5ForConditionalGeneration
import torch.nn as nn
import math
import os
from typing import Optional, Union

MODEL_NAME = "VietAI/vit5-base"
dtype = torch.float16
class MultiQuery_attention(T5Attention):
    def __init__(self, config, has_relative_attention_bias=False,
        layer_idx: Optional[int] = None):
        super().__init__(config, has_relative_attention_bias,layer_idx)
        self.head = config.num_heads
        self.d_model = config.d_model
        self.d_h = self.d_model // self.head
        self.q = nn.Linear(self.d_model, self.d_model, bias= False, dtype = dtype) 
        self.k = nn.Linear(self.d_model, self.d_h,bias= False, dtype = dtype) 
        self.v = nn.Linear(self.d_model, self.d_h, bias= False, dtype = dtype)
        self.o = nn.Linear(self.d_model, self.d_model, bias=False, dtype = dtype)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
        cache_position=None,
    ):
        batch_size, seq_len = hidden_states.shape[:2]
        
        dtype = hidden_states.dtype
        device = hidden_states.device
        q = self.q(hidden_states)  # [B, T, d_model]

        key_value_encoder = hidden_states if key_value_states is None else key_value_states
        k = self.k(key_value_encoder)  # [B, T, d_h]
        v = self.v(key_value_encoder)  # [B, T, d_h]

        q = q.view(batch_size, seq_len, self.head, self.d_h).transpose(1, 2)  # [B, head, T, d_h]
        k = k.unsqueeze(1)  # [B, 1, T, d_h]
        v = v.unsqueeze(1)  # [B, 1, T, d_h]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_h)

        # Position bias
        if position_bias is None and self.has_relative_attention_bias:
            context_position = torch.arange(seq_len, dtype=dtype, device=hidden_states.device)[:, None]
            memory_position = torch.arange(seq_len, dtype=dtype, device=hidden_states.device)[None, :]
            relative_position = memory_position - context_position
            relative_position_bucket = self._relative_position_bucket(relative_position)
            position_bias = self.relative_attention_bias(relative_position_bucket)
            position_bias = position_bias.permute(2, 0, 1).unsqueeze(0)  # [1, head, T, T]

        if position_bias is not None:
            scores = scores + position_bias

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = nn.functional.softmax(scores, dim=-1)
         # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask
            
        attn_output = torch.matmul(attn_weights, v)  # [B, head, T, d_h]

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        attn_output = self.o(attn_output)
        attn_output = self.dropout(attn_output)

        outputs = (attn_output, position_bias)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs
    

def convert_model(model):
    for block in model.encoder.block:
        block.layer[0].SelfAttention = MultiQuery_attention(config = model.config)


    for block in model.decoder.block:
        block.layer[0].SelfAttention = MultiQuery_attention(config = model.config)
        block.layer[1].EncDecAttention = MultiQuery_attention(config = model.config)
    
    return model 


def freeze_params(customzied_model): 
    # Đóng băng tất cả tham số
    for param in customzied_model.parameters():
        param.requires_grad = False

    # Mở train phần SelfAttention encoder
    for block in customzied_model.encoder.block:
        for param in block.layer[0].SelfAttention.parameters():
            param.requires_grad = True

    # Mở train phần SelfAttention và EncDecAttention decoder
    for block in customzied_model.decoder.block:
        for param in block.layer[0].SelfAttention.parameters():
            param.requires_grad = True
        for param in block.layer[1].EncDecAttention.parameters():
            param.requires_grad = True
    return customzied_model 


# load model ---> custom 
def load_customized_model():
    bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,                        
    bnb_4bit_compute_dtype=torch.float16,    
    bnb_4bit_quant_type="nf4",              
    bnb_4bit_use_double_quant=True         
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=bnb_config
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total initial model params: {total_params}")
    customzied_model = convert_model(model)
    total_redesigned_params = sum(p.numel() for p in customzied_model.parameters())
    print(f"After redesignning - Total params: {total_redesigned_params}, comparing to the intial Model: {round(total_redesigned_params/total_params * 100, 5)}%")
    freezed_model = freeze_params(customzied_model) 
    trained_params = sum(p.numel() for p in freezed_model.parameters() if p.requires_grad == True)


    print(f"After freezing - Total params: {total_redesigned_params}, Trainable params: {round(trained_params/total_redesigned_params * 100, 5)}%")
    print(f"After freezing - Trainable params comparing to the inital Model: {round(trained_params/total_params * 100, 5)}%")

    return freezed_model,  tokenizer

