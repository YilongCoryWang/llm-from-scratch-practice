import torch
from torch import nn

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        #每个token都有自己的query，key和value（既作为query提问，也作为key索引，也作为value答案），用于计算相似度
        # 假设输入序列：["The", "cat", "sits", "here"]
        # 每个token经过投影后：
        # Token "The":  query = q0, key = k0, value = v0
        # Token "cat":  query = q1, key = k1, value = v1
        # Token "sits": query = q2, key = k2, value = v2
        # Token "here": query = q3, key = k3, value = v3
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        #计算每个token对其他token的相似度，值越大，两个token越相关
        # attn_scores[0][1] = q0 · k1
        #           = Token "The"的query · Token "cat"的key
        #           = Token "The" 对 Token "cat" 的相似度
        attn_scores = queries @ keys.transpose(1, 2)
        #防止看到未来的token
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        #最后一个维度（token维度）上做softmax，每个token的注意力权重之和 = 1
        #attn_scores / keys.shape[-1] ** 0.5缩放，防止 softmax 饱和（梯度消失）：
        # 假设点积很大
        # attn_scores = [100, 50, 10, 5]
        # 直接softmax
        # softmax([100, 50, 10, 5]) ≈ [1.0, 0.0, 0.0, 0.0]
        # 只有一个值接近1，其他都接近0。梯度几乎为0 → 梯度消失！
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        #随机丢弃一些注意力权重，防止过拟合。只在训练时生效，评估时自动关闭
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)
inputs = torch.tensor([
    [0.43, 0.15, 0.89],
    [0.55, 0.87, 0.66],
    [0.57, 0.86, 0.64],
    [0.22, 0.58, 0.33],
    [0.77, 0.25, 0.10],
    [0.05, 0.80, 0.55],
])
print("=== step1: inputs ===")
print("inputs.shape:", inputs.shape)
print("inputs:", inputs)
print()

d_in = inputs.shape[1] # 3
d_out = 2

batch = torch.stack((inputs, inputs), dim=0)
print("=== step2: create batch with stack ===")
print("batch.shape:", batch.shape)
print("batch:", batch)
print()

context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)

print("=== step3: context_vecs = ca(batch) ===")
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs.shape)
print("context_vecs:", context_vecs)