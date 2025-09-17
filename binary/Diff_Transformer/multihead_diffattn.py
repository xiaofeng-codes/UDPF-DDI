import math
import torch
import torch.nn.functional as F
from torch import nn

# from .kernel.rotary import apply_rotary_emb
# from flash_attn import flash_attn_func
from .rms_norm import RMSNorm  # 使用自定义的RMSNorm实现


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    对输入的KV张量进行重复操作，用于多头注意力机制中的KV头扩展。

    参数:
        x (torch.Tensor): 输入的KV张量，形状为 (bs, n_kv_heads, slen, head_dim)。
        n_rep (int): 每个KV头需要重复的次数。

    返回:
        torch.Tensor: 重复后的张量，形状为 (bs, n_kv_heads * n_rep, slen, head_dim)。
    """
    bs, n_kv_heads, slen, head_dim = x.shape  # 获取输入张量的形状
    if n_rep == 1:
        return x  # 如果不需要重复，直接返回原张量
    return (
        x[:, :, None, :, :]  # 在第2维增加一个维度
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)  # 扩展张量
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)  # 重塑张量形状
    )


def lambda_init_fn(depth):
    """
    根据当前层深度初始化lambda值。

    参数:
        depth (int): 当前层的索引（深度）。

    返回:
        float: 计算得到的lambda初始值。
    """
    return 0.8 - 0.6 * math.exp(-0.3 * depth)



class MultiheadDiffAttn(nn.Module):
    def __init__(
            self,
            embed_dim,  # 嵌入维度
            depth,  # 当前层索引
            num_heads,  # 注意力头数
            num_kv_heads=None,  # KV头数（默认为None，表示使用MHA）
    ):
        super().__init__()
        self.embed_dim = embed_dim  # 嵌入维度

        # 注意力头数，设置为基线Transformer的一半
        self.num_heads = num_heads

        # KV头数，如果使用GQA则设置为基线Transformer的一半
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.n_rep = self.num_heads // self.num_kv_heads  # 每个KV头对应的Q头数

        self.head_dim = embed_dim // num_heads // 2  # 每个头的维度
        self.scaling = self.head_dim ** -0.5  # 缩放因子

        # 线性变换层：Q, K, V, 输出
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # 初始化lambda参数
        self.lambda_init = lambda_init_fn(depth)  # 根据深度初始化lambda
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))

        # 子层归一化
        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(
            self,
            x,  # 输入张量
            attn_mask=None,  # 注意力掩码（可选）
    ):
        bsz, tgt_len, embed_dim = x.size()  # 获取输入形状
        src_len = tgt_len  # 源序列长度等于目标序列长度

        # 线性变换得到Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 重塑Q, K, V的形状
        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)

        # 应用旋转位置编码
        q = apply_rotary_pos_emb(q)
        k = apply_rotary_pos_emb(k)

        # 调整Q, K, V的维度顺序
        offset = src_len - tgt_len
        q = q.transpose(1, 2)
        k = repeat_kv(k.transpose(1, 2), self.n_rep)
        v = repeat_kv(v.transpose(1, 2), self.n_rep)
        q *= self.scaling  # 缩放Q

        # 计算注意力权重
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        # if attn_mask is None:
        #     # 如果没有提供掩码，则生成上三角掩码
        #     attn_mask = torch.triu(
        #         torch.zeros([tgt_len, src_len])
        #         .float()
        #         .fill_(float("-inf"))
        #         .type_as(attn_weights),
        #         1 + offset,
        #     )
        attn_weights = torch.nan_to_num(attn_weights)  # 处理NaN值
        if attn_mask is not None:
            attn_weights += attn_mask  # 应用掩码
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)  # Softmax归一化

        # 计算lambda值
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        # 调整注意力权重形状并应用lambda
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]

        # 计算注意力输出
        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)  # 子层归一化
        attn = attn * (1 - self.lambda_init)  # 应用lambda_init
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)  # 重塑形状

        # 输出线性变换
        attn = self.out_proj(attn)      # 不使用残差连接大法
        return attn

def rotate_half(x):
    """将输入向量的后半部分旋转（复数乘法的一种实现方式）"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)




def apply_rotary_pos_emb(q, theta=1e-4):
    """
    应用旋转位置编码到查询q上
    Args:
        q: 查询张量，形状为 (batch_size, seq_len, num_heads, d_head)
        sin_vals: 正弦值，形状为 (seq_len, d_head//2)
        cos_vals: 余弦值，形状为 (seq_len, d_head//2)
    Returns:
        q_rotated: 旋转后的查询张量
    """
    batch_size, seq_len, num_heads, head_dim = q.shape
    # 生成位置索引（0到seq_len-1）
    position = torch.arange(seq_len, device=q.device).unsqueeze(-1)
    # 计算频率因子：theta_i = theta^(2i/d), i从0到d/2-1
    freq = 1.0 / (theta ** (2 * torch.arange(0, head_dim // 2, device=q.device).float() / head_dim))
    # 计算旋转角度：m * theta_i（m为位置，theta_i为频率）
    rot_angle = position * freq  # 形状 (seq_len, head_dim//2)
    # 将角度转换为复数形式 [cos(angle), sin(angle)]
    cos_vals = torch.cos(rot_angle)  # 形状 (seq_len, head_dim//2)
    sin_vals = torch.sin(rot_angle)  # 形状 (seq_len, head_dim//2)


    # 调整形状以便广播
    cos_vals = cos_vals.view(1, -1, 1, cos_vals.size(-1), 1)  # (1, seq_len, 1, d_head//2, 1)
    sin_vals = sin_vals.view(1, -1, 1, sin_vals.size(-1), 1)  # 同上

    # 拆分q的最后维度为 (d_head//2, 2)
    q_reshaped = q.view(*q.size()[:-1], -1, 2)  # (..., d_head//2, 2)

    # 应用旋转
    q_rotated = torch.stack(
        [
            q_reshaped[..., 0] * cos_vals.squeeze(-1) - q_reshaped[..., 1] * sin_vals.squeeze(-1),
            q_reshaped[..., 0] * sin_vals.squeeze(-1) + q_reshaped[..., 1] * cos_vals.squeeze(-1)
        ],
        dim=-1
    )

    # 恢复原始形状
    return q_rotated.view(q.size())


