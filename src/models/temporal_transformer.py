import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from models.model_utils.reverse_layer import ReverseLayerF


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class TemporalTransformer(nn.Module):
    def __init__(self, frames_per_clip, num_classes=1, dim=256, depth=6, heads=12, mlp_dim=1024, pool = 'cls', modality_embedding=False, dim_head = 64, dropout = 0., emb_dropout = 0., predict_fake_label=False, fake_classes=1):
        super().__init__()

        self.pos_embedding = nn.Parameter(torch.randn(1, frames_per_clip + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()

        self.modality_embedding = modality_embedding
        if self.modality_embedding:
            self.v_embedding = nn.Parameter(torch.randn(1, frames_per_clip + 1, dim//2))
            self.a_embedding = nn.Parameter(torch.randn(1, frames_per_clip + 1, dim//2))
            self.dropout2 = nn.Dropout(emb_dropout)


        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        if predict_fake_label:
                self.fake_type_mlp_head = nn.Sequential(
                    nn.LayerNorm(dim),
                    # nn.Linear(dim, fake_classes),
                    nn.Linear(dim, dim//2),
                    nn.LayerNorm(dim//2),
                    nn.Linear(dim//2, fake_classes),
                )

    def forward(self, x, predict_fake_label=False, alpha=0):
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        if self.modality_embedding:
            x += torch.cat((self.v_embedding, self.a_embedding), dim=-1)[:, :(n+1)]
            x = self.dropout2(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)

        feature = x

        if predict_fake_label:
            reverse_x = ReverseLayerF.apply(x, alpha)
            return self.mlp_head(x), self.fake_type_mlp_head(reverse_x), feature
        else:
            return self.mlp_head(x), feature

class TemporalTransformerWithoutMLP(nn.Module):
    def __init__(self, frames_per_clip, num_classes=1, dim=256, depth=6, heads=12, mlp_dim=1024, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.pos_embedding = nn.Parameter(torch.randn(1, frames_per_clip + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x):
        b, n, _ = x.shape

        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x)
        
        return x