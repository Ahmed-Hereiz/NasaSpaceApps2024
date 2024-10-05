from dataclasses import  dataclass
from typing import Tuple
import torch 
from torch import nn


@dataclass
class VitConfig:
      hidden_dim:int = 384
      num_blocks:int = 6
      kernel_size:int = 3
      bias:bool = True
      qkv_bias:bool = False
      use_layer_scale: bool = False
      num_classes:int = 1
      num_heads: int = 6
      patch_size:int = 7
      mlp_fac: int = 4
      fc_intermediate_dim: int = 128
      in_chs: int =128
      img_size: Tuple[int] = (4508,)
      drop_rate:float = 0.1
      attn_drop: float = 0.1
      embd_drop_rate: float = 0.1
      mlp_drop_rate: float = 0.2
      layer_scale_init: float = 1e-4
      mlp_act_str: str = "GELU"
      act_final: str = "ReLU"


class PatchEmbed(nn.Module):
    def __init__(
        self,
        config
    ):
        super(PatchEmbed, self).__init__()
        img_size = config.img_size
        patch_size = config.patch_size
        self.grid_size = tuple((sz // patch_size for sz in img_size))
        self.num_patches = 1
        for sz in self.grid_size:
           self.num_patches = int (sz * self.num_patches)

        if len(img_size) == 3:
            self.proj = nn.Conv3d(config.in_chs, config.hidden_dim, kernel_size=config.kernel_size, padding=1)
        else:
           self.proj = nn.Sequential(
               nn.Conv1d(config.in_chs, config.hidden_dim, kernel_size=config.patch_size, stride=config.patch_size),
               nn.GELU(),
               nn.Conv1d(config.hidden_dim,config.hidden_dim, stride=2, kernel_size=config.kernel_size, padding=1),
               nn.GELU(),
           )
        inp = torch.randn((1, config.in_chs) + img_size)
        with torch.no_grad():
           o = self.proj(inp)
           o = o.flatten(2).transpose(1, 2)
        self.num_patches  = int(o.shape[1])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        #self.distilation_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        self.num_patches += 1
        self.pos_embd = nn.Parameter(torch.zeros(1, self.num_patches, config.hidden_dim))

        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        #torch.nn.init.trunc_normal_(self.distilation_token, std=0.02)
        torch.nn.init.trunc_normal_(self.pos_embd, std=0.02)
        self.drop = nn.Dropout(config.embd_drop_rate)
        #self.norm = nn.LayerNorm(config.hidden_dim, eps=1e-6)

    def add_special_tokens(self, hiddens):
        bs, _, _ = hiddens.shape
        cls_token = self.cls_token.expand(bs, -1, -1)
        #distilation_tok = self.distilation_token.expand(bs, -1, -1)
        return torch.cat((cls_token, hiddens), dim=1)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
          # BCHW -> BNC
        x = self.add_special_tokens(x)
        x = x + self.pos_embd
        return self.drop(x)
    

class VitAttention(nn.Module):
  def __init__(self, config):
      super(VitAttention, self).__init__()
      self.wqkv = nn.Linear(config.hidden_dim, 3 * config.hidden_dim, config.qkv_bias)
      self.wo = nn.Linear(config.hidden_dim, config.hidden_dim, config.qkv_bias)
      self.num_heads = config.num_heads
      self.head_dim = config.hidden_dim // self.num_heads
      self.attn_drop = config.attn_drop
      self.hidden_dim = config.hidden_dim

  def forward(self, hidden_states):
      bs = hidden_states.size(0)
      qkv = self.wqkv(hidden_states)
      q, k, v = torch.chunk(qkv, 3, dim=-1)
      q = q.view(bs, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
      k = k.view(bs, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
      v = v.view(bs, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
      attention_scores = nn.functional.scaled_dot_product_attention(
          q, k, v, dropout_p=self.attn_drop)

      attention_scores = attention_scores.permute(0, 2, 1, 3).reshape(bs, -1, self.num_heads * self.head_dim)
      return self.wo(attention_scores)


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class VitBlock(nn.Module):
  def __init__(self, config):
      super(VitBlock, self).__init__()
      self.norm1 = nn.LayerNorm(config.hidden_dim, 1e-6)
      self.norm2 = nn.LayerNorm(config.hidden_dim, 1e-6)
      mlp_dim = int(config.mlp_fac * config.hidden_dim)
      self.attn = VitAttention(config)
      self.attn_drop = nn.Dropout(config.attn_drop)
      self.mlp = nn.Sequential(
          nn.Linear(config.hidden_dim, mlp_dim),
          getattr(nn, config.mlp_act_str)(),
          nn.Linear(mlp_dim, config.hidden_dim),
          nn.Dropout(config.mlp_drop_rate)
      )

  def forward(self, hidden_states):
      norm_hidden_states = self.norm1(hidden_states)
      attn_output = self.attn(norm_hidden_states)
      hidden_states = hidden_states + self.attn_drop(attn_output)

      norm_hidden_states = self.norm2(hidden_states)
      mlp_out = self.mlp(norm_hidden_states)
      hidden_states = hidden_states + mlp_out
      return hidden_states

class VitBlockLayerScale(nn.Module):
  def __init__(self, config):
      super(VitBlockLayerScale, self).__init__()
      self.norm1 = nn.LayerNorm(config.hidden_dim, 1e-6)
      self.norm2 = nn.LayerNorm(config.hidden_dim, 1e-6)
      mlp_dim = int(config.mlp_fac * config.hidden_dim)
      self.attn = VitAttention(config)
      self.mlp = nn.Sequential(
          nn.Linear(config.hidden_dim, mlp_dim),
          getattr(nn, config.mlp_act_str)(),
          nn.Linear(mlp_dim, config.hidden_dim),
          nn.Dropout(config.mlp_drop_rate)
      )
      self.gamma_1 = nn.Parameter(config.layer_scale_init * torch.ones(config.hidden_dim,), requires_grad=True)
      self.gamma_2 = nn.Parameter(config.layer_scale_init * torch.ones(config.hidden_dim,), requires_grad=True)
      self.drop_path = DropPath(config.drop_rate)

  def forward(self, hidden_states):
      norm_hidden_states = self.norm1(hidden_states)
      attn_output = self.attn(norm_hidden_states)
      hidden_states = hidden_states + self.drop_path(self.gamma_1 * attn_output)

      norm_hidden_states = self.norm2(hidden_states)
      mlp_out = self.mlp(norm_hidden_states)
      hidden_states = hidden_states + self.drop_path(self.gamma_2 * mlp_out)
      return hidden_states


class BaseModel(nn.Module):
  def set_drop(self, drop):
      for n, mod in self.named_modules():
        if isinstance(mod, nn.Dropout):
          mod.p = drop

        if isinstance(mod, VitAttention):
           mod.attn_drop = drop

  def infer(self):
      self.eval()
      for p in self.parameters():
        p.requires_grad = False

      self.set_drop(0.0)

  def train(self, mode=True):
      super().train(mode)
      for p in self.parameters():
        p.requires_grad = True
      self.set_drop(0.3)

class VitEncoder(nn.Module):
  def __init__(self, config):
      super(VitEncoder, self).__init__()
      self.patch_embd = PatchEmbed(config)

      self.norm_final = nn.LayerNorm(config.hidden_dim, 1e-6)
      self.patch_size = config.patch_size
      self.blocks = nn.ModuleList([])
      for _ in range(config.num_blocks):
        if config.use_layer_scale:
          self.blocks.append(VitBlockLayerScale(config))
        else:
          self.blocks.append(VitBlock(config))
      self.init_weights()


  def init_weights(self):
      def _init_weights(m):
          if isinstance(m, nn.Linear):
              nn.init.trunc_normal_(m.weight, std=.02)
              if isinstance(m, nn.Linear) and m.bias is not None:
                  nn.init.constant_(m.bias, 0)
          elif isinstance(m, nn.LayerNorm):
              nn.init.constant_(m.bias, 0)
              nn.init.constant_(m.weight, 1.0)
      self.apply(_init_weights)

  def forward(self, x):
      x = self.patch_embd(x)
      for blc in self.blocks:
        x = blc(x)
      return x


class VitModel(BaseModel):
  def __init__(self, config:VitConfig):
      super(VitModel, self).__init__()
      self.encoder = VitEncoder(config)
      self.norm = nn.LayerNorm(config.hidden_dim, 1e-6)
      self.fc = nn.Sequential(
        nn.Dropout(config.drop_rate), 
        nn.Linear(config.hidden_dim, config.fc_intermediate_dim),
        getattr(nn, config.acts_final)(),
        nn.Linear(config.fc_intermediate_dim, config.num_classes),
        nn.Sigmoid(),
      )

  def forward_feats(self, inputs):
      hidden_states = self.encoder(inputs)
      hidden_states = self.norm(hidden_states)
      return hidden_states
  
  def forward(self, inputs):
      hidden_states = self.forward_feats(inputs)
      hidden_states = hidden_states[:, 0, :]
      return self.fc(hidden_states)      
