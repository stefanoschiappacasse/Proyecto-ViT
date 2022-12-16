import torch
from torch import nn, Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


class PatchEmbedding(nn.Module):
    """Esta clase permite crear el patch embedding del modelo ViT. 
    Este contiene la proyecciones de los patches en conjunto con el 
    token de clasificación y los positional embeddings.

    Args:
        in_channels (int): cantidad de canales de la imagen de entrada
        patch_size (int): tamaño del patch
        emb_size (int): tamaño del embedding luego de la transformación lineal
        img_size (int): tamaño de la imagen de entrada
    """

    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size * patch_size * in_channels, emb_size)
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn(num_patches + 1, emb_size))
                    
    def forward(self, x: Tensor) -> Tensor:
        batch_size, C, H, W = x.shape
        P = self.patch_size
        n_row = int((H*W)/(P**2))
        n_col = int((P**2)*C)
        x = x.view(batch_size, n_row, n_col)
        x = self.projection(x)
        cls_tokens = self.cls_token.repeat(batch_size,1,1)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions
        return x

class MultiHeadAttention(nn.Module):
    """Esta clase ejecuta el multihead attention para los patch embeddings que
    entran como input.

    Args:
        emb_size (int): tamaño del patch embedding de entrada
        num_heads (int): cantidad de cabezas de atención a utilizar
        dropout (float): porcentaje de dropout a utilizar
    """
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    """Esta clase es la que se encarga de realizar la conexión 
    residual en el momento que corresponda.

    Args:
        fn (nn.Module): bloque de la red a la cual hay que hacerle una conexión residual
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Module):
    """Esta clase es la que realiza la pasada feed forward luego de haber realizado 
    el mecanismo de atención.

    Args:
        emb_size (int): tamaño del embedding que entra a la red feed forward
        expansion (int): tamaño de la expansión que se realiza dentro de la red
        drop_p (int): cantidad de dropout utilizado en esta red feed forward
    """
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
      super().__init__()
      self.ffnn = nn.Sequential(
          nn.Linear(emb_size, expansion * emb_size),
          nn.GELU(),
          nn.Dropout(drop_p),
          nn.Linear(expansion * emb_size, emb_size),
      )
    def forward(self, x):
      x = self.ffnn(x)
      return x


class TransformerEncoderBlock(nn.Module):
    """Esta clase es la que contiene un bloque encoder que cuenta
    con el bloque del multihead attention y con la parte lineal.

    Args:
        emb_size (int): tamaño el embedding de entrada
        drop_p (float): dropout utilizado en el bloque de atención y el lineal
        forward_expansion (int): factor de expansión que se utiliza en el bloque lineal
        forward_drop_p (float): dropout utilizado en la red feed forward
    """
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
      super().__init__()
      self.multiheadBlock = ResidualAdd(nn.Sequential(
              nn.LayerNorm(emb_size),
              MultiHeadAttention(emb_size, **kwargs),
              nn.Dropout(drop_p)
          ))
      self.linearBlock = ResidualAdd(nn.Sequential(
              nn.LayerNorm(emb_size),
              FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
              nn.Dropout(drop_p)
          ))

    def forward(self, x):
      x = self.multiheadBlock(x)
      x = self.linearBlock(x)
      return x


class TransformerEncoder(nn.Module):
    """Clase que contiene la secuencia de bloques transformer.

    Args:
        depth (int): profundidad o cantidad de bloques transformer a utilizar.
    """
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__()
        self.transformerList = nn.ModuleList([TransformerEncoderBlock(**kwargs) for _ in range(depth)])
        
    def forward(self, x):
        for i in range(len(self.transformerList)):
            x = self.transformerList[i](x)
        return x


class ClassificationHead(nn.Sequential):
    """Bloque de arquitectura que contiene la cabeza de clasificación
    del modelo.

    Args:
        emb_size (int): tamaño del patch embedding
        n_classes (int): cantidad de clases a utilizar
    """
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, n_classes))

