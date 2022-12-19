""" Unión de las distintas partes del modelo en una sola clase"""

from ViT.ViT_parts import *

class ViT(nn.Module):
  def __init__(self,     
              in_channels: int = 3,
              patch_size: int = 16,
              emb_size: int = 768,
              img_size: int = 224,
              depth: int = 12,
              n_classes: int = 1000,
              **kwargs):
    super().__init__()
    self.patchEmbedding = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
    self.encoder = TransformerEncoder(depth, emb_size=emb_size, **kwargs)
    self.clf = ClassificationHead(emb_size, n_classes)

  def forward(self, x):
    x = self.patchEmbedding(x)
    x = self.encoder(x)
    x = self.clf(x)
    return x