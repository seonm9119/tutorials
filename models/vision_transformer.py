import torch
import torch.nn as nn
from .vit.sublayers import PatchEmbed
from .vit.blocks import Block

import math
from functools import partial
from timm.models.layers import trunc_normal_

class VisionTransformer(nn.Module):

    def __init__(self, img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 num_classes=0, 
                 embed_dim=768, 
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size=img_size, 
                                      patch_size=patch_size, 
                                      in_chans=in_chans, 
                                      embed_dim=embed_dim)
        
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  
        
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, 
                  num_heads=num_heads, 
                  mlp_ratio=mlp_ratio, 
                  qkv_bias=qkv_bias, 
                  qk_scale=qk_scale,
                  drop=drop_rate, 
                  attn_drop=attn_drop_rate, 
                  drop_path=dpr[i], 
                  norm_layer=norm_layer) for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

       
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


    def forward(self, x):
        B, _, w, h = x.shape
        x = self.patch_embed(x)  

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.pos_drop(x)
    
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]



def vit_tiny(patch_size=16, **kwargs):
    return VisionTransformer(patch_size=patch_size, 
                              embed_dim=192, 
                              depth=12, 
                              num_heads=3, 
                              mlp_ratio=4,
                              qkv_bias=True, 
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)



def vit_small(patch_size=16, **kwargs):
    return VisionTransformer(patch_size=patch_size, 
                              embed_dim=384, 
                              depth=12, 
                              num_heads=6, 
                              mlp_ratio=4,
                              qkv_bias=True, 
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)



def vit_base(patch_size=16, **kwargs):
    return VisionTransformer(patch_size=patch_size, 
                              embed_dim=768, 
                              depth=12, 
                              num_heads=12, 
                              mlp_ratio=4,
                              qkv_bias=True, 
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def vit_large(patch_size=16, **kwargs):
    return VisionTransformer(patch_size=patch_size, 
                              embed_dim=1024, 
                              depth=24, 
                              num_heads=16, 
                              mlp_ratio=4,
                              qkv_bias=True, 
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
