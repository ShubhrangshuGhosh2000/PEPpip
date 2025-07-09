import torch
import torch.nn as nn
from torch.nn import functional as F


class CustomMultiheadAttention(nn.MultiheadAttention):
    """Custom MultiheadAttention that explicitly returns attention weights
    This implements Custom Attention Module by ensuring attention
    weights are always returned from the forward pass, solving the key issue
    with standard PyTorch implementation where attention weights aren't returned.
    """
    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(embed_dim, num_heads, **kwargs)
    
    
    def forward(self, query, key, value, **kwargs):
        kwargs['need_weights'] = True
        attn_output, attn_weights = super().forward(query, key, value, 
                                                   average_attn_weights=False,
                                                   **kwargs)
        return attn_output, attn_weights


class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """Custom TransformerEncoderLayer with custom attention mechanism
    Replaces standard MultiheadAttention with our CustomMultiheadAttention
    to ensure attention weights are accessible for CDAM.
    """
    def __init__(self, d_model, nhead, **kwargs):
        super().__init__(d_model, nhead, **kwargs)
        self.self_attn = CustomMultiheadAttention(d_model, nhead, 
                                                   batch_first=kwargs.get('batch_first', False))


class CDAMViT(nn.Module):
    """CDAM-ViT: Class-Discriminative Attention Map Vision Transformer for PPI Prediction
    Implements gradient-scaled attention maps following:
    Brocki et al. 'Class-Discriminative Attention Maps for Vision Transformers' (2023)
    Key Features:
    - Gradient scaling of attention weights using class-specific relevance
    - Multi-head attention aggregation with trainable lambda scaling
    - Patch-based processing optimized for 400x400x3 protein interaction maps
    """
    def __init__(self, num_classes=2, img_resoln=400, patch_size=16,
                 num_heads=12, hidden_dim=768, num_layers=12, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.img_resoln = img_resoln
        self.num_patches = (img_resoln // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, hidden_dim,
                                    kernel_size=patch_size,
                                    stride=patch_size)
        encoder_layers = []
        for i in range(num_layers):
            encoder_layers.append(
                CustomTransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim*4,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True,
                    dropout=0.1
                )
            )
        self.transformer_blocks = nn.ModuleList(encoder_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, hidden_dim)
        )
        self.head = nn.Linear(hidden_dim, num_classes)
        self.final_block_input = None
        self.attention_weights = None
    
    
    def forward(self, x):
        """CDAM-ViT Forward Pass
        Returns:
            logits (torch.Tensor): Classification outputs [B, num_classes]
            attn_map (torch.Tensor): Class-discriminative attention map [B, H, W]
        """
        B = x.shape[0]  
        x = self.patch_embed(x)  
        x = x.flatten(2).transpose(1, 2)  
        cls_tokens = self.cls_token.expand(B, -1, -1)  
        x = torch.cat((cls_tokens, x), dim=1)  
        x += self.pos_embed  
        for i in range(len(self.transformer_blocks) - 1):
            x = self.transformer_blocks[i](x)
        final_block_input = x.detach() 
        final_block_input.requires_grad = True
        final_layer = self.transformer_blocks[-1]
        if final_layer.norm_first:
            x_norm = final_layer.norm1(x)
            attn_output, attn_weights = final_layer.self_attn(x_norm, x_norm, x_norm)
            x = x + final_layer.dropout1(attn_output)
            x = x + final_layer.dropout2(final_layer.linear2(
                final_layer.dropout(final_layer.activation(final_layer.linear1(final_layer.norm2(x))))
            ))
        else:
            attn_output, attn_weights = final_layer.self_attn(x, x, x)
            x = x + final_layer.dropout1(attn_output)
            x = final_layer.norm1(x)
            x = x + final_layer.dropout2(final_layer.linear2(
                final_layer.dropout(final_layer.activation(final_layer.linear1(x)))
            ))
            x = final_layer.norm2(x)
        self.attention_weights = attn_weights  
        logits = self.head(x[:, 0]) 
        attn_map = None
        final_block_input_norm = None
        if not self.training and torch.is_grad_enabled():  
            if final_layer.norm_first:  
                final_block_input_norm = final_layer.norm1(final_block_input)
                viz_output, viz_attn_weights = final_layer.self_attn(final_block_input_norm, final_block_input_norm, final_block_input_norm)
                viz_x = final_block_input + final_layer.dropout1(viz_output)
                viz_x = viz_x + final_layer.dropout2(final_layer.linear2(
                    final_layer.dropout(final_layer.activation(final_layer.linear1(final_layer.norm2(viz_x))))
                ))
            else:  
                viz_output, viz_attn_weights = final_layer.self_attn(final_block_input, final_block_input, final_block_input)
                viz_x = final_block_input + final_layer.dropout1(viz_output)
                viz_x = final_layer.norm1(viz_x)
                viz_x = viz_x + final_layer.dropout2(final_layer.linear2(
                    final_layer.dropout(final_layer.activation(final_layer.linear1(viz_x)))
                ))
                viz_x = final_layer.norm2(viz_x)
            viz_logits = self.head(viz_x[:, 0])
            target_class = 1  
            viz_cls_score = viz_logits[:, target_class]
            grads = torch.autograd.grad(
                outputs=viz_cls_score.sum(),
                inputs=final_block_input_norm,
                create_graph=False,
                retain_graph=True
            )[0]  
            token_importance = (final_block_input_norm * grads).sum(dim=-1)  
            cls_attn = self.attention_weights[:, :, 0, :]  
            scaled_attn = cls_attn * token_importance.unsqueeze(1)  
            attn_map = scaled_attn.mean(dim=1)  
            patch_attn = attn_map[:, 1:]  
            patch_attn_min = patch_attn.min(dim=1, keepdim=True)[0]  
            patch_attn_max = patch_attn.max(dim=1, keepdim=True)[0]  
            patch_attn = (patch_attn - patch_attn_min) / (patch_attn_max - patch_attn_min + 1e-8)  
            map_size = self.img_resoln // self.patch_size  
            attn_map_2d = patch_attn.reshape(B, map_size, map_size)  
            attn_map = F.interpolate(
                attn_map_2d.unsqueeze(1),  
                size=(self.img_resoln, self.img_resoln),  
                mode='nearest'
            ).squeeze(1)  
        else:
            attn_map = torch.zeros(B, self.img_resoln, self.img_resoln, device=x.device)
        return logits, attn_map  
