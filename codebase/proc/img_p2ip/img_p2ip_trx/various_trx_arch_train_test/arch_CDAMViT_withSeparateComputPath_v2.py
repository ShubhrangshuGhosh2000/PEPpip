import torch
import torch.nn as nn
from torch.nn import functional as F

# Custom MultiheadAttention class that explicitly returns attention weights
class CustomMultiheadAttention(nn.MultiheadAttention):
    """Custom MultiheadAttention that explicitly returns attention weights
    
    This implements Custom Attention Module by ensuring attention
    weights are always returned from the forward pass, solving the key issue
    with standard PyTorch implementation where attention weights aren't returned.
    """
    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(embed_dim, num_heads, **kwargs)
        

    def forward(self, query, key, value, **kwargs):
        # Call parent class forward method which returns (attn_output, attn_weights)
        kwargs['need_weights'] = True
        attn_output, attn_weights = super().forward(query, key, value, 
                                                #    need_weights=True, 
                                                   average_attn_weights=False,
                                                   **kwargs)
        # Explicitly return both output and attention weights
        return attn_output, attn_weights
# End of CustomMultiheadAttention class


class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """Custom TransformerEncoderLayer with custom attention mechanism
    
    Replaces standard MultiheadAttention with our CustomMultiheadAttention
    to ensure attention weights are accessible for CDAM.
    """
    def __init__(self, d_model, nhead, **kwargs):
        super().__init__(d_model, nhead, **kwargs)
        # Replace the standard self-attention with our custom version
        self.self_attn = CustomMultiheadAttention(d_model, nhead, 
                                                   batch_first=kwargs.get('batch_first', False))
# End of CustomTransformerEncoderLayer class


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
                 num_heads=12, hidden_dim=768, num_layers=12,
                 grad_scale_lambda=0.7, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.img_resoln = img_resoln
        self.num_patches = (img_resoln // patch_size) ** 2
        
        # Patch Embedding
        self.patch_embed = nn.Conv2d(3, hidden_dim,
                                    kernel_size=patch_size,
                                    stride=patch_size)
        
        # ######################################################
        # Instead of using nn.TransformerEncoder, build transformer layers with our custom implementation -Start
        # ######################################################
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
        
        # Manual transformer encoder implementation to access layers easily
        self.transformer_blocks = nn.ModuleList(encoder_layers)
        # ######################################################
        # Instead of using nn.TransformerEncoder, build transformer layers with our custom implementation -End
        # ######################################################
        
        # Class Token and Positional Embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, hidden_dim)
        )
        
        # Classifier Head
        self.head = nn.Linear(hidden_dim, num_classes)
        
        # Attention Scaling Parameters
        self.grad_scale_lambda = grad_scale_lambda
        
        # Storage for final block's input tokens and attention weights
        self.final_block_input = None
        self.attention_weights = None
    # End of __init__() method


    def forward(self, x):
        """CDAM-ViT Forward Pass
        
        Returns:
            logits (torch.Tensor): Classification outputs [B, num_classes]
            attn_map (torch.Tensor): Class-discriminative attention map [B, H, W]
        """
        # B: Batch size; 
        # H: Number of attention heads; 
        # N: Number of patches (625 for 400x400 input with 16x16 patches)
        # N+1: Number of tokens (1 class token + N patches )
        # D: Hidden dimension (768) = 16x16x3 = a single flattened patch dimension
        
        B = x.shape[0]  # Batch size
        # --- Input Processing ---
        x = self.patch_embed(x)  # (B, C, Ht, W) -> (B, D, h, w) where Ht, W are original image height and width (400 each) respectively; h=Ht/16=400/16=25, w=W/16=400/16=25 are number of (16x16) patches along Ht and W respectively.
        x = x.flatten(2).transpose(1, 2)  # (B, h*w, D) = (B, N, D) where N = h*w = Number of patches (625 for 400x400 input with 16x16 patches) = Number of image tokens
        
        # --- Token & Position Embedding ---
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (1, 1, D) => (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, N+1, D] where N+1 is number of tokens (1 class token + N patches)
        x += self.pos_embed  # Learned positional embeddings
        
        # --- Process through all transformer blocks EXCEPT the final one ---
        for i in range(len(self.transformer_blocks) - 1):
            x = self.transformer_blocks[i](x)
        
        # --- Critical CDAM Section: Process final transformer block separately ---
        # Store the input to the final transformer block for gradient calculation
        # This directly implements the CDAM approach: only consider the final block
        final_block_input = x.detach() # [B, N+1, D] where N+1 is number of tokens (1 class token + N patches)
        final_block_input.requires_grad = True
        
        # Process through final transformer block
        final_layer = self.transformer_blocks[-1]
        
        # Forward pass through final transformer layer
        # (Following the structure of TransformerEncoderLayer but with access to attention)
        if final_layer.norm_first:
            x_norm = final_layer.norm1(x)
            # Get both the attention output and attention weights
            attn_output, attn_weights = final_layer.self_attn(x_norm, x_norm, x_norm)
            x = x + final_layer.dropout1(attn_output)
            x = x + final_layer.dropout2(final_layer.linear2(
                final_layer.dropout(final_layer.activation(final_layer.linear1(final_layer.norm2(x))))
            ))
        else:
            # Get both the attention output and attention weights
            attn_output, attn_weights = final_layer.self_attn(x, x, x)
            x = x + final_layer.dropout1(attn_output)
            x = final_layer.norm1(x)
            x = x + final_layer.dropout2(final_layer.linear2(
                final_layer.dropout(final_layer.activation(final_layer.linear1(x)))
            ))
            x = final_layer.norm2(x)
        
        # Store attention weights for gradient scaling
        self.attention_weights = attn_weights  # (B, H, N+1, N+1); Explanation below.
        # Shape of attn: (B, H, N+1, N+1); Full attention matrix from a transformer encode layer; 
        # Represents attention scores between all pairs of tokens; Each position [i,j] shows how much token i attends to token j.
        
        # --- Classifier and CDAM Calculation ---
        # Get logits from the final [CLS] token
        logits = self.head(x[:, 0]) # (B, 2); Explanation below.
        # x: Transformer output tensor of shape (B, N+1, D) where: B = batch size; N+1 = 626 tokens (1 class token + 625 image patches) and D = 768 (hidden dimension)
        # x[:, 0] indicates the first token ([CLS] token) across the batch: (B, D)
        # self.head(x[:, 0]) implies Linear projection: (B, D) to (B, num_classes);
        # For PPI: (B, 768) => (B, 2)
        # CLS token aggregates information from all patches through self-attention and the final classification decision is made using only this token.
        # The CLS token acts as a "summary" of the entire protein interaction map.
        
        # --- CDAM: Calculate gradient-scaled attention maps in evaluation mode ---
        if not self.training:  # In evaluation mode
            # Force gradient tracking for the entire CDAM calculation
            with torch.enable_grad():
                # Temporarily set relevant layers to train mode to ensure they track gradients
                # Store the original modes of different layers of the model (to be used later for the original modes restoration)
                final_layer_mode = final_layer.training
                head_mode = self.head.training

                try:
                    final_layer.train(True)  # Set to training mode temporarily
                    self.head.train(True)    # Set classifier to training mode
                    # This is critical: calculating gradients of class-logits w.r.t. token activations in the final transformer block ONLY (as specified in the paper)
                    # Create a parallel computation path for visualization
                    # Process final_block_input through final transformer block (separate from main path)
                    if final_layer.norm_first:
                        viz_norm = final_layer.norm1(final_block_input)
                        viz_output, viz_attn_weights = final_layer.self_attn(viz_norm, viz_norm, viz_norm)
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
                    # End of if-else block: if final_layer.norm_first:
                    
                    # Compute visualization-specific logits from this parallel path
                    viz_logits = self.head(viz_x[:, 0])
                    target_class = 1  # PPI positive class
                    viz_cls_score = viz_logits[:, target_class]
                
                    # Calculate gradients for token relevance along this parallel computation path for visualization
                    grads = torch.autograd.grad(
                        outputs=viz_cls_score.sum(),
                        inputs=final_block_input,
                        create_graph=False,
                        retain_graph=True
                    )[0]  # Shape: [B, N+1, D]
                    
                    # CDAM formula: S_i,c = ∑_j T_ij * ∂f_c/∂T_ij
                    # Element-wise multiply tokens with their gradients and sum over embedding dim
                    print(f'###########################\n grads: {grads}\n ##################')
                    token_importance = (final_block_input * grads).sum(dim=-1)  # [B, N+1]
                    print(f'###########################\n token_importance: {token_importance}\n ##################')
                    
                    # Extract CLS token's attention to all tokens (including itself)
                    cls_attn = self.attention_weights[:, :, 0, :]  # (B, H, N+1); Explanation below. 
                    # Breaking it down:
                    # self.attention_weights.shape: (B, H, N+1, N+1)
                    # self.attention_weights[:, :]: Keep all batches and all heads
                    # self.attention_weights[:, :, 0]: Extract only the class token's row (index 0)
                    # self.attention_weights[:, :, 0, :]: Extract attention from class token to all tokens (including the class token itself)
                    # Shape of result: (B, H, N+1)
                    # viz_attn_weights can also be use in place of self.attention_weights
                    
                    # Scale attention by token importance (core CDAM operation)
                    # Broadcasting token_importance [B, N+1] -> [B, 1, N+1]
                    scaled_attn = cls_attn * token_importance.unsqueeze(1)  # (B, H, N+1)
                    
                    # Average over attention heads
                    attn_map = scaled_attn.mean(dim=1)  # [B, N+1]
                    
                    # Remove CLS token attention and reshape to 2D feature map
                    patch_attn = attn_map[:, 1:]  # [B, N] - remove CLS token
                    map_size = self.img_resoln // self.patch_size  # 25 for 400×400 input with 16x16 patches
                    attn_map_2d = patch_attn.reshape(B, map_size, map_size)  # (B, N) => (B, h, w) i.e. (B, 625) => (B, 25, 25)
                    
                    # Upsample to original image resolution
                    attn_map = F.interpolate(
                        attn_map_2d.unsqueeze(1),  # [B, 1, h, w]
                        size=(self.img_resoln, self.img_resoln),  # (Ht, W) = (400, 400)
                        mode='bilinear'
                    ).squeeze(1)  # [B, Ht, W] = [B, 400, 400]
                finally:
                    # Restore original modes regardless of errors
                    final_layer.train(final_layer_mode)
                    self.head.train(head_mode)
        else:
            # Simplified computation for training or when gradients disabled
            attn_map = torch.zeros(B, self.img_resoln, self.img_resoln, device=x.device)
        
        return logits, attn_map  # logits.shape: (B,2); attn_map.shape: [B, Ht, W] = [B, 400, 400]
    # End of forward() method
# End of CDAMViT class
