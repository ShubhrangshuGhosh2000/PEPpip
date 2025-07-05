import torch
import torch.nn as nn
from torch.nn import functional as F

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
        
        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim*4,
                activation='gelu',
                batch_first=True,
                norm_first=True,
                dropout=0.1
            ),
            num_layers=num_layers
        )

        # Class Token and Positional Embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, hidden_dim)
        )
        
        # Classifier Head
        self.head = nn.Linear(hidden_dim, num_classes)
        
        # Attention Scaling Parameters
        self.grad_scale_lambda = grad_scale_lambda

        # Gradient & Attention Tracking
        self.attention_activations = []
        self.gradient_buffer = None
        self._register_hooks()


    def _register_hooks(self):
        """Register forward/backward module-level and tensor-level hooks for CDAM gradient scaling
            Refer Pytorch hooks video: https://www.youtube.com/watch?v=syLFCVYua6Q

            B: Batch size; 
            H: Number of attention heads; 
            N: Number of patches (625 for 400x400 input with 16x16 patches)
            N+1: Number of tokens (1 class token + N patches )
            D: Hidden dimension (768) = 16x16x3 = a single flattened patch dimension
        """
        def attention_hook(module, inputs, outputs):  # module-level forward post-hook
            """Store attention weights post-softmax from all transformer heads in self.attention_activations for gradient scaling later"""
            self.attention_activations.append(outputs[1].detach())  # outputs[1] = attention matrix (B, H, N+1, N+1)

        def gradient_hook(grad):  # tensor level hook (for backward graph)
            """Capture gradients of class logits w.r.t token activations in self.gradient_buffer for CDAM scaling"""
            self.gradient_buffer = grad  # self.gradient_buffer.shape: (B, num_layers, H, N+1)
            return grad  # Preserve original gradient flow

        # Register module-level forward post-hooks on all transformer layers
        self.hooks = []
        for layer in self.transformer.layers:
            self.hooks.append(
                # Attaches attention_hook to all transformer layers' self-attention modules
                layer.self_attn.register_forward_hook(attention_hook)
            )
            
        # Register tensor level hook (for backward graph) on class token
        self.cls_token.register_hook(gradient_hook)  # Attaches gradient_hook to class token parameter


    def forward(self, x):
        """CDAM-ViT Forward Pass
        Returns:
            logits (torch.Tensor): Classification outputs [B, num_classes]
            attn_map (torch.Tensor): Class-discriminative attention map [B, H, W]
        """
        # Input Processing
        B = x.shape[0]   # Batch size
        x = self.patch_embed(x)  # (B, C, H, W) -> (B, D, h, w) where H, W are original image height and width (400 each) respectively; h=H/16=400/16=25, w=W/16=400/16=25 are number of (16x16) patches along H and W respectively.
        x = x.flatten(2).transpose(1, 2)  # [B, h*w, D] = [B, N, D] where N = h*w = Number of patches (625 for 400x400 input with 16x16 patches) = Number of image tokens
        
        # Add Class Token and Position Embedding
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (1, 1, D) => (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, N+1, D] where N+1 is number of tokens (1 class token + N patches )
        x += self.pos_embed  # Learned positional embeddings
        
        # Transformer Forward with Attention Capture at each layer
        self.attention_activations.clear()
        x = self.transformer(x)  # (B, N+1, D)
        
        # CDAM Gradient Scaling
        if self.training and x.requires_grad:
            # Compute gradients for positive class (PPI=1)
            logits = self.head(x[:, 0])   # Class token => (B, num_classes)
            loss = F.cross_entropy(logits, torch.ones(B, device=x.device, dtype=torch.long))
            loss.backward(retain_graph=True)  # Populates gradient_buffer through gradient_hook;
            
        # Attention Scaling
        scaled_attn = []
        for layer_idx, attn in enumerate(self.attention_activations):
            # Gradient scaling (Eq.2 from CDAM paper)
            # Gradients quantify each attention head's contribution to PPI prediction (gradients of class logits w.r.t token activations).
            grad = self.gradient_buffer[:, layer_idx] if self.gradient_buffer is not None else 1.0  # (B, H, N+1); Explanation below.
            # Shape of self.gradient_buffer: (B, num_layers, H, N+1) where 
            # B = batch size;  num_layers = number of transformer encoder layers; H = number of attention heads; N+1 = number of tokens (1 class token + N patches )
            # Shape of grad: (B, H, N+1);  Here, extracting gradients for a specific layer across all heads.
            
            scaled = attn * grad.abs().mean(dim=-1, keepdim=True)  # (B, H, N+1, N+1); Explanation below.
            # Shape of attn: (B, H, N+1, N+1); Full attention matrix from a transformer encode layer; 
            # Represents attention scores between all pairs of tokens; Each position [i,j] shows how much token i attends to token j.
            # Shape of grad.abs().mean(dim=-1, keepdim=True): (B, H, 1); 
            # Taking mean across token dimension (N+1); Represents average gradient magnitude per attention head.
            # Shape of scaled: (B, H, N+1, N+1); Same shape as attn; Each attention score is scaled by the corresponding head's gradient magnitude.

            scaled_attn.append(scaled[:, :, 0, 1:])  # Focus only image patch tokens  => (B, H, N); Explanation below.
            # Purpose of scaled[:, :, 0, 1:] => Extract class token's attention to image patches only.
            # Breaking it down:
            # scaled[:, :]: Keep all batches and all heads
            # scaled[:, :, 0]: Extract only the class token's row (index 0)
            # scaled[:, :, 0, 1:]: Extract attention from class token to all image patch tokens (skipping class token itself)
            # Shape of result: (B, H, N)
            # Why do this?:
            # Class token (index 0) is used for final classification.
            # Patch tokens (indices 1:N+1) represent image regions.
            # Class attention to image patches shows which image regions most influenced the prediction.
            # This directly implements the core CDAM insight: "the class token's attention to patches, scaled by class-specific gradients".
            # The final output scaled[:, :, 0, 1:] precisely isolates the attention pattern from the classification token to each image patch, which 
            # after upsampling creates the class-discriminative attention map highlighting regions relevant to protein-protein interaction prediction.
        # End of for loop: for layer_idx, attn in enumerate(self.attention_activations):

        # Attention Aggregation
        cls_attn = torch.stack(scaled_attn, dim=1).mean(dim=(1, 2))  # (B, N); Explanation below.
        # Operation Breakdown:
        # torch.stack(scaled_attn, dim=1):
        # Input: scaled_attn which is a list of tensors (one per layer with shape [B, H, N]).
        # Action: Stacks all layer tensors along a new dimension (dim=1)
        # Output: Single tensor with shape [B, L, H, N] where L = number of layers (12 by default).
        # Purpose: Creates a 4D tensor where we can efficiently operate on all layers at once
        #
        # .mean(dim=(1, 2)):
        # Input: Tensor with shape [B, L, H, N]
        # Action: Averages values across both dimensions 1 (layers) and 2 (heads).
        # Output: Tensor with shape [B, N].
        # Purpose: Creates a consensus attention map by averaging across all layers and heads
        #
        # Final Result:
        # cls_attn has shape [B, N] (e.g., [625] for batch size 32). Each value represents the average importance of that patch across all layers and heads.
        # Higher values indicate patches more relevant to the positive class (PPI=1)
        
        # Gradient Stabilization (Sec 3.3)
        # Mixes raw and detached gradients to prevent divergence
        if self.training:
            cls_attn = (self.grad_scale_lambda * cls_attn + 
                       (1 - self.grad_scale_lambda) * cls_attn.detach())
        
        # Output Processing
        logits = self.head(x[:, 0])  # (B, 2); Explanation below.
        # x: Transformer output tensor of shape (B, N+1, D) where: B = batch size; N+1 = 626 tokens (1 class token + 625 image patches) and D = 768 (hidden dimension)
        # x[:, 0] indicates the first token ([CLS] token) across the batch: (B, D)
        # self.head(x[:, 0]) implies Linear projection: (B, D) to (B, num_classes);
        # For PPI: (B, 768) => (B, 2)
        # CLS token aggregates information from all patches through self-attention and the final classification decision is made using only this token.
        # The class token acts as a "summary" of the entire protein interaction map.
        
        # Reshape & Upsample Attention
        map_size = self.img_resoln // self.patch_size  # 25 for 400×400 input
        attn_map = cls_attn.reshape(B, map_size, map_size)  # (B, N) => (B, h, w) i.e. (B, 625) => (B, 25, 25)
        attn_map = F.interpolate(attn_map.unsqueeze(1), 
                               size=(self.img_resoln, self.img_resoln),
                               mode='bilinear').squeeze(1) # (B, 400, 400)
        
        return logits, attn_map
    

    def __del__(self):
        """Cleanup hooks to prevent memory leaks"""
        for hook in self.hooks:
            hook.remove()
