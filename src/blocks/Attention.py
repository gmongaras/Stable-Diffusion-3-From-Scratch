import torch
from torch import nn
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
from src.blocks.patchify import patchify, unpatchify


class Attention(nn.Module):
    def __init__(self, dim, num_heads = 8, attn_type = "cosine", causal=False, rotary_dim=None, emb_dim=None, layer_idx=None):
        super().__init__()

        self.layer_idx = layer_idx

        # If the attention type is "both", even indices are softmax while odd indices are cosine
        if attn_type == "both":
            attn_type = "softmax" if layer_idx % 2 == 0 else "cosine"
        
        # Projections
        self.query_proj = nn.Linear(dim, dim if emb_dim == None else emb_dim, bias = False)
        self.key_proj = nn.Linear(dim, dim if emb_dim == None else emb_dim, bias = False)
        self.value_proj = nn.Linear(dim, dim if emb_dim == None else emb_dim, bias = False)
        self.out_proj = nn.Linear(dim if emb_dim == None else emb_dim, dim if emb_dim == None else emb_dim, bias = False)
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = (dim if emb_dim == None else emb_dim) // num_heads
        if attn_type == "softmax":
            self.scale = self.head_dim ** -0.5

            # Softmax attention also needs q k norms
            self.q_norm = nn.RMSNorm(dim, dim)
            self.k_norm = nn.RMSNorm(dim, dim)

        elif attn_type == "cosine":
            self.norm_const = nn.Parameter(0.5*torch.ones(1, num_heads, 1, 1, dtype=self.query_proj.weight.dtype).to(self.query_proj.weight.device))
        elif attn_type == "cosine2":
            pass
        elif attn_type == "cosine3":
            pass
        elif attn_type == "cosine4":
            pass
        elif attn_type == "cosine_norm":
            pass
        else:
            raise RuntimeError(f"attn_type must be either softmax or cosine, but got {attn_type}")
        self.attn_type = attn_type
        self.causal = causal
        self.rotary_dim = rotary_dim



        self.randomize = False
        if self.randomize:
            self.idx = {
                ((i//8)**2)//4: torch.randperm((((i//8)**2)//4)*dim).to(torch.long).to(self.query_proj.weight.device)
                for i in range(128, 257, 16)
            }
            self.inverses = {
                ((i//8)**2)//4: torch.empty_like(self.idx[(((i//8)**2)//4)]).to(torch.long).to(self.query_proj.weight.device)
                for i in range(128, 257, 16)
            }
            for k, v in self.idx.items():
                self.inverses[k][v] = torch.arange(v.size(0)).to(torch.long).to(self.query_proj.weight.device)
        
        
        
        
    def forward(self, x):
        N, C, d = x.shape


        if self.randomize:
            # Unpatchify the images
            shp = x.shape
            x = x.reshape(x.shape[0], -1)
            # Randomly shuffle the last index
            x = x[:, self.idx[shp[1]]]
            # Patchify the images
            x = x.reshape(shp)
            


        # RMSNorm if softmax
        # Project the queries, keys, and values (N, C, d) --> (N, H, C, d//H)
        if self.attn_type == "softmax":
            # Add RMS norm if softmax
            queries = self.q_norm(self.query_proj(x)).reshape(N, C, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            keys = self.k_norm(self.key_proj(x)).reshape(N, C, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            values = self.value_proj(x).reshape(N, C, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        else:
            queries = self.query_proj(x).reshape(N, C, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            keys = self.key_proj(x).reshape(N, C, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            values = self.value_proj(x).reshape(N, C, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Normalize if cosine attention
        if self.attn_type == "cosine" or self.attn_type == "cosine2":
            queries = torch.nn.functional.normalize(queries, dim=-1, p=2)
            keys = torch.nn.functional.normalize(keys, dim=-1, p=2)

            
        # Softmax attention
        if self.attn_type == "softmax":
            # Create mask
            if self.causal:
                mask = torch.tril(torch.ones(N, self.num_heads, C, C, requires_grad=False)).bool().to(x.device)
                    
            # Flash attention
            # attn = flash_attn_func(queries, keys, values, causal=self.causal)

            attn = (queries @ keys.mT) / self.scale
            
            if self.causal:
                attn = attn.masked_fill(mask, float('-inf')).softmax(dim=-1)
            else:
                attn = attn.softmax(dim=-1)

            attn = attn @ values
            
        # Cosine attention
        elif self.attn_type == "cosine":
            """
            # Inner product
            # denom = (queries@keys.mT).sum(-1, keepdim=True) #(queries * keys.sum(-2, keepdim=True)).sum(-1, keepdim=True)
            inner = (queries@keys.mT + 1) / 2
            denom = (inner).sum(-1, keepdim=True)
            # sign = torch.sign(denom)
            # denom = denom.abs_().clamp_(1)
            # denom *= sign
            num = inner / denom
            attn = num @ values

            """
            if self.causal:
                # Create mask
                mask = torch.tril(torch.ones(N, self.num_heads, C, C, requires_grad=False)).bool().to(x.device)
                
                # We need to normalize the values
                values = values / ((mask).sum(-1, keepdims=True)**self.norm_const.sigmoid()).clamp(min=1)
                
                # Inner product
                attn = ((queries @ keys.mT) * mask) @ values

            # Can be optimized if not causal
            else:
                # Normalization term
                # v = ((values.shape[2]*torch.ones(values.shape[2]).unsqueeze(0).repeat(values.shape[1], 1)[None, :, :, None].to(values.device))**self.norm_const.sigmoid().to(values.device))
                
                # We need to normalize the values
                values = values / (values.shape[2]**self.norm_const.sigmoid().to(values.device))

                # Inner product
                attn = queries @ (keys.mT @ values)
        
        elif self.attn_type == "cosine2":
            # # Create mask
            # mask = torch.tril(torch.ones(N, self.num_heads, C, C, requires_grad=False)).bool().to(x.device) if self.causal \
            #         else torch.ones(N, self.num_heads, C, C, requires_grad=False).bool().to(x.device)

            prod = (((queries @ keys.mT)+1))# * mask

            attn = prod / prod.sum(-1, keepdim=True)

            attn = attn @ values


        elif self.attn_type == "cosine3":
            # Create mask
            mask = torch.tril(torch.ones(N, self.num_heads, C, C, requires_grad=False)).bool().to(x.device) if self.causal \
                    else torch.ones(N, self.num_heads, C, C, requires_grad=False).bool().to(x.device)

            prod = (((queries @ keys.mT))) * mask

            attn = prod / prod.abs().sum(-1, keepdim=True)

            attn = attn @ values

        elif self.attn_type == "cosine4":
            # Get norms of keys and queries
            keys_norm = keys.norm(dim=-1, keepdim=True)
            query_norm = queries.norm(dim=-1, keepdim=True)

            scale = 1/(self.head_dim**0.5)
            
            # Inner product
            attn = ((queries @ keys.mT) * scale) + (query_norm * keys_norm.mT) * scale

            # Normalize
            attn = attn / attn.sum(dim=-1, keepdim=True)

            attn = attn @ values

        elif self.attn_type == "cosine_norm":
            # Get norms of keys and queries
            key_norm = keys.norm(dim=-1, keepdim=True)
            query_norm = queries.norm(dim=-1, keepdim=True)

            # Inner product
            attn_weights = torch.matmul(queries, keys.mT)
            # attn_weights_denom = (query_norm * key_norm.mT).sum(-1, keepdim=True)
            attn_weights_denom = (query_norm * key_norm.sum(-2, keepdim=True))

            # Weight normalization
            attn_weights = attn_weights / attn_weights_denom

            attn = attn_weights @ values



        if self.randomize:
            attn = self.out_proj(attn.permute(0, 2, 1, 3).reshape(N, C, self.dim))
            # Unpatchify the images
            attn = attn.reshape(attn.shape[0], -1)
            # Inverse the random shuffle
            attn = attn[:, self.inverses[shp[1]]]
            # Patchify the images
            return attn.reshape(shp)



        # Output projection
        return self.out_proj(attn.permute(0, 2, 1, 3).reshape(N, C, -1))
