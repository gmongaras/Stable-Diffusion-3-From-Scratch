import torch
from torch import nn
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
from src.blocks.patchify import patchify, unpatchify
from src.blocks.rotary_embedding import RotaryEmbedding


class Attention(nn.Module):
    def __init__(self, dim, num_heads = 8, attn_type = "cosine", causal=False, emb_dim=None, positional_encoding="absolute", layer_idx=None, dual=False, last=False):
        super().__init__()

        self.positional_encoding = positional_encoding

        self.layer_idx = layer_idx
        # Dual blocks have two streams concatenated
        self.dual = dual
        # The last layer doesn't have to have a context output
        self.last = last

        # If the attention type is "both", even indices are softmax while odd indices are cosine
        if attn_type == "both":
            attn_type = "softmax" if layer_idx % 2 == 0 else "cosine"
        
        # Projections
        if self.dual:
            self.query_proj_x = nn.Linear(dim, dim if emb_dim == None else emb_dim, bias = False)
            self.key_proj_x = nn.Linear(dim, dim if emb_dim == None else emb_dim, bias = False)
            self.value_proj_x = nn.Linear(dim, dim if emb_dim == None else emb_dim, bias = False)
            self.out_proj_x = nn.Linear(dim if emb_dim == None else emb_dim, dim if emb_dim == None else emb_dim, bias = False)
            self.query_proj_c = nn.Linear(dim, dim if emb_dim == None else emb_dim, bias = False)
            self.key_proj_c = nn.Linear(dim, dim if emb_dim == None else emb_dim, bias = False)
            self.value_proj_c = nn.Linear(dim, dim if emb_dim == None else emb_dim, bias = False)
            if not self.last:
                self.out_proj_c = nn.Linear(dim if emb_dim == None else emb_dim, dim if emb_dim == None else emb_dim, bias = False)
        else:
            self.query_proj = nn.Linear(dim, dim if emb_dim == None else emb_dim, bias = False)
            self.key_proj = nn.Linear(dim, dim if emb_dim == None else emb_dim, bias = False)
            self.value_proj = nn.Linear(dim, dim if emb_dim == None else emb_dim, bias = False)
            self.out_proj = nn.Linear(dim if emb_dim == None else emb_dim, dim if emb_dim == None else emb_dim, bias = False)
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = (dim if emb_dim == None else emb_dim) // num_heads
        if attn_type == "softmax" or attn_type == "softmax_flash":
            self.scale = self.head_dim ** -0.5

            # Softmax attention also needs q k norms
            if self.dual:
                self.q_norm_x = nn.RMSNorm(dim, dim)
                self.k_norm_x = nn.RMSNorm(dim, dim)
                self.q_norm_c = nn.RMSNorm(dim, dim)
                self.k_norm_c = nn.RMSNorm(dim, dim)
            else:
                self.q_norm = nn.RMSNorm(dim, dim)
                self.k_norm = nn.RMSNorm(dim, dim)

        elif attn_type == "cosine":
            self.norm_const = nn.Parameter(0.5*torch.ones(1, num_heads, 1, 1, dtype=self.query_proj_x.weight.dtype).to(self.query_proj_x.weight.device))
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
        

        # Rotary embeddings
        if positional_encoding == "RoPE":
            self.rotary_emb = RotaryEmbedding(self.head_dim)
        
        
        
        
    def forward(self, x, c=None):
        B, N, d = x.shape
        B, M, d = c.shape if self.dual else (B, N, d)

        if self.dual:
            assert c is not None, "Dual attention requires context tensor c"


        # RMSNorm if softmax
        # Project the queries, keys, and values (N, C, d) --> (N, H, C, d//H)
        if self.attn_type == "softmax" or self.attn_type == "softmax_flash":
            if self.dual:
                queries_x = self.q_norm_x(self.query_proj_x(x)).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                keys_x = self.k_norm_x(self.key_proj_x(x)).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                values_x = self.value_proj_x(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                queries_c = self.q_norm_c(self.query_proj_c(c)).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                keys_c = self.k_norm_c(self.key_proj_c(c)).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                values_c = self.value_proj_c(c).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            else:
                queries = self.q_norm(self.query_proj(x)).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                keys = self.k_norm(self.key_proj(x)).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                values = self.value_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        else:
            if self.dual:
                queries_x = self.query_proj_x(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                keys_x = self.key_proj_x(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                values_x = self.value_proj_x(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                queries_c = self.query_proj_c(c).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                keys_c = self.key_proj_c(c).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                values_c = self.value_proj_c(c).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            else:
                queries = self.query_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                keys = self.key_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                values = self.value_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Normalize if cosine attention
        if self.attn_type == "cosine" or self.attn_type == "cosine2":
            if self.dual:
                queries_x = torch.nn.functional.normalize(queries_x, dim=-1, p=2)
                keys_x = torch.nn.functional.normalize(keys_x, dim=-1, p=2)
                queries_c = torch.nn.functional.normalize(queries_c, dim=-1, p=2)
                keys_c = torch.nn.functional.normalize(keys_c, dim=-1, p=2)
            else:
                queries = torch.nn.functional.normalize(queries, dim=-1, p=2)
                keys = torch.nn.functional.normalize(keys, dim=-1, p=2)


        # Apply rotary embeddings only to the image
        if self.positional_encoding == "RoPE":
            if self.dual:
                queries_x = self.rotary_emb.rotate_queries_or_keys(queries_x)
                keys_x = self.rotary_emb.rotate_queries_or_keys(keys_x)
            else:
                queries[:, :, :N] = self.rotary_emb.rotate_queries_or_keys(queries[:, :, :N])
                keys[:, :, :N] = self.rotary_emb.rotate_queries_or_keys(keys[:, :, :N])

        
        # Concat if dual
        if self.dual:
            queries = torch.cat([queries_x, queries_c], dim=2)
            keys = torch.cat([keys_x, keys_c], dim=2)
            values = torch.cat([values_x, values_c], dim=2)
            N_old = N
            N = N + M

            
        # Softmax attention
        if self.attn_type == "softmax":
            # Create mask
            if self.causal:
                mask = torch.tril(torch.ones(B, self.num_heads, N, N, requires_grad=False)).bool().to(x.device)
                    
            # Flash attention
            # attn = flash_attn_func(queries, keys, values, causal=self.causal)

            attn = (queries @ keys.mT) * self.scale
            
            if self.causal:
                attn = attn.masked_fill(mask, float('-inf')).softmax(dim=-1)
            else:
                attn = attn.softmax(dim=-1)

            attn = attn @ values

        # Flash attention
        elif self.attn_type == "softmax_flash":
            # Create mask
            if self.causal:
                mask = torch.tril(torch.ones(B, self.num_heads, N, N, requires_grad=False)).bool().to(x.device)
                    
            # Flash attention
            attn = flash_attn_func(queries.to(torch.bfloat16), keys.to(torch.bfloat16), values.to(torch.bfloat16), causal=self.causal).to(queries.dtype)
            
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
                mask = torch.tril(torch.ones(B, self.num_heads, N, N, requires_grad=False)).bool().to(x.device)
                
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
            mask = torch.tril(torch.ones(B, self.num_heads, N, N, requires_grad=False)).bool().to(x.device) if self.causal \
                    else torch.ones(B, self.num_heads, N, N, requires_grad=False).bool().to(x.device)

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



        # Split if dual
        if self.dual:
            attn_x, attn_c = attn[:, :, :N_old], attn[:, :, N_old:]


        # Remove heads
        if self.dual:
            attn_x = attn_x.permute(0, 2, 1, 3).reshape(B, N_old, -1)
            attn_c = attn_c.permute(0, 2, 1, 3).reshape(B, M, -1)
        else:
            attn = attn.permute(0, 2, 1, 3).reshape(B, N, -1)



        # Output projection
        if self.dual:
            return self.out_proj_x(attn_x), (self.out_proj_c(attn_c) if not self.last else attn_c)
        else:
            return self.out_proj(attn)
