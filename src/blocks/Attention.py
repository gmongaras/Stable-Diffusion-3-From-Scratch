import torch
from torch import nn
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
from src.blocks.patchify import patchify, unpatchify
from src.blocks.rotary_embedding import RotaryEmbedding, apply_rotary_emb
import src.blocks.rotary_embedding_2d as rotary_embedding_2d
import src.blocks.rotary_embedding_2d_v2 as rotary_embedding_2d_v2


class Attention(nn.Module):
    def __init__(self, dim, num_heads = 8, attn_type = "cosine", causal=False, emb_dim=None, positional_encoding="absolute", RoPE_Scale=1, kv_merge_attn=False, qk_half_dim=False, layer_idx=None, dual=False, last=False):
        super().__init__()

        self.positional_encoding = positional_encoding
        self.kv_merge_attn = kv_merge_attn
        self.RoPE_Scale = RoPE_Scale

        self.layer_idx = layer_idx
        # Dual blocks have two streams concatenated
        self.dual = dual
        # The last layer doesn't have to have a context output
        self.last = last

        # If the attention type is "both", even indices are softmax while odd indices are cosine
        if attn_type == "both":
            attn_type = "softmax" if layer_idx % 2 == 0 else "cosine"

        dim_qk = dim // 2 if qk_half_dim else dim
        
        # Projections
        if self.dual:
            self.query_proj_x = nn.Linear(dim, dim_qk if emb_dim == None else emb_dim, bias = False)
            self.key_proj_x = nn.Linear(dim, dim_qk if emb_dim == None else emb_dim, bias = False)
            self.value_proj_x = nn.Linear(dim, dim if emb_dim == None else emb_dim, bias = False)
            self.out_proj_x = nn.Linear(dim if emb_dim == None else emb_dim, dim if emb_dim == None else emb_dim, bias = False)
            self.query_proj_c = nn.Linear(dim, dim_qk if emb_dim == None else emb_dim, bias = False)
            self.key_proj_c = nn.Linear(dim, dim_qk if emb_dim == None else emb_dim, bias = False)
            self.value_proj_c = nn.Linear(dim, dim if emb_dim == None else emb_dim, bias = False)
            if not self.last:
                self.out_proj_c = nn.Linear(dim if emb_dim == None else emb_dim, dim if emb_dim == None else emb_dim, bias = False)
        else:
            self.query_proj = nn.Linear(dim, dim_qk if emb_dim == None else emb_dim, bias = False)
            self.key_proj = nn.Linear(dim, dim_qk if emb_dim == None else emb_dim, bias = False)
            self.value_proj = nn.Linear(dim, dim if emb_dim == None else emb_dim, bias = False)
            self.out_proj = nn.Linear(dim if emb_dim == None else emb_dim, dim if emb_dim == None else emb_dim, bias = False)
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim_qk = (dim_qk if emb_dim == None else emb_dim) // num_heads
        self.head_dim = (dim if emb_dim == None else emb_dim) // num_heads
        if attn_type == "softmax" or attn_type == "softmax_flash":
            self.scale = self.head_dim ** -0.5

            # Softmax attention also needs q k norms
            if self.dual:
                self.q_norm_x = nn.RMSNorm(self.head_dim_qk)
                self.k_norm_x = nn.RMSNorm(self.head_dim_qk)
                self.q_norm_c = nn.RMSNorm(self.head_dim_qk)
                self.k_norm_c = nn.RMSNorm(self.head_dim_qk)
            else:
                self.q_norm = nn.RMSNorm(self.head_dim_qk)
                self.k_norm = nn.RMSNorm(self.head_dim_qk)

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
        elif attn_type == "relu":
            pass
        elif attn_type == "silu":
            pass
        elif attn_type == "exp":
            pass
        else:
            raise RuntimeError(f"attn_type must be either softmax or cosine, but got {attn_type}")
        self.attn_type = attn_type
        self.causal = causal
        

        # Rotary embeddings
        if positional_encoding == "RoPE":
            self.rotary_emb = RotaryEmbedding(self.head_dim_qk, use_xpos=False, interpolate_factor=1/RoPE_Scale)
        elif positional_encoding == "RoPE2d":
            # Divide by 2 since we are applying to two dimensions. One to one half, the other to the other half
            self.rotary_emb = RotaryEmbedding(self.head_dim_qk//2, use_xpos=False, interpolate_factor=1/RoPE_Scale)
            # self.rotary_emb = rotary_embedding_2d.precompute_freqs_cis_2d(
            #     dim=self.head_dim_qk,
            #     height=1024//8,
            #     width=1024//8,
            #     theta=100_000.0,
            # )
        elif positional_encoding == "RoPE2dV2":
            # Divide by 2 since we are applying to two dimensions. One to one half, the other to the other half
            self.rotary_emb = rotary_embedding_2d_v2.RoPE2D(self.head_dim_qk, interpolate_factor=1/RoPE_Scale)
            # self.rotary_emb = rotary_embedding_2d.precompute_freqs_cis_2d(
            #     dim=self.head_dim_qk,
            #     height=1024//8,
            #     width=1024//8,
            #     theta=100_000.0,
            # )
        
        
        
        
    def forward(self, x, c=None, orig_shape=None):
        B, N, d = x.shape
        B, M, d = c.shape if self.dual else (B, N, d)

        if self.dual:
            assert c is not None, "Dual attention requires context tensor c"


        # RMSNorm if softmax
        # Project the queries, keys, and values (N, C, d) --> (N, H, C, d//H)
        if self.attn_type == "softmax" or self.attn_type == "softmax_flash":
            if self.dual:
                queries_x = self.q_norm_x(self.query_proj_x(x).reshape(B, N, self.num_heads, self.head_dim_qk).permute(0, 2, 1, 3))
                keys_x = self.k_norm_x(self.key_proj_x(x).reshape(B, N, self.num_heads, self.head_dim_qk).permute(0, 2, 1, 3))
                values_x = self.value_proj_x(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                queries_c = self.q_norm_c(self.query_proj_c(c).reshape(B, M, self.num_heads, self.head_dim_qk).permute(0, 2, 1, 3))
                keys_c = self.k_norm_c(self.key_proj_c(c).reshape(B, M, self.num_heads, self.head_dim_qk).permute(0, 2, 1, 3))
                values_c = self.value_proj_c(c).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            else:
                queries = self.q_norm(self.query_proj(x).reshape(B, N, self.num_heads, self.head_dim_qk).permute(0, 2, 1, 3))
                keys = self.k_norm(self.key_proj(x).reshape(B, N, self.num_heads, self.head_dim_qk).permute(0, 2, 1, 3))
                values = self.value_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        else:
            if self.dual:
                queries_x = self.query_proj_x(x).reshape(B, N, self.num_heads, self.head_dim_qk).permute(0, 2, 1, 3)
                keys_x = self.key_proj_x(x).reshape(B, N, self.num_heads, self.head_dim_qk).permute(0, 2, 1, 3)
                values_x = self.value_proj_x(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                queries_c = self.query_proj_c(c).reshape(B, M, self.num_heads, self.head_dim_qk).permute(0, 2, 1, 3)
                keys_c = self.key_proj_c(c).reshape(B, M, self.num_heads, self.head_dim_qk).permute(0, 2, 1, 3)
                values_c = self.value_proj_c(c).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            else:
                queries = self.query_proj(x).reshape(B, N, self.num_heads, self.head_dim_qk).permute(0, 2, 1, 3)
                keys = self.key_proj(x).reshape(B, N, self.num_heads, self.head_dim_qk).permute(0, 2, 1, 3)
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
                # queries_x, keys_x = self.rotary_emb.rotate_queries_and_keys(queries_x, keys_x, seq_dim=-2)
            else:
                queries[:, :, :N] = self.rotary_emb.rotate_queries_or_keys(queries[:, :, :N])
                keys[:, :, :N] = self.rotary_emb.rotate_queries_or_keys(keys[:, :, :N])
                # queries[:, :, :N], keys[:, :, :N] = self.rotary_emb.rotate_queries_and_keys(queries[:, :, :N], keys[:, :, :N], seq_dim=-2)
        if self.positional_encoding == "RoPE2d":
            if self.dual:
                #"""
                # The height and width are divded by 2 since the patch size is 2
                height = orig_shape[-2] // 2
                width = orig_shape[-1] // 2

                # Convert back to a 2D image.
                queries_x = queries_x.reshape(B, self.num_heads, height, width, self.head_dim_qk)
                keys_x = keys_x.reshape(B, self.num_heads, height, width, self.head_dim_qk)
                
                # Get the frequencies
                freqs = self.rotary_emb.get_axial_freqs(height, width)

                # Apply RoPE
                queries_x = apply_rotary_emb(freqs, queries_x)
                keys_x = apply_rotary_emb(freqs, keys_x)

                # Flatten back to (B, H, N, d)
                queries_x = queries_x.reshape(B, self.num_heads, -1, self.head_dim_qk)
                keys_x = keys_x.reshape(B, self.num_heads, -1, self.head_dim_qk)
                """

                # The height and width are divded by 2 since the patch size is 2
                height = orig_shape[-2] // 2
                width = orig_shape[-1] // 2

                # Convert back to a 2D image.
                queries_x = queries_x.reshape(B, self.num_heads, height, width, self.head_dim_qk)
                keys_x = keys_x.reshape(B, self.num_heads, height, width, self.head_dim_qk)
                
                # Get the frequencies
                freqs = self.rotary_emb[:height, :width].to(queries_x.device)

                # Apply RoPE
                queries_x, keys_x = rotary_embedding_2d.apply_rotary_emb(queries_x, keys_x, freqs)

                # Flatten back to (B, H, N, d)
                queries_x = queries_x.reshape(B, self.num_heads, -1, self.head_dim_qk)
                keys_x = keys_x.reshape(B, self.num_heads, -1, self.head_dim_qk)
                """
            else:
                assert False
                # queries[:, :, :N] = self.rotary_emb.rotate_queries_or_keys(queries[:, :, :N])
                # keys[:, :, :N] = self.rotary_emb.rotate_queries_or_keys(keys[:, :, :N])
                queries[:, :, :N], keys[:, :, :N] = self.rotary_emb.rotate_queries_and_keys(queries[:, :, :N], keys[:, :, :N], seq_dim=-2)
        if self.positional_encoding == "RoPE2dV2":
            if self.dual:
                #"""
                # The height and width are divded by 2 since the patch size is 2
                height = orig_shape[-2] // 2
                width = orig_shape[-1] // 2

                # Convert back to a 2D image.
                queries_x = queries_x.reshape(B, self.num_heads, height, width, self.head_dim_qk)
                keys_x = keys_x.reshape(B, self.num_heads, height, width, self.head_dim_qk)
                
                # Apply RoPE
                queries_x = self.rotary_emb(queries_x)
                keys_x = self.rotary_emb(keys_x)

                # Flatten back to (B, H, N, d)
                queries_x = queries_x.reshape(B, self.num_heads, -1, self.head_dim_qk)
                keys_x = keys_x.reshape(B, self.num_heads, -1, self.head_dim_qk)
            else:
                assert False
        # No positional encoding for the text

        # If merging, we merge the keys and values along the sequence dimension
        if self.kv_merge_attn:
            if self.dual:
                # Should be a factor of 2
                assert keys_x.shape[2] % 2 == 0, "Merge attention requires an even number of keys"
                assert keys_c.shape[2] % 2 == 0, "Merge attention requires an even number of keys"
                keys_x = (keys_x[:, :, ::2] + keys_x[:, :, 1::2]) / 2
                values_x = (values_x[:, :, ::2] + values_x[:, :, 1::2]) / 2
                keys_c = (keys_c[:, :, ::2] + keys_c[:, :, 1::2]) / 2
                values_c = (values_c[:, :, ::2] + values_c[:, :, 1::2]) / 2
            else:
                assert keys.shape[2] % 2 == 0, "Merge attention requires an even number of keys"
                keys = (keys[:, :, ::2] + keys[:, :, 1::2]) / 2
                values = (values[:, :, ::2] + values[:, :, 1::2]) / 2
        
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
            attn = flash_attn_func(queries.transpose(1, 2).to(torch.bfloat16), keys.transpose(1, 2).to(torch.bfloat16), values.transpose(1, 2).to(torch.bfloat16), causal=self.causal, softmax_scale=self.scale).transpose(1, 2).to(queries.dtype)
            
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
        
        elif self.attn_type == "relu":
            # Get norms of keys and queries
            keys = keys.relu()
            queries = queries.relu()

            attn = (queries @ (keys.mT @ values)) / \
                (queries @ keys.mT.sum(-1, keepdim=True))
            
        elif self.attn_type == "silu":
            # Get norms of keys and queries
            keys = torch.nn.functional.silu(keys)
            queries = torch.nn.functional.silu(queries)

            attn = (queries @ (keys.mT @ values)) / \
                (queries @ keys.mT.sum(-1, keepdim=True))
            
        elif self.attn_type == "exp":
            # Get norms of keys and queries
            keys = keys.exp()
            queries = queries.exp()

            attn = (queries @ (keys.mT @ values)) / \
                (queries @ keys.mT.sum(-1, keepdim=True))



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
