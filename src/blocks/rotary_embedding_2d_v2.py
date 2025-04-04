import torch
import torch.nn as nn


class RoPE2D(nn.Module):
    def __init__(self, dim, interpolate_factor=1):
        super().__init__()
        self.dim = (dim // 3) * 3
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim, 3).float() / self.dim))
        self.register_buffer(
            "inv_freq",
            inv_freq
        )
        self.interpolate_factor = interpolate_factor

    def forward(self, x):
        len_ = (x.shape[-1] // 3) * 3
        x_subset = x[..., :len_]

        with torch.no_grad():
            # x: (batch, heads, length, width, dim)
            length = x.shape[2]
            width = x.shape[3]
            pos_len = torch.arange(length, device=x.device).float().unsqueeze(1) / self.interpolate_factor
            pos_wid = torch.arange(width, device=x.device).float().unsqueeze(1) / self.interpolate_factor

            # Thetas are rotated by the length. Alphas are rotated by the width.
            thetas = (pos_len * self.inv_freq)[None, None, :, None, :] # (1, length, 1, dim/3)
            alphas = (pos_wid * self.inv_freq)[None, None, None, :, :] # (1, 1, width, dim/3)

            # Compute sin and cosines for thetas and alphas
            thetas_sin, thetas_cos = thetas.sin(), thetas.cos()
            alphas_sin, alphas_cos = alphas.sin(), alphas.cos()

        # Get x indices
        x1, x2, x3 = x_subset[..., 0::3], x_subset[..., 1::3], x_subset[..., 2::3] # (1, length, width, dim/3)

        # Multiply and concatenate
        x_rot = torch.cat([
            x1 * thetas_cos  +   x2 * -thetas_sin * alphas_cos   +   x3 * thetas_sin * alphas_sin,
            x1 * thetas_sin  +   x2 * thetas_cos * alphas_cos    +   x3 * -thetas_cos * alphas_sin,
                                 x2 * alphas_sin                 +   x3 * alphas_cos
        ], dim=-1)

        x[..., :len_] = x_rot

        return x