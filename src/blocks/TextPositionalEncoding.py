import torch
import torch.nn as nn

class TextPositionalEncoding(nn.Module):
    def __init__(self, d: int, max_len: int = 154, learnable: bool = False):
        """
        Positional Encoding module for tensors of shape (N, seq_len, d).

        Args:
            d (int): Dimensionality of the embedding.
            max_len (int): Maximum sequence length.
            learnable (bool): If True, use learnable positional embeddings. Default is False (sinusoidal).
        """
        super(TextPositionalEncoding, self).__init__()
        
        self.d = d
        self.max_len = max_len

        if learnable:
            # Learnable positional encoding
            self.positional_embeddings = nn.Parameter(torch.zeros(max_len, d))
            nn.init.normal_(self.positional_embeddings, mean=0.0, std=0.02)
        else:
            # Sinusoidal positional encoding
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d, 2) * -(torch.log(torch.tensor(10000.0)) / d))
            pe = torch.zeros(max_len, d)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('positional_embeddings', pe)

    def forward(self, x):
        """
        Add positional encodings to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (N, seq_len, d).

        Returns:
            torch.Tensor: Tensor with positional encodings added.
        """
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")

        positional_encodings = self.positional_embeddings[:seq_len, :]
        return x + positional_encodings.unsqueeze(0)