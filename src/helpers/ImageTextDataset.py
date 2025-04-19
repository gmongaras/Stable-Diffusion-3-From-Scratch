import torch
from torch.utils.data import Dataset

class ImageTextDataset(Dataset):
    def __init__(self, image_dataset, text_encoder, tokenizer, device, dummy_text="A default caption", transform=None):
        """
        Custom dataset wrapper for image and dummy text encoding.

        Args:
            image_dataset: Original dataset for images (e.g., ImageNet).
            text_encoder: Text encoder (e.g., self.VAE_T5_CLIP).
            tokenizer: Tokenizer for the text encoder.
            device: Device for text encoding (e.g., 'cpu' or 'cuda').
            dummy_text: Dummy text to encode for all images.
            transform: Image transforms.
        """
        self.image_dataset = image_dataset
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.device = device
        self.dummy_text = dummy_text
        self.transform = transform

        # Pre-encode the dummy text
        self.encoded_text = self._encode_text(dummy_text)

    def _encode_text(self, text):
        """Encodes the text using the text encoder."""
        with torch.inference_mode():
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            outputs = self.text_encoder.encoder(**inputs)  # Assuming T5-like encoder
            return outputs.last_hidden_state.squeeze(0)  # Shape: (seq_len, hidden_dim)

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        # Load the image
        image, _ = self.image_dataset[idx]
        return image, self.encoded_text
