from transformers import CLIPProcessor, CLIPModel
from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM
from transformers.models.gemma.tokenization_gemma_fast import GemmaTokenizerFast
import open_clip
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os







# Class for VAE + CLIP + T5
class VAE_T5_CLIP_inference:
    def __init__(self, device):
        self.device = device

        # Load in the VAE
        self.VAE = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-schnell", subfolder="vae", cache_dir="./models/VAE", device=self.device).eval()
        # self.VAE = AutoencoderKL.from_single_file(url, config="./models/VAE/FLUX_config.json", cache_dir="./pretrained_models/VAE", device=self.device).eval()
        self.VAE_downsample = 8
        # Freeze all VAE parameters
        for param in self.VAE.parameters():
            param.requires_grad = False
        # Store locally to prevent issues with DDP
        self.VAE = self.VAE.eval().to(dtype=torch.float16, device=self.device)

        # Passes image data through the VAE and then samples from the latent distribution
        @torch.no_grad()
        @torch.inference_mode()
        def forward_VAE_and_sample(x):
            # 1. Encode
            # 2. Sample from the latent distribution
            # 3. Normalize the latent representation
            return self.VAE.encode(x).latent_dist.sample() * self.VAE.config.scaling_factor + self.VAE.config.shift_factor
        # forward_VAE_and_sample = torch.compile(forward_VAE_and_sample, backend="inductor")
        self.forward_VAE_and_sample = forward_VAE_and_sample




        # Load CLIP model (400M version) - https://huggingface.co/facebook/metaclip-l14-400m
        # Or larger version (2.5B) - https://huggingface.co/facebook/metaclip-h14-fullcc2.5b
        self.CLIP_processor = CLIPProcessor.from_pretrained("facebook/metaclip-l14-400m", cache_dir="./models/CLIP", use_fast=True)
        self.CLIP_model = CLIPModel.from_pretrained("facebook/metaclip-l14-400m", cache_dir="./models/CLIP").to(self.device)
        for param in self.CLIP_model.parameters():
            param.requires_grad = False
        self.CLIP_model = self.CLIP_model.eval().half()
        # We want to use the text projection layer as the final output which
        # also decreases the variance of the output.
        self.CLIP_text_proj = self.CLIP_model.text_projection
        # Delete vision model
        # del self.CLIP_model.vision_model
        # del self.CLIP_model.visual_projection
        # model_CLIP = torch.compile(model_CLIP, backend="inductor")
        @torch.no_grad()
        @torch.inference_mode()
        def CLIP_encode_text(text):
            text = self.CLIP_processor(text, return_tensors="pt", padding=True, truncation=True).to(device=self.device)
            return self.CLIP_text_proj(self.CLIP_model.text_model(**text).pooler_output)
        self.CLIP_encode_text = CLIP_encode_text





        # Gemma 2B - https://huggingface.co/google/gemma-2-2b
        with open(".env", "r") as f:
            token = f.read().strip()
        self.Gemma_tokenizer = GemmaTokenizerFast.from_pretrained("google/gemma-2-2b", cache_dir="./models/Gemma2b", legacy=False, token=token)
        self.Gemma_model = Gemma2ForCausalLM.from_pretrained("google/gemma-2-2b", cache_dir="./models/Gemma2b", token=token).half().eval().to(self.device)
        @torch.no_grad()
        @torch.inference_mode()
        def Gemma_encode_text(text): # Output of shape (B, 128, 2304)
            tokenized = self.Gemma_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128).to(self.device)
            return self.Gemma_model(**tokenized, output_hidden_states=True, num_logits_to_keep=1).hidden_states[-1]
        self.Gemma_encode_text = Gemma_encode_text

        torch.cuda.empty_cache()
    











    # Given text, return the pooled and bloacked embeddings
    @torch.no_grad()
    @torch.inference_mode()
    def text_to_embedding(self, text):
        # Encode text using Gemma - (B, 128, 2304)
        text_hidden = self.Gemma_encode_text(text)

        # Get pooled embedding from CLIP - (B, 768)
        text_pooled = self.CLIP_encode_text(text)

        return text_hidden, text_pooled