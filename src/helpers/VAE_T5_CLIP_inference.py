from transformers import CLIPProcessor, CLIPModel
import open_clip
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM







# Class for VAE + CLIP + T5
class VAE_T5_CLIP_inference:
    def __init__(self, device):
        self.device = device

        # Load in the VAE
        self.VAE = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema", cache_dir="./pretrained_models/VAE", device=self.device).eval()
        self.VAE_downsample = 8
        # Freeze all VAE parameters
        for param in self.VAE.parameters():
            param.requires_grad = False
        # Store locally to prevent issues with DDP
        self.VAE = self.VAE.eval().to(dtype=torch.float16, device=self.device)
        # Passes image data through the VAE and then samples from the latent distribution
        @torch.no_grad()
        @torch.inference_mode()
        @torch.compile
        def forward_VAE_and_sample(x):
            # 1. Encode
            # 2. Sample from the latent distribution
            # 3. Normalize the latent representation
            return self.VAE.encode(x).latent_dist.sample() * self.VAE.config.scaling_factor
        self.forward_VAE_and_sample = forward_VAE_and_sample




        # CLIP L/4 - https://huggingface.co/openai/clip-vit-large-patch14
        CLIPL4 = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir="./models/CLIP")
        self.CLIPL4 = CLIPL4.text_model
        # self.CLIPL4_proj = CLIPL4.text_projection
        del CLIPL4
        self.CLIPL4_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir="./models/CLIP")
        for param in self.CLIPL4.parameters():
            param.requires_grad = False
        self.CLIPL4 = self.CLIPL4.eval().half().to(self.device)
        @torch.no_grad()
        @torch.inference_mode()
        @torch.compile
        def model_CLIPL4(text):
            return self.CLIPL4(**text)
        def CLIPL4_encode_text(text):
            text = self.CLIPL4_processor(text, return_tensors="pt", padding="max_length", truncation=False)
            return model_CLIPL4(text)
        # Main function used to encode text using CLIP L/4
        self.CLIPL4_encode_text = CLIPL4_encode_text
        # self.CLIPL4_proj = torch.compile(self.CLIPL4_proj).eval().half().to(self.device)

        # CLIP G/14 - https://huggingface.co/laion/CLIP-ViT-g-14-laion2B-s34B-b88K
        # model, _, _ = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-g-14-laion2B-s34B-b88K', precision="fp16", device="cpu", cache_dir="./models/CLIP")
        # self.CLIPG14_tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-g-14-laion2B-s34B-b88K')
        # CLIP G/14 - https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k
        model, _, _ = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k', precision="fp16", device="cpu", cache_dir="./models/CLIP")
        self.CLIPG14_tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')
        self.CLIPG14_token_embedding = model.token_embedding.to(dtype=torch.float16, device=self.device)
        self.CLIPG14_positional_embedding = model.positional_embedding.to(dtype=torch.float16, device=self.device)
        self.CLIPG14_transformer = model.transformer.to(device=self.device)
        self.CLIPG14_ln_final = model.ln_final.to(device=self.device)
        self.CLIPG14_text_projection = model.text_projection.to(dtype=torch.float16, device=self.device)
        self.CLIPG14_attn_mask = model.attn_mask.to(dtype=torch.float16, device=self.device)
        del model
        @torch.no_grad()
        @torch.inference_mode()
        @torch.compile
        def model_CLIPG14(text, cast_dtype):
            x = self.CLIPG14_token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
            x = x + self.CLIPG14_positional_embedding.to(cast_dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.CLIPG14_transformer(x, attn_mask=self.CLIPG14_attn_mask)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.CLIPG14_ln_final(x)  # [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x_pooled = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.CLIPG14_text_projection
            return x, x_pooled
        def CLIPG14_encode_text(text):
            text = self.CLIPG14_tokenizer(text).to(self.device)
            cast_dtype = self.CLIPG14_transformer.get_cast_dtype()
            return model_CLIPG14(text, cast_dtype)
        # Main function used to encode text using CLIP G/14
        self.CLIPG14_encode_text = CLIPG14_encode_text

        # T5 XXL - https://huggingface.co/google/t5-v1_1-xxl
        # NOTE: Size is limited to 77 tokens, although it was trained on 512
        self.T5_tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-xxl", cache_dir="./models/T5", legacy=False)
        self.T5_model = torch.compile(AutoModelForSeq2SeqLM.from_pretrained("google/t5-v1_1-xxl", cache_dir="./models/T5").encoder.to(torch.float16)).eval().to(self.device)
        @torch.no_grad()
        @torch.inference_mode()
        def T5_encode_text(text):
            return self.T5_model(**(self.T5_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=77).to(self.device))).last_hidden_state
        self.T5_encode_text = T5_encode_text

        torch.cuda.empty_cache()
    











    # Given text, return the pooled and bloacked embeddings
    @torch.no_grad()
    @torch.inference_mode()
    def text_to_embedding(self, text):
        # Encode text using T5
        T5_output = self.T5_encode_text(text)

        # Tokenize the class strings
        batch_class_CLIPL4 = self.CLIPL4_processor(text, return_tensors="pt", padding="max_length", truncation=True).to(device=self.device)
        # Encode text using CLIP L/4
        CLIPL4_output = self.CLIPL4(**batch_class_CLIPL4)
        CLIPL4_hidden = CLIPL4_output.last_hidden_state
        CLIPL4_pooled = CLIPL4_output.pooler_output

        # Encode text using CLIP G/14
        CLIPG14_output, CLIPG14_pooled = self.CLIPG14_encode_text(text)

        # Create large tensor for parallel text stream (B, 154, 4096)
        # CLIPL4_hidden - [77, 768]
        # CLIPG14_output - [77, 1280]
        # T5_output - [77, 4096]
        # -------------------------------------------------------------
        # |  [CLIPL4_hidden - (77, 768)]  | [T5_output - (77, 4096)]  |
        # | [CLIPG14_output - (77, 1280)] |             ...           |
        # |    [zeros - (77, 2048)]       |             ...           |
        # -------------------------------------------------------------
        text_hidden = torch.cat([
            torch.cat([
                CLIPL4_hidden, 
                CLIPG14_output, 
                torch.zeros(CLIPL4_hidden.shape[0], CLIPL4_hidden.shape[1], 2048, dtype=T5_output.dtype, device=T5_output.device)], 
            dim=2), 
            T5_output],
            dim=1
        )
        # Create small conditioning vector (B, 2048)
        # CLIPL4_pooled - [768]
        # CLIPG14_pooled - [1280]
        # -------------------------------------------------------------
        # |  [CLIPL4_pooled - (768)]  | [CLIPG14_pooled - (1280)]  |
        # -------------------------------------------------------------
        text_pooled = torch.cat([CLIPL4_pooled, CLIPG14_pooled], dim=1)

        return text_hidden, text_pooled