import torch
from diffusers.loaders import AttnProcsLayers

from modules.BEATs.BEATs import BEATs, BEATsConfig
from modules.AudioToken.embedder import FGAEmbedder
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import LoRAAttnProcessor


class AudioTokenWrapper(torch.nn.Module):
    """Simple wrapper module for Stable Diffusion that holds all the models together"""

    def __init__(
        self,
        args,
        accelerator,
    ):

        super().__init__()
        # Load scheduler and models
        from modules.clip_text_model.modeling_clip import CLIPTextModel
        self.text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
        )
        self.vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
        )

        checkpoint = torch.load(
            'models/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt')
        cfg = BEATsConfig(checkpoint['cfg'])
        self.aud_encoder = BEATs(cfg)
        self.aud_encoder.load_state_dict(checkpoint['model'])
        self.aud_encoder.predictor = None
        input_size = 768 * 3

        if args.pretrained_model_name_or_path == "CompVis/stable-diffusion-v1-4":
            self.embedder = FGAEmbedder(input_size=input_size, output_size=768)

        else:
            self.embedder = FGAEmbedder(input_size=input_size, output_size=1024)

        self.vae.eval()
        self.unet.eval()
        self.text_encoder.eval()
        self.aud_encoder.eval()

        if 'lora' in args and args.lora:
            # Set correct lora layers
            lora_attn_procs = {}
            for name in self.unet.attn_processors.keys():
                cross_attention_dim = None if name.endswith(
                    "attn1.processor") else self.unet.config.cross_attention_dim
                if name.startswith("mid_block"):
                    hidden_size = self.unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = self.unet.config.block_out_channels[block_id]

                lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size,
                                                          cross_attention_dim=cross_attention_dim)

            self.unet.set_attn_processor(lora_attn_procs)
            self.lora_layers = AttnProcsLayers(self.unet.attn_processors)

        if args.data_set == 'train':

            # Freeze vae, unet, text_enc and aud_encoder
            self.vae.requires_grad_(False)
            self.unet.requires_grad_(False)
            self.text_encoder.requires_grad_(False)
            self.aud_encoder.requires_grad_(False)
            self.embedder.requires_grad_(True)
            self.embedder.train()

            if 'lora' in args and args.lora:
                self.unet.train()

        if args.data_set == 'test':

            from transformers import CLIPTextModel
            self.text_encoder = CLIPTextModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
            )

            self.embedder.eval()
            embedder_learned_embeds = args.learned_embeds
            self.embedder.load_state_dict(torch.load(embedder_learned_embeds, map_location=accelerator.device))

            if 'lora' in args and args.lora:
                self.lora_layers.eval()
                lora_layers_learned_embeds = args.learned_embeds_lora
                self.lora_layers.load_state_dict(torch.load(lora_layers_learned_embeds, map_location=accelerator.device))
                self.unet.load_attn_procs(lora_layers_learned_embeds)
