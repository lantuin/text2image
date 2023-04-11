import torch
from torch import autocast
import numpy as np

from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL
from diffusers import LMSDiscreteScheduler
from diffusers import UNet2DConditionModel
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from tqdm import tqdm
from PIL import Image

class ImageDiffusionModel:

     def __init__(self, vae, tokenizer, text_encoder, unet,
                scheduler_LMS, scheduler_DDIM):
        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.unet = unet
        self.scheduler_LMS = scheduler_LMS
        self.scheduler_DDIM  = scheduler_DDIM
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

     def get_text_embeds(self, text):
         # tokenize the text
         text_input = self.tokenizer(text,
                                     padding='max_length',
                                     max_length=tokenizer.model_max_length,
                                     truncation=True,
                                     return_tensors='pt')
         # embed the text
         with torch.no_grad():
             text_embeds = self.text_encoder(text_input.input_ids.to(self.device))[0]
         return text_embeds

     def get_prompt_embeds(self, prompt):
         # get conditional prompt embeddings
         cond_embeds = self.get_text_embeds(prompt)
         # get unconditional prompt embeddings
         uncond_embeds = self.get_text_embeds([''] * len(prompt))
         # concatenate the above 2 embeds
         prompt_embeds = torch.cat([uncond_embeds, cond_embeds])
         return prompt_embeds

     def get_img_latents(self,
                         text_embeds,
                         height=512, width=512, 
                         num_inference_steps=50, 
                         guidance_scale=7.5, 
                         img_latents=None):
         # if no image latent is passed, start reverse diffusion with random noise
         if img_latents is None:
             img_latents = torch.randn((text_embeds.shape[0] // 2, self.unet.in_channels,\
                                        height // 8, width // 8)).to(self.device)
         # set the number of inference steps for the scheduler
         self.scheduler_LMS.set_timesteps(num_inference_steps)
         # scale the latent embeds
         img_latents = img_latents * self.scheduler_LMS.sigmas[0]
         # use autocast for automatic mixed precision (AMP) inference
         with autocast('cuda'):
             for i, t in tqdm(enumerate(self.scheduler_LMS.timesteps)):
                 # do a single forward pass for both the conditional and unconditional latents
                 latent_model_input = torch.cat([img_latents] * 2)
                 sigma = self.scheduler_LMS.sigmas[i]
                 latent_model_input = latent_model_input / ((sigma ** 2 + 1) ** 0.5)
                 # predict noise residuals
                 with torch.no_grad():
                     noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds)['sample']
                 # separate predictions for unconditional and conditional outputs
                 noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                 # perform guidance
                 noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                 # remove the noise from the current sample i.e. go from x_t to x_{t-1}
                 #img_latents = self.scheduler_LMS.step(noise_pred, i, img_latents)['prev_sample']
                 img_latents = self.scheduler_LMS.step(noise_pred, t, img_latents)['prev_sample']
         return img_latents

     def decode_transform_img_latents(self, img_latents):
         img_latents = 1 / 0.18215 * img_latents
         
         with torch.no_grad():
           imgs = self.vae.decode(img_latents)

         imgs = (imgs.sample / 2 + 0.5).clamp(0, 1)
         imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
         imgs = (imgs * 255).round().astype('uint8')
         imgs = [Image.fromarray(image) for image in imgs]
         return imgs

     def transform_imgs(self, imgs):
         # transform images from the range [-1, 1] to [0, 1]
         imgs = (imgs / 2 + 0.5).clamp(0, 1)
         # permute the channels and convert to numpy arrays
         imgs = imgs.permute(0, 2, 3, 1).numpy()
         # scale images to the range [0, 255] and convert to int
         imgs = (imgs * 255).round().astype('uint8')        
         # convert to PIL Image objects
         imgs = [Image.fromarray(img) for img in imgs]
         return imgs

     def prompt_to_img(self, 
                       prompts, 
                       height=512, width=512, 
                       num_inference_steps=50, 
                       guidance_scale=7.5, 
                       img_latents=None):
         # convert prompt to a list
         if isinstance(prompts, str):
             prompts = [prompts]
         # get prompt embeddings
         text_embeds = self.get_prompt_embeds(prompts)
         # get image embeddings
         img_latents = self.get_img_latents(text_embeds,
                                       height, width,
                                       num_inference_steps,
                                       guidance_scale, 
                                       img_latents)
         # decode the image embeddings
         imgs = self.decode_transform_img_latents(img_latents)
         # convert decoded image to suitable PIL Image format
         # imgs = self.transform_imgs(imgs)
         return imgs

device = 'cuda'
# Load autoencoder
vae = AutoencoderKL.from_pretrained('CompVis/stable-diffusion-v1-4', 
                                    subfolder='vae').to(device)
# Load tokenizer and the text encoder
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14').to(device)
# Load UNet model
unet = UNet2DConditionModel.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='unet').to(device)
# Load schedulers
scheduler_LMS = LMSDiscreteScheduler(beta_start=0.00085, 
                                 beta_end=0.012, 
                                 beta_schedule='scaled_linear', 
                                 num_train_timesteps=1000)

scheduler_DDIM = DDIMScheduler(beta_start=0.00085, 
                               beta_end=0.012, 
                               beta_schedule='scaled_linear', 
                               num_train_timesteps=1000)

model = ImageDiffusionModel(vae, tokenizer, text_encoder, unet, scheduler_LMS, scheduler_DDIM)

prompts = ["A really giant cute pink barbie doll on the top of Burj Khalifa", 
           "A green, scary aesthetic dragon breathing fire near a group of heroic firefighters"]



imgs = model.prompt_to_img(prompts)

#imgs[0].save(f'result_0.jpg')
#imgs[1].save(f'result_1.jpg')

var = 0

while var < len(prompts):
    imgs[var].save(f'result_{var}.jpg')
    #print(imgs[var])
    var += 1
