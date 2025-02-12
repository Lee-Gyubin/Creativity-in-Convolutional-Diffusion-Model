import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from perioconv2d import DiffusionModel2D
from local_consistency import compute_dominant_pixel_ratio



####################### setting
image_size = (1, 16, 16) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_steps = 100  # Diffusion steps(4 or 8 steps are possible)(kernel size should be odd number)
conv_size = 3
num_layer = 1
channel_size = 4
################################ for consistency check(kernel size should be odd number)
test_size = 3
######################################## for file
model_save_name = f"step{n_steps}_perioconv{conv_size}_layer{num_layer}_channel{channel_size}.pth"

output_dir = f"step{n_steps}_perioconv{conv_size}_layer{num_layer}_channel{channel_size}"
os.makedirs(output_dir, exist_ok=True)
output_dir_seq = f"denoising_sequence_step{n_steps}_perioconv{conv_size}_layer{num_layer}_channel{channel_size}"
os.makedirs(output_dir_seq, exist_ok=True)


####################### model part
model = DiffusionModel2D(channel_size, conv_size).to(device)
model.load_state_dict(torch.load(model_save_name))
model.eval()

####################### scheduling part

def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    
    return torch.tensor(betas_clipped, dtype=dtype)

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#beta_start = 0.0001
#beta_end = 0.02
#beta = torch.linspace(beta_start, beta_end, n_steps).to(device)  # 선형 스케줄
betas = cosine_beta_schedule(n_steps)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
sqrt_alphas_cumprod =  torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod=torch.sqrt(1. - alphas_cumprod)

log_one_minus_alphas_cumprod=torch.log(1. - alphas_cumprod)
sqrt_recip_alphas_cumprod=torch.sqrt(1. / alphas_cumprod)
sqrt_recipm1_alphas_cumprod=torch.sqrt(1. / alphas_cumprod - 1)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
posterior_mean_coef2 = (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)
posterior_log_variance_clipped= torch.log(torch.clamp(posterior_variance, min=1e-20))


##########################################
num_samples = 10 
num_batch = 100

local_consist = []
pre_consist = []

for i in range(num_samples):
    x_t = torch.randn(num_batch, *image_size).to(device)
    original_noise = x_t
    
    #TODO refine some code for sequence figure
    #x_t_s = [x_t]
    x_t_s = []
    with tqdm(reversed(range(0, n_steps)), colour="#6565b5", total=n_steps) as sampling_steps:
        for time_step in sampling_steps:
            #x_t = sample_one_step(model, x_t, time_step)
            t = torch.full((x_t.shape[0],), time_step, device=x_t.device, dtype=torch.long)
            
            b, *_, device = *x_t.shape, x_t.device
            
            x_0_pred = model(x_t, t) #이게 noise 값
            #TODO: clamp the boundary
            x_0_pred.clamp_(-1., 1.)
            
            model_mean = (
                extract(posterior_mean_coef1, t, x_t.shape) * x_0_pred +
                extract(posterior_mean_coef2, t, x_t.shape) * x_t
            )
            #print(t) #-> ([1])
            #print(posterior_variance)# ([8])
            #posterior_variance = extract(posterior_variance, t, x_t.shape)
            model_log_variance = extract(posterior_log_variance_clipped, t, x_t.shape)

            noise = torch.randn_like(x_t)
            # no noise when t == 0
            nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x_t.shape) - 1)))
            x_t = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
            
            sampling_steps.set_postfix(ordered_dict={"step": time_step + 1, "sample": len(x_t_s)})

            x_t_s.append(x_t[0].unsqueeze(0))
        ########## for statistic
        local_consist.append((x_t > 0.5).float())
        pre_consist.append((original_noise > 0.5).float())
        
        ########################################## saving part
        # sampling steps
        col = n_steps
    
        selected_steps = np.linspace(0, len(x_t_s) - 1, col, dtype=int)
        ############################
    
        fig, axes = plt.subplots(1, col, figsize=(2*col, 2))  
        for s, step in enumerate(selected_steps):
            img = x_t_s[step].detach().cpu().numpy()  # (1, 1, 16, 16)
            img = img.squeeze(0).squeeze(0)  # (16, 16)
        
            axes[s].imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            axes[s].set_title(f"Time step t : {n_steps - step}")
            axes[s].axis("off")

        fig_path = os.path.join(output_dir_seq, f"sample_{i + 1}_denoising_sequence.png")
        plt.savefig(fig_path, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        print(f"\n Denoising sequence saved: {fig_path}")
    
        ########### Binarized final image(optional)   
        #print(x_t) 
        x_t = x_t[0].unsqueeze(0)
        
        x_t = (x_t > 0.5).float()  # 0.5 criterior
        final_image = x_t.detach().cpu().squeeze(0).squeeze(0).numpy()
        
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(final_image, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(-0.5, 16, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 16, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle=":", linewidth=0.5)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

        save_path = os.path.join(output_dir, f"Binarized_generated_{i + 1}.png")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        print(f"\n Generated Image Saved: {save_path}")

local_consist = torch.cat(local_consist, dim = 0)
pre_consist = torch.cat(pre_consist, dim = 0)
#print(local_consist.shape)

local_ratio, local_std = compute_dominant_pixel_ratio(local_consist, test_size)
pre_ratio , pre_std = compute_dominant_pixel_ratio(pre_consist, test_size)
#print(pre_ratio)s
print("Image Generation Complete!")
print(f"local consistency ratio for {num_samples * num_batch} samples is initial noise : {pre_ratio.mean():.4f} ± {pre_std:.4f}, , generated : {local_ratio.mean():.4f} ± {local_std:.4f}")