import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import BinaryImageDataset
from perioconv2d import DiffusionModel2D
import numpy as np



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

################################################
batch_size = 100
epochs = 3000
learning_rate = 0.0001
###############################################
n_steps = 100  # Diffusion steps
conv_size = 3
channel_size = 4
num_layer = 1
######################################## for file
model_save_name = f"step{n_steps}_perioconv{conv_size}_layer{num_layer}_channel{channel_size}.pth"
loss_file_name = f"loss_step{n_steps}_perioconv{conv_size}_layer{num_layer}_channel{channel_size}"

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

#DataLoad
dataset = BinaryImageDataset(size=500)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#Model part
model = DiffusionModel2D(channel_size, conv_size).to(device)
criterion = nn.MSELoss()  # DDPM은 원본 노이즈와 예측 노이즈 간의 차이를 학습
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Loss list
loss_history = []


print("Training Started...")

for epoch in range(epochs):
    total_loss = 0.0
    for images in dataloader:
        images = images.to(device)
        t = torch.randint(0, n_steps, size=(images.shape[0],), device=device)#, dtype=torch.float32)  # 랜덤한 t 선택
        #print(f't is : {t}')
        # Noisy image generate
        epsilon = torch.randn_like(images)
        
        # predict the noise added from $x_{t-1}$ to $x_t$
        x_t = (
            extract(sqrt_alphas_cumprod, t, images.shape) * images +
            extract(sqrt_one_minus_alphas_cumprod, t, images.shape) * epsilon
        )
        
        #TODO : think more about this part..
        #x_t = torch.clip(x_t, -2.0, 2.0)
        
        x_0_pred = model(x_t, t)
        
        # 손실 계산 (MSE: 예측한 노이즈 vs 실제 노이즈)
        loss = criterion(x_0_pred, images)#, reduction="none")

        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    # average Loss save
    avg_loss = total_loss / len(dataloader)
    loss_history.append(avg_loss)
    #print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.6f}")
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch}], Loss: {avg_loss:.6f}")
    
# save model
torch.save(model.state_dict(), model_save_name)
print(f"Training Complete! Model saved : {model_save_name}")

# final loss plot save
plt.figure(figsize=(8, 5))
plt.plot(loss_history, label="Training Loss", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("DDPM Training Loss")
plt.legend()
plt.grid()

os.makedirs("training_plots", exist_ok = True)
plt.savefig(f"training_plots/{loss_file_name}.png")
plt.close()
print(f"Final loss plot saved at training_plots/{loss_file_name}.png")