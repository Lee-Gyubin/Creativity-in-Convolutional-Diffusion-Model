import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math


# **Sinusoidal Timestep Embedding**
def timestep_embedding(timesteps, dim=16, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
    freqs = freqs.to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# **Residual UNet 기반 Diffusion Model**
class ResidualUNetDiffusionModel(nn.Module):
    def __init__(self):
        super(ResidualUNetDiffusionModel, self).__init__()

        # **Encoder (Downsampling)**
        self.enc1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1)  # 16x16 → 8x8
        self.bn1 = nn.BatchNorm2d(16)
        self.enc2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)  # 8x8 → 4x4
        self.bn2 = nn.BatchNorm2d(32)

        # **Decoder (Upsampling)**
        self.up1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1)  # 4x4 → 8x8
        self.bn3 = nn.BatchNorm2d(16)
        self.up2 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1)  # 8x8 → 16x16

        # **Time Embedding**
        self.time_emb = nn.Linear(1, 16)

    def forward(self, x, t):
        # Time Embedding (Sinusoidal)
        t_emb = timestep_embedding(t)[:, :1]
        #print(x.shape)
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)  # (batch, 1, 1, 1)
        
        x = x + t_emb
        # **Encoder**
        x1 = self.enc1(x)  # (batch, 16, 8, 8)
        x1 = self.bn1(x1)
        x1 = F.relu(x1) + t_emb
        
        x2 = self.enc2(x1)  # (batch, 32, 4, 4)
        x2 = self.bn2(x2)
        x2 = F.relu(x2) + t_emb

        # **Decoder**
        x3 = self.up1(x2) + x1 + t_emb # (batch, 16, 8, 8)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)

        x4 = self.up2(x3) + x + t_emb  # (batch, 1, 16, 16)
        
        # **Residual Connection (Skip Connection)**
        #x4 = x4 + x  # Residual connection with original input

        x4 = torch.tanh(x4)  # Activation

        return x4


# **Test Model**
if __name__ == "__main__":
    model = ResidualUNetDiffusionModel()
    test_input = torch.randn(4, 1, 16, 16)
    test_time = torch.tensor([[0.1], [0.5], [0.8], [1.0]])

    output = model(test_input, test_time)
    print("Output Size:", output.shape)

    output_dir = "pre_image"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(output.shape[0]):
        img = output[i, 0].detach().cpu().numpy()
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.title(f"Output Image {i}")

        save_path = os.path.join(output_dir, f"output_{i}.png")
        plt.savefig(save_path)
        print(f"Image Saved: {save_path}")