import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math


def timestep_embedding(timesteps, dim=16, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding



def periodic_padding(x, pad):
    """
    Args:
        x: (batch, channels, height, width)
        pad: padding size (int)

    Returns:
        periodic conv applied tensor
    """
    if pad == 0:
        return x

    left_pad = x[:, :, :, -pad:]
    right_pad = x[:, :, :, :pad]
    x = torch.cat([left_pad, x, right_pad], dim=3)  

    top_pad = x[:, :, -pad:, :]
    bottom_pad = x[:, :, :pad, :]
    x = torch.cat([top_pad, x, bottom_pad], dim=2)  

    return x

class PeriodicConv2D(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(PeriodicConv2D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        #TODO: change stride 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)

    def forward(self, x):
        # augment the data for periodic convolution
        x_padded = periodic_padding(x, self.padding)
        
        return self.conv(x_padded)


#TODO : change the time embedding

class DiffusionModel2D(nn.Module):
    """
    - 3x3 Periodic Convolution Layer
    """
    def __init__(self, channel = 2, conv_size=3):
        super(DiffusionModel2D, self).__init__()
        self.channel = channel
        self.conv1 = PeriodicConv2D(in_channels=1, out_channels=channel, kernel_size=conv_size)
        self.bn1 = nn.BatchNorm2d(channel)  # BatchNorm 추가
        #self.conv2 = PeriodicConv2D(in_channels=1, out_channels=1, kernel_size=3)
        #self.bn2 = nn.BatchNorm2d(1)  # BatchNorm 추가
        self.time_emb = nn.Linear(1, 1)
        self.linear = nn.Linear(channel,1)

    def forward(self, x, t):
        """
        Args:
            x: (batch, 1, 16, 16)
            t: (batch, 1) - (Diffusion Step)
        """
        #t_emb = self.time_emb(t)  # (batch, 1)
        #print(t)
        t_emb = timestep_embedding(t)[:, :self.channel]
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1) # (batch, 1, 1, 1)
        #print(t_emb.shape)
        x = self.conv1(x) + t_emb # (batch, out_channels, 16, 16)
        #x = torch.sigmoid(x)
        x = self.bn1(x)
        x = torch.tanh(x)
        x = x.permute(0, 2, 3, 1)  # (batch, 16, 16, 4)
        x = self.linear(x)  # (batch, 16, 16, 1)
        x = x.permute(0, 3, 1, 2)
        x = torch.tanh(x)
        #x = self.conv2(x) + t_emb
        #x = self.bn2(x)
        #x = torch.tanh(x)
        #x = torch.sigmoid(x)

        return x

if __name__ == "__main__":

    model = DiffusionModel2D()

    test_input = torch.randn(4, 1, 16, 16)  
    test_time = torch.tensor([[0.1], [0.5], [0.8], [1.0]])  
    
    output = model(test_input, test_time)
    print("output size:", output.shape) 
    
    
    output_dir = "pre_image"
    os.makedirs(output_dir, exist_ok=True)  

    for i in range(output.shape[0]):
        img = output[i, 0].detach().cpu().numpy()  
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.title(f"Output Image {i}")

        save_path = os.path.join(output_dir, f"output_{i}.png")
        plt.savefig(save_path)
        print(f"image saved : {save_path}")