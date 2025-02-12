import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import random
import os


class BinaryImageDataset(Dataset):
    def __init__(self, size=100): 
        self.size = size
        #self.data = [2*torch.zeros((1, 16, 16))-1, 2*torch.ones((1, 16, 16))-1]
        self.data = [torch.zeros((1, 16, 16)), torch.ones((1, 16, 16))]
        
    def __len__(self):
        return self.size  
    
    def __getitem__(self, idx):
        return random.choice(self.data)  
    


if __name__ == "__main__":
    
    dataset = BinaryImageDataset(size=200)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    for batch in dataloader:
        print("Batch Shape:", batch.shape)
        output_dir = "dataset"
        os.makedirs(output_dir, exist_ok=True) 
        
        for i in range(batch.shape[0]):
            img = batch[i,0]
            #Binarization is also possible
            #img = (batch[i,0]>0.5).float()
            img = img.detach().cpu().numpy() 
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            
            ax.set_xticks(np.arange(-0.5, 16, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, 16, 1), minor=True)
            ax.grid(which="minor", color="gray", linestyle=":", linewidth=0.5)
            ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
            
            save_path = os.path.join(output_dir, f"Data_{i+1}.png")
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)
            print(f"Data saved : {save_path}")
            
        break  
    