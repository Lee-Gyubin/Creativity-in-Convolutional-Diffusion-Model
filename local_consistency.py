import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from perioconv2d import periodic_padding


def compute_dominant_pixel_ratio(binary_image, kernel = 3):
    """
    Args:
        binary_image (torch.Tensor): (batch, 1, 16, 16)
        
    Returns:
        ratio: (batch, 1) 
    """
    pad = kernel // 2 
    # 이미지를 (1, 1, 16, 16) 형태로 변환 (Batch, Channel, Height, Width)
    image = binary_image#.unsqueeze(0).unsqueeze(0)
    binary_image = periodic_padding(binary_image, pad)
    
    # 3×3 커널을 사용하여 주변 픽셀 합 계산
    recep = torch.ones((1, 1, kernel, kernel), dtype=torch.float32)
    neighbor_sum = F.conv2d(binary_image, recep, padding=0)
    
    # 3×3 내에서 지배적인 값 결정 (4.5보다 크면 1, 아니면 0)
    dominant_color = (neighbor_sum >= (kernel**2 / 2)).float()
    
    # 현재 픽셀과 지배적인 값이 같은 경우 (1) / 다르면 (0)
    agreement = (image == dominant_color).float()
    
    # 일치하는 픽셀 비율 계산
    match_ratio = agreement.mean([2,3])#.mean()
    match_std = match_ratio.std()

    return match_ratio, match_std


if __name__ == '__main__':
    # 예제: 샘플 이진화된 이미지 (랜덤 0,1 값)
    binary_image = torch.randint(0, 2, (10, 1, 16, 16))
    binary_image = binary_image.float()

    # 비율 계산
    ratio, std = compute_dominant_pixel_ratio(binary_image, kernel = 3)
    mean_ratio = ratio.mean().item()
    #print(ratio.shape)
    #print(ratio.shape)
    #print(f"mean local consistent ratio: {mean_ratio:.4f}")
    print(f"mean local consistency ratio : {mean_ratio:.4f} ± {std:.4f}")