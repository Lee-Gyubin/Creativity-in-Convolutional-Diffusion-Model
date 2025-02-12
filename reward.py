import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from perioconv2d import periodic_padding


def create_white_image(size=16):
    """
    16x16 이미지에서 위쪽 절반은 흰색(1), 아래쪽 절반은 검은색(0)인 이미지 생성.
    
    Returns:
        torch.Tensor: (16,16) 크기의 0과 1로 이루어진 이미지 텐서
    """
    image = torch.ones((size, size), dtype=torch.float32)  # 전체를 0(검정)으로 초기화
    #image[:size//2, :] = 1  # 위쪽 절반을 1(흰색)으로 변경
    return image


def create_black_image(size=16):
    """
    16x16 이미지에서 위쪽 절반은 흰색(1), 아래쪽 절반은 검은색(0)인 이미지 생성.
    
    Returns:
        torch.Tensor: (16,16) 크기의 0과 1로 이루어진 이미지 텐서
    """
    image = torch.zeros((size, size), dtype=torch.float32)  # 전체를 0(검정)으로 초기화
    #image[:size//2, :] = 1  # 위쪽 절반을 1(흰색)으로 변경
    return image


def create_checkerboard(size=16):
    """
    16x16 체커보드 패턴(흰-검 반복) 생성.
    - 흰색(1), 검은색(0) 패턴 반복
    """

    grid = torch.tensor([[((i + j) % 2) for j in range(size)] for i in range(size)], dtype=torch.float32)
    return grid


def create_half_white_black_image(size=16):
    """
    16x16 이미지에서 위쪽 절반은 흰색(1), 아래쪽 절반은 검은색(0)인 이미지 생성.
    
    Returns:
        torch.Tensor: (16,16) 크기의 0과 1로 이루어진 이미지 텐서
    """
    image = torch.zeros((size, size), dtype=torch.float32)  # 전체를 0(검정)으로 초기화
    image[:size//2, :] = 1  # 위쪽 절반을 1(흰색)으로 변경
    return image

def create_half_black_white_image(size=16):
    """
    16x16 이미지에서 위쪽 절반은 흰색(1), 아래쪽 절반은 검은색(0)인 이미지 생성.
    
    Returns:
        torch.Tensor: (16,16) 크기의 0과 1로 이루어진 이미지 텐서
    """
    image = torch.ones((size, size), dtype=torch.float32)  # 전체를 0(검정)으로 초기화
    image[:size//2, :] = 0  # 위쪽 절반을 1(흰색)으로 변경
    return image

def create_quadrant_image(size=16):
    """
    1, 4사분면은 1, 2, 3사분면은 0인 16x16 이미지 생성
    
    Returns:
        torch.Tensor: (16,16) 크기의 0과 1로 이루어진 이미지
    """
    image = torch.zeros((size, size), dtype=torch.float32)  # 전체 0(검정)으로 초기화
    
    half_size = size // 2  # 8
    image[:half_size, half_size:] = 1  # 1사분면 (왼쪽 위)
    image[half_size:, :half_size] = 1  # 4사분면 (오른쪽 아래)

    return image


def has_single_3x3_block_periodic(images, reward_img):
    """
    Check if there is exactly one 3x3 block in a periodic setting.
    Args:
        images: (batch, 1, 16, 16) binary tensor
    Returns:
        (batch,) tensor with 1 if exactly one 3x3 block exists, otherwise 0
    """
    # Apply periodic padding with size 1
    padded_images = periodic_padding(images, pad=1)

    # 3x3 filter (all ones)
    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32)

    # Convolution operation
    conv_output = F.conv2d(padded_images.float(), kernel, stride=1, padding=0)  # (batch, 1, 16, 16)

    # Find where the sum is 9 (fully filled 3x3 block)
    mask = (conv_output == 9).float()  # (batch, 1, 16, 16)

    # Count occurrences of 3x3 blocks per image
    count_per_image = mask.view(mask.shape[0], -1).sum(dim=1)  # (batch,)

    # Return 1 if there is exactly one block, else 0
    return (count_per_image == 1).float()*100



def compute_l1_distance(image, grid):
    """
    16x16 이미지와 체커보드 패턴의 L1 거리 계산.
    
    Args:
        image (torch.Tensor): (num_particles, 1, 16,16) 크기의 0 또는 1 이진 이미지
        grid (torch.Tensor): 기준 체커보드 패턴
    
    Returns:
        float: 이미지와 체커보드 간의 L1 거리
    """
    image = (image > 0.5).float()
    l1_distance = torch.abs(image - grid)#.sum(dim = 0).item()  # 절대 차이 합산
    l1_distance = l1_distance.squeeze()
    #print(l1_distance.shape)
    
    return -l1_distance.sum(dim=[1,2]) 


def projection_difference(image, target=None):
    """
    2D 이미지(16x16)의 투영값을 계산하고, 원하는 투영값과 차이를 반환하는 함수.
    
    Args:
        image (torch.Tensor): (batch, 1, 16, 16) 크기의 binary tensor
        target_proj_y (torch.Tensor or None): (batch, 16) 크기의 y축 기준 목표 투영값
        target_proj_x (torch.Tensor or None): (batch, 16) 크기의 x축 기준 목표 투영값
        
    Returns:
        torch.Tensor: (batch,) 크기의 총 차이 값
    """
    batch_size = image.shape[0]

    # y축(dim=0) 기준 projection (각 열에 1이 하나라도 있으면 1, 없으면 0)
    proj_y = (image.sum(dim=2) > 0).float()  # (batch, 16)
    #print(proj_y.shape)

    # x축(dim=1) 기준 projection (각 행에 1이 하나라도 있으면 1, 없으면 0)
    proj_x = (image.sum(dim=3) > 0).float()  # (batch, 16)

    #print(proj_x.shape)
    #print(target_proj_x.shape)
    # 원하는 projection 값과 비교
    diff_y = torch.abs(proj_y - target[0].unsqueeze(0)) if target is not None else torch.zeros_like(proj_y)
    diff_x = torch.abs(proj_x - target[1].unsqueeze(0)) if target is not None else torch.zeros_like(proj_x)

    #print(diff_x.shape)
    # 총 차이 계산
    total_diff = diff_y.sum(dim=-1) + diff_x.sum(dim=-1)  # (batch,)
    #print(total_diff.shape)
    return total_diff.squeeze()



if __name__ == '__main__':
    
    # 체커보드 패턴 생성
    checkerboard = create_checkerboard().unsqueeze(0).unsqueeze(0)

    # 랜덤 0,1 이진 이미지 생성
    test_image = torch.randint(0, 2, (8, 1, 16, 16), dtype=torch.float32)

    # L1 거리 계산
    l1_dist = compute_l1_distance(test_image, checkerboard)

    print("L1 Distance:", l1_dist)
    print(l1_dist.shape)
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(checkerboard.squeeze(0).squeeze(0), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Checkerboard Pattern")
    axes[0].axis("off")

    axes[1].imshow(test_image[0].squeeze(0), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Test Image\nL1 Distance: {l1_dist[0]:.1f}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()