import torch
import torch.nn.functional as F
from perioconv2d import periodic_padding

'''
def periodic_padding(x, pad):
    """
    Periodic padding (wrap-around padding)
    Args:
        x: (batch, channels, height, width)
        pad: padding size (int)
    Returns:
        Tensor with periodic padding applied
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
'''

def has_single_3x3_block_periodic(images):
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
    return (count_per_image == 1).float()

# ====== Test ======
# 16x16 이미지에서 periodic condition에서 3x3 블록 테스트
if __name__=='__main__':
    image = torch.zeros((1, 1, 16, 16))
    ## case 1 
    image[:, :, 14:17, 14:17] = 1  # 우측 하단에 3x3 블록 생성 (주기적 조건에 의해 0,0으로 wrap-around됨)
    image[:, :, :1, 14:17] = 1  # 우측 하단에 3x3 블록 생성 (주기적 조건에 의해 0,0으로 wrap-around됨)
    image[:, :, :1, :1] = 1
    image[:, :,14:17, :1] = 1
    print(image)

    ###case 2
    image = torch.zeros((1, 1, 16, 16))
    image[:, :, 8:11, 14:17] = 1  # 우측 하단에 3x3 블록 생성 (주기적 조건에 의해 0,0으로 wrap-around됨)
    image[:, :, 8:11, :1] = 1  # 우측 하단에 3x3 블록 생성 (주기적 조건에 의해 0,0으로 wrap-around됨)
    #image[:, :, :1, :1] = 1
    #image[:, :,14:17, :1] = 1
    print(image)

    # 실행
    result = has_single_3x3_block_periodic(image)
    print(result)  # tensor([1]) → 하나만 있으면 1, 없거나 여러 개면 0