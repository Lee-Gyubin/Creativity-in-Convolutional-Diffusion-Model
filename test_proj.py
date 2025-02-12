import torch

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

# ====== 테스트 ======
# 16x16 이미지에서 특정 패턴 설정
image = torch.zeros((8, 1, 16, 16))
image[:, :, 5:8, 7] = 1  # 특정 패턴 삽입

# 원하는 projection 설정 (예제)
target_proj_y = torch.zeros((1, 16))  # y축(열 기준) 전체 0
target_proj_y[:, 7] = 1  # 7번 열만 1

target_proj_x = torch.zeros((1, 16))  # x축(행 기준) 전체 0
target_proj_x[:, 5:8] = 1  # 5~7번 행만 1

#print(image)
# 실행
diff = projection_difference(image, [target_proj_y, target_proj_x])
print(diff)  # tensor([0]) → 완벽히 맞으면 0, 다르면 차이 값