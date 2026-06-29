import math
import numpy as np
import torch.nn as nn
import torch
from sklearn.neighbors import BallTree

from modules import CrossFusionModule
from .extrinsic import EXTRINSIC_INDEX

def _create_uniform_pixel_coords_image(resolution: np.ndarray):
    pixel_x_coords = np.reshape(
        np.tile(np.arange(resolution[1]), [resolution[0]]),
        (resolution[0], resolution[1], 1)).astype(np.float32)
    pixel_y_coords = np.reshape(
        np.tile(np.arange(resolution[0]), [resolution[1]]),
        (resolution[1], resolution[0], 1)).astype(np.float32)
    pixel_y_coords = np.transpose(pixel_y_coords, (1, 0, 2))
    uniform_pixel_coords = np.concatenate(
        (pixel_x_coords, pixel_y_coords, np.ones_like(pixel_x_coords)), -1)
    return uniform_pixel_coords

def _transform(coords, trans):
    h, w = coords.shape[:2]
    coords = np.reshape(coords, (h * w, -1))
    coords = np.transpose(coords, (1, 0))
    transformed_coords_vector = np.matmul(trans, coords)
    transformed_coords_vector = np.transpose(
        transformed_coords_vector, (1, 0))
    return np.reshape(transformed_coords_vector,
                      (h, w, -1))

def _pixel_to_world_coords(pixel_coords, cam_proj_mat_inv):
    h, w = pixel_coords.shape[:2]
    pixel_coords = np.concatenate(
        [pixel_coords, np.ones((h, w, 1))], -1)
    world_coords = _transform(pixel_coords, cam_proj_mat_inv)
    world_coords_homo = np.concatenate(
        [world_coords, np.ones((h, w, 1))], axis=-1)
    return world_coords_homo

def pointcloud_from_depth_and_camera_params(
        depth: np.ndarray, intrinsics: np.ndarray=np.array([[282.2225057, 0.,           116.01908976],
                                                            [  0.,        282.06123454, 114.71401672],
                                                            [  0.,        0.,           1.        ]]),
        extrinsics: np.ndarray=np.eye(4)) -> np.ndarray:
    """Converts depth (in meters) to point cloud in word frame.
    :return: A numpy array of size (width, height, 3)
    """
    upc = _create_uniform_pixel_coords_image(depth.shape)
    pc = upc * np.expand_dims(depth, -1)
    C = np.expand_dims(extrinsics[:3, 3], 0).T
    R = extrinsics[:3, :3]
    R_inv = R.T  # inverse of rot matrix is transpose
    R_inv_C = np.matmul(R_inv, C)
    extrinsics = np.concatenate((R_inv, -R_inv_C), -1)
    cam_proj_mat = np.matmul(intrinsics, extrinsics)
    cam_proj_mat_homo = np.concatenate(
        [cam_proj_mat, [np.array([0, 0, 0, 1])]])
    cam_proj_mat_inv = np.linalg.inv(cam_proj_mat_homo)[0:3]
    world_coords_homo = np.expand_dims(_pixel_to_world_coords(
        pc, cam_proj_mat_inv), 0)
    world_coords = world_coords_homo[..., :-1][0]
    return world_coords



# Depth Encoder for Film
class FilmDepthEncoder(nn.Module):

    def __init__(self, input_dim=3, output_dim=1024, encoding=False):
        super(FilmDepthEncoder, self).__init__()
        self.input_dim = input_dim
        self.encoding = encoding
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 128, kernel_size=7, stride=2, padding=3, bias=True), nn.ReLU(),    # 224 -> 112
            nn.Conv2d(128, 512, kernel_size=7, stride=2, padding=3, bias=True), nn.ReLU(),  # 112 -> 56
            nn.Conv2d(512, output_dim, kernel_size=7, stride=2, padding=3, bias=True), nn.ReLU(), # 56 -> 28
            nn.AdaptiveAvgPool2d((16, 16))
        )

    def depth_preprocess(self, x): # 点云是以np的形式输入的, 存储的坐标没有归一化
        # input x: [H, W, 3]
        # output x: [3, H, W]
        x = x / 10.0
        x = torch.tensor(x, dtype=torch.float32)
        return x.permute(2, 0, 1)

    def forward(self, x):
        if self.encoding:
            x = rotary_position_encoding_3d(x, feature_dim=self.input_dim)
        x = self.encoder(x)
        return x
    

# Depth Encoder for Film
class DiTDepthEncoder(nn.Module):

    def __init__(self, input_dim=3, output_dim=4096, encoding=False):
        super(DiTDepthEncoder, self).__init__()
        self.input_dim = input_dim
        self.encoding = encoding
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=True), nn.InstanceNorm2d(64), nn.ReLU(),    # 224 -> 112
            nn.Conv2d(64, 128, kernel_size=7, stride=2, padding=3, bias=True), nn.InstanceNorm2d(128), nn.ReLU(),         # 112 -> 56
            nn.Conv2d(128, 128, kernel_size=7, stride=2, padding=3, bias=True), nn.InstanceNorm2d(128),                   # 56 -> 28
            nn.AdaptiveMaxPool2d((16, 16))
        )
        self.adapter = nn.Conv2d(128, output_dim, kernel_size=1) 

        self.ee_fusion = CrossFusionModule(embedding_dim=output_dim, num_attn_heads=8, num_layers=2)

        #### point cloud bank ####
        self.pc_bank = None
        self.pc_bank_batch_idx = None
        self.pc_embedding = nn.Parameter((3 ** -0.5) * torch.randn(2, 3))

    def depth_preprocess(self, x, **kwargs): # 中心化
        # input x: list[dict]   depth: [F, N, 3]
        # output x: list[dict]
        center = kwargs.get("center", None)
        extrinsic_id = kwargs.get("extrinsic_id", None)
        if center is None:
            center = np.zeros((1, 3))
        if extrinsic_id is not None:
            extrinsic_id = [EXTRINSIC_INDEX[int(e)] for e in extrinsic_id[0]]
        c = list(x.values())[0].shape[-1] # H, W, C 的形状， C=1 or C=3
        if c == 1:
            # from depth to pcd (1, H, W, 1) -> (H, W) -> (H, W, 3)
            x = {k: torch.tensor(pointcloud_from_depth_and_camera_params(v[0, ..., 0],
                                extrinsics=extrinsic_id[i] if extrinsic_id is not None else np.eye(4))[None] - 
                                center[:, None, None]).permute(0, 3, 1, 2) for i, (k, v) in enumerate(x.items())} 
        else:
            x = {k: torch.tensor(v - center[:, None, None]).permute(0, 3, 1, 2) for k, v in x.items()} 
        return x

    def forward(self, x:dict):
        outputs = {}
        v = list(x.values())
        x = torch.cat(v, dim=0) # [B*M, 3, H, W]
        pos = nn.functional.adaptive_avg_pool2d(x, (16, 16)).flatten(2) # [B*M, 3, 16*16]
        outputs.update({"pos": pos.reshape(-1, len(v), 3, 256).permute(1, 3, 0, 2)}) # [B, M, 3, 256] -> [M, 256, B, 3]
        BM = x.shape[0]
        B = BM // len(v) # batch size
        # 提取末端位姿的点云patch
        with torch.no_grad():
            ee_patch = (pos ** 2).sum(dim=1).argmin(dim=-1) # [B*M], 离ee最近
        if self.encoding:
            sel_pc = rotary_position_encoding_3d(sel_pc, feature_dim=self.input_dim) # [B*M, 60]
        x = self.encoder(x) # [B*M, C, 16, 16]
        x_adapt = self.adapter(x).flatten(-2) # [B*M, 4096, 256]

        # 提取末端位姿的点云patch embedding
        ee_patch_feature = ee_patch.view(BM, 1, 1).expand(-1, x_adapt.size(1), -1)   # [BM, C, 1]
        ee_feature = torch.gather(x_adapt, dim=2, index=ee_patch_feature)           # [BM, C, 1]
        
        ee_patch_pos = ee_patch.view(BM, 1, 1).expand(-1, pos.size(1), -1)   # [BM, C, 1]
        ee_pos = torch.gather(pos, dim=2, index=ee_patch_pos)                   # [BM, 3, 1]
        outputs.update({"ee_pos": ee_pos.reshape(B, -1, 3, 1).permute(1, 3, 0, 2)}) # [B, M, 3, 1] -> [M, 1, B, 3]

        # 全局特征提取, 使用ee_feature作为query
        ee_feature = self.ee_fusion(queries=ee_feature.permute(2, 0, 1), 
                                    values=x_adapt.permute(2, 0, 1), 
                                    query_pos=ee_pos.permute(2, 0, 1),
                                    value_pos=pos.permute(2, 0, 1))         # [BM, C, 1]
        outputs.update({"ee_feature": ee_feature.reshape(1, B, len(v), -1).permute(2, 0, 1, 3)}) # [M, 1, B, C]
        outputs.update({"x_adapt": x_adapt.reshape(B, len(v), -1, x_adapt.shape[-1]).permute(1, 3, 0, 2)})       # [M, 256, B, C]

        return outputs  # [B, M, 256, 4096], [B, 1, 4096]

    
### 位置编码
def rotary_position_encoding_3d(XYZ: torch.Tensor, feature_dim: int = 60) -> torch.Tensor:
    """
    3D Rotary Position Encoding for image-like 3D tensor input.

    Args:
        XYZ: Tensor of shape [B, 3, H, W], representing 3D coordinates per pixel.
        feature_dim: Total output feature dimension, must be divisible by 6.

    Returns:
        Tensor of shape [B, feature_dim * 2, H, W] — last dim contains [cos, sin] flattened.
    """
    assert feature_dim % 6 == 0, "feature_dim must be divisible by 6"

    B, C, H, W = XYZ.shape
    assert C == 3, "Input must have 3 channels for x/y/z"

    # Convert to [B, H, W, 3]
    XYZ = XYZ.permute(0, 2, 3, 1).contiguous()  # [B, H, W, 3]
    x, y, z = XYZ[..., 0:1], XYZ[..., 1:2], XYZ[..., 2:3]

    dim_each = feature_dim // 3  # per axis
    div_term = torch.exp(
        torch.arange(0, dim_each // 2, dtype=torch.float32, device=XYZ.device)
        * (-math.log(10000.0) / (dim_each // 2))
    ).view(1, 1, 1, -1)  # [1, 1, 1, d//6]

    def encode(pos):
        sin = torch.sin(pos * div_term)
        cos = torch.cos(pos * div_term)
        sin = torch.stack([sin, sin], dim=-1).view(B, H, W, -1)
        cos = torch.stack([cos, cos], dim=-1).view(B, H, W, -1)
        return sin, cos

    sinx, cosx = encode(x)
    siny, cosy = encode(y)
    sinz, cosz = encode(z)

    sin_pos = torch.cat([sinx, siny, sinz], dim=-1)  # [B, H, W, D]
    cos_pos = torch.cat([cosx, cosy, cosz], dim=-1)

    position_code = torch.stack([cos_pos, sin_pos], dim=-1)  # [B, H, W, D, 2]
    position_code = position_code[..., ::2, :].flatten(-2)  # [B, H, W, D]

    # Permute to [B, D, H, W]
    position_code = position_code.permute(0, 3, 1, 2).contiguous()

    return position_code

### 剪裁点云
def crop_point_cloud(points, center=None, radius=None, **kwargs):
    """
    Args:
        points: np.ndarray of shape [N, 3]
        center: np.ndarray of shape [3]
        radius: np.ndarray of shape [3]
    Returns:
        cropped_points: np.ndarray of shape [M, 3], where M <= N
    """
    if center is None or radius is None:
        return points

    lower = center - radius
    upper = center + radius

    mask = np.all((points >= lower) & (points <= upper), axis=1)  # shape [N]

    return mask


def fps_with_balltree(points, n_samples):
    tree = BallTree(points)
    n_points = points.shape[0]
    sampled_indices = np.zeros(n_samples, dtype=int)
    sampled_indices[0] = np.random.randint(n_points)
    min_distances = np.full(n_points, np.inf)
    
    for i in range(1, n_samples):
        # 只查询最近的一个邻居（已选点）
        dist, _ = tree.query(points, k=1, return_distance=True)
        min_distances = np.minimum(min_distances, dist[:,0])
        sampled_indices[i] = np.argmax(min_distances)
    
    return sampled_indices