import numpy as np
import torch
import math
import random
import collections
from scipy.linalg import expm, norm

def rotate_angle_vector(theta, v):
    '''
        theta: B 1
        v:  B 3
    '''
    cos_a = torch.cos(theta)
    sin_a = torch.sin(theta)
    x, y, z = v[:, 0:1], v[:, 1:2], v[:, 2:3]
    
    R = torch.stack([
        torch.cat([cos_a+(1-cos_a)*x*x, (1-cos_a)*x*y-sin_a*z, (1-cos_a)*x*z+sin_a*y], dim=-1) , # [b1 b1 b1]
        torch.cat([(1-cos_a)*y*x+sin_a*z, cos_a+(1-cos_a)*y*y, (1-cos_a)*y*z-sin_a*x], dim=-1) ,
        torch.cat([(1-cos_a)*z*x-sin_a*y, (1-cos_a)*z*y+sin_a*x, cos_a+(1-cos_a)*z*z], dim=-1) 
    ], dim = 1)

    return R

def rotate_theta_phi(angles):
    '''
        angles: B, 2
    '''
    assert len(angles.shape) == 2
    B = angles.size(0)
    theta, phi = angles[:, 0:1], angles[:, 1:2]

    v1 = torch.Tensor([[0, 0, 1]]).expand(B, -1) # B 3
    v2 = torch.cat([torch.sin(theta) , -torch.cos(theta), torch.zeros_like(theta)], dim=-1) # B 3

    R1_inv = rotate_angle_vector(-theta, v1)
    R2_inv = rotate_angle_vector(-phi, v2)
    R_inv = R1_inv @ R2_inv

    return R_inv

def rotate_point_clouds(pc, rotation_matrix, use_normals=False):
    '''
        Input: 
            pc  B N 3
            R   3 3
        Output:
            B N 3
    '''
    if not use_normals:
        new_pc = torch.einsum('bnc, dc -> bnd', pc, rotation_matrix.float().to(pc.device))
    else:
        new_pc = torch.einsum('bnc, dc -> bnd', pc[:, :, :3], rotation_matrix.float().to(pc.device))
        new_normal = torch.einsum('bnc, dc -> bnd', pc[:, :, 3:], rotation_matrix.float().to(pc.device))
        new_pc = torch.cat([new_pc, new_normal], dim=-1)
    return new_pc

def rotate_point_clouds_batch(pc, rotation_matrix, use_normals=False):
    '''
        Input: 
            pc  B N 3
            R   B 3 3
        Output:
            B N 3
    '''
    if not use_normals:
        new_pc = torch.einsum('bnc, bdc -> bnd', pc, rotation_matrix.float().to(pc.device))
    else:
        new_pc = torch.einsum('bnc, bdc -> bnd', pc[:, :, :3], rotation_matrix.float().to(pc.device))
        new_normal = torch.einsum('bnc, bdc -> bnd', pc[:, :, 3:], rotation_matrix.float().to(pc.device))
        new_pc = torch.cat([new_pc, new_normal], dim=-1)
    return new_pc

def rotate_point_clouds_batch_groups(pc, rotation_matrix, use_normals=False):
    '''
    输入:
        pc: 点云数据，形状为 B G M 3
        rotation_matrix: 旋转矩阵，形状为 B 3 3
    输出:
        旋转后的点云数据，形状为 B G M 3
    '''
    B, G, M, _ = pc.shape

    # 扩展旋转矩阵的形状以匹配点云数据的批次和组的数量
    # 新的形状将是 B G 3 3，这样每个分组都可以应用一个旋转矩阵
    rotation_matrix_expanded = rotation_matrix.unsqueeze(1).expand(B, G, 3, 3).float().to(pc.device)
    
    if not use_normals:
        # 应用旋转矩阵到每个分组
        new_pc = torch.einsum('bgmc, bgdc -> bgmd', pc, rotation_matrix_expanded)
    else:
        # 如果点云数据包含法线信息，分别对点和法线应用旋转
        new_pc = torch.einsum('bgmc, bgdc -> bgmd', pc[:, :, :, :3], rotation_matrix_expanded)
        new_normal = torch.einsum('bgmc, bgdc -> bgmd', pc[:, :, :, 3:], rotation_matrix_expanded)
        new_pc = torch.cat([new_pc, new_normal], dim=-1)

    return new_pc

class PatchPointcloudRandomRotate(object):
    def __init__(self, use_normals=False):
        super(PatchPointcloudRandomRotate, self).__init__()
        self.use_normals = use_normals

    def __call__(self, inputs):

        neighborhood, center = inputs 
        # 随机生成旋转角度
        angle = torch.stack([
            torch.rand(center.size(0)) * 1.9 + 0.04,  # 0.04 ~ 1.94pi
            (torch.rand(center.size(0)) * 0.2 - 0.4)],  # -0.4 ~ -0.2 pi
            dim=-1) * math.pi
        
        # 生成旋转矩阵
        rotation_matrix = rotate_theta_phi(angle)
        
        # 应用旋转变换
        neighborhood = rotate_point_clouds_batch_groups(neighborhood, rotation_matrix, use_normals=self.use_normals).contiguous()
        center = rotate_point_clouds_batch(center, rotation_matrix, use_normals=self.use_normals).contiguous()

        return neighborhood, center

class CenterRandomRotate(object):
    def __init__(self, use_normals=False):
        super(CenterRandomRotate, self).__init__()
        self.use_normals = use_normals

    def __call__(self, inputs):

        pc, center = inputs 
        # 随机生成旋转角度
        angle = torch.stack([
            torch.rand(pc.size(0)) * 1.9 + 0.04,  # 0.04 ~ 1.94pi
            (torch.rand(pc.size(0)) * 0.2 - 0.4)],  # -0.4 ~ -0.2 pi
            dim=-1) * math.pi
        
        # 生成旋转矩阵
        rotation_matrix = rotate_theta_phi(angle)
        
        # 应用旋转变换
        rotated_pc = rotate_point_clouds_batch(pc, rotation_matrix, use_normals=self.use_normals).contiguous()
        center = rotate_point_clouds_batch(center, rotation_matrix, use_normals=self.use_normals).contiguous()
        return rotated_pc, center
    
class CentercloudJitter(object):
    def __init__(self, use_normals=False, std=0.01, clip=0.05):
        super(CentercloudJitter, self).__init__()
        self.use_normals = use_normals
        self.jitter_patch = PointcloudJitterPatch(std=std, clip=clip)  # 用于neighborhood
        self.jitter = PointcloudJitter(std=std, clip=clip)  # 用于center

    def __call__(self, inputs):
        pc, center = inputs 
        # 对neighborhood应用抖动效果，适用于 B N 3 维度
        pc = self.jitter(pc)
        # 对center应用抖动效果，适用于 B N 3 维度
        center = self.jitter(center)

        return pc, center

class PointcloudRandomRotate(object):
    def __init__(self, use_normals=False):
        super(PointcloudRandomRotate, self).__init__()
        self.use_normals = use_normals

    def __call__(self, pc):
        # 随机生成旋转角度
        angle = torch.stack([
            torch.rand(pc.size(0)) * 1.9 + 0.04,  # 0.04 ~ 1.94pi
            (torch.rand(pc.size(0)) * 0.2 - 0.4)],  # -0.4 ~ -0.2 pi
            dim=-1) * math.pi
        
        # 生成旋转矩阵
        rotation_matrix = rotate_theta_phi(angle)
        
        # 应用旋转变换
        rotated_pc = rotate_point_clouds_batch(pc, rotation_matrix, use_normals=self.use_normals).contiguous()
        return rotated_pc

class PointcloudRotate(object):
    def __init__(self, angle=[0.0, 1.0, 0.0]):
        self.angle = np.array(angle) * np.pi

    @staticmethod
    def M(axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

    def __call__(self, data):
        if hasattr(data, 'keys'):
            device = data['pos'].device
        else:
            device = data.device

        if isinstance(self.angle, collections.Iterable):
            rot_mats = []
            for axis_ind, rot_bound in enumerate(self.angle):
                theta = 0
                axis = np.zeros(3)
                axis[axis_ind] = 1
                if rot_bound is not None:
                    theta = np.random.uniform(-rot_bound, rot_bound)
                rot_mats.append(self.M(axis, theta))
            # Use random order
            np.random.shuffle(rot_mats)
            rot_mat = torch.tensor(rot_mats[0] @ rot_mats[1] @ rot_mats[2], dtype=torch.float32, device=device)
        else:
            raise ValueError()

        """ DEBUG
        from openpoints.dataset import vis_multi_points
        old_points = data.cpu().numpy()
        # old_points = data['pos'].numpy()
        # new_points = (data['pos'] @ rot_mat.T).numpy()
        new_points = (data @ rot_mat.T).cpu().numpy()
        vis_multi_points([old_points, new_points])
        End of DEBUG"""

        if hasattr(data, 'keys'):
            data['pos'] = data['pos'] @ rot_mat.T
            if 'normals' in data:
                data['normals'] = data['normals'] @ rot_mat.T
        else:
            data = data @ rot_mat.T
        return data

class PointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
            
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda()) + torch.from_numpy(xyz2).float().cuda()
            
        return pc

class PointcloudRotatePerturbation(object):
    def __init__(self, angle_sigma=0.06, angle_clip=0.18):
        self.angle_sigma, self.angle_clip = angle_sigma, angle_clip

    def __call__(self, pc):
        # pc is expected to be a PyTorch tensor of shape [B, N, 3]
        rotated_data = pc.clone()
        for k in range(pc.shape[0]):
            angles = np.clip(self.angle_sigma * np.random.randn(3), -self.angle_clip, self.angle_clip)
            Rx = np.array([[1, 0, 0],
                           [0, np.cos(angles[0]), -np.sin(angles[0])],
                           [0, np.sin(angles[0]), np.cos(angles[0])]])
            Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                           [0, 1, 0],
                           [-np.sin(angles[1]), 0, np.cos(angles[1])]])
            Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                           [np.sin(angles[2]), np.cos(angles[2]), 0],
                           [0, 0, 1]])
            R = np.dot(Rz, np.dot(Ry, Rx))
            shape_pc = pc[k, ...].cpu().numpy()
            rotated_data[k, ...] = torch.tensor(np.dot(shape_pc, R), dtype=pc.dtype, device=pc.device)
        return rotated_data

class PointcloudJitter(object):
    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            jittered_data = pc.new(pc.size(1), 3).normal_(
                mean=0.0, std=self.std
            ).clamp_(-self.clip, self.clip)
            pc[i, :, 0:3] += jittered_data
            
        return pc

class PointcloudJitterPatch(object):
    def __init__(self, std=0.01, clip=0.05):
        self.std = std
        self.clip = clip

    def __call__(self, pc):
        # pc 形状为 B G M 3
        B, G, M, _ = pc.shape
        for i in range(B):
            for j in range(G):
                jittered_data = pc.new(M, 3).normal_(
                    mean=0.0, std=self.std
                ).clamp_(-self.clip, self.clip)
                pc[i, j, :, :3] += jittered_data
        return pc

# class PatchPointcloudJitter(object):
#     def __init__(self, use_normals=False, std=0.01, clip=0.05):
#         super(PatchPointcloudJitter, self).__init__()
#         self.use_normals = use_normals
#         self.jitter_patch = PointcloudJitterPatch(std=std, clip=clip)  # 用于neighborhood
#         self.jitter = PointcloudJitter(std=std, clip=clip)  # 用于center

#     def __call__(self, inputs):
#         neighborhood, center = inputs 
#         # 对neighborhood应用抖动效果，适用于 B G M 3 维度
#         neighborhood = self.jitter_patch(neighborhood)
#         # 对center应用抖动效果，适用于 B N 3 维度
#         center = self.jitter(center)

#         return neighborhood, center

class PointcloudScale(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2.):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda())
            
        return pc

class PointcloudTranslate(object):
    def __init__(self, translate_range=0.2):
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
            
            pc[i, :, 0:3] = pc[i, :, 0:3] + torch.from_numpy(xyz2).float().cuda()
            
        return pc

class PointcloudRandomInputDropout(object):
    def __init__(self, max_dropout_ratio=0.5):
        assert max_dropout_ratio >= 0 and max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            dropout_ratio = np.random.random() * self.max_dropout_ratio  # 0~0.875
            drop_idx = np.where(np.random.random((pc.size()[1])) <= dropout_ratio)[0]
            if len(drop_idx) > 0:
                cur_pc = pc[i, :, :]
                cur_pc[drop_idx.tolist(), 0:3] = cur_pc[0, 0:3].repeat(len(drop_idx), 1)  # set to the first point
                pc[i, :, :] = cur_pc

        return pc

class RandomHorizontalFlip(object):


  def __init__(self, upright_axis = 'z', is_temporal=False):
    """
    upright_axis: axis index among x,y,z, i.e. 2 for z
    """
    self.is_temporal = is_temporal
    self.D = 4 if is_temporal else 3
    self.upright_axis = {'x': 0, 'y': 1, 'z': 2}[upright_axis.lower()]
    # Use the rest of axes for flipping.
    self.horz_axes = set(range(self.D)) - set([self.upright_axis])


  def __call__(self, coords):
    bsize = coords.size()[0]
    for i in range(bsize):
        if random.random() < 0.95:
            for curr_ax in self.horz_axes:
                if random.random() < 0.5:
                    coord_max = torch.max(coords[i, :, curr_ax])
                    coords[i, :, curr_ax] = coord_max - coords[i, :, curr_ax]
    return coords
  

class PatchPointcloudRotatePerturbation(object):
    def __init__(self, angle_sigma=0.06, angle_clip=0.18):
        super(PatchPointcloudRotatePerturbation, self).__init__()
        self.angle_sigma = angle_sigma
        self.angle_clip = angle_clip

    def __call__(self, inputs):
        neighborhood, center = inputs

        B = neighborhood.shape[0]
        device = neighborhood.device
        dtype = neighborhood.dtype

        # 生成每个样本的随机旋转角度，形状为 [B, 3]
        angles = self.angle_sigma * torch.randn(B, 3, device=device, dtype=dtype)
        angles = torch.clamp(angles, -self.angle_clip, self.angle_clip)

        # 计算角度的正弦和余弦值
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)

        # 构建旋转矩阵，形状为 [B, 3, 3]
        ones = torch.ones(B, 1, device=device, dtype=dtype)
        zeros = torch.zeros(B, 1, device=device, dtype=dtype)

        # 构建绕 X 轴的旋转矩阵 Rx
        Rx = torch.stack([
            ones[:, 0], zeros[:, 0], zeros[:, 0],
            zeros[:, 0], cos_angles[:, 0], -sin_angles[:, 0],
            zeros[:, 0], sin_angles[:, 0], cos_angles[:, 0]
        ], dim=1).view(B, 3, 3)

        # 构建绕 Y 轴的旋转矩阵 Ry
        Ry = torch.stack([
            cos_angles[:, 1], zeros[:, 0], sin_angles[:, 1],
            zeros[:, 0], ones[:, 0], zeros[:, 0],
            -sin_angles[:, 1], zeros[:, 0], cos_angles[:, 1]
        ], dim=1).view(B, 3, 3)

        # 构建绕 Z 轴的旋转矩阵 Rz
        Rz = torch.stack([
            cos_angles[:, 2], -sin_angles[:, 2], zeros[:, 0],
            sin_angles[:, 2], cos_angles[:, 2], zeros[:, 0],
            zeros[:, 0], zeros[:, 0], ones[:, 0]
        ], dim=1).view(B, 3, 3)

        # 合并旋转矩阵，得到最终的旋转矩阵 R，形状为 [B, 3, 3]
        R = torch.bmm(Rz, torch.bmm(Ry, Rx))

        # 将 neighborhood 和 center 重塑为 [B, -1, 3]
        neighborhood_flat = neighborhood.view(B, -1, 3)  # [B, G*M, 3]
        center_flat = center.view(B, -1, 3)  # [B, N, 3]

        # 在第二个维度上拼接
        combined = torch.cat([neighborhood_flat, center_flat], dim=1)  # [B, K, 3], K = G*M + N

        # 应用旋转矩阵，使用批量矩阵乘法
        rotated_combined = torch.bmm(combined, R.transpose(1, 2))  # [B, K, 3]

        # 拆分回 rotated_neighborhood 和 rotated_center
        num_neighborhood_points = neighborhood_flat.shape[1]
        rotated_neighborhood_flat = rotated_combined[:, :num_neighborhood_points, :]  # [B, G*M, 3]
        rotated_center_flat = rotated_combined[:, num_neighborhood_points:, :]  # [B, N, 3]

        # 重塑 rotated_neighborhood
        G = neighborhood.shape[1]
        M = neighborhood.shape[2]
        rotated_neighborhood = rotated_neighborhood_flat.view(B, G, M, 3)  # [B, G, M, 3]
        rotated_center = rotated_center_flat.view_as(center)  # [B, N, 3]

        return rotated_neighborhood, rotated_center



class PatchPointcloudRotate(object):
    def __init__(self, use_normals=False, default_angles=None):
        super(PatchPointcloudRotate, self).__init__()
        self.use_normals = use_normals
        if default_angles is None:
            # 设置默认的旋转角度为绕 Y 轴旋转 180 度
            self.default_angles = torch.tensor([0, math.pi, 0])  # [0, π, 0]
        else:
            self.default_angles = default_angles

    def __call__(self, inputs, angles=None):
        """
        inputs: tuple of (neighborhood, center)
            - neighborhood: [B, G, M, 3] 或 [B, G, M, 6]（如果包含法向量）
            - center: [B, N, 3] 或 [B, N, 6]
        angles: torch.Tensor 或 numpy.ndarray,形状为 [B, 3] 或 [3]，可选
            - 每个样本的旋转角度,分别绕 x、y、z 轴旋转（单位：弧度）
            - 如果为 None,将使用 default_angles
            # 例如，绕 Z 轴旋转 90 度
                angles = torch.tensor([0, 0, math.pi / 2])  # [0, 0, π/2]
                rotated_neighborhood, rotated_center = augment((neighborhood, center), angles)
        """
        neighborhood, center = inputs
        B = center.size(0)  # 批量大小

        # 如果未提供 angles，则使用 default_angles
        if angles is None:
            angles = self.default_angles

        # 确保 angles 是 torch.Tensor，并在正确的设备上
        if not isinstance(angles, torch.Tensor):
            angles = torch.tensor(angles, dtype=neighborhood.dtype, device=neighborhood.device)
        else:
            angles = angles.to(dtype=neighborhood.dtype, device=neighborhood.device)

        # 如果 angles 是形状为 [3] 的张量，扩展为 [B, 3]
        if angles.ndim == 1:
            angles = angles.unsqueeze(0).repeat(B, 1)  # [B, 3]
        elif angles.shape[0] != B:
            # 如果 angles 的第一个维度不等于批量大小 B，抛出错误
            raise ValueError(f"Angles must have shape [B, 3] or [3], but got {angles.shape}")

        # 构建每个样本的旋转矩阵，形状为 [B, 3, 3]
        Rs = []
        for b in range(B):
            angle = angles[b]
            Rx = torch.tensor([[1, 0, 0],
                               [0, torch.cos(angle[0]), -torch.sin(angle[0])],
                               [0, torch.sin(angle[0]), torch.cos(angle[0])]], dtype=neighborhood.dtype, device=neighborhood.device)
            Ry = torch.tensor([[torch.cos(angle[1]), 0, torch.sin(angle[1])],
                               [0, 1, 0],
                               [-torch.sin(angle[1]), 0, torch.cos(angle[1])]], dtype=neighborhood.dtype, device=neighborhood.device)
            Rz = torch.tensor([[torch.cos(angle[2]), -torch.sin(angle[2]), 0],
                               [torch.sin(angle[2]), torch.cos(angle[2]), 0],
                               [0, 0, 1]], dtype=neighborhood.dtype, device=neighborhood.device)
            R = torch.mm(Rz, torch.mm(Ry, Rx))  # 合并旋转矩阵
            Rs.append(R.unsqueeze(0))
        Rs = torch.cat(Rs, dim=0)  # [B, 3, 3]

        # 定义旋转函数，处理是否包含法向量的情况
        def rotate_point_cloud(pc, Rs, use_normals=False):
            # pc: [B, ..., 3] 或 [B, ..., 6]
            B = pc.shape[0]
            if use_normals:
                # 分离坐标和法向量
                coord, normals = pc[..., :3], pc[..., 3:]
                # 重塑为 [B, -1, 3]
                coord_flat = coord.view(B, -1, 3)
                normals_flat = normals.view(B, -1, 3)
                # 应用旋转
                rotated_coord = torch.bmm(coord_flat, Rs.transpose(1, 2))
                rotated_normals = torch.bmm(normals_flat, Rs.transpose(1, 2))
                # 合并
                rotated_pc_flat = torch.cat([rotated_coord, rotated_normals], dim=2)
                # 恢复原始形状
                rotated_pc = rotated_pc_flat.view_as(pc)
            else:
                # 仅处理坐标
                coord_flat = pc.view(B, -1, 3)
                rotated_coord = torch.bmm(coord_flat, Rs.transpose(1, 2))
                rotated_pc = rotated_coord.view_as(pc)
            return rotated_pc

        # 对 neighborhood 和 center 应用旋转
        rotated_neighborhood = rotate_point_cloud(neighborhood, Rs, use_normals=self.use_normals)
        rotated_center = rotate_point_cloud(center, Rs, use_normals=self.use_normals)

        return rotated_neighborhood.contiguous(), rotated_center.contiguous()


class PatchPointcloudJitter(object):
    def __init__(self, std=0.01, clip=0.05):
        """
        Args:
            std (float): 抖动的标准差，控制噪声的强度。
            clip (float): 抖动的截断值，防止噪声过大。
        """
        super(PatchPointcloudJitter, self).__init__()
        self.std = std
        self.clip = clip

    def __call__(self, inputs):
        """
        Args:
            inputs (tuple): 包含两个元素的元组 (neighborhood, center)
                - neighborhood: [B, G, M, 3]
                - center: [B, N, 3]
        Returns:
            tuple: 经过抖动的数据 (jittered_neighborhood, jittered_center)
        """
        neighborhood, center = inputs

        # 对 neighborhood 应用抖动
        jittered_neighborhood = self.jitter_pointcloud(neighborhood)

        # 对 center 应用抖动
        jittered_center = self.jitter_pointcloud(center)

        return jittered_neighborhood, jittered_center

    def jitter_pointcloud(self, pc):
        """
        对点云添加随机抖动噪声。

        Args:
            pc (torch.Tensor): 点云数据，形状为 [B, ..., 3]
        Returns:
            torch.Tensor: 经过抖动的点云数据，形状与输入相同
        """
        B = pc.shape[0]
        device = pc.device
        dtype = pc.dtype

        # 生成随机噪声，形状与 pc[..., :3] 相同
        noise = torch.clamp(
            self.std * torch.randn_like(pc[..., :3], device=device, dtype=dtype),
            min=-self.clip, max=self.clip
        )

        # 将噪声添加到点云坐标上
        jittered_pc = pc.clone()
        jittered_pc[..., :3] += noise

        return jittered_pc



class HierarchicalJitter(object):
    def __init__(self, std=0.01, clip=0.05):
        """
        Args:
            std (float): 抖动的标准差，控制噪声的强度。
            clip (float): 抖动的截断值，防止噪声过大。
        """
        super(HierarchicalJitter, self).__init__()
        self.std = std
        self.clip = clip

    def __call__(self, inputs):
        """
        Args:
            inputs (tuple): 包含两个列表的元组 (neighborhoods, centers)
                - neighborhoods: List[Tensor]，每个元素形状为 [B, G_i, M_i, 3]
                - centers: List[Tensor]，每个元素形状为 [B, N_i, 3]
        Returns:
            tuple: 经过抖动的数据 (jittered_neighborhoods, jittered_centers)
        """
        neighborhoods, centers = inputs
        B = neighborhoods[0].shape[0]
        device = neighborhoods[0].device
        dtype = neighborhoods[0].dtype

        # 对每个尺度的数据进行抖动
        jittered_neighborhoods = []
        jittered_centers = []

        for i in range(len(neighborhoods)):
            neighborhood = neighborhoods[i]
            center = centers[i]

            # 生成抖动噪声，形状与 neighborhood 和 center 的坐标部分相同
            noise_neighborhood = torch.clamp(
                self.std * torch.randn_like(neighborhood[..., :3]),
                min=-self.clip, max=self.clip
            )
            noise_center = torch.clamp(
                self.std * torch.randn_like(center[..., :3]),
                min=-self.clip, max=self.clip
            )

            # 添加抖动噪声
            jittered_neighborhood = neighborhood.clone()
            jittered_neighborhood[..., :3] += noise_neighborhood

            jittered_center = center.clone()
            jittered_center[..., :3] += noise_center

            jittered_neighborhoods.append(jittered_neighborhood)
            jittered_centers.append(jittered_center)

        return jittered_neighborhoods, jittered_centers


class HierarchicalRotatePerturbation(object):
    def __init__(self, angle_sigma=0.06, angle_clip=0.18):
        """
        Args:
            angle_sigma (float): 旋转角度的标准差，控制旋转角度的随机性。
            angle_clip (float): 旋转角度的截断值，防止旋转角度过大。
        """
        super(HierarchicalRotatePerturbation, self).__init__()
        self.angle_sigma = angle_sigma
        self.angle_clip = angle_clip

    def __call__(self, inputs):
        """
        Args:
            inputs (tuple): 包含两个列表的元组 (neighborhoods, centers)
                - neighborhoods: List[Tensor]，每个元素形状为 [B, G_i, M_i, 3]
                - centers: List[Tensor]，每个元素形状为 [B, N_i, 3]
        Returns:
            tuple: 经过旋转扰动的数据 (rotated_neighborhoods, rotated_centers)
        """
        neighborhoods, centers = inputs
        B = neighborhoods[0].shape[0]
        device = neighborhoods[0].device
        dtype = neighborhoods[0].dtype

        # 生成每个样本的随机旋转角度，形状为 [B, 3]
        angles = self.angle_sigma * torch.randn(B, 3, device=device, dtype=dtype)
        angles = torch.clamp(angles, -self.angle_clip, self.angle_clip)

        # 计算角度的正弦和余弦值
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)

        # 构建旋转矩阵，形状为 [B, 3, 3]
        ones = torch.ones(B, 1, device=device, dtype=dtype)
        zeros = torch.zeros(B, 1, device=device, dtype=dtype)

        # 构建绕 X 轴的旋转矩阵 Rx
        Rx = torch.stack([
            ones[:, 0], zeros[:, 0], zeros[:, 0],
            zeros[:, 0], cos_angles[:, 0], -sin_angles[:, 0],
            zeros[:, 0], sin_angles[:, 0], cos_angles[:, 0]
        ], dim=1).view(B, 3, 3)

        # 构建绕 Y 轴的旋转矩阵 Ry
        Ry = torch.stack([
            cos_angles[:, 1], zeros[:, 0], sin_angles[:, 1],
            zeros[:, 0], ones[:, 0], zeros[:, 0],
            -sin_angles[:, 1], zeros[:, 0], cos_angles[:, 1]
        ], dim=1).view(B, 3, 3)

        # 构建绕 Z 轴的旋转矩阵 Rz
        Rz = torch.stack([
            cos_angles[:, 2], -sin_angles[:, 2], zeros[:, 0],
            sin_angles[:, 2], cos_angles[:, 2], zeros[:, 0],
            zeros[:, 0], zeros[:, 0], ones[:, 0]
        ], dim=1).view(B, 3, 3)

        # 合并旋转矩阵，得到最终的旋转矩阵 R，形状为 [B, 3, 3]
        R = torch.bmm(Rz, torch.bmm(Ry, Rx))  # [B, 3, 3]

        # 对 neighborhoods 和 centers 的每一层进行旋转
        rotated_neighborhoods = []
        rotated_centers = []

        for i in range(len(neighborhoods)):
            neighborhood = neighborhoods[i]
            center = centers[i]

            # neighborhood: [B, G_i, M_i, 3]
            # center: [B, N_i, 3]

            # 将 neighborhood 和 center 重塑为 [B, -1, 3]
            neighborhood_flat = neighborhood.view(B, -1, 3)
            center_flat = center.view(B, -1, 3)

            # 对 neighborhood 和 center 应用旋转
            rotated_neighborhood = torch.bmm(neighborhood_flat, R.transpose(1, 2)).view_as(neighborhood)
            rotated_center = torch.bmm(center_flat, R.transpose(1, 2)).view_as(center)

            rotated_neighborhoods.append(rotated_neighborhood)
            rotated_centers.append(rotated_center)

        return rotated_neighborhoods, rotated_centers


class JitterConsistent(object):
    def __init__(self, std=0.01, clip=0.05):
        """
        Args:
            std (float): 抖动的标准差，控制噪声的强度。
            clip (float): 抖动的截断值，防止噪声过大。
        """
        super(JitterConsistent, self).__init__()
        self.std = std
        self.clip = clip

    def __call__(self, inputs):
        """
        Args:
            inputs (tuple): 包含两个列表和两个张量的元组 (neighborhoods, centers, neighborhood, center)
                - neighborhoods: List[Tensor]，每个元素形状为 [B, G_i, M_i, 3]
                - centers: List[Tensor]，每个元素形状为 [B, N_i, 3]
                - neighborhood: Tensor, 形状为 [B, G, M, 3]
                - center: Tensor, 形状为 [B, N, 3]
        Returns:
            tuple: 经过抖动的数据 (jittered_neighborhoods, jittered_centers, jittered_neighborhood, jittered_center)
        """
        neighborhoods, centers, neighborhood, center = inputs
        B = neighborhoods[0].shape[0]
        device = neighborhoods[0].device
        dtype = neighborhoods[0].dtype

        # 生成抖动噪声
        noise = torch.clamp(
            self.std * torch.randn(B, 1, 1, 3, device=device, dtype=dtype),
            min=-self.clip, max=self.clip
        )

        # 对 neighborhoods 和 centers 的每一层进行抖动
        jittered_neighborhoods = []
        jittered_centers = []

        for i in range(len(neighborhoods)):
            neighborhood_i = neighborhoods[i] + noise
            center_i = centers[i] + noise.squeeze(2)
            jittered_neighborhoods.append(neighborhood_i)
            jittered_centers.append(center_i)

        # 对单独的 neighborhood 和 center 进行抖动
        jittered_neighborhood = neighborhood + noise
        jittered_center = center + noise.squeeze(2)

        return jittered_neighborhoods, jittered_centers, jittered_neighborhood, jittered_center


class RotatePerturbationConsistent(object):
    def __init__(self, angle_sigma=0.06, angle_clip=0.18):
        """
        Args:
            angle_sigma (float): 旋转角度的标准差，控制旋转角度的随机性。
            angle_clip (float): 旋转角度的截断值，防止旋转角度过大。
        """
        super(RotatePerturbationConsistent, self).__init__()
        self.angle_sigma = angle_sigma
        self.angle_clip = angle_clip

    def __call__(self, inputs):
        """
        Args:
            inputs (tuple): 包含两个列表和两个张量的元组 (neighborhoods, centers, neighborhood, center)
                - neighborhoods: List[Tensor]，每个元素形状为 [B, G_i, M_i, 3]
                - centers: List[Tensor]，每个元素形状为 [B, N_i, 3]
                - neighborhood: Tensor，形状为 [B, G, M, 3]
                - center: Tensor，形状为 [B, N, 3]
        Returns:
            tuple: 经过旋转扰动的数据 (rotated_neighborhoods, rotated_centers, rotated_neighborhood, rotated_center)
        """
        neighborhoods, centers, neighborhood, center = inputs
        B = neighborhoods[0].shape[0]
        device = neighborhoods[0].device
        dtype = neighborhoods[0].dtype

        # 生成每个样本的随机旋转角度，形状为 [B, 3]
        angles = self.angle_sigma * torch.randn(B, 3, device=device, dtype=dtype)
        angles = torch.clamp(angles, -self.angle_clip, self.angle_clip)

        # 计算角度的正弦和余弦值
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)

        # 构建旋转矩阵，形状为 [B, 3, 3]
        ones = torch.ones(B, 1, device=device, dtype=dtype)
        zeros = torch.zeros(B, 1, device=device, dtype=dtype)

        # 构建绕 X 轴的旋转矩阵 Rx
        Rx = torch.stack([
            ones[:, 0], zeros[:, 0], zeros[:, 0],
            zeros[:, 0], cos_angles[:, 0], -sin_angles[:, 0],
            zeros[:, 0], sin_angles[:, 0], cos_angles[:, 0]
        ], dim=1).view(B, 3, 3)

        # 构建绕 Y 轴的旋转矩阵 Ry
        Ry = torch.stack([
            cos_angles[:, 1], zeros[:, 0], sin_angles[:, 1],
            zeros[:, 0], ones[:, 0], zeros[:, 0],
            -sin_angles[:, 1], zeros[:, 0], cos_angles[:, 1]
        ], dim=1).view(B, 3, 3)

        # 构建绕 Z 轴的旋转矩阵 Rz
        Rz = torch.stack([
            cos_angles[:, 2], -sin_angles[:, 2], zeros[:, 0],
            sin_angles[:, 2], cos_angles[:, 2], zeros[:, 0],
            zeros[:, 0], zeros[:, 0], ones[:, 0]
        ], dim=1).view(B, 3, 3)

        # 合并旋转矩阵，得到最终的旋转矩阵 R，形状为 [B, 3, 3]
        R = torch.bmm(Rz, torch.bmm(Ry, Rx))  # [B, 3, 3]

        # 对 neighborhoods 和 centers 的每一层进行旋转
        rotated_neighborhoods = []
        rotated_centers = []

        for i in range(len(neighborhoods)):
            neighborhood_i = neighborhoods[i]
            center_i = centers[i]

            # neighborhood_i: [B, G_i, M_i, 3]
            # center_i: [B, N_i, 3]

            # 将 neighborhood_i 和 center_i 重塑为 [B, -1, 3]
            neighborhood_flat = neighborhood_i.view(B, -1, 3)
            center_flat = center_i.view(B, -1, 3)

            # 应用旋转
            rotated_neighborhood_flat = torch.bmm(neighborhood_flat, R.transpose(1, 2))
            rotated_center_flat = torch.bmm(center_flat, R.transpose(1, 2))

            # 恢复原始形状
            rotated_neighborhood_i = rotated_neighborhood_flat.view_as(neighborhood_i)
            rotated_center_i = rotated_center_flat.view_as(center_i)

            rotated_neighborhoods.append(rotated_neighborhood_i)
            rotated_centers.append(rotated_center_i)

        # 对单独的 neighborhood 和 center 进行旋转
        neighborhood_flat = neighborhood.view(B, -1, 3)
        center_flat = center.view(B, -1, 3)

        rotated_neighborhood_flat = torch.bmm(neighborhood_flat, R.transpose(1, 2))
        rotated_center_flat = torch.bmm(center_flat, R.transpose(1, 2))

        rotated_neighborhood = rotated_neighborhood_flat.view_as(neighborhood)
        rotated_center = rotated_center_flat.view_as(center)

        return rotated_neighborhoods, rotated_centers, rotated_neighborhood, rotated_center