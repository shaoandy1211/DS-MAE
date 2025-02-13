import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .build import MODELS
import random
from timm.models.layers import DropPath, trunc_normal_
from torch_scatter import scatter
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from utils.logger import *
from utils.misc import fps
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from knn_cuda import KNN
from .modules import knn_point, square_distance, index_points

class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True):
        super(ConvBNReLU1D, self).__init__()
        self.act = nn.GELU()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)
        
class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True):
        super(ConvBNReLURes1D, self).__init__()
        self.act = nn.GELU()
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(in_channels=channel, out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)

class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):  # [b, d, g]
        return self.operation(x)

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, out_channel, blocks=1, groups=1, res_expansion=1.0, bias=True):
        super(PointNetFeaturePropagation, self).__init__()
        self.fuse = ConvBNReLU1D(in_channel, out_channel, 1, bias=bias)
        self.extraction = PosExtraction(out_channel, blocks, groups=groups,
                                        res_expansion=res_expansion, bias=bias)


    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: sampled input points position data, [B, S, 3]
            points1: input points data, [B, D', N]
            points2: input points data, [B, D'', S]
        Return:
            new_points: upsampled points data, [B, D''', N]
        # """
        # if points1 is not None:
        #     points1 = points1.permute(0, 2, 1)
        # if points2 is not None:
        #     points2 = points2.permute(0, 2, 1)


        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            # points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)

        new_points = self.fuse(new_points)
        new_points = self.extraction(new_points)
        new_points = new_points.permute(0, 2, 1)

        return new_points

class Token_Embed(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        if in_c == 3:
            self.first_conv = nn.Sequential(
                nn.Conv1d(in_c, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, 256, 1)
            )
            self.second_conv = nn.Sequential(
                nn.Conv1d(512, 512, 1),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, out_c, 1)
            )

        else:
            self.first_conv = nn.Sequential(
                nn.Conv1d(in_c, in_c, 1),
                nn.BatchNorm1d(in_c),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_c, in_c, 1)
            )
            self.second_conv = nn.Sequential(
                nn.Conv1d(in_c * 2, out_c, 1),
                nn.BatchNorm1d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv1d(out_c, out_c, 1)
            )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , c = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, c)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.out_c)

class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        # self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz, center=None):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        if center is None:
            center = fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        # _, idx = self.knn(xyz, center)
            
        idx = knn_point(self.group_size, xyz, center)
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center, idx


# Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y = None, mask=None):
        if y is None:
            y = x

        B, N, C = y.shape
        kv = self.kv(y).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            # mask = mask * float('-inf') 
            mask = mask * - 100000.0
            attn = attn + mask.unsqueeze(1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., with_cross=False, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)      
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.with_cross = with_cross
        if self.with_cross:
            self.cross_norm = norm_layer(dim)

    def forward(self, *x, **kwargs):
        if self.with_cross:
            x, z = x
            x = x + self.drop_path(self.attn(self.norm1(x), y = self.cross_norm(z)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        else:
            x, vis_mask = x
            x = x + self.drop_path(self.attn(self.norm1(x), mask = vis_mask))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

class Encoder_Block(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, with_cross=False,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, vis_mask = None):
        for _, block in enumerate(self.blocks):
            x = block(x, vis_mask)
        return x

class Decoder_Block(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, with_cross=True,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, z, return_token_num = None):
        for i, block in enumerate(self.blocks):
            if i == 0:
                x = block(x, z)
            else:
                x = block(x, x)
        if return_token_num is not None:
            x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
        return x
    
class MAE_Encoder(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.mask_ratio = config.mask_ratio
        self.drop_path_rate = config.drop_path_rate
        self.num_heads = config.num_heads
        # MAE argparse
        self.encoder_dim = config.encoder_dim
        self.depth = config.encoder_depth
        # MAE token_embed and pos_embed
        self.token_embed = Token_Embed(in_c=3, out_c=self.encoder_dim)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.encoder_dim),
        )
        # MAE encoder block
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.encoder_block = Encoder_Block(
            embed_dim=self.encoder_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )
        self.norm = nn.LayerNorm(self.encoder_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def rand_mask(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device)  # B G

    def forward(self, neighborhood, center, mask = None, eval = False):
        # generate mask
        if mask is None:
            mask = self.rand_mask(center, noaug = eval)  # B G
            
        group_input_tokens = self.token_embed(neighborhood)  # B G C

        batch_size, seq_len, C = group_input_tokens.size()
        x_vis = group_input_tokens[~mask].reshape(batch_size, -1, C)
        masked_center = center[~mask].reshape(batch_size, -1, 3)
        pos = self.pos_embed(masked_center)         
        # transformer
        x_vis = self.encoder_block(x_vis + pos)
        x_vis = self.norm(x_vis)
        
        return x_vis, mask    
    
class M2AE_Encoder(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.mask_ratio = config.mask_ratio
        self.drop_path_rate = config.drop_path_rate
        self.num_heads = config.num_heads
        # M2AE argparse
        self.encoder_depths = config.encoder_depths
        self.encoder_dims =  config.encoder_dims
        self.local_radius = config.local_radius
        # M2AE token_embed and pos_embed
        self.token_embeds = nn.ModuleList()
        self.pos_embeds = nn.ModuleList()
        for i in range(len(self.encoder_dims)):
            if i == 0:
                self.token_embeds.append(Token_Embed(in_c=3, out_c=self.encoder_dims[i]))
            else:
                self.token_embeds.append(Token_Embed(in_c=self.encoder_dims[i - 1], out_c=self.encoder_dims[i]))
            
            self.pos_embeds.append(nn.Sequential(
                            nn.Linear(3, self.encoder_dims[i]),
                            nn.GELU(),
                            nn.Linear(self.encoder_dims[i], self.encoder_dims[i]),
                        ))
        # M2AE encoder block
        self.encoder_blocks = nn.ModuleList()
        depth_count = 0
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.encoder_depths))]
        for i in range(len(self.encoder_depths)):
            self.encoder_blocks.append(Encoder_Block(
                            embed_dim=self.encoder_dims[i],
                            depth=self.encoder_depths[i],
                            drop_path_rate=dpr[depth_count: depth_count + self.encoder_depths[i]],
                            num_heads=self.num_heads,
                        ))
            depth_count += self.encoder_depths[i]

        self.norms = nn.ModuleList()
        for i in range(len(self.encoder_depths)):
            self.norms.append(nn.LayerNorm(self.encoder_dims[i]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def rand_mask(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool().cuda()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device)  # B G
    
    def local_att_mask(self, xyz, radius, dist=None):
        with torch.no_grad():
            if dist is None or dist.shape[1] != xyz.shape[1]:
                dist = torch.cdist(xyz, xyz, p=2)
            # entries that are True in the mask do not contribute to self-attention
            # so points outside the radius are not considered
            mask = dist >= radius
        return mask, dist

    def forward(self, neighborhoods, centers, idxs, bool_masked_pos = None, eval = False):
        # generate mask at the highest level
        if bool_masked_pos is None:
            # generate mask at the highest level
            bool_masked_pos = []
            if eval:
                # no mask
                B, G, _ = centers[-1].shape
                bool_masked_pos.append(torch.zeros(B, G).bool().cuda())
            else:
                # mask_index: 1, mask; 0, vis
                bool_masked_pos.append(self.rand_mask(centers[-1]))
            # Multi-scale Masking by back-propagation
            for i in range(len(neighborhoods) - 1, 0, -1):
                b, g, k, _ = neighborhoods[i].shape
                idx = idxs[i].reshape(b * g, -1)
                idx_masked = ~(bool_masked_pos[-1].reshape(-1).unsqueeze(-1)) * idx
                idx_masked = idx_masked.reshape(-1).long()
                masked_pos = torch.ones(b * centers[i - 1].shape[1]).cuda().scatter(0, idx_masked, 0).bool()
                bool_masked_pos.append(masked_pos.reshape(b, centers[i - 1].shape[1]))
            # hierarchical encoding
            bool_masked_pos = list(reversed(bool_masked_pos))

        x_vis_list = []
        mask_vis_list = []
        xyz_dist = None
        for i in range(len(centers)):
            # 1st-layer encoder, conduct token embedding
            if i == 0:
                group_input_tokens = self.token_embeds[i](neighborhoods[0])
            # intermediate layers, conduct token merging
            else:
                b, g1, _ = x_vis.shape
                b, g2, k2, _ = neighborhoods[i].shape
                x_vis_neighborhoods = x_vis.reshape(b * g1, -1)[idxs[i], :].reshape(b, g2, k2, -1)
                group_input_tokens = self.token_embeds[i](x_vis_neighborhoods)

            # visible_index
            bool_vis_pos = ~(bool_masked_pos[i])
            batch_size, seq_len, C = group_input_tokens.size()

            # Due to Multi-scale Masking different, samples of a batch have varying numbers of visible tokens
            # find the longest visible sequence in the batch
            vis_tokens_len = bool_vis_pos.long().sum(dim=1)
            max_tokens_len = torch.max(vis_tokens_len)
            # use the longest length (max_tokens_len) to construct tensors
            x_vis = torch.zeros(batch_size, max_tokens_len, C).cuda()
            masked_center = torch.zeros(batch_size, max_tokens_len, 3).cuda()
            mask_vis = torch.ones(batch_size, max_tokens_len, max_tokens_len).cuda()
            
            for bz in range(batch_size):
                # inject valid visible tokens
                vis_tokens = group_input_tokens[bz][bool_vis_pos[bz]]
                x_vis[bz][0: vis_tokens_len[bz]] = vis_tokens
                # inject valid visible centers
                vis_centers = centers[i][bz][bool_vis_pos[bz]]
                masked_center[bz][0: vis_tokens_len[bz]] = vis_centers
                # the mask for valid visible tokens/centers
                mask_vis[bz][0: vis_tokens_len[bz], 0: vis_tokens_len[bz]] = 0
            
            if self.local_radius[i] > 0:
                mask_radius, xyz_dist = self.local_att_mask(masked_center, self.local_radius[i], xyz_dist)
                # disabled for pre-training, this step would not change mask_vis by *
                mask_vis_att = mask_radius * mask_vis
            else:
                mask_vis_att = mask_vis

            pos = self.pos_embeds[i](masked_center)
            x = x_vis + pos

            x_vis = self.encoder_blocks[i](x, mask_vis_att)
            x_vis_list.append(x_vis)
            mask_vis_list.append(~(mask_vis[:, :, 0].bool()))

            if i == len(centers) - 1:
                pass
            else:
                group_input_tokens[bool_vis_pos] = x_vis[~(mask_vis[:, :, 0].bool())]
                x_vis = group_input_tokens

        for i in range(len(x_vis_list)):
            x_vis_list[i] = self.norms[i](x_vis_list[i])

        return x_vis_list, mask_vis_list, bool_masked_pos

from torchvision import transforms
from datasets import data_transforms

train_transform = transforms.Compose(
    [
        data_transforms.JitterConsistent(), 
        data_transforms.RotatePerturbationConsistent(), 
    ]
)

# Pretrain model
@MODELS.register_module()
class DSMAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        print_log(f'[args] {config}', logger='DSMAE')
        
        # Extract configuration parameters
        self.trans_dim = config.trans_dim  # Transformer dimension size
        self.decoder_depth = config.decoder_depth  # Depth of decoder
        self.drop_path_rate = config.drop_path_rate  # Drop path rate for regularization

        self.decoder_depths = config.decoder_depths  # Depths of hierarchical decoder layers
        self.decoder_dims = config.decoder_dims  # Embedding dimensions of each hierarchical decoder
        self.decoder_up_blocks = config.decoder_up_blocks  # Number of upsampling blocks for each decoder
        
        # Patching setup for MAE and M2AE
        self.group_size = config.group_size  # Group size for MAE
        self.num_group = config.num_group  # Number of groups for MAE
        self.mae_group_divider = Group(num_group=self.num_group, group_size=self.group_size)  # Grouping for MAE
        print_log(f'[MAE] divide point cloud into G {self.num_group} x S {self.group_size} points ...', logger='MAE')
        
        # Patching setup for multi-scale M2AE
        self.group_sizes = config.group_sizes  # List of group sizes for M2AE
        self.num_groups = config.num_groups  # List of number of groups for M2AE
        self.m2ae_group_dividers = nn.ModuleList([
            Group(num_group=self.num_groups[i], group_size=self.group_sizes[i])
            for i in range(len(self.group_sizes))
        ])
        for i in range(len(self.group_sizes)):
            print_log(f'[M2AE] divide point cloud into G {self.num_groups[i]} x S {self.group_sizes[i]} points ...', logger='M2AE')
        
        # Encoder initialization
        self.MAE_encoder = MAE_Encoder(config)  # MAE encoder
        self.M2AE_encoder = M2AE_Encoder(config)  # M2AE encoder for hierarchical features
        
        # Masked token initialization for masked modeling
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_dims[0]))
        self.mask_token_ = nn.Parameter(torch.zeros(1, 1, self.decoder_dims[0]))
        trunc_normal_(self.mask_token, std=.02)  # Truncated normal initialization
        trunc_normal_(self.mask_token_, std=.02)
        
        # Decoder setup
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]  # Stochastic depth schedule
        self.decoder = Decoder_Block(
            embed_dim=self.trans_dim,  # Embedding dimension of transformer
            depth=self.decoder_depth,  # Number of layers in decoder block
            drop_path_rate=dpr,  # Drop path rates
            num_heads=config.num_heads,  # Number of attention heads
        )
        
        # Hierarchical decoder setup
        self.h_decoder = nn.ModuleList()  # List to store hierarchical decoder blocks
        self.decoder_pos_embeds = nn.ModuleList()  # List to store positional embeddings for each decoder layer
        self.token_prop = nn.ModuleList()  # List to store token propagation layers for each decoder layer

        # Initialize hierarchical decoder and token propagation layers
        depth_count = 0
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(self.decoder_depths))]
        for i in range(len(self.decoder_dims)):
            # Add decoder block for each hierarchical layer
            self.h_decoder.append(Decoder_Block(
                embed_dim=self.decoder_dims[i],
                depth=self.decoder_depths[i],
                drop_path_rate=dpr[depth_count: depth_count + self.decoder_depths[i]],
                num_heads=config.num_heads,
            ))
            depth_count += self.decoder_depths[i]
            
            # Add positional embedding layer for each hierarchical decoder
            self.decoder_pos_embeds.append(nn.Sequential(
                nn.Linear(3, self.decoder_dims[i]),  # Linear layer to project positional info to embedding dimension
                nn.GELU(),  # Activation function
                nn.Linear(self.decoder_dims[i], self.decoder_dims[i])  # Linear layer to project back to embedding dimension
            ))
            
            # Add token propagation layers, starting from the second layer
            if i > 0:
                self.token_prop.append(PointNetFeaturePropagation(
                    self.decoder_dims[i] + self.decoder_dims[i - 1], self.decoder_dims[i],
                    blocks=self.decoder_up_blocks[i - 1], groups=1, res_expansion=1.0, bias=True
                ))
        
        # Layer normalization for the final hierarchical decoder output
        self.h_decoder_norm = nn.LayerNorm(self.decoder_dims[-1])
        
        # Prediction heads for reconstruction of point clouds
        self.rec_head_h = nn.Conv1d(self.decoder_dims[-1], 3 * self.group_sizes[0], 1)  # For hierarchical reconstruction
        self.rec_head_h_ = nn.Conv1d(self.decoder_dims[-1], 3 * self.group_sizes[0], 1)  # For hierarchical reconstruction
        self.rec_head_g = nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)  # For global reconstruction
        self.rec_head_g_ = nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)  # For global reconstruction
        
        # Loss function for reconstruction
        self.rec_loss = ChamferDistanceL2().cuda()  # Chamfer distance loss for measuring reconstruction quality

    def forward(self, pts, eval=False, vis = False, **kwargs):
        """
        Forward pass of the DSMAE model.
        Args:
            pts: Input point cloud data.
            eval: Whether to perform evaluation mode or training mode.
        Returns:
            Either the contrastive features or the combined loss (contrastive + reconstruction).
        """
        # Multi-scale representation extraction for point clouds using M2AE
        neighborhoods, centers, idxs = [], [], []
        for i, divider in enumerate(self.m2ae_group_dividers):
            # Perform multi-scale grouping; initial grouping uses original points
            if i == 0:
                neighborhood, center, idx = divider(pts)
            else:
                neighborhood, center, idx = divider(centers[-1])
            neighborhoods.append(neighborhood)
            centers.append(center)
            idxs.append(idx)
        
        # Perform additional grouping for MAE
        neighborhood, center, idx = self.mae_group_divider(pts, centers[-1])
        
        # Data augmentation using a transformation function
        neighborhoods_, centers_, neighborhood_, center_ = train_transform((neighborhoods, centers, neighborhood, center))
        
        if eval:
            # In evaluation mode, return the concatenated features from encoders
            x_list, _, _ = self.M2AE_encoder(neighborhoods, centers, idxs, eval=True)
            x_g, _ = self.MAE_encoder(neighborhood, center, eval=True)
            x_h = x_list[-1]
            concat_f = torch.cat([x_g.max(1)[0], x_h.max(1)[0]], dim=-1)  # Concatenate max-pooled features
            return concat_f
        
        # Multi-scale encoding using M2AE encoder
        x_vis_list, mask_vis_list, masks = self.M2AE_encoder(neighborhoods, centers, idxs)
        x_vis_list_, mask_vis_list_, masks_ = self.M2AE_encoder(neighborhoods_, centers_, idxs, bool_masked_pos=masks.copy())

        x_vis, mask = self.MAE_encoder(neighborhood, center, mask=masks[-1])
        x_vis_, mask_ = self.MAE_encoder(neighborhood_, center_, mask=mask)
        
        # Reverse lists for hierarchical decoding
        centers = list(reversed(centers))
        centers_ = list(reversed(centers_))
        x_vis_list = list(reversed(x_vis_list))
        x_vis_list_ = list(reversed(x_vis_list_))
        masks = list(reversed(masks))
        masks_ = list(reversed(masks_))
        neighborhoods = list(reversed(neighborhoods))
        neighborhoods_ = list(reversed(neighborhoods_))

        for i in range(len(self.decoder_dims)):
            center = centers[i]
            center_ = centers_[i]
            if i == 0:
                # First layer of hierarchical decoding
                x_vis_h, x_vis_h_, mask = x_vis_list[i], x_vis_list_[i], masks[i]
                B, _, C = x_vis.shape

                center_0 = torch.cat((center[~mask].reshape(B, -1, 3), center[mask].reshape(B, -1, 3)), dim=1)
                
                # Generate positional embeddings for visible and masked tokens
                pos_emd_vis = self.decoder_pos_embeds[i](center[~mask]).reshape(B, -1, C)
                pos_emd_mask = self.decoder_pos_embeds[i](center[mask]).reshape(B, -1, C)
                pos_emd_vis_ = self.decoder_pos_embeds[i](center_[~mask]).reshape(B, -1, C)
                pos_emd_mask_ = self.decoder_pos_embeds[i](center_[mask]).reshape(B, -1, C)
                
                # Concatenate visible and masked positional embeddings
                pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)
                pos_full_ = torch.cat([pos_emd_vis_, pos_emd_mask_], dim=1)
                
                # Expand mask tokens to match the batch size and number of masked points
                N = pos_emd_mask.shape[1]
                mask_token = self.mask_token.repeat(B, N, 1)
                mask_token_ = self.mask_token_.repeat(B, N, 1)
                
                # Concatenate visible features and mask tokens
                x_h = torch.cat([x_vis_h, mask_token], dim=1)
                x_h_ = torch.cat([x_vis_h_, mask_token_], dim=1)
                # global
                x_g = torch.cat([x_vis, mask_token], dim=1)
                x_g_ = torch.cat([x_vis_, mask_token_], dim=1)
                
                # Add positional embeddings to the tokens and pass through decoder block
                q_h = x_h + pos_full
                q_g = x_g + pos_full
                kv_h = x_h_ + pos_full_
                kv_g = x_g_ + pos_full_

                x_h = self.h_decoder[i](q_h, kv_h)
                x_h_ = self.h_decoder[i](q_h, kv_g)

                x_g = self.decoder(q_g, kv_g, N)
                x_g_ = self.decoder(q_g, kv_h, N)

            else:
                x_vis = x_vis_list[i]
                bool_vis_pos = ~masks[i]
                mask_vis = mask_vis_list[i]
                B, N, _ = center.shape
                _, _, C = x_vis.shape
                
                # Initialize an empty tensor for full encoded features
                x_full_en = torch.zeros(B, N, C).cuda()
                x_full_en[bool_vis_pos] = x_vis[mask_vis]  # Assign visible features to appropriate positions
                
                # Token propagation from previous layer
                if i == 1:
                    x_h = self.token_prop[i - 1](center, center_0, x_full_en, x_h)
                    x_h_ = self.token_prop[i - 1](center, center_0, x_full_en, x_h_)
                else:
                    x_h = self.token_prop[i - 1](center, centers[i - 1], x_full_en, x_h)
                    x_h_ = self.token_prop[i - 1](center, centers[i - 1], x_full_en, x_h_)
                
                # Add positional embeddings and pass through decoder block
                pos_full = self.decoder_pos_embeds[i](center)

                x = x_h + pos_full
                x_ = x_h_ + pos_full
                x_h = self.h_decoder[i](x, x_)
                x_h_ = self.h_decoder[i](x_, x)
        
        # Final reconstruction using hierarchical decoder normalization
        x_h = self.h_decoder_norm(x_h)
        x_h_ = self.h_decoder_norm(x_h_)
        B, N, C = x_h.shape

        x_h = x_h[masks[-2]].reshape(-1, C)
        x_h_ = x_h_[masks[-2]].reshape(-1, C)
        L, _ = x_h.shape
        
        # Predict reconstructed points from decoder output
        rec_points_h = self.rec_head_h(x_h.unsqueeze(-1)).reshape(L, -1, 3)
        rec_points_h_ = self.rec_head_h_(x_h_.unsqueeze(-1)).reshape(L, -1, 3)
        gt_points_h = neighborhoods[-2][masks[-2]].reshape(L, -1, 3)
        
        # Global reconstruction
        B, M, C = x_g.shape
        rec_points_g = self.rec_head_g(x_g.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)
        rec_points_g_ = self.rec_head_g_(x_g_.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)
        gt_points_g = neighborhood_[mask].reshape(B * M, -1, 3)
        
        # Compute Chamfer distance loss for reconstruction quality
        loss_rec_h = self.rec_loss(rec_points_h, gt_points_h) + self.rec_loss(rec_points_h_, gt_points_h)
        loss_rec_g = self.rec_loss(rec_points_g, gt_points_g) + self.rec_loss(rec_points_g_, gt_points_g)
        loss_rec = loss_rec_h + loss_rec_g
        
        # Return the combined contrastive and reconstruction loss
        if vis:  # Visualization mode
            B, G, S, _ = neighborhood.shape  # Get batch size, groups, group size, and channels

            gt = neighborhood + centers[0].unsqueeze(2)
            
            # M = int(mask.sum(dim=1)[0].item())
            # MAE branch
            mae_vis = neighborhood[~mask] + centers[0][~mask].unsqueeze(1)
            # 将 rec_points_g reshape 成 [B, -1, 3]
            mae_rebuild = rec_points_g + centers[0][mask].unsqueeze(1)
            mae_full = torch.cat([mae_vis.reshape(-1, 3), mae_rebuild.reshape(-1, 3)], dim=0)  # Fully reconstructed MAE

            # M2AE branch: Extract visible and masked points
            m2ae_vis = neighborhoods[-2][~masks[-2]] + centers[-2][~masks[-2]].unsqueeze(1)
            # m2ae_vis = neighborhoods[-1][~masks[-1]] + centers[-1][~masks[-1]].unsqueeze(1)
            m2ae_rebuild = rec_points_h + centers[-2][masks[-2]].unsqueeze(1)
            m2ae_full = torch.cat([m2ae_vis.reshape(-1, 3), m2ae_rebuild.reshape(-1, 3)], dim=0)  # Fully reconstructed M2AE

            return gt.reshape(-1, 3), mae_vis.reshape(-1, 3), \
                    mae_rebuild.reshape(-1, 3), mae_full.reshape(-1, 3), \
                    m2ae_vis.reshape(-1, 3), m2ae_rebuild.reshape(-1, 3), m2ae_full.reshape(-1, 3) , pts.reshape(-1, 3)
        else:
            return loss_rec




# finetune model
@MODELS.register_module()
class DSMAE_CLS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        print_log(f'[args] {config}', logger='DSMAE_CLS')
        
        # Extract configuration parameters
        self.trans_dim = config.trans_dim  # Transformer dimension size
        self.cls_dim = config.cls_dim # Dimension size of classification head
        
        # Patching setup for MAE and M2AE
        self.group_size = config.group_size  # Group size for MAE
        self.num_group = config.num_group  # Number of groups for MAE
        self.mae_group_divider = Group(num_group=self.num_group, group_size=self.group_size)  # Grouping for MAE
        print_log(f'[MAE] divide point cloud into G {self.num_group} x S {self.group_size} points ...', logger='MAE')
        
        # Patching setup for multi-scale M2AE
        self.group_sizes = config.group_sizes  # List of group sizes for M2AE
        self.num_groups = config.num_groups  # List of number of groups for M2AE
        self.m2ae_group_dividers = nn.ModuleList([
            Group(num_group=self.num_groups[i], group_size=self.group_sizes[i])
            for i in range(len(self.group_sizes))
        ])
        for i in range(len(self.group_sizes)):
            print_log(f'[M2AE] divide point cloud into G {self.num_groups[i]} x S {self.group_sizes[i]} points ...', logger='M2AE')
        
        # Encoder initialization
        self.MAE_encoder = MAE_Encoder(config)  # MAE encoder
        self.M2AE_encoder = M2AE_Encoder(config)  # M2AE encoder for hierarchical features
        
        if hasattr(config, 'type'):
            if config.type == "linear":
                self.cls_head_finetune = nn.Sequential(
                    nn.Linear(self.trans_dim * 2, self.cls_dim)
                )
                # raise ValueError
            else:
                self.cls_head_finetune = nn.Sequential(
                    nn.Linear(self.trans_dim * 2, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(256, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(256, self.cls_dim)
                )            
        else:    
            self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * 2, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
            )


    # def build_loss_func(self):
    #     self.loss_ce = nn.CrossEntropyLoss()

    # def get_loss_acc(self, ret, gt):
    #     loss = self.loss_ce(ret, gt.long())
    #     pred = ret.argmax(-1)
    #     acc = (pred == gt).sum() / float(gt.size(0))
    #     return loss, acc * 100
    
    # for smooth loss
        self.smooth = config.smooth
    
    def get_loss_acc(self, ret, gt):
        loss = self.smooth_loss(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def smooth_loss(self, pred, gt):
        eps = self.smooth
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gt.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
        return loss
        
    def load_model_from_ckpt(self, ckpt_path):
        print_log(f'[DSMAE_CLS] Load DSMAE pretrain from: {ckpt_path}', logger='DSMAE_CLS')
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder') :
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('M2AE_encoder'):
                    base_ckpt[k[len('M2AE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)
            if incompatible.missing_keys:
                print_log(
                    f'missing_keys:\n{get_missing_parameters_message(incompatible.missing_keys)}',
                    logger='DSMAE_CLS'
                )
            if incompatible.unexpected_keys:
                print_log(
                    f'unexpected_keys:\n{get_unexpected_parameters_message(incompatible.unexpected_keys)}',
                    logger='DSMAE_CLS'
                )
            print_log(f'[DSMAE_CLS] Successful Loading the ckpt from {ckpt_path}', logger='DSMAE_CLS')
        else:
            print_log('Training from scratch!!!', logger='DSMAE_CLS')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):
        # Multi-scale representation extraction for point clouds using M2AE
        neighborhoods, centers, idxs = [], [], []
        for i, divider in enumerate(self.m2ae_group_dividers):
            # Perform multi-scale grouping; initial grouping uses original points
            if i == 0:
                neighborhood, center, idx = divider(pts)
            else:
                neighborhood, center, idx = divider(centers[-1])
            neighborhoods.append(neighborhood)
            centers.append(center)
            idxs.append(idx)
        
        # Perform additional grouping for MAE
        neighborhood, center, idx = self.mae_group_divider(pts, centers[-1])
        
        # In evaluation mode, return the concatenated features from encoders
        x_list, _, _ = self.M2AE_encoder(neighborhoods, centers, idxs, eval=True)
        x_g, _ = self.MAE_encoder(neighborhood, center, eval=True)
        x_h = x_list[-1]
        concat_f = torch.cat([x_g.max(1)[0], x_h.max(1)[0]], dim=-1)  # Concatenate max-pooled features
        ret = self.cls_head_finetune(concat_f)
        return ret
        
        