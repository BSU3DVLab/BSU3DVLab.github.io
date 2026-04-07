"""
SkeletonX 风格的小样本分类框架 - 针对 Subject 级风险等级分类
核心组件：
1. 样本对构造模块（DASP/SADP）
2. 特征解耦模块（动作特征 vs Subject特征）
3. 跨样本特征聚合模块
4. 动作感知损失函数
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional
import random

# 导入ST-GCN工具函数
from model.stgcn import import_class, conv_init, bn_init, conv_branch_init, unit_tcn


class SubjectPairDataset(Dataset):
    """
    Pose级样本对数据集 - 构建 DASP 和 SADP 样本对
    DASP (Different Action Same Person): 同Subject不同Pose + 同风险标签
    SADP (Same Action Different Person): 同Pose不同Subject + 同风险标签
    
    数据结构: {subject_id: {pose_id: [T, V, C]}}
    """
    def __init__(self, subject_data: Dict, subject_labels: Dict,
                 use_DASP: bool = True, use_SADP: bool = True,
                 pairs_per_sample: int = 2,
                 transform=None, noise_std: float = 0.0,
                 seed: int = 408):
        """
        Args:
            subject_data: Dict[subject_id] -> Dict[pose_id] -> data [T, V, C]
            subject_labels: Dict[subject_id] -> risk_level (0=低, 1=中, 2=高)
            use_DASP: 是否使用DASP样本对
            use_SADP: 是否使用SADP样本对
            pairs_per_sample: 每个原始样本生成多少个配对样本
            transform: 数据增强变换
            noise_std: 高斯噪声标准差
            seed: 随机种子
        """
        self.subject_data = subject_data
        self.subject_labels = subject_labels
        self.transform = transform
        self.noise_std = noise_std
        self.use_DASP = use_DASP
        self.use_SADP = use_SADP
        self.pairs_per_sample = pairs_per_sample
        self.seed = seed
        
        # 获取所有subject和pose
        self.subject_ids = list(subject_data.keys())
        self.n_subject = len(self.subject_ids)
        
        # 获取每个subject的pose列表
        self.subject_poses = {}
        for sid in self.subject_ids:
            self.subject_poses[sid] = list(subject_data[sid].keys())
        
        # 构建所有样本索引 (subject_id, pose_id)
        self.samples = []
        for sid in self.subject_ids:
            for pid in self.subject_poses[sid]:
                self.samples.append((sid, pid))
        
        # 创建标签到subject的映射
        self.label_to_subjects = {}
        for sid in self.subject_ids:
            label = int(subject_labels[sid])
            if label not in self.label_to_subjects:
                self.label_to_subjects[label] = []
            self.label_to_subjects[label].append(sid)
        
        # 预生成样本对（但每次epoch会动态选择）
        self.pairs = self._generate_pairs()
        
        print(f"[SubjectPairDataset] 构建了 {len(self.pairs)} 个样本对 "
              f"(DASP: {sum(1 for p in self.pairs if p[2] == 'DASP')}, "
              f"SADP: {sum(1 for p in self.pairs if p[2] == 'SADP')})")
    
    def _generate_pairs(self):
        """动态生成样本对"""
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        pairs = []
        
        # 遍历每个样本，生成配对
        for sid_a, pid_a in self.samples:
            label_a = int(self.subject_labels[sid_a])
            
            # 生成 SADP 样本对（同Pose不同Subject）
            if self.use_SADP:
                # 找到同风险等级的其他subject
                same_label_subjects = [s for s in self.label_to_subjects[label_a] if s != sid_a]
                if same_label_subjects:
                    # 随机选择1个同风险subject
                    sid_b = random.choice(same_label_subjects)
                    # 确保该subject有这个pose
                    if pid_a in self.subject_data[sid_b]:
                        for _ in range(self.pairs_per_sample):
                            pairs.append((sid_a, pid_a, sid_b, pid_a, 'SADP'))
            
            # 生成 DASP 样本对（同Subject不同Pose）
            if self.use_DASP:
                # 找到同subject的其他pose
                other_poses = [p for p in self.subject_poses[sid_a] if p != pid_a]
                if other_poses:
                    # 随机选择1个其他pose
                    pid_b = random.choice(other_poses)
                    for _ in range(self.pairs_per_sample):
                        pairs.append((sid_a, pid_a, sid_a, pid_b, 'DASP'))
        
        random.shuffle(pairs)
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        sid_a, pid_a, sid_b, pid_b, pair_type = self.pairs[idx]
        
        # 获取数据
        data_a = self.subject_data[sid_a][pid_a].copy()
        data_b = self.subject_data[sid_b][pid_b].copy()
        
        # 数据增强
        if self.transform is not None:
            data_a = self.transform(data_a)
            data_b = self.transform(data_b)
        
        # 添加噪声
        if self.noise_std > 0:
            data_a = self._add_noise(data_a)
            data_b = self._add_noise(data_b)
        
        # 标签（使用subject标签）
        label_a = int(self.subject_labels[sid_a])
        label_b = int(self.subject_labels[sid_b])
        
        # 转换为张量 [C, T, V]
        if isinstance(data_a, np.ndarray):
            data_a = torch.from_numpy(data_a).float()
        if isinstance(data_b, np.ndarray):
            data_b = torch.from_numpy(data_b).float()
        
        return data_a, data_b, label_a, label_b, pair_type, f"{sid_a}_{pid_a}", f"{sid_b}_{pid_b}"
    
    def on_epoch_end(self):
        """每个epoch结束时重新生成样本对，增加多样性"""
        self.pairs = self._generate_pairs()
    
    def _add_noise(self, data):
        """添加高斯噪声"""
        if isinstance(data, np.ndarray):
            noise = np.random.normal(0, self.noise_std, data.shape)
            return np.clip(data + noise, -10, 10)
        else:
            noise = torch.randn_like(data) * self.noise_std
            return torch.clamp(data + noise, -10, 10)


class ST_DecoupleNet(nn.Module):
    """
    特征解耦网络 - 将特征拆分为"动作相关特征"和"Subject相关特征"
    动作特征（action_feat）: 捕捉动作模式，用于分类
    Subject特征（subject_feat）: 捕捉个体差异，用于多样性增强
    """
    def __init__(self, n_channel: int, n_frame: int, n_joint: int, n_person: int = 1):
        super().__init__()
        self.n_channel = n_channel
        self.n_frame = n_frame
        self.n_joint = n_joint
        self.n_person = n_person
        
        # 动作特征分支：专注于时序模式
        self.action_branch = nn.Sequential(
            nn.Conv2d(n_channel, n_channel // 2, kernel_size=1),
            nn.BatchNorm2d(n_channel // 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((n_frame, 1))  # 池化空间维度，保留时间
        )
        
        # Subject特征分支：专注于空间模式
        self.subject_branch = nn.Sequential(
            nn.Conv2d(n_channel, n_channel // 2, kernel_size=1),
            nn.BatchNorm2d(n_channel // 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, n_joint))  # 池化时间维度，保留空间
        )
    
    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feat: 输入特征 [N*M, C, T, V]
        Returns:
            subject_feat: Subject特征 [N*M, C//2, 1, V]
            action_feat: 动作特征 [N*M, C//2, T, 1]
        """
        # 解耦为两个特征
        subject_feat = self.subject_branch(feat)  # [N*M, C//2, 1, V]
        action_feat = self.action_branch(feat)    # [N*M, C//2, T, 1]
        
        return subject_feat, action_feat


class ST_FeatureAggrNet(nn.Module):
    """
    跨样本特征聚合网络 - 融合原始样本和配对样本的解耦特征
    使用拼接策略（concat），论文验证最稳定的无参聚合方式
    """
    def __init__(self, n_channel: int, aggr_mode: str = 'concat'):
        super().__init__()
        self.n_channel = n_channel
        self.aggr_mode = aggr_mode
        
        assert aggr_mode == 'concat', f"Only support concat mode, got {aggr_mode}"
    
    def forward(self, 
                feat_a: Tuple[torch.Tensor, torch.Tensor],
                feat_b: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feat_a: 原始样本的 (subject_feat, action_feat)
            feat_b: 配对样本的 (subject_feat, action_feat)
        Returns:
            cross_a: 聚合后的特征A [N, C]
            cross_b: 聚合后的特征B [N, C]
        """
        subject_feat_a, action_feat_a = feat_a
        subject_feat_b, action_feat_b = feat_b
        
        # 池化得到特征向量
        # subject_feat: [N*M, C//2, 1, V] -> [N, M, C//2] -> [N, C//2]
        subject_a_pooled = subject_feat_a.mean(-1).squeeze(-1)  # [N*M, C//2]
        subject_b_pooled = subject_feat_b.mean(-1).squeeze(-1)
        
        # action_feat: [N*M, C//2, T, 1] -> [N, M, C//2] -> [N, C//2]
        action_a_pooled = action_feat_a.squeeze(-1).mean(-1)   # [N*M, C//2]
        action_b_pooled = action_feat_b.squeeze(-1).mean(-1)
        
        # 拼接聚合策略
        # cross_a = subject_a + action_b (原始样本的subject + 配对样本的动作)
        # cross_b = subject_b + action_a (配对样本的subject + 原始样本的动作)
        cross_a = torch.cat([subject_a_pooled, action_b_pooled], dim=1)
        cross_b = torch.cat([subject_b_pooled, action_a_pooled], dim=1)
        
        return cross_a, cross_b


class LightSTGCNBackbone(nn.Module):
    """
    轻量级ST-GCN Backbone - 参数量控制在~800K
    适用于小样本学习场景
    """
    def __init__(self, in_channels: int, num_class: int = 3, 
                 num_point: int = 17, num_frame: int = 30, 
                 num_person: int = 1, graph: str = 'graph.ntu_rgb_d.Graph',
                 graph_args: dict = None, adaptive: bool = True,
                 num_set: int = 3):
        super().__init__()
        
        self.num_class = num_class
        self.num_point = num_point
        self.num_frame = num_frame
        self.num_person = num_person
        
        # 导入图结构
        if graph_args is None:
            graph_args = {'labeling_mode': 'spatial'}
        Graph = import_class(graph)
        self.graph = Graph(**graph_args)
        
        A = np.stack([np.eye(num_point)] * num_set, axis=0)
        
        # 轻量级通道配置
        base_channel = 32  # 从64减少到32
        
        # 数据标准化层
        self.data_bn = nn.BatchNorm1d(in_channels * num_point)
        
        # 精简的ST-GCN层：6层（从10层减少）
        # 通道变化: 32 -> 32 -> 32 -> 64 -> 64 -> 128
        self.stgcn_layers = nn.ModuleList([
            # 第1层: 3->32 (不使用残差)
            TCN_GCN_Lite(3, base_channel, A, residual=False, adaptive=adaptive),
            # 第2层: 32->32
            TCN_GCN_Lite(base_channel, base_channel, A, adaptive=adaptive),
            # 第3层: 32->32
            TCN_GCN_Lite(base_channel, base_channel, A, adaptive=adaptive),
            # 第4层: 32->64 (下采样)
            TCN_GCN_Lite(base_channel, base_channel * 2, A, stride=2, adaptive=adaptive),
            # 第5层: 64->64
            TCN_GCN_Lite(base_channel * 2, base_channel * 2, A, adaptive=adaptive),
            # 第6层: 64->128 (下采样)
            TCN_GCN_Lite(base_channel * 2, base_channel * 4, A, stride=2, adaptive=adaptive),
        ])
        
        # 全局池化 + 分类层
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.drop_out = nn.Dropout(0.5)
        self.fc = nn.Linear(base_channel * 4, num_class)
        
        # 初始化
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
    
    def forward(self, x: torch.Tensor, get_hidden_feat: bool = False) -> torch.Tensor:
        """
        Args:
            x: 输入 [N, C, T, V] 或 [N, T, V, C]
            get_hidden_feat: 是否返回隐藏特征
        Returns:
            输出或隐藏特征
        """
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        
        # 通过ST-GCN层
        for i, layer in enumerate(self.stgcn_layers):
            x = layer(x)
        
        # 全局池化
        x = self.global_pool(x)  # [N*M, C, 1, 1]
        x = x.view(N * M, -1)    # [N*M, C]
        
        if get_hidden_feat:
            return x  # 返回隐藏特征 [N*M, 128]
        
        # 分类
        x = self.drop_out(x)
        x = self.fc(x)
        return x
    
    def forward_to_hidden_feat(self, x: torch.Tensor) -> torch.Tensor:
        """获取隐藏特征"""
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        
        for i, layer in enumerate(self.stgcn_layers):
            x = layer(x)
        
        x = self.global_pool(x)
        return x.view(N * M, -1)
    
    def forward_hidden_feat(self, hidden_feat: torch.Tensor) -> torch.Tensor:
        """从隐藏特征到分类输出"""
        x = self.drop_out(hidden_feat)
        x = self.fc(x)
        return x


class TCN_GCN_Lite(nn.Module):
    """
    精简版TCN-GCN单元 - 减少参数量
    """
    def __init__(self, in_channels: int, out_channels: int, A: np.ndarray, 
                 stride: int = 1, residual: bool = True, adaptive: bool = True):
        super().__init__()
        
        self.gcn = unit_gcn_lite(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn = unit_tcn(out_channels, out_channels, kernel_size=5, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.tcn(self.gcn(x)) + self.residual(x))


class unit_gcn_lite(nn.Module):
    """
    精简版GCN单元 - 减少通道数和参数
    """
    def __init__(self, in_channels: int, out_channels: int, A: np.ndarray, adaptive: bool = True):
        super().__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = A.shape[0]
        self.adaptive = adaptive
        
        if adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        
        # 精简：只使用1个子集（原版使用3个）
        self.conv_d = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1)
        ])
        
        # 简化残差连接
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        conv_branch_init(self.conv_d[0], self.num_subset)
    
    def L2_norm(self, A):
        A_norm = torch.norm(A, 2, dim=1, keepdim=True) + 1e-4
        return A / A_norm
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V = x.size()
        
        if self.adaptive:
            A = self.PA
            A = self.L2_norm(A)
        else:
            A = self.A.cuda(x.get_device())
        
        # 简化：只使用1个子集
        A1 = A[0]  # 使用第一个邻接矩阵
        A2 = x.view(N, C * T, V)
        y = self.conv_d[0](torch.matmul(A2, A1).view(N, C, T, V))
        
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)
        
        return y


class SubjectFewShotClassifier(nn.Module):
    """
    基于SkeletonX的小样本Subject分类器 - 轻量版
    集成：LightST-GCN backbone + 特征解耦 + 跨样本聚合
    参数量控制在~1.0M
    """
    def __init__(self, input_channels: int, num_classes: int = 3,
                 n_frame: int = 30, n_joint: int = 17,
                 gcn_model: nn.Module = None,
                 use_DASP: bool = True, use_SADP: bool = True,
                 aggr_mode: str = 'concat',
                 drop_rate: float = 0.5,
                 use_light_gcn: bool = True):  # 默认使用轻量版
        super().__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.use_DASP = use_DASP
        self.use_SADP = use_SADP
        
        # 1. ST-GCN Backbone (使用轻量版)
        if gcn_model is not None:
            self.gcn_model = gcn_model
        elif use_light_gcn:
            # 使用轻量级ST-GCN Backbone (参数量约~800K)
            self.gcn_model = LightSTGCNBackbone(
                in_channels=input_channels,
                num_class=num_classes,
                num_point=n_joint,
                num_frame=n_frame,
                num_person=1,
                graph='graph.ntu_rgb_d.Graph',
                graph_args={'labeling_mode': 'spatial'},
                adaptive=True
            )
        else:
            # 使用原始ST-GCN Backbone
            try:
                from model.stgcn import Model as STGCNModel
            except (ModuleNotFoundError, ImportError):
                import sys
                import os
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from model.stgcn import Model as STGCNModel
            
            self.gcn_model = STGCNModel(
                num_class=num_classes,
                num_point=n_joint,
                num_frame=n_frame,
                num_person=1,
                graph='graph.ntu_rgb_d.Graph',
                graph_args={'labeling_mode': 'spatial'},
                in_channels=input_channels,
                adaptive=True
            )
        
        # 获取GCN输出通道数
        gcn_output_dim = self.gcn_model.fc.in_features if hasattr(self.gcn_model, 'fc') else 128
        
        # 2. 特征解耦模块 (适配输出维度)
        self.decouple_net = ST_DecoupleNet(
            n_channel=gcn_output_dim,
            n_frame=n_frame,
            n_joint=n_joint
        )
        
        # 3. 跨样本特征聚合模块
        self.feat_aggr_net = ST_FeatureAggrNet(
            n_channel=gcn_output_dim,
            aggr_mode=aggr_mode
        )
        
        # 4. 分类头
        decoupled_dim = gcn_output_dim  # subject + action 各一半，拼接后恢复原维度
        
        self.classifier = nn.Sequential(
            nn.Linear(decoupled_dim, decoupled_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(decoupled_dim // 2, num_classes)
        )
        
        # 5. 相似度损失权重
        self.w_DA = 0.1
        self.w_DS = 0.1
        
    def forward_to_hidden_feat(self, x: torch.Tensor) -> torch.Tensor:
        """获取GCN隐藏特征 - 兼容两种backbone，返回4D特征用于decouple_net"""
        # 检测输入格式并进行转换
        if x.dim() == 4:
            N, C, T, V = x.shape
            # 转换为 [N, C, T, V, M]
            x = x.unsqueeze(-1)  # [N, C, T, V, M], M=1
        
        # 如果已经是 5D [N, C, T, V, M]，直接使用
        
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.gcn_model.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        
        # 通过ST-GCN层，返回4D特征用于decouple_net
        for i, layer in enumerate(self.gcn_model.stgcn_layers):
            x = layer(x)
        
        # 返回4D特征 [N*M, C, T, V]
        return x
    
    def forward_hidden_feat(self, hidden_feat: torch.Tensor) -> torch.Tensor:
        """从隐藏特征到分类输出 - 兼容两种backbone"""
        # 兼容两种backbone
        if hasattr(self.gcn_model, 'forward_hidden_feat'):
            return self.gcn_model.forward_hidden_feat(hidden_feat)
        else:
            # 原始ST-GCN需要展平再通过fc
            x = self.gcn_model.drop_out(hidden_feat)
            x = self.gcn_model.fc(x)
            return x
    
    def forward(self, x: torch.Tensor,
                x_dasp: Optional[torch.Tensor] = None,
                x_sadp: Optional[torch.Tensor] = None,
                label: Optional[torch.Tensor] = None,
                label_dasp: Optional[torch.Tensor] = None,
                label_sadp: Optional[torch.Tensor] = None,
                eval: bool = False) -> Dict:
        """
        前向传播
        Args:
            x: 原始样本 [N, C, T, V]
            x_dasp: DASP配对样本 [N, C, T, V]
            x_sadp: SADP配对样本 [N, C, T, V]
            label: 原始样本标签
            label_dasp: DASP配对样本标签
            label_sadp: SADP配对样本标签
            eval: 评估模式
        Returns:
            output: 包含各类输出和损失的字典
        """
        output = {}
        losses = {}
        
        # 1. 原始样本处理
        hidden_feat = self.forward_to_hidden_feat(x)  # [N, C, T, V]
        subject_feat, action_feat = self.decouple_net(hidden_feat)
        
        # 原始样本特征
        feat_origin = torch.cat([
            subject_feat.mean(-1).squeeze(),  # [N, C//2]
            action_feat.squeeze(-1).mean(-1)   # [N, C//2]
        ], dim=1)  # [N, C]
        
        output['x'] = self.forward_hidden_feat(feat_origin)  # 分类logits
        output['feat_origin'] = feat_origin
        
        if eval:
            return output
        
        # 计算原始样本损失
        if label is not None:
            losses['loss_origin'] = F.cross_entropy(output['x'], label)
            output['acc_origin'] = (output['x'].argmax(dim=1) == label).float().mean()
        
        # 2. DASP样本对处理（同Subject不同风险）
        if x_dasp is not None and self.use_DASP:
            hidden_feat_dasp = self.forward_to_hidden_feat(x_dasp)
            subject_feat_dasp, action_feat_dasp = self.decouple_net(hidden_feat_dasp)
            
            # 跨样本聚合
            cross_a, cross_b = self.feat_aggr_net(
                (subject_feat, action_feat),
                (subject_feat_dasp, action_feat_dasp)
            )
            
            # DASP原始样本输出
            feat_dasp_origin = torch.cat([
                subject_feat_dasp.mean(-1).squeeze(),
                action_feat_dasp.squeeze(-1).mean(-1)
            ], dim=1)
            output['x_dasp'] = self.forward_hidden_feat(feat_dasp_origin)
            output['x_dasp_cross_a'] = self.forward_hidden_feat(cross_a)
            output['x_dasp_cross_b'] = self.forward_hidden_feat(cross_b)
            
            # DASP损失
            if label_dasp is not None:
                losses['loss_DASP_origin'] = F.cross_entropy(output['x_dasp'], label_dasp)
                losses['loss_DASP_cross_a'] = F.cross_entropy(output['x_dasp_cross_a'], label)
                losses['loss_DASP_cross_b'] = F.cross_entropy(output['x_dasp_cross_b'], label_dasp)
                
                output['acc_dasp'] = (output['x_dasp'].argmax(dim=1) == label_dasp).float().mean()
            
            # 动作相似性损失：同Subject的动作特征应该相似
            action_sim = F.cosine_similarity(
                action_feat.squeeze(-1).mean(-1),
                action_feat_dasp.squeeze(-1).mean(-1),
                dim=1
            )
            losses['loss_action_sim'] = self.w_DA * (1 - action_sim).mean()
            
            output['action_sim'] = action_sim.mean()
        
        # 3. SADP样本对处理（同风险不同Subject）
        if x_sadp is not None and self.use_SADP:
            hidden_feat_sadp = self.forward_to_hidden_feat(x_sadp)
            subject_feat_sadp, action_feat_sadp = self.decouple_net(hidden_feat_sadp)
            
            # 跨样本聚合
            cross_a, cross_b = self.feat_aggr_net(
                (subject_feat, action_feat),
                (subject_feat_sadp, action_feat_sadp)
            )
            
            # SADP原始样本输出
            feat_sadp_origin = torch.cat([
                subject_feat_sadp.mean(-1).squeeze(),
                action_feat_sadp.squeeze(-1).mean(-1)
            ], dim=1)
            output['x_sadp'] = self.forward_hidden_feat(feat_sadp_origin)
            output['x_sadp_cross_a'] = self.forward_hidden_feat(cross_a)
            output['x_sadp_cross_b'] = self.forward_hidden_feat(cross_b)
            
            # SADP损失
            if label_sadp is not None:
                losses['loss_SADP_origin'] = F.cross_entropy(output['x_sadp'], label_sadp)
                losses['loss_SADP_cross_a'] = F.cross_entropy(output['x_sadp_cross_a'], label)
                losses['loss_SADP_cross_b'] = F.cross_entropy(output['x_sadp_cross_b'], label_sadp)
                
                output['acc_sadp'] = (output['x_sadp'].argmax(dim=1) == label_sadp).float().mean()
            
            # Subject相似性损失：同风险等级的Subject特征应该相似
            subject_sim = F.cosine_similarity(
                subject_feat.mean(-1).squeeze(),
                subject_feat_sadp.mean(-1).squeeze(),
                dim=1
            )
            losses['loss_subject_sim'] = self.w_DS * (1 - subject_sim).mean()
            
            output['subject_sim'] = subject_sim.mean()
        
        # 总损失
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        output['losses'] = losses
        
        return output
    
    def train(self, mode: bool = True):
        super().train(mode)
        self.gcn_model.train(mode)
    
    def eval(self):
        super().eval()
        self.gcn_model.eval()


def create_pair_collate_fn(use_DASP: bool = True, use_SADP: bool = True):
    """
    创建样本对批次整理函数 - 适配pose级别数据
    确保原始样本、DASP样本和SADP样本的批次大小一致
    """
    def collate_fn(batch):
        # batch: list of tuples (data_a, data_b, label_a, label_b, pair_type, key_a, key_b)
        batch_size = len(batch)
        
        # 分离不同类型的样本对
        dasp_samples = []
        sadp_samples = []
        original_samples = []
        
        for data_a, data_b, label_a, label_b, pair_type, key_a, key_b in batch:
            if pair_type == 'DASP':
                dasp_samples.append((data_a, data_b, label_a, label_b, key_a, key_b))
            else:  # 'SADP'
                sadp_samples.append((data_a, data_b, label_a, label_b, key_a, key_b))
        
        # 收集所有唯一的原始样本
        sample_keys = set()
        for samples_list in [dasp_samples, sadp_samples]:
            for data_a, data_b, _, _, key_a, key_b in samples_list:
                sample_keys.add(key_a)
                sample_keys.add(key_b)
        
        # 创建key到原始样本的映射
        key_to_original = {}
        for samples_list in [dasp_samples, sadp_samples]:
            for data_a, data_b, _, _, key_a, key_b in samples_list:
                if key_a not in key_to_original:
                    key_to_original[key_a] = (data_a, key_a)
                if key_b not in key_to_original:
                    key_to_original[key_b] = (data_b, key_b)
        
        n_original = len(key_to_original)
        
        # 填充DASP和SADP列表
        if use_DASP and len(dasp_samples) > 0:
            while len(dasp_samples) < n_original:
                dasp_samples.append(dasp_samples[0])  # 重复第一个样本
        
        if use_SADP and len(sadp_samples) > 0:
            while len(sadp_samples) < n_original:
                sadp_samples.append(sadp_samples[0])  # 重复第一个样本
        
        # 构建最终批次
        original_data = []
        original_labels = []
        dasp_data = []
        sadp_data = []
        
        for key, (data, _) in key_to_original.items():
            original_data.append(data)
            original_labels.append(int(batch[0][2]))  # 使用任意一个label
        
        # 处理DASP批次
        if use_DASP and len(dasp_samples) > 0:
            dasp_data = [s[0] for s in dasp_samples[:n_original]]
        
        # 处理SADP批次
        if use_SADP and len(sadp_samples) > 0:
            sadp_data = [s[0] for s in sadp_samples[:n_original]]
        
        # 堆叠张量
        def stack_tensors(tensor_list):
            if not tensor_list:
                return None
            # 获取第一个张量的尺寸
            first = tensor_list[0]
            shape = tuple(first.shape)  # [C, T, V]
            dtype = first.dtype
            device = first.device
            
            # 创建零张量
            result = torch.zeros((len(tensor_list),) + shape, dtype=dtype, device=device)
            for i, t in enumerate(tensor_list):
                result[i] = t
            
            # 转换为 [N, C, T, V] 格式（原始是 [N, T, V, C]）
            if len(shape) == 3 and shape[0] != shape[1]:  # [C, T, V] -> [N, C, T, V]
                result = result.permute(0, 3, 1, 2)  # [N, T, V, C] -> [N, C, T, V]
            
            return result
        
        batch_dict = {
            'original': {
                'data': stack_tensors(original_data),
                'label': torch.tensor(original_labels, dtype=torch.long)
            }
        }
        
        if use_DASP and len(dasp_data) > 0:
            batch_dict['dasp'] = {
                'data': stack_tensors(dasp_data),
                'label': torch.tensor([s[2] for s in dasp_samples[:n_original]], dtype=torch.long)
            }
        
        if use_SADP and len(sadp_data) > 0:
            batch_dict['sadp'] = {
                'data': stack_tensors(sadp_data),
                'label': torch.tensor([s[2] for s in sadp_samples[:n_original]], dtype=torch.long)
            }
        
        return batch_dict
    
    return collate_fn


class SimpleSubjectDataset(Dataset):
    """简单Subject数据集"""
    def __init__(self, subject_data: Dict, subject_labels: np.ndarray, transform=None, noise_std: float = 0.0):
        self.subject_ids = list(subject_data.keys())
        self.subject_data = subject_data
        self.subject_labels = subject_labels
        self.transform = transform
        self.noise_std = noise_std
    
    def __len__(self):
        return len(self.subject_ids)
    
    def __getitem__(self, idx):
        sid = self.subject_ids[idx]
        data = self.subject_data[sid].copy()
        label = int(self.subject_labels[sid])
        
        if self.transform is not None:
            data = self.transform(data)
        
        if self.noise_std > 0:
            data = self._add_noise(data)
        
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        
        return data, label, sid
    
    def _add_noise(self, data):
        if isinstance(data, np.ndarray):
            noise = np.random.normal(0, self.noise_std, data.shape)
            return np.clip(data + noise, -10, 10)
        else:
            noise = torch.randn_like(data) * self.noise_std
            return torch.clamp(data + noise, -10, 10)


def build_fewshot_model(config: Dict, train_data: Dict, train_labels: np.ndarray,
                        gcn_model: nn.Module = None) -> SubjectFewShotClassifier:
    """
    构建小样本模型
    
    Args:
        config: 配置字典
        train_data: 训练数据 - 格式为 {subject_id: [T, V, C]}
        train_labels: 训练标签
        gcn_model: 预训练的GCN模型
    
    Returns:
        model: SubjectFewShotClassifier实例
    """
    # 获取数据维度信息
    # train_data格式: {subject_id: {pose_id: data [T, V, C]}}
    first_subject = list(train_data.keys())[0]
    sample_data = train_data[first_subject]
    
    # 获取数据的形状
    if isinstance(sample_data, np.ndarray):
        data_shape = sample_data.shape  # [T, V, C]
        n_frame = data_shape[0]     # T
        n_joint = data_shape[1]     # V
        n_channel = data_shape[2]   # C
    elif isinstance(sample_data, dict):
        # 嵌套字典格式: {pose_id: data [T, V, C]}
        first_pose = list(sample_data.keys())[0]
        pose_data = sample_data[first_pose]
        data_shape = pose_data.shape
        n_frame = data_shape[0]
        n_joint = data_shape[1]
        n_channel = data_shape[2]
    else:
        # 兜底处理
        data_shape = sample_data.shape
        n_channel, n_frame, n_joint = data_shape
    
    print(f"[build_fewshot_model] 数据维度推断: C={n_channel}, T={n_frame}, V={n_joint}")
    
    # 创建模型
    model = SubjectFewShotClassifier(
        input_channels=n_channel,
        num_classes=config.get('num_classes', 3),
        n_frame=n_frame,
        n_joint=n_joint,
        gcn_model=gcn_model,
        use_DASP=config.get('use_DASP', True),
        use_SADP=config.get('use_SADP', True),
        aggr_mode=config.get('aggr_mode', 'concat'),
        drop_rate=config.get('drop_rate', 0.5)
    )
    
    print(f"[build_fewshot_model] 创建模型，参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def build_fewshot_dataloader(train_data: Dict, train_labels: np.ndarray,
                             batch_size: int = 16, shuffle: bool = True,
                             use_DASP: bool = True, use_SADP: bool = True,
                             num_workers: int = 0, **kwargs) -> DataLoader:
    """
    构建小样本DataLoader
    """
    dataset = SubjectPairDataset(
        subject_data=train_data,
        subject_labels=train_labels,
        use_DASP=use_DASP,
        use_SADP=use_SADP,
        noise_std=kwargs.get('noise_std', 0.01)
    )
    
    collate_fn = create_pair_collate_fn(use_DASP=use_DASP, use_SADP=use_SADP)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    return dataloader


def train_fewshot_epoch(model: nn.Module, dataloader: DataLoader,
                        optimizer: torch.optim.Optimizer,
                        device: torch.device) -> Dict:
    """
    训练一个epoch
    """
    model.train()
    total_losses = []
    total_accs = []
    
    for batch_idx, batch in enumerate(dataloader):
        # 解析批次数据 - collate_fn返回字典格式
        original = batch['original']
        if isinstance(original, dict):
            original_data = original['data']  # [batch_size, C, T, V]
            original_labels = original['label']
        else:
            original_data, original_labels = original
        # 数据格式已经是 [batch_size, C, T, V]，直接移动到设备
        original_data = original_data.to(device)
        original_labels = original_labels.to(device)
        
        # 获取配对数据
        x_dasp, label_dasp_a, label_dasp_b = None, None, None
        x_sadp, label_sadp_a, label_sadp_b = None, None, None
        
        if batch.get('dasp') is not None:
            dasp = batch['dasp']
            if isinstance(dasp, dict):
                x_dasp = dasp['data']  # [batch_size, C, T, V]
                label_dasp_a = dasp['label'].to(device)
            else:
                dasp_a, dasp_labels_a, dasp_labels_b = dasp
                x_dasp = dasp_a.to(device)
                label_dasp_a = dasp_labels_a.to(device)
                label_dasp_b = dasp_labels_b.to(device)
        
        if batch.get('sadp') is not None:
            sadp = batch['sadp']
            if isinstance(sadp, dict):
                x_sadp = sadp['data']  # [batch_size, C, T, V]
                label_sadp_a = sadp['label'].to(device)
            else:
                sadp_a, sadp_labels_a, sadp_labels_b = sadp
                x_sadp = sadp_a.to(device)
                label_sadp_a = sadp_labels_a.to(device)
                label_sadp_b = sadp_labels_b.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(
            x=original_data,
            x_dasp=x_dasp,
            x_sadp=x_sadp,
            label=original_labels,
            label_dasp=label_dasp_a,
            label_sadp=label_sadp_a
        )
        
        # 反向传播
        loss = output['losses']['total_loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_losses.append(loss.item())
        if 'acc_origin' in output:
            total_accs.append(output['acc_origin'].item())
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}: Loss={loss.item():.4f}, "
                  f"Acc={output.get('acc_origin', 0):.4f}")
    
    avg_loss = np.mean(total_losses)
    avg_acc = np.mean(total_accs) if total_accs else 0
    
    return {'loss': avg_loss, 'acc': avg_acc}


def evaluate_fewshot_model(model: nn.Module, test_data: Dict, test_labels: np.ndarray,
                           device: torch.device, batch_size: int = 16) -> Dict:
    """
    评估小样本模型
    """
    model.eval()
    
    # 创建测试数据集
    test_dataset = SimpleSubjectDataset(
        subject_data=test_data,
        subject_labels=test_labels,
        noise_std=0.0
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels, sids in test_loader:
            data = data.to(device)
            
            # 前向传播
            output = model(x=data, eval=True)
            preds = output['x'].argmax(dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # 计算指标
    accuracy = (all_preds == all_labels).mean()
    
    # 分类报告
    from sklearn.metrics import classification_report
    report = classification_report(all_labels, all_preds, 
                                   target_names=['低风险', '中风险', '高风险'],
                                   output_dict=True)
    
    return {
        'accuracy': accuracy,
        'predictions': all_preds,
        'labels': all_labels,
        'report': report
    }


if __name__ == "__main__":
    # 简单测试
    print("=== SubjectFewShotClassifier 模块测试 ===")
    
    # 创建模拟数据
    n_subject = 33
    n_frame = 30
    n_joint = 17
    n_channel = 3
    
    # 模拟subject数据
    subject_data = {
        i: np.random.randn(n_channel, n_frame, n_joint).astype(np.float32)
        for i in range(n_subject)
    }
    
    # 模拟标签（低/中/高风险）
    subject_labels = np.array([i % 3 for i in range(n_subject)])
    
    # 测试数据集
    print("\n1. 测试SubjectPairDataset...")
    pair_dataset = SubjectPairDataset(
        subject_data=subject_data,
        subject_labels=subject_labels,
        use_DASP=True,
        use_SADP=True,
        noise_std=0.01
    )
    
    # 测试模型
    print("\n2. 测试SubjectFewShotClassifier...")
    model = SubjectFewShotClassifier(
        input_channels=n_channel,
        num_classes=3,
        n_frame=n_frame,
        n_joint=n_joint
    )
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   模型参数量: {total_params:,}")
    print(f"   GCN backbone参数量: {sum(p.numel() for p in model.gcn_model.parameters()):,}")
    
    # 测试前向传播
    print("\n3. 测试前向传播...")
    test_input = torch.randn(8, n_channel, n_frame, n_joint)
    output = model(x=test_input, eval=True)
    print(f"   输出形状: {output['x'].shape}")
    
    print("\n4. 测试带样本对的前向传播...")
    output = model(
        x=test_input,
        x_dasp=torch.randn(8, n_channel, n_frame, n_joint),
        x_sadp=torch.randn(8, n_channel, n_frame, n_joint),
        label=torch.randint(0, 3, (8,)),
        label_dasp=torch.randint(0, 3, (8,)),
        label_sadp=torch.randint(0, 3, (8,))
    )
    print(f"   损失项: {list(output['losses'].keys())}")
    
    print("\n=== 测试完成 ===")
