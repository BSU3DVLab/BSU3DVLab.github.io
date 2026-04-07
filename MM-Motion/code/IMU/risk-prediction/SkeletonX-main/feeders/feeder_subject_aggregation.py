"""
受试者级(Subject-Level)数据聚合模块
将pose级别的样本聚合为subject级别，每个受试者的所有pose数据聚合为1个特征向量
"""

import numpy as np
import torch
from torch.utils.data import Dataset


def extract_statistical_features(pose_sequence):
    """
    从pose序列中提取统计特征
    
    Args:
        pose_sequence: numpy array of shape (C, T, V, M) 或 (C, T, V)
    
    Returns:
        feature_vector: 聚合后的统计特征向量
    """
    # 处理多人的情况，取第一个人的数据
    if pose_sequence.ndim == 4:
        # C, T, V, M -> 取 M=0
        pose_sequence = pose_sequence[:, :, :, 0]
    
    C, T, V = pose_sequence.shape
    features = []
    
    # 对每个通道、每个关节点提取统计特征
    for c in range(C):
        for v in range(V):
            channel_data = pose_sequence[c, :, v]
            
            # 时序统计特征
            mean_val = np.mean(channel_data)
            std_val = np.std(channel_data)
            max_val = np.max(channel_data)
            min_val = np.min(channel_data)
            median_val = np.median(channel_data)
            
            # 差分特征（反映变化率）
            diff = np.diff(channel_data)
            diff_mean = np.mean(diff) if len(diff) > 0 else 0
            diff_std = np.std(diff) if len(diff) > 0 else 0
            
            # 能量特征
            energy = np.sum(channel_data ** 2)
            
            # 极值比率
            range_val = max_val - min_val
            
            features.extend([
                mean_val, std_val, max_val, min_val, 
                median_val, diff_mean, diff_std, energy, range_val
            ])
    
    return np.array(features)


def aggregate_subject_features(data, labels, subject_ids):
    """
    将pose级别数据聚合为subject级别
    
    Args:
        data: pose数据，shape (N, C, T, V, M)
        labels: 标签，shape (N,)
        subject_ids: 每个样本对应的subject ID，shape (N,)
    
    Returns:
        subject_data: 聚合后的subject数据，shape (S, C*T*V*9) 其中9是统计特征数
        subject_labels: 每个subject的标签，shape (S,)
        subject_ids_unique: 唯一的subject ID列表
    """
    unique_subjects = np.unique(subject_ids)
    n_features = data.shape[1] * data.shape[3] * 9  # C * V * 9个统计特征
    
    subject_data = []
    subject_labels = []
    
    for subj_id in unique_subjects:
        # 获取该subject的所有样本索引
        mask = subject_ids == subj_id
        subj_data = data[mask]
        subj_labels = labels[mask]
        
        # 聚合该subject的所有pose数据
        aggregated_features = []
        for i in range(len(subj_data)):
            pose_features = extract_statistical_features(subj_data[i])
            aggregated_features.append(pose_features)
        
        # 计算所有pose的特征均值作为subject特征
        subject_feature = np.mean(aggregated_features, axis=0)
        
        # 该subject的标签取众数（出现最多的类别）
        from scipy import stats
        subject_label = int(stats.mode(subj_labels, keepdims=True)[0][0])
        
        subject_data.append(subject_feature)
        subject_labels.append(subject_label)
    
    return np.array(subject_data), np.array(subject_labels), unique_subjects


class SubjectLevelDataset(Dataset):
    """
    受试者级数据集
    每个样本代表一个受试者，特征由该受试者的所有pose聚合而成
    """
    
    def __init__(self, data_path, label_path=None, split='train', 
                 normalization=False, debug=False, use_mmap=False,
                 noise_augmentation=0.0, subject_info_path=None):
        """
        Args:
            data_path: .npz数据文件路径
            label_path: 标签文件路径（可选）
            split: 数据分割 ('train', 'test')
            normalization: 是否归一化
            debug: 调试模式，只使用前100个样本
            use_mmap: 是否使用内存映射
            noise_augmentation: 高斯噪声强度
            subject_info_path: subject信息文件路径
        """
        self.debug = debug
        self.data_path = data_path
        self.split = split
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.noise_augmentation = noise_augmentation
        
        # 加载数据
        self.load_data()
        
        # 加载或生成subject信息
        if subject_info_path is not None and os.path.exists(subject_info_path):
            self.load_subject_info(subject_info_path)
        else:
            self.generate_synthetic_subject_info()
        
        # 聚合为subject级别
        self.aggregate_to_subject_level()
        
        if normalization:
            self.normalize()
    
    def load_data(self):
        """加载原始pose级别数据"""
        npz_data = np.load(self.data_path, mmap_mode='r' if self.use_mmap else None)
        
        if self.split == 'train':
            self.pose_data = npz_data['x_train']
            self.pose_labels = np.where(npz_data['y_train'] > 0)[1]
        elif self.split == 'test':
            self.pose_data = npz_data['x_test']
            self.pose_labels = np.where(npz_data['y_test'] > 0)[1]
        else:
            raise NotImplementedError('Only train/test splits supported')
        
        if self.debug:
            self.pose_data = self.pose_data[:100]
            self.pose_labels = self.pose_labels[:100]
        
        print(f"[SubjectDataset] Loaded {len(self.pose_data)} pose samples for {self.split}")
    
    def generate_synthetic_subject_info(self):
        """
        生成合成的subject信息
        由于没有真实的subject ID，我们假设连续的同类别样本属于同一个subject
        每个subject包含多个pose样本
        """
        n_samples = len(self.pose_labels)
        
        # 计算需要多少个subject（每个subject大约5-20个pose）
        min_poses_per_subject = 5
        max_poses_per_subject = 20
        
        self.pose_subject_ids = np.zeros(n_samples, dtype=int)
        self.subject_ids = []
        
        current_subject = 0
        i = 0
        while i < n_samples:
            # 每个subject随机分配5-20个pose
            n_poses = np.random.randint(min_poses_per_subject, max_poses_per_subject + 1)
            end_idx = min(i + n_poses, n_samples)
            
            # 确保同一subject内的样本标签一致
            target_label = self.pose_labels[i]
            
            for j in range(i, end_idx):
                self.pose_subject_ids[j] = current_subject
                self.subject_ids.append(current_subject)
            
            current_subject += 1
            i = end_idx
        
        self.n_subjects = current_subject
        
        print(f"[SubjectDataset] Generated {self.n_subjects} synthetic subjects")
        print(f"[SubjectDataset] Subjects per class: ", end="")
        
        for cls in range(3):
            n_subj = len(set(self.pose_subject_ids[self.pose_labels == cls]))
            print(f"Class {cls}: {n_subj}, ", end="")
        print()
    
    def aggregate_to_subject_level(self):
        """将pose级别数据聚合为subject级别"""
        from tqdm import tqdm
        
        print(f"[SubjectDataset] Aggregating {len(self.pose_data)} poses to {self.n_subjects} subjects...")
        
        n_features = self.pose_data.shape[1] * self.pose_data.shape[3] * 9  # C * V * 9
        self.subject_data = np.zeros((self.n_subjects, n_features))
        self.subject_labels = np.zeros(self.n_subjects, dtype=int)
        
        for subj_id in tqdm(range(self.n_subjects)):
            mask = self.pose_subject_ids == subj_id
            subj_poses = self.pose_data[mask]
            
            # 聚合该subject所有pose的特征
            pose_features = []
            for pose in subj_poses:
                features = extract_statistical_features(pose)
                pose_features.append(features)
            
            # 取均值作为subject特征
            self.subject_data[subj_id] = np.mean(pose_features, axis=0)
            
            # subject标签取众数
            from scipy import stats
            self.subject_labels[subj_id] = int(stats.mode(self.pose_labels[mask], keepdims=True)[0][0])
        
        print(f"[SubjectDataset] Aggregated data shape: {self.subject_data.shape}")
        print(f"[SubjectDataset] Subject labels distribution: {np.bincount(self.subject_labels)}")
    
    def normalize(self):
        """对subject级别数据进行归一化"""
        self.mean = self.subject_data.mean(axis=0, keepdims=True)
        self.std = self.subject_data.std(axis=0, keepdims=True) + 1e-8
        self.subject_data = (self.subject_data - self.mean) / self.std
    
    def add_noise(self, data):
        """添加高斯噪声进行数据增强"""
        if self.noise_augmentation > 0:
            noise = np.random.normal(0, self.noise_augmentation, data.shape)
            data = data + noise
        return data
    
    def __len__(self):
        return len(self.subject_data)
    
    def __getitem__(self, index):
        feature = self.subject_data[index].astype(np.float32)
        
        # 添加高斯噪声（仅训练集）
        if self.split == 'train' and self.noise_augmentation > 0:
            feature = self.add_noise(feature)
        
        label = self.subject_labels[index]
        return torch.FloatTensor(feature), label, index


def create_subject_dataset(data_path, split='train', noise_augmentation=0.1, batch_size=16):
    """
    便捷函数：创建subject级别数据集和DataLoader
    
    Args:
        data_path: 数据文件路径
        split: 'train' 或 'test'
        noise_augmentation: 高斯噪声强度
        batch_size: 批次大小
    
    Returns:
        dataloader: PyTorch DataLoader
    """
    from torch.utils.data import DataLoader
    
    dataset = SubjectLevelDataset(
        data_path=data_path,
        split=split,
        normalization=True,
        noise_augmentation=noise_augmentation
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        drop_last=(split == 'train' and len(dataset) > batch_size)
    )
    
    return dataloader


if __name__ == "__main__":
    # 测试subject级别数据聚合
    import os
    
    data_path = "work_dir/imu_data/imu_data.npz"
    
    print("=" * 60)
    print("测试受试者级数据聚合模块")
    print("=" * 60)
    
    # 创建训练集
    train_loader = create_subject_dataset(
        data_path=data_path,
        split='train',
        noise_augmentation=0.1,
        batch_size=8
    )
    
    print(f"\n训练集批次数: {len(train_loader)}")
    
    # 遍历一个批次
    for batch_idx, (features, labels, indices) in enumerate(train_loader):
        print(f"\n批次 {batch_idx}:")
        print(f"  特征形状: {features.shape}")
        print(f"  标签形状: {labels.shape}")
        print(f"  标签分布: {labels.tolist()}")
        break
    
    print("\n模块测试完成!")
