import torch
import numpy as np
from torch.utils.data import DataLoader
from model.subject_fewshot import SubjectPairDataset, create_pair_collate_fn, SubjectFewShotClassifier

# 创建模拟数据
n_subjects = 15
n_frame = 8
n_joint = 21
n_channel = 3

train_data = {}
train_labels = {}

for i in range(n_subjects):
    subject_id = f'subject_{i}'
    label = i % 3
    base_data = np.random.randn(n_frame, n_joint, n_channel).astype(np.float32)
    train_data[subject_id] = base_data
    train_labels[subject_id] = label

print('数据格式:')
print(f'  train_data[subject_0] shape: {train_data["subject_0"].shape}')
print(f'  (应该是 [T, V, C] = [{n_frame}, {n_joint}, {n_channel}])')

# 创建数据集
dataset = SubjectPairDataset(
    subject_data=train_data,
    subject_labels=train_labels,
    use_DASP=True,
    use_SADP=True
)

# 创建 DataLoader
collate_fn = create_pair_collate_fn(use_DASP=True, use_SADP=True)
loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, shuffle=True)

# 检查批次
print('\n检查批次数据形状:')
for batch in loader:
    print(f"\n原始数据形状: {batch['original'][0].shape if batch['original'] else None}")
    print(f"DASP数据形状: {batch['dasp'][0].shape if batch['dasp'] else None}")
    print(f"SADP数据形状: {batch['sadp'][0].shape if batch['sadp'] else None}")
    break

# 测试模型
print('\n测试模型:')

model = SubjectFewShotClassifier(
    input_channels=n_channel,
    num_classes=3,
    n_frame=n_frame,
    n_joint=n_joint
)

# 使用原始数据
x = batch['original'][0]
print(f'原始数据 x 形状: {x.shape}')
hidden_feat = model.forward_to_hidden_feat(x)
print(f'hidden_feat 形状: {hidden_feat.shape}')

# 检查 x 的格式
if batch['dasp'] is not None:
    x_dasp = batch['dasp'][0]
    print(f'\nDASP数据 x_dasp 形状: {x_dasp.shape}')
    hidden_feat_dasp = model.forward_to_hidden_feat(x_dasp)
    print(f'hidden_feat_dasp 形状: {hidden_feat_dasp.shape}')
    
    # 检查 decouple_net
    subject_feat, action_feat = model.decouple_net(hidden_feat)
    subject_feat_dasp, action_feat_dasp = model.decouple_net(hidden_feat_dasp)
    
    print(f'\nsubject_feat 形状: {subject_feat.shape}')
    print(f'action_feat 形状: {action_feat.shape}')
    print(f'subject_feat_dasp 形状: {subject_feat_dasp.shape}')
    print(f'action_feat_dasp 形状: {action_feat_dasp.shape}')
    
    # 检查池化后
    subject_a_pooled = subject_feat.mean(-1).squeeze(-1)
    subject_b_pooled = subject_feat_dasp.mean(-1).squeeze(-1)
    action_a_pooled = action_feat.squeeze(-1).mean(-1)
    action_b_pooled = action_feat_dasp.squeeze(-1).mean(-1)
    
    print(f'\n池化后:')
    print(f'subject_a_pooled 形状: {subject_a_pooled.shape}')
    print(f'subject_b_pooled 形状: {subject_b_pooled.shape}')
    print(f'action_a_pooled 形状: {action_a_pooled.shape}')
    print(f'action_b_pooled 形状: {action_b_pooled.shape}')
    
    # 尝试拼接
    print(f'\n尝试拼接...')
    print(f'  subject_a_pooled: {subject_a_pooled.shape}')
    print(f'  action_b_pooled: {action_b_pooled.shape}')
    try:
        cross_a = torch.cat([subject_a_pooled, action_b_pooled], dim=1)
        print(f'  cross_a 形状: {cross_a.shape}')
    except Exception as e:
        print(f'  拼接失败: {e}')
