import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================
# 1. 完整 33 人风险映射表 (MM-Motion)
# ==========================================
risk_map = {
    'subject01': 0, 'subject02': 0, 'subject07': 0, 'subject10': 0, 'subject14': 0, 'subject29': 0, 
    'subject03': 1, 'subject04': 1, 'subject05': 1, 'subject08': 1, 'subject09': 1, 
    'subject11': 1, 'subject12': 1, 'subject13': 1, 'subject15': 1, 'subject16': 1, 
    'subject21': 1, 'subject30': 1, 'subject31': 1, 'subject33': 1, 'subject34': 1, 
    'subject35': 1, 'subject36': 1, 'subject37': 1, 'subject38': 1, 'subject39': 1, 
    'subject06': 2, 'subject17': 2, 'subject18': 2, 'subject19': 2, 'subject20': 2, 
    'subject40': 2, 'subject41': 2  
} 

# ==========================================
# 🚨 绝对严谨的数据集切分 
# ==========================================
all_subjects = list(risk_map.keys())

# 强制圈定 01 到 07 为测试集 (严格 7 人)
test_subjects = [f'subject{i:02d}' for i in range(1, 8)]
# 剩余全部为训练集 (严格 26 人)
train_subjects = [s for s in all_subjects if s not in test_subjects]

print(f"📊 数据集切分校验：训练集 {len(train_subjects)} 人, 测试集 {len(test_subjects)} 人。")

# 下半身关节索引
LOWER_BODY_JOINTS = [0, 12, 13, 14, 15, 16, 17, 18, 19]

def load_skeleton_file(file_path):
    try:
        df = pd.read_csv(file_path)
        joint_data = []
        for i in range(32):
            cols = [f'joint_{i}_confidence', f'joint_{i}_x', f'joint_{i}_y', f'joint_{i}_z']
            if all(col in df.columns for col in cols):
                joint_data.append(df[cols].values)
        if len(joint_data) == 32:
            return np.stack(joint_data, axis=1)
        return None
    except Exception as e:
        return None

def extract_features(data, use_lower_body_only=False):
    if data is None or len(data) == 0:
        return None
    
    # 提取 (time, 32, 3) 坐标
    coords = data[:, :, 1:]
    
    # 骨盆绝对中心化
    pelvis_pos = coords[:, 0:1, :]
    coords = coords - pelvis_pos
    
    # 动作 10-15 物理静音上半身
    if use_lower_body_only:
        mask = np.zeros((1, 32, 1))
        mask[0, LOWER_BODY_JOINTS, 0] = 1
        coords = coords * mask
        
    return coords.reshape(coords.shape[0], -1)

def load_subject_data(subject_id, data_dir='F:\\AAA-data', window_size=150, step_size=30):
    subject_num = int(subject_id.replace('subject', ''))
    subject_folder = f'subject{subject_num:02d}'
    
    X_list, y_list, risk_list = [], [], []
    
    for pose_id in range(1, 17):
        pose_folder = f'pose{pose_id:02d}'
        for trial in [1, 2]:
            master_dir = os.path.join(data_dir, subject_folder, pose_folder, str(trial), 'Kinect-data', 'Master-Kinect', 'skeleton')
            sub_dir = os.path.join(data_dir, subject_folder, pose_folder, str(trial), 'Kinect-data', 'Sub-Kinect', 'skeleton')
            
            if not os.path.exists(master_dir) or not os.path.exists(sub_dir): continue
            
            master_csvs = [f for f in os.listdir(master_dir) if f.endswith('.csv')]
            sub_csvs = [f for f in os.listdir(sub_dir) if f.endswith('.csv')]
            
            if not master_csvs or not sub_csvs: continue
            
            master_data = load_skeleton_file(os.path.join(master_dir, master_csvs[0]))
            sub_data = load_skeleton_file(os.path.join(sub_dir, sub_csvs[0]))
            
            if master_data is not None and sub_data is not None:
                use_lower_body = pose_id in [10, 11, 12, 13, 14, 15]
                master_features = extract_features(master_data, use_lower_body)
                sub_features = extract_features(sub_data, use_lower_body)
                
                if master_features is not None and sub_features is not None:
                    min_len = min(len(master_features), len(sub_features))
                    # 拼接为主副机位 192 维
                    combined = np.concatenate([master_features[:min_len], sub_features[:min_len]], axis=1)
                    
                    if len(combined) >= window_size:
                        for i in range(0, len(combined) - window_size + 1, step_size):
                            X_list.append(combined[i:i+window_size])
                            y_list.append(pose_id - 1)
                            risk_list.append(risk_map.get(subject_id, 1))
                    else:
                        pad_width = ((0, window_size - len(combined)), (0, 0))
                        X_list.append(np.pad(combined, pad_width, mode='edge'))
                        y_list.append(pose_id - 1)
                        risk_list.append(risk_map.get(subject_id, 1))
                        
    return X_list, y_list, risk_list

def get_data(data_dir='F:\\AAA-data'):
    print("📥 正在加载双视角 Kinect 数据 (集成学习强制切分模式)...")
    X_tr, y_tr, r_tr = [], [], []
    X_te, y_te, r_te = [], [], []
    
    for sid in train_subjects:
        X, y, r = load_subject_data(sid, data_dir)
        X_tr.extend(X); y_tr.extend(y); r_tr.extend(r)
        
    for sid in test_subjects:
        X, y, r = load_subject_data(sid, data_dir)
        X_te.extend(X); y_te.extend(y); r_te.extend(r)
        
    X_tr, X_te = np.array(X_tr), np.array(X_te)
    
    # 🌟 物理边界绝对裁剪：粉碎异常黑洞
    X_tr = np.clip(X_tr, -2.0, 2.0) / 2.0
    X_te = np.clip(X_te, -2.0, 2.0) / 2.0
    
    print(f"✅ 数据处理完毕！训练集: {X_tr.shape}, 测试集: {X_te.shape}")
    return X_tr, X_te, np.array(y_tr), np.array(y_te), np.array(r_tr), np.array(r_te)