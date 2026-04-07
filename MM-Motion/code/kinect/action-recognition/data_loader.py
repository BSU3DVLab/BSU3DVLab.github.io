import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ==========================================
# 1. 全局配置与风险映射表
# ==========================================
risk_map = {
    'subject01': 0, 'subject02': 0, 'subject07': 0, 'subject10': 0, 'subject14': 0, 'subject29': 0, # High
    'subject03': 1, 'subject04': 1, 'subject05': 1, 'subject08': 1, 'subject09': 1, 
    'subject11': 1, 'subject12': 1, 'subject13': 1, 'subject15': 1, 'subject16': 1, 
    'subject21': 1, 'subject30': 1, 'subject31': 1, 'subject33': 1, 'subject34': 1, 
    'subject35': 1, 'subject36': 1, 'subject37': 1, 'subject38': 1, 'subject39': 1, # Mid
    'subject06': 2, 'subject17': 2, 'subject18': 2, 'subject19': 2, 'subject20': 2, 
    'subject40': 2, 'subject41': 2 # Low
}

# 明确划分训练集与测试集 (25人 vs 8人)
train_subjects = [f'subject{i:02d}' for i in list(range(1, 22)) + [29, 30, 31, 33, 34, 35, 36]]
test_subjects = [f'subject{i:02d}' for i in [37, 38, 39, 40, 41]]

# Kinect 下半身关节索引 (0: Pelvis 为核心)
LOWER_BODY_JOINTS = [0, 12, 13, 14, 15, 16, 17, 18, 19]
ALL_JOINTS = list(range(32))

# ==========================================
# 2. 核心数据解析函数
# ==========================================
def load_skeleton_file(file_path):
    """加载单个骨架CSV文件并提取 3D 坐标 (x, y, z)"""
    try:
        df = pd.read_csv(file_path)
        joint_data = []
        for i in range(32):
            cols = [f'joint_{i}_confidence', f'joint_{i}_x', f'joint_{i}_y', f'joint_{i}_z']
            if all(col in df.columns for col in cols):
                joint_data.append(df[cols].values)
        
        if len(joint_data) == 32:
            data = np.stack(joint_data, axis=1) # 形状: (timesteps, 32, 4)
            return data
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_features(data, use_lower_body_only=False):
    """提取坐标并执行【骨盆归一化】(平移不变性处理)"""
    if data is None or len(data) == 0:
        return None
    
    joints = LOWER_BODY_JOINTS if use_lower_body_only else ALL_JOINTS
    coords = data[:, joints, 1:]  # 切片去除 confidence，只留 (x,y,z)
    
    # 归一化：所有坐标减去 Pelvis(骨盆) 的坐标，消除受试者站位远近的干扰
    if 0 in joints:
        pelvis_idx = joints.index(0)
        pelvis_pos = coords[:, pelvis_idx:pelvis_idx+1, :]
        coords = coords - pelvis_pos
    
    features = coords.reshape(coords.shape[0], -1) # 展平为 (timesteps, n_joints * 3)
    return features

# ==========================================
# 3. 增强版：引入滑动窗口机制与暴力文件抓取
# ==========================================
def load_subject_data(subject_id, data_dir='F:\\AAA-data', window_size=150, step_size=30):
    """加载受试者数据，执行双机位融合、滑动窗口切片，无视文件名规则抓取CSV"""
    subject_num = int(subject_id.replace('subject', ''))
    subject_folder = f'subject{subject_num:02d}'
    
    X_list, y_list, risk_list = [], [], []
    
    for pose_id in range(1, 17):
        pose_folder = f'pose{pose_id:02d}'
        for trial in [1, 2]:
            master_dir = os.path.join(data_dir, subject_folder, pose_folder, str(trial), 'Kinect-data', 'Master-Kinect', 'skeleton')
            sub_dir = os.path.join(data_dir, subject_folder, pose_folder, str(trial), 'Kinect-data', 'Sub-Kinect', 'skeleton')
            
            # 检查文件夹是否存在
            if not os.path.exists(master_dir) or not os.path.exists(sub_dir):
                continue
                
            # 【终极极简雷达】：无视文件名，只要是 CSV 就直接抓过来！
            master_csvs = [f for f in os.listdir(master_dir) if f.endswith('.csv')]
            sub_csvs = [f for f in os.listdir(sub_dir) if f.endswith('.csv')]
            
            # 如果文件夹里连一个 CSV 都没有，跳过
            if not master_csvs or not sub_csvs:
                continue
                
            master_path = os.path.join(master_dir, master_csvs[0])
            sub_path = os.path.join(sub_dir, sub_csvs[0])
            
            master_data = load_skeleton_file(master_path)
            sub_data = load_skeleton_file(sub_path)
            
            if master_data is not None and sub_data is not None:
                use_lower_body = pose_id in [10, 11, 12, 13, 14, 15]
                
                master_features = extract_features(master_data, use_lower_body)
                sub_features = extract_features(sub_data, use_lower_body)
                
                if master_features is not None and sub_features is not None:
                    # 1. 时间轴最短截断对齐 (主副机位严格同步)
                    min_len = min(len(master_features), len(sub_features))
                    master_features = master_features[:min_len]
                    sub_features = sub_features[:min_len]
                    
                    # 2. 双视角特征拼接
                    combined_features = np.concatenate([master_features, sub_features], axis=1)
                    
                    # 3. 滑动窗口切片 (大幅扩增数据量)
                    if len(combined_features) >= window_size:
                        for i in range(0, len(combined_features) - window_size + 1, step_size):
                            X_list.append(combined_features[i:i+window_size])
                            y_list.append(pose_id - 1) # 类别变为 0-15
                            risk_list.append(risk_map.get(subject_id, 1))
                    else:
                        # 对于极少数不足 150 帧的短动作，进行边缘静止填充
                        pad_width = ((0, window_size - len(combined_features)), (0, 0))
                        padded_features = np.pad(combined_features, pad_width, mode='edge')
                        X_list.append(padded_features)
                        y_list.append(pose_id - 1)
                        risk_list.append(risk_map.get(subject_id, 1))
                        
    return X_list, y_list, risk_list

def align_feature_dimensions(X_list):
    """处理半身动作与全身动作的特征维度不一致问题，为半身动作补零"""
    if not X_list:
        return np.array([]), 0
        
    max_features = max(x.shape[1] for x in X_list)
    
    X_aligned = []
    for x in X_list:
        if x.shape[1] < max_features:
            pad_width = ((0, 0), (0, max_features - x.shape[1]))
            x_padded = np.pad(x, pad_width, mode='constant', constant_values=0)
            X_aligned.append(x_padded)
        else:
            X_aligned.append(x)
            
    return np.array(X_aligned), max_features

# ==========================================
# 4. 统筹与标准化输出
# ==========================================
def get_data(data_dir='F:\\AAA-data'):
    print("📥 正在加载双机位 Kinect 骨架数据 (启用滑动窗口机制)...")
    
    X_train_list, y_train_list, r_train_list = [], [], []
    X_test_list, y_test_list, r_test_list = [], [], []
    
    for subject_id in train_subjects:
        X, y, r = load_subject_data(subject_id, data_dir)
        X_train_list.extend(X)
        y_train_list.extend(y)
        r_train_list.extend(r)
        
    for subject_id in test_subjects:
        X, y, r = load_subject_data(subject_id, data_dir)
        X_test_list.extend(X)
        y_test_list.extend(y)
        r_test_list.extend(r)
        
    # 统一特征维度 (半身动作补零至192维)
    X_all_list = X_train_list + X_test_list
    if not X_all_list:
        raise ValueError("❌ 灾难性错误：没有加载到任何数据，请检查 data_dir 路径是否正确！")
        
    X_train_aligned, n_features = align_feature_dimensions(X_train_list)
    X_test_aligned, _ = align_feature_dimensions(X_test_list)
    
    y_train = np.array(y_train_list)
    y_test = np.array(y_test_list)
    r_train = np.array(r_train_list)
    r_test = np.array(r_test_list)
    
    # 彻底的标准化 (StandardScaler)
    scaler = StandardScaler()
    n_samples_tr, n_timesteps, _ = X_train_aligned.shape
    n_samples_te = X_test_aligned.shape[0]
    
    X_train = scaler.fit_transform(X_train_aligned.reshape(-1, n_features)).reshape(n_samples_tr, n_timesteps, n_features)
    X_test = scaler.transform(X_test_aligned.reshape(-1, n_features)).reshape(n_samples_te, n_timesteps, n_features)
    
    print(f"\n✅ 数据加载与多视角对齐完成！")
    print(f"👉 训练集形状: {X_train.shape} (经过滑窗扩增，数据大军集结完毕！)")
    print(f"👉 测试集形状: {X_test.shape}")
    print(f"👉 统一后的特征维度: {n_features}")
    
    return X_train, X_test, y_train, y_test, r_train, r_test

if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, r_tr, r_te = get_data()