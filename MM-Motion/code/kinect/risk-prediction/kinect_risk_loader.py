import os
import numpy as np
import pandas as pd

# ==========================================
# 1. 🌟 33人新版风险映射表 (高=0, 中=1, 低=2)
# ==========================================
risk_map = {
    'subject01': 0, 'subject02': 0, 'subject04': 0, 'subject07': 0, 'subject08': 0, 
    'subject10': 0, 'subject11': 0, 'subject12': 0, 'subject14': 0, 'subject29': 0, 
    'subject30': 0, 'subject34': 0, 'subject37': 0, 'subject38': 0, # 高风险 14
    
    'subject03': 1, 'subject05': 1, 'subject09': 1, 'subject13': 1, 'subject15': 1, 
    'subject16': 1, 'subject21': 1, 'subject33': 1, 'subject35': 1, 'subject36': 1, 
    'subject39': 1, # 中风险 11
    
    'subject06': 2, 'subject17': 2, 'subject18': 2, 'subject19': 2, 'subject20': 2, 
    'subject31': 2, 'subject40': 2, 'subject41': 2  # 低风险 8
}

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
    if data is None or len(data) == 0: return None
    coords = data[:, :, 1:]
    
    pelvis_pos = coords[:, 0:1, :]
    coords = coords - pelvis_pos
    
    if use_lower_body_only:
        mask = np.zeros((1, 32, 1))
        mask[0, LOWER_BODY_JOINTS, 0] = 1
        coords = coords * mask
        
    return coords.reshape(coords.shape[0], -1)

def load_subject_data(subject_id, data_dir='F:\\AAA-data', window_size=150, step_size=30):
    subject_num = int(subject_id.replace('subject', ''))
    subject_folder = f'subject{subject_num:02d}'
    
    X_list, y_risk_list, p_pose_list = [], [], []
    
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
                    combined = np.concatenate([master_features[:min_len], sub_features[:min_len]], axis=1)
                    
                    if len(combined) >= window_size:
                        for i in range(0, len(combined) - window_size + 1, step_size):
                            X_list.append(combined[i:i+window_size])
                            y_risk_list.append(risk_map[subject_id])
                            p_pose_list.append(pose_id - 1)
                    else:
                        pad_width = ((0, window_size - len(combined)), (0, 0))
                        X_list.append(np.pad(combined, pad_width, mode='edge'))
                        y_risk_list.append(risk_map[subject_id])
                        p_pose_list.append(pose_id - 1)
                        
    return X_list, y_risk_list, p_pose_list

def get_all_subjects_data(data_dir='F:\\AAA-data'):
    print("📥 正在读取全量 33 人数据至内存，为 5 折交叉验证做准备...")
    all_data = {}
    for sid in risk_map.keys():
        X, y, p = load_subject_data(sid, data_dir)
        if len(X) > 0:
            X = np.array(X)
            # 全局排毒与物理截断
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            X = np.clip(X, -2.0, 2.0) / 2.0
            all_data[sid] = {'X': X, 'y': np.array(y), 'p': np.array(p)}
    print(f"✅ 成功加载 {len(all_data)} 位受试者的数据！")
    return all_data