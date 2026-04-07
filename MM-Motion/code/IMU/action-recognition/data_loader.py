import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 指定的受试者列表 (共33人)
selected_subjects = list(range(1, 22)) + list(range(29, 32)) + list(range(33, 42))
selected_subject_ids = [f"subject{id:02d}" for id in selected_subjects]

# 动作标签映射 (pose01 到 pose16 映射为 0-15)
pose_labels = {
    f'pose{i:02d}': i-1 for i in range(1, 17)
}

def load_imu_data(data_dir, subject_ids, window_size=150, step_size=75):
    """
    加载IMU数据，进行滑窗处理，并执行【物理级特征截肢】
    只保留下肢（骨盆、大腿、小腿、脚）的传感器信号，彻底摒弃上半身噪音！
    """
    X = []
    y = []
    subject_list = []
    
    standard_columns = None 
    
    # 【核心手术刀】：只保留这些关键部位开头的列！
    # 包含了：骨盆(Hips)、右大腿(RightUpLeg)、右小腿(RightLeg)、右脚(RightFoot) 以及左侧对应部位
    target_joints = [
        'Hips-', 
        'RightUpLeg-', 'RightLeg-', 'RightFoot-', 
        'LeftUpLeg-', 'LeftLeg-', 'LeftFoot-'
    ]
    
    for subject_id in subject_ids:
        subject_path = os.path.join(data_dir, subject_id)
        if not os.path.exists(subject_path):
            continue
        
        for pose_name in os.listdir(subject_path):
            if not pose_name.startswith('pose'):
                continue
            
            pose_path = os.path.join(subject_path, pose_name)
            if not os.path.isdir(pose_path):
                continue
            
            label = pose_labels.get(pose_name, -1)
            if label == -1:
                continue
            
            for trial in os.listdir(pose_path):
                trial_path = os.path.join(pose_path, trial)
                if not os.path.isdir(trial_path):
                    continue
                
                for file_name in os.listdir(trial_path):
                    # 确保是 IMU 数据文件
                    if file_name.endswith('.csv') and 'N' in file_name.upper():
                        file_path = os.path.join(trial_path, file_name)
                        try:
                            df = pd.read_csv(file_path)
                            
                            if standard_columns is None:
                                valid_cols = []
                                for col in df.columns:
                                    # 先排除时间和序号等垃圾列
                                    if 'time' in str(col).lower() or 'frame' in str(col).lower() or 'unnamed' in str(col).lower():
                                        continue
                                    
                                    # 【截肢逻辑】：只有列名包含 target_joints 里的部位，才会被选中！
                                    if any(joint in str(col) for joint in target_joints):
                                        valid_cols.append(col)
                                
                                standard_columns = valid_cols
                                print(f"✅ 截肢手术成功！全身 1071 维 -> 下肢专属 {len(standard_columns)} 维 (依据文件: {file_name})")
                            
                            # 严格按照精简后的 standard_columns 提取数据
                            missing_cols = [col for col in standard_columns if col not in df.columns]
                            if missing_cols:
                                print(f"⚠️ 跳过 {file_path}: 缺少标准特征列")
                                continue
                                
                            imu_data = df[standard_columns].values.astype(float)
                            
                            # 滑窗处理
                            if len(imu_data) >= window_size:
                                for i in range(0, len(imu_data) - window_size + 1, step_size):
                                    X.append(imu_data[i:i+window_size])
                                    y.append(label)
                                    subject_list.append(subject_id)
                            else:
                                padding = np.zeros((window_size - len(imu_data), imu_data.shape[1]))
                                window_data = np.vstack((imu_data, padding))
                                X.append(window_data)
                                y.append(label)
                                subject_list.append(subject_id)
                                
                        except Exception as e:
                            print(f"❌ Error loading {file_path}: {e}")
                            
    return np.array(X), np.array(y), subject_list

def split_and_scale_data(X, y, subject_list, train_size=25, test_size=8):
    """按受试者划分训练集和测试集，并进行防止数据泄露的全局标准化"""
    unique_subjects = list(set(subject_list))
    
    train_subjects, test_subjects = train_test_split(
        unique_subjects, train_size=train_size, test_size=test_size, random_state=42
    )
    
    train_indices = [i for i, subj in enumerate(subject_list) if subj in train_subjects]
    test_indices = [i for i, subj in enumerate(subject_list) if subj in test_subjects]
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    # ========== 全局标准化 ==========
    scaler = StandardScaler()
    
    N_tr, W, F = X_train.shape
    N_te = X_test.shape[0]
    
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, F)).reshape(N_tr, W, F)
    X_test_scaled = scaler.transform(X_test.reshape(-1, F)).reshape(N_te, W, F)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def get_data():
    """获取训练和测试数据供模型调用"""
    data_dir = "F:\\AAA-data"
    print("正在加载并清洗下肢专属动作序列，请稍候...")
    X, y, subject_list = load_imu_data(data_dir, selected_subject_ids)
    
    if len(X) == 0:
        raise ValueError("❌ 没有加载到任何有效数据！请检查路径或文件过滤规则。")
        
    X_train, X_test, y_train, y_test = split_and_scale_data(X, y, subject_list)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data()
    print("\n" + "="*40)
    print(f"🎉 数据集手术完成，加载成功！")
    print(f"👉 训练集形状: {X_train.shape}")
    print(f"👉 测试集形状: {X_test.shape}")
    print(f"👉 锁定的特征维度: {X_train.shape[-1]} 维 (纯净的下半身信号！)")
    print("="*40 + "\n")