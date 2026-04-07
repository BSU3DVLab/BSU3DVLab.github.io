import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 风险映射表
risk_map = {
    'subject01': 0, 'subject02': 0, 'subject07': 0, 'subject10': 0, 'subject14': 0, 'subject29': 0, # High
    'subject03': 1, 'subject04': 1, 'subject05': 1, 'subject08': 1, 'subject09': 1, 'subject11': 1, 
    'subject12': 1, 'subject13': 1, 'subject15': 1, 'subject16': 1, 'subject21': 1, 'subject30': 1, 
    'subject31': 1, 'subject33': 1, 'subject34': 1, 'subject35': 1, 'subject36': 1, 'subject37': 1, 
    'subject38': 1, 'subject39': 1, # Mid
    'subject06': 2, 'subject17': 2, 'subject18': 2, 'subject19': 2, 'subject20': 2, 'subject40': 2, 'subject41': 2  # Low
}

def get_data():
    data_dir = "F:\\AAA-data"
    selected_subjects = list(risk_map.keys())
    
    X, y, r, subjects = [], [], [], [] # r 为风险等级
    standard_columns = None
    target_joints = ['Hips-', 'RightUpLeg-', 'RightLeg-', 'RightFoot-', 'LeftUpLeg-', 'LeftLeg-', 'LeftFoot-']
    
    for sub_id in selected_subjects:
        sub_path = os.path.join(data_dir, sub_id)
        if not os.path.exists(sub_path): continue
        for pose in [f'pose{i:02d}' for i in range(1, 17)]:
            pose_path = os.path.join(sub_path, pose)
            if not os.path.exists(pose_path): continue
            for trial in os.listdir(pose_path):
                t_path = os.path.join(pose_path, trial)
                for f in os.listdir(t_path):
                    if f.endswith('.csv') and 'N' in f.upper():
                        df = pd.read_csv(os.path.join(t_path, f))
                        if standard_columns is None:
                            standard_columns = [c for c in df.columns if any(j in c for j in target_joints) and not any(k in c.lower() for k in ['time','frame','unnamed'])]
                        
                        data = df[standard_columns].values.astype(float)
                        # 滑窗... (简化逻辑，保留核心)
                        if len(data) >= 150:
                            for i in range(0, len(data)-150+1, 75):
                                X.append(data[i:i+150])
                                y.append(int(pose[-2:])-1)
                                r.append(risk_map[sub_id])
                                subjects.append(sub_id)

    X, y, r = np.array(X), np.array(y), np.array(r)
    
    # 分层切分
    u_subs = sorted(list(set(subjects)))
    u_risks = [risk_map[s] for s in u_subs]
    tr_subs, te_subs = train_test_split(u_subs, train_size=25, test_size=8, stratify=u_risks, random_state=42)
    
    tr_idx = [i for i, s in enumerate(subjects) if s in tr_subs]
    te_idx = [i for i, s in enumerate(subjects) if s in te_subs]
    
    # 标准化
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[tr_idx].reshape(-1, 147)).reshape(-1, 150, 147)
    X_te = scaler.transform(X[te_idx].reshape(-1, 147)).reshape(-1, 150, 147)
    
    return X_tr, X_te, y[tr_idx], y[te_idx], r[tr_idx], r[te_idx]