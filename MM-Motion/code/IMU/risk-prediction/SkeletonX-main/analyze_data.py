import os
import numpy as np
import pandas as pd

# 数据根目录
data_root = 'F:\\AAA-data'

# 检查目录结构
def check_directory_structure():
    print("检查目录结构...")
    subjects = [d for d in os.listdir(data_root) if d.startswith('subject')]
    print(f"找到 {len(subjects)} 个受试者文件夹")
    
    # 统计有多少个有效的CSV文件
    valid_files = []
    for subject in subjects:
        subject_path = os.path.join(data_root, subject)
        if not os.path.isdir(subject_path):
            continue
        
        poses = [d for d in os.listdir(subject_path) if d.startswith('pose')]
        for pose in poses:
            pose_path = os.path.join(subject_path, pose)
            if not os.path.isdir(pose_path):
                continue
            
            trials = [d for d in os.listdir(pose_path) if d in ['1', '2']]
            for trial in trials:
                trial_path = os.path.join(pose_path, trial)
                if not os.path.isdir(trial_path):
                    continue
                
                csv_files = [f for f in os.listdir(trial_path) if f.endswith('.csv')]
                for csv_file in csv_files:
                    valid_files.append(os.path.join(trial_path, csv_file))
    
    print(f"找到 {len(valid_files)} 个有效的CSV文件")
    return valid_files

# 分析示例CSV文件
def analyze_csv_file(file_path):
    print(f"\n分析文件: {file_path}")
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 打印文件基本信息
        print(f"文件形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        print(f"前5行数据:")
        print(df.head())
        
        # 检查数据类型
        print(f"\n数据类型:")
        print(df.dtypes)
        
        # 检查是否有缺失值
        print(f"\n缺失值情况:")
        print(df.isnull().sum())
        
        return df
    except Exception as e:
        print(f"读取文件出错: {e}")
        return None

# 主函数
if __name__ == '__main__':
    valid_files = check_directory_structure()
    
    if valid_files:
        # 分析第一个有效文件
        df = analyze_csv_file(valid_files[0])
        
        # 如果有更多文件，随机选择几个分析
        if len(valid_files) > 5:
            import random
            random_files = random.sample(valid_files[1:], 4)
            for file in random_files:
                analyze_csv_file(file)
