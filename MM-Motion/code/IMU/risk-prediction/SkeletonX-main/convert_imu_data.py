import os
import numpy as np
import pandas as pd
import logging
import argparse
from sklearn.model_selection import train_test_split

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 数据根目录（默认值，可以通过命令行参数覆盖）
data_root = 'F:\\AAA-data'
# 保存转换后数据的目录
save_dir = './work_dir/imu_data'
# 最大帧率
max_fps = 500
# 关节数量 - 根据提供的列名，我们只提取主要关节的位置数据
# 包括：Hips, RightUpLeg, RightLeg, RightFoot, LeftUpLeg, LeftLeg, LeftFoot,
#       Spine, Spine1, Spine2, Neck, Neck1, Head, RightShoulder, RightArm,
#       RightForeArm, RightHand, LeftShoulder, LeftArm, LeftForeArm, LeftHand
num_joints = 21
# 人数
num_persons = 1
# 通道数（x, y, z）
num_channels = 3

# 受试者风险等级
risk_levels = {
    1: '高风险', 2: '高风险', 3: '高风险', 4: '中风险', 5: '低风险',
    6: '低风险', 7: '高风险', 8: '高风险', 9: '中风险', 10: '高风险',
    11: '中风险', 12: '高风险', 13: '中风险', 14: '高风险', 15: '高风险',
    16: '高风险', 17: '低风险', 18: '低风险', 19: '低风险', 20: '低风险',
    21: '高风险', 29: '高风险', 30: '高风险', 31: '低风险', 33: '低风险',
    34: '高风险', 35: '高风险', 36: '中风险', 37: '高风险', 38: '中风险',
    39: '高风险', 40: '低风险', 41: '低风险'
}

# 确保保存目录存在
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 定义关节名称和顺序
joint_names = [
    'Hips',
    'RightUpLeg', 'RightLeg', 'RightFoot',
    'LeftUpLeg', 'LeftLeg', 'LeftFoot',
    'Spine', 'Spine1', 'Spine2',
    'Neck', 'Neck1', 'Head',
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
    'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand'
]

# 读取CSV文件并转换为关节坐标格式
def read_imu_csv(file_path):
    """
    读取IMU CSV文件并转换为关节坐标格式
    从CSV文件中提取关节位置数据（Joint-Posi-x, Joint-Posi-y, Joint-Posi-z）
    """
    try:
        # 自动检测分隔符
        df = pd.read_csv(file_path, sep=None, engine='python')
        
        # 清理列名：去除空格、转换为小写
        original_columns = df.columns.tolist()
        df.columns = [col.strip().lower() for col in df.columns]
        
        logger.info(f"文件 {file_path} 包含 {len(df.columns)} 列：{original_columns[:10]}...")  # 只显示前10列
        
        # 提取每个关节的x、y、z坐标数据
        joint_data_list = []
        for joint_name in joint_names:
            # 查找关节位置的列（使用小写进行匹配）
            joint_name_lower = joint_name.lower()
            x_col = f'{joint_name_lower}-joint-posi-x'
            y_col = f'{joint_name_lower}-joint-posi-y'
            z_col = f'{joint_name_lower}-joint-posi-z'
            
            # 打印查找的列名以便调试
            logger.info(f"尝试查找 {joint_name} 的列：{x_col}, {y_col}, {z_col}")
            
            found_x = found_y = found_z = None
            found_cols = []
            
            # 精确匹配尝试
            if x_col in df.columns:
                found_x = x_col
            if y_col in df.columns:
                found_y = y_col
            if z_col in df.columns:
                found_z = z_col
            
            # 如果精确匹配失败，尝试更灵活的匹配方式
            if not all([found_x, found_y, found_z]):
                for col in df.columns:
                    if joint_name_lower in col and 'posi' in col:
                        found_cols.append(col)
                
                # 对找到的列进行排序和匹配
                if len(found_cols) >= 3:
                    # 尝试根据列名中的x/y/z或位置排序
                    for col in found_cols:
                        if 'x' in col or col.endswith('0'):
                            found_x = col
                        elif 'y' in col or col.endswith('1'):
                            found_y = col
                        elif 'z' in col or col.endswith('2'):
                            found_z = col
                    
                    # 如果仍有缺失，使用排序后的前3个
                    if not all([found_x, found_y, found_z]):
                        found_cols.sort()
                        found_x = found_cols[0]
                        found_y = found_cols[1]
                        found_z = found_cols[2]
            
            # 检查是否找到所有坐标列
            if all([found_x, found_y, found_z]):
                # 提取关节位置数据
                joint_pos = df[[found_x, found_y, found_z]].values
                joint_data_list.append(joint_pos)
                logger.info(f"成功找到 {joint_name} 的位置数据列：{found_x}, {found_y}, {found_z}")
            else:
                logger.error(f"文件 {file_path} 缺少 {joint_name} 的位置数据列")
                logger.error(f"文件中包含的相关列：{found_cols}")
                return None
        
        # 将所有关节数据合并为 (帧数 × 关节数 × 通道数) 格式
        num_frames = df.shape[0]
        joint_data = np.stack(joint_data_list, axis=1)  # (帧数 × 关节数 × 通道数)
        
        # 转换为 (帧数 × 人数 × 关节数 × 通道数) 格式
        # 这里人数固定为1
        joint_data = joint_data.reshape(num_frames, 1, num_joints, num_channels)
        
        logger.info(f"成功读取文件 {file_path}，生成 {joint_data.shape} 大小的数据")
        return joint_data
    except Exception as e:
        logger.error(f"读取文件 {file_path} 出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# 控制帧率
def control_fps(data, original_fps, max_fps):
    """
    控制帧率，确保不超过max_fps
    """
    if original_fps <= max_fps:
        return data
    
    # 计算降采样因子
    downsample_factor = int(np.ceil(original_fps / max_fps))
    
    # 降采样
    downsampled_data = data[::downsample_factor]
    
    logger.info(f"帧率从 {original_fps} 降低到 {original_fps / downsample_factor:.2f}，帧数从 {data.shape[0]} 减少到 {downsampled_data.shape[0]}")
    
    return downsampled_data

# 收集所有有效数据
def collect_data():
    """
    收集所有有效的IMU数据文件
    只处理XXXXNX.csv格式的文件（其中X是数字，N是固定字符）
    """
    logger.info("开始收集IMU数据...")
    
    all_data = []
    all_labels = []
    all_subjects = []
    valid_subjects = 0
    
    # 风险等级到数值标签的映射
    risk_to_label = {
        '低风险': 0,
        '中风险': 1,
        '高风险': 2
    }
    
    # 遍历每个受试者（从1到41）
    for subject_idx in range(1, 42):  # 1-41个受试者
        subject_found = False
        
        # 尝试不同的目录命名格式
        subject_dirs = [
            os.path.join(data_root, f'subject{subject_idx:02d}'),
            os.path.join(data_root, f'subject_{subject_idx}'),
            os.path.join(data_root, f'subject{subject_idx}')
        ]
        
        subject_path = None
        for sd in subject_dirs:
            if os.path.exists(sd):
                subject_path = sd
                break
        
        if not subject_path:
            logger.info(f"受试者{subject_idx}的目录不存在，跳过")
            continue
        
        logger.info(f"找到受试者{subject_idx}的目录: {subject_path}")
        
        # 检查受试者是否在risk_levels字典中
        if subject_idx not in risk_levels:
            logger.warning(f"受试者{subject_idx}的风险等级未定义")
            continue
        
        # 获取受试者的风险等级
        risk_level = risk_levels[subject_idx]
        
        # 转换为数值标签
        label = risk_to_label[risk_level]
        
        # 查找XXXXNX.csv格式的文件
        nx_files_found = []
        
        # 遍历所有可能的子目录
        for root, dirs, files in os.walk(subject_path):
            for file in files:
                if file.endswith('.csv') and 'N' in file and len(file) >= 6:
                    # 检查文件名格式：XXXXNX.csv
                    # 前4位是数字，第5位是N，后面是数字和.csv
                    try:
                        # 验证文件名格式
                        prefix = file[:4]
                        n_pos = file[4]
                        suffix = file[5:].split('.')[0]
                        
                        if n_pos == 'N' and prefix.isdigit() and suffix.isdigit():
                            nx_files_found.append(os.path.join(root, file))
                    except:
                        continue
        
        if not nx_files_found:
            logger.info(f"受试者{subject_idx}没有XXXXNX.csv格式的文件，跳过")
            continue
        
        # 找到了XXXXNX.csv文件，处理这些文件
        valid_subjects += 1
        logger.info(f"在受试者{subject_idx}目录下找到{len(nx_files_found)}个XXXXNX.csv格式的文件")
        
        for csv_file in nx_files_found:
            try:
                # 读取IMU数据
                data = read_imu_csv(csv_file)
                
                if data is None:
                    logger.warning(f"读取CSV文件失败: {csv_file}")
                    continue
                
                # 控制帧率
                original_fps = 1000  # 需要根据实际情况调整
                data = control_fps(data, original_fps, max_fps)
                
                # 添加到数据集
                all_data.append(data)
                all_labels.append(label)  # 使用风险等级作为标签：0-低风险，1-中风险，2-高风险
                all_subjects.append(subject_idx)
                
                logger.info(f"已添加: 受试者{subject_idx}, 风险等级{risk_level}, 文件{csv_file}, 帧数{data.shape[0]}")
            except Exception as e:
                logger.error(f"处理CSV文件时出错: {csv_file}, 错误: {str(e)}")
                import traceback
                logger.error(f"错误堆栈: {traceback.format_exc()}")
                continue
    
    logger.info(f"数据收集完成，共收集到{len(all_data)}个样本")
    logger.info(f"找到{valid_subjects}个符合条件的受试者")
    return all_data, all_labels, all_subjects

# 划分训练集和测试集
def split_dataset(data, labels, subjects):
    """
    按照4:1的比例划分训练集和测试集，基于受试者的风险等级
    """
    logger.info("开始划分数据集...")
    
    # 创建受试者到样本索引的映射
    subject_to_indices = {}
    for idx, subject in enumerate(subjects):
        if subject not in subject_to_indices:
            subject_to_indices[subject] = []
        subject_to_indices[subject].append(idx)
    
    logger.info(f"subject_to_indices: {subject_to_indices}")
    
    # 获取所有唯一的受试者
    unique_subjects = list(subject_to_indices.keys())
    
    # 按照风险等级分组
    high_risk_subjects = [s for s in unique_subjects if risk_levels.get(s) == '高风险']
    medium_risk_subjects = [s for s in unique_subjects if risk_levels.get(s) == '中风险']
    low_risk_subjects = [s for s in unique_subjects if risk_levels.get(s) == '低风险']
    
    logger.info(f"高风险受试者: {len(high_risk_subjects)} 个: {high_risk_subjects}")
    logger.info(f"中风险受试者: {len(medium_risk_subjects)} 个: {medium_risk_subjects}")
    logger.info(f"低风险受试者: {len(low_risk_subjects)} 个: {low_risk_subjects}")
    
    # 对每个风险等级的受试者进行4:1划分
    def split_risk_group(subjects):
        if len(subjects) < 5:
            # 受试者数量不足5个，全部用于训练
            return subjects, []
        
        # 随机打乱
        np.random.shuffle(subjects)
        
        # 4:1划分
        split_idx = int(len(subjects) * 0.8)
        train_subjects = subjects[:split_idx]
        test_subjects = subjects[split_idx:]
        
        return train_subjects, test_subjects
    
    high_train, high_test = split_risk_group(high_risk_subjects)
    medium_train, medium_test = split_risk_group(medium_risk_subjects)
    low_train, low_test = split_risk_group(low_risk_subjects)
    
    # 合并训练集和测试集受试者
    train_subjects = high_train + medium_train + low_train
    test_subjects = high_test + medium_test + low_test
    
    logger.info(f"训练集受试者: {len(train_subjects)} 个: {train_subjects}")
    logger.info(f"测试集受试者: {len(test_subjects)} 个: {test_subjects}")
    
    # 获取训练集和测试集的样本索引
    train_indices = []
    for subject in train_subjects:
        if subject in subject_to_indices:
            train_indices.extend(subject_to_indices[subject])
    
    test_indices = []
    for subject in test_subjects:
        if subject in subject_to_indices:
            test_indices.extend(subject_to_indices[subject])
    
    # 获取训练集和测试集数据
    train_data = [data[idx] for idx in train_indices]
    train_labels = [labels[idx] for idx in train_indices]
    
    test_data = [data[idx] for idx in test_indices] if test_indices else train_data[:len(train_data)//5]  # 如果测试集为空，使用训练集的一部分作为测试集
    test_labels = [labels[idx] for idx in test_indices] if test_indices else train_labels[:len(train_labels)//5]
    
    logger.info(f"训练集样本数: {len(train_data)}")
    logger.info(f"测试集样本数: {len(test_data)}")
    
    return train_data, train_labels, test_data, test_labels

# 将数据转换为模型可接受的格式
def convert_to_model_format(data_list, labels):
    """
    将数据转换为模型可接受的格式：(N, C, T, V, M)
    其中：
    N: 样本数量
    C: 通道数（x, y, z）
    T: 时间帧数量（需要统一）
    V: 关节数
    M: 人数
    """
    logger.info("开始转换数据格式...")
    
    # 处理空数据列表的情况
    if not data_list:
        logger.info("数据列表为空，返回空数组")
        return np.array([]), np.array([])
    
    # 找到最大帧数
    max_frames = max(data.shape[0] for data in data_list)
    logger.info(f"最大帧数: {max_frames}")
    
    N = len(data_list)
    C = num_channels
    T = max_frames
    V = num_joints
    M = num_persons
    
    # 创建统一格式的数据数组
    model_data = np.zeros((N, C, T, V, M))
    model_labels = np.zeros(N, dtype=int)
    
    for i in range(N):
        data = data_list[i]
        label = labels[i]
        
        # 数据形状：(帧数, 人数, 关节数, 通道数)
        current_frames = data.shape[0]
        
        # 将数据填充到统一形状
        # 通道维度需要调整到第二维
        model_data[i, :, :current_frames, :, :] = data.transpose(3, 0, 2, 1)
        model_labels[i] = label
    
    return model_data, model_labels

# 保存数据为.npz格式
def save_data(train_data, train_labels, test_data, test_labels):
    """
    将处理后的数据保存为模型可接受的.npz格式
    """
    logger.info("开始保存数据...")
    
    # 将训练集数据转换为模型格式
    train_model_data, train_model_labels = convert_to_model_format(train_data, train_labels)
    
    # 将测试集数据转换为模型格式
    test_model_data, test_model_labels = convert_to_model_format(test_data, test_labels)
    
    # 转换为one-hot编码
    num_classes = 3  # 3个风险等级
    
    # 处理训练集标签
    if len(train_model_labels) > 0:
        train_one_hot = np.eye(num_classes)[train_model_labels]
    else:
        train_one_hot = np.array([])
    
    # 处理测试集标签
    if len(test_model_labels) > 0:
        test_one_hot = np.eye(num_classes)[test_model_labels]
    else:
        test_one_hot = np.array([])
    
    # 保存为.npz格式
    save_path = os.path.join(save_dir, 'imu_data.npz')
    np.savez(save_path,
             x_train=train_model_data,
             y_train=train_one_hot,
             x_test=test_model_data,
             y_test=test_one_hot)
    
    logger.info(f"数据保存完成，路径: {save_path}")
    logger.info(f"训练集数据形状: {train_model_data.shape}")
    logger.info(f"训练集标签形状: {train_one_hot.shape}")
    logger.info(f"测试集数据形状: {test_model_data.shape}")
    logger.info(f"测试集标签形状: {test_one_hot.shape}")

# 主函数
def main():
    """
    主函数
    """
    # 声明全局变量
    global data_root
    
    parser = argparse.ArgumentParser(description='IMU数据转换工具')
    parser.add_argument('--data-root', type=str, default=data_root, help='数据根目录')
    parser.add_argument('--single-file', type=str, default=None, help='单个CSV文件路径（用于测试）')
    
    args = parser.parse_args()
    
    # 更新数据根目录
    data_root = args.data_root
    
    logger.info("开始处理IMU数据...")
    
    if args.single_file:
        # 处理单个文件
        logger.info(f"处理单个文件: {args.single_file}")
        
        # 读取文件
        joint_data = read_imu_csv(args.single_file)
        
        if joint_data is None:
            logger.error("读取文件失败")
            return
        
        # 控制帧率
        joint_data = control_fps(joint_data, 1000, max_fps)  # 假设原始帧率为1000
        
        # 转换为模型格式
        data_list = [joint_data]
        labels = [0]  # 假设是第一个动作
        subjects = [1]
        
        # 保存数据（只保存训练集，因为只有一个样本）
        save_data(data_list, labels, [], [])
    else:
        # 收集所有数据
        all_data, all_labels, all_subjects = collect_data()
        
        if not all_data:
            logger.error("没有找到有效数据")
            return
        
        # 划分数据集
        train_data, train_labels, test_data, test_labels = split_dataset(all_data, all_labels, all_subjects)
        
        if not train_data or not test_data:
            logger.error("数据集划分失败")
            return
        
        # 保存数据
        save_data(train_data, train_labels, test_data, test_labels)
    
    logger.info("IMU数据处理完成")

if __name__ == '__main__':
    main()