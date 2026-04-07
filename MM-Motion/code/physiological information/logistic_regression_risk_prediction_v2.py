import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置科研绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# 定义33个subject列表
target_subjects = [
    'subject01', 'subject02', 'subject03', 'subject04', 'subject05',
    'subject06', 'subject07', 'subject08', 'subject09', 'subject10',
    'subject11', 'subject12', 'subject13', 'subject14', 'subject15',
    'subject16', 'subject17', 'subject18', 'subject19', 'subject20',
    'subject21', 'subject29', 'subject30', 'subject31', 'subject33',
    'subject34', 'subject35', 'subject36', 'subject37', 'subject38',
    'subject39', 'subject40', 'subject41'
]

# 风险标签映射
def get_risk_label(subject_id):
    risk_mapping = {
        'subject01': '高风险', 'subject02': '高风险', 'subject03': '中风险',
        'subject04': '中风险', 'subject05': '中风险', 'subject06': '低风险',
        'subject07': '高风险', 'subject08': '中风险', 'subject09': '中风险',
        'subject10': '高风险', 'subject11': '中风险', 'subject12': '中风险',
        'subject13': '中风险', 'subject14': '高风险', 'subject15': '中风险',
        'subject16': '中风险', 'subject17': '低风险', 'subject18': '低风险',
        'subject19': '低风险', 'subject20': '低风险', 'subject21': '中风险',
        'subject29': '高风险', 'subject30': '中风险', 'subject31': '中风险',
        'subject33': '中风险', 'subject34': '中风险', 'subject35': '中风险',
        'subject36': '中风险', 'subject37': '中风险', 'subject38': '中风险',
        'subject39': '中风险', 'subject40': '低风险', 'subject41': '低风险'
    }
    return risk_mapping.get(subject_id, '中风险')

# 加载问卷数据
def load_questionnaire_data():
    file_path = r"D:\Ankle-project\问卷数据\问卷数据受试者排序版.xlsx"
    try:
        df = pd.read_excel(file_path)
        print(f"成功加载问卷数据，共 {len(df)} 条记录")
        print(f"数据列：{list(df.columns)}")
        return df
    except Exception as e:
        print(f"加载数据失败：{e}")
        return None

# 数据预处理
def preprocess_data(df):
    # 清理列名
    df.columns = [col.strip() for col in df.columns]
    
    # 添加subject标识
    df['subject'] = [f'subject{i:02d}' for i in range(1, len(df) + 1)]
    print(f"添加subject标识后的数据行数：{len(df)}")
    
    # 提取目标subject的数据
    df = df[df['subject'].isin(target_subjects)]
    print(f"提取目标subject后的数据行数：{len(df)}")
    
    # 使用标注风险等级作为标签
    if '标注风险等级' in df.columns:
        df['risk_label'] = df['标注风险等级']
        print(f"标注风险等级分布：{df['risk_label'].value_counts()}")
    else:
        # 如果没有标注风险等级，使用原始风险标签
        df['risk_label'] = df['subject'].apply(get_risk_label)
        print(f"风险标签分布：{df['risk_label'].value_counts()}")
    
    # 编码风险标签
    le = LabelEncoder()
    df['risk_encoded'] = le.fit_transform(df['risk_label'])
    print(f"风险标签编码：{dict(zip(le.classes_, range(len(le.classes_))))}")
    
    return df, le

# 特征处理
def process_features(df):
    # 选择特征列
    feature_cols = []
    
    # 数值特征
    numeric_cols = ['3、年龄', '4、身高（cm）', '6、鞋码大小', '总分']
    
    # 分类特征
    categorical_cols = ['2、性别', '7、运动专项', '8、运动等级', '9、是否有如下疾病', '10、足形',
                       '11、踝关节感觉疼痛', '12、踝关节感觉不稳定', '13、急转身时感觉踝关节不稳定',
                       '14、下楼梯时踝关节感觉不稳定', '15、单腿站立时踝关节感觉不稳定',
                       '18、将要发生明显的崴脚动作时，能控制住', '19、曾经脚踝扭伤过', '20、脚踝扭伤次数']
    
    # 检查哪些列存在
    available_numeric = [col for col in numeric_cols if col in df.columns]
    available_categorical = [col for col in categorical_cols if col in df.columns]
    
    print(f"\n可用的数值特征：{available_numeric}")
    print(f"可用的分类特征：{available_categorical}")
    
    # 处理数值特征
    X_numeric = df[available_numeric].fillna(0)
    # 转换为数值类型
    for col in X_numeric.columns:
        X_numeric[col] = pd.to_numeric(X_numeric[col], errors='coerce').fillna(0)
    
    # 处理分类特征
    if available_categorical:
        # 对分类特征进行one-hot编码
        encoder = OneHotEncoder(handle_unknown='ignore')
        X_categorical = encoder.fit_transform(df[available_categorical].fillna('未知')).toarray()
        
        # 获取编码后的列名
        encoded_cols = encoder.get_feature_names_out(available_categorical)
        

        
        # 合并特征
        X = np.hstack([X_numeric, X_categorical])
        feature_names = available_numeric + list(encoded_cols)
    else:
        X = X_numeric.values
        feature_names = available_numeric
    
    print(f"\n总特征数量：{len(feature_names)}")
    print(f"前10个特征：{feature_names[:10]}")
    
    return X, feature_names

# 特征选择和模型训练
def train_logistic_regression(df, le):
    # 处理特征
    X, feature_names = process_features(df)
    y = df['risk_encoded'].values
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 对运动专项特征进行缩放，减小权重
    # 找到运动专项特征的索引
    sport_feature_indices = [i for i, feat in enumerate(feature_names) if '7、运动专项' in feat]
    # 缩小运动专项特征的权重
    for idx in sport_feature_indices:
        X_scaled[:, idx] *= 0.01  # 缩小100倍
    
    # 提高脚踝扭伤4次及以上特征的权重
    # 找到脚踝扭伤4次及以上特征的索引
    ankle_sprain_indices = [i for i, feat in enumerate(feature_names) if '20、脚踝扭伤次数_4次及以上' in feat]
    # 增加该特征的权重
    for idx in ankle_sprain_indices:
        X_scaled[:, idx] *= 50  # 放大50倍
    
    # 划分训练集和测试集（26个训练，7个测试）
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=7, train_size=26, random_state=42, stratify=y)
    
    print(f"\n训练集大小：{len(X_train)}")
    print(f"测试集大小：{len(X_test)}")
    
    # 训练随机森林模型
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # 评估模型
    print("\n模型评估：")
    print(f"准确率：{accuracy_score(y_test, y_pred):.4f}")
    print("\n分类报告：")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("\n混淆矩阵：")
    print(cm)
    
    # 可视化混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('混淆矩阵', fontweight='bold')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.savefig('X-experiments/confusion_matrix.png', bbox_inches='tight')
    plt.close()
    
    return model, feature_names, scaler, le, X_train, X_test, y_train, y_test

# 提取并展示重要特征
def extract_important_features(model, feature_names, le):
    # 获取特征重要性
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance = dict(zip(feature_names, importances))
        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        print("\n\n前10个重要特征：")
        top_features = []
        for i, (feat, importance) in enumerate(sorted_importance[:10]):
            print(f"{i+1}. {feat}: {importance:.4f}")
            top_features.append((feat, importance))
        
        # 可视化特征重要性
        plt.figure(figsize=(12, 6))
        features, importance = zip(*top_features)
        # 处理长特征名
        features = [f[:30] + '...' if len(f) > 30 else f for f in features]
        plt.barh(features[::-1], importance[::-1], color='skyblue')
        plt.xlabel('特征重要性')
        plt.ylabel('特征')
        plt.title('前10个重要特征', fontweight='bold')
        plt.tight_layout()
        plt.savefig('X-experiments/feature_importance.png', bbox_inches='tight')
        plt.close()
        
        return top_features
    elif hasattr(model, 'coef_'):
        # 逻辑回归的特征重要性
        coef = model.coef_
        # 对于多分类，每个类别都有一组权重
        for i, class_name in enumerate(le.classes_):
            print(f"\n{class_name} 类别的重要特征：")
            feature_weights = dict(zip(feature_names, coef[i]))
            # 按绝对值排序
            sorted_weights = sorted(feature_weights.items(), key=lambda x: abs(x[1]), reverse=True)
            # 显示前10个
            for feat, weight in sorted_weights[:10]:
                print(f"  {feat}: {weight:.4f}")
        
        # 计算平均权重绝对值
        mean_abs_weights = np.mean(np.abs(coef), axis=0)
        feature_importance = dict(zip(feature_names, mean_abs_weights))
        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        print("\n\n前10个权重最大的特征（所有类别平均）：")
        top_features = []
        for i, (feat, importance) in enumerate(sorted_importance[:10]):
            print(f"{i+1}. {feat}: {importance:.4f}")
            top_features.append((feat, importance))
        
        # 可视化特征重要性
        plt.figure(figsize=(12, 6))
        features, importance = zip(*top_features)
        # 处理长特征名
        features = [f[:30] + '...' if len(f) > 30 else f for f in features]
        plt.barh(features[::-1], importance[::-1], color='skyblue')
        plt.xlabel('平均权重绝对值')
        plt.ylabel('特征')
        plt.title('前10个重要特征', fontweight='bold')
        plt.tight_layout()
        plt.savefig('X-experiments/feature_importance.png', bbox_inches='tight')
        plt.close()
        
        return top_features
    else:
        print("模型没有特征重要性属性")
        return []

# 主函数
def main():
    print("=== 随机森林风险预测实验 ===")
    print(f"目标受试者数量：{len(target_subjects)}")
    print(f"受试者列表：{target_subjects}")
    
    # 加载数据
    df = load_questionnaire_data()
    if df is None:
        return
    
    # 预处理数据
    df_processed, le = preprocess_data(df)
    
    # 训练模型
    model, feature_names, scaler, le, X_train, X_test, y_train, y_test = train_logistic_regression(df_processed, le)
    
    # 提取重要特征
    top_features = extract_important_features(model, feature_names, le)
    
    # 保存结果
    results = {
        'subjects': target_subjects,
        'feature_count': len(feature_names),
        'top_features': top_features,
        'train_size': len(X_train),
        'test_size': len(X_test)
    }
    
    # 保存特征重要性到文件
    feature_df = pd.DataFrame(top_features, columns=['Feature', 'Importance'])
    feature_df.to_csv('X-experiments/top_features.csv', index=False, encoding='utf-8-sig')
    
    # 保存处理后的数据
    try:
        df_processed.to_csv('X-experiments/processed_data.csv', index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f"保存处理后的数据失败：{e}")
    
    print("\n=== 实验完成 ===")
    print(f"结果已保存到 X-experiments 目录")
    print(f"特征重要性文件：X-experiments/top_features.csv")
    print(f"混淆矩阵图片：X-experiments/confusion_matrix.png")
    print(f"特征重要性图片：X-experiments/feature_importance.png")
    print(f"处理后的数据：X-experiments/processed_data.csv")

if __name__ == "__main__":
    main()
