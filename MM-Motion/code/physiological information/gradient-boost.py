import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
import os

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# 核心配置优化 - 大幅提高关键特征权重
# ------------------------------------------------------------------------------
EXCEL_PATH = "D:\Ankle-project\问卷数据\问卷数据受试者排序版.xlsx"
CAIT_LABEL_COL = "标注风险等级"

# 大幅提高扭伤相关特征的权重
TARGET_FEATURE_WEIGHTS = {
    "扭伤次数": 10.0,  # 大幅提高权重
    "最近扭伤时间": 8.0,
    "曾经脚踝扭伤过": 6.0,
    "脚踝问题综合得分": 4.0,
    "恢复状况": 4.0,
    "翻转方向": 3.0,
    "脚踝不稳定": 3.0,
    "脚踝疼痛": 3.0
}

# 强制排除的列（包括总分）
EXCLUDE_COLS = ['总分', '风险等级', '序号', '1、姓名', CAIT_LABEL_COL]


# ------------------------------------------------------------------------------
# 数据读取和预处理优化
# ------------------------------------------------------------------------------
def load_and_preprocess_data(file_path):
    """优化数据加载和预处理"""
    try:
        df = pd.read_excel(file_path, sheet_name="Sheet1", engine="openpyxl")
    except:
        df = pd.read_excel(file_path)

    # 基础列筛选
    base_cols = [col for col in df.columns if 'Unnamed' not in str(col)]
    df = df[base_cols].copy()

    return df


def enhanced_feature_engineering(df):
    """增强特征工程，特别关注扭伤相关特征"""
    df_processed = df.copy()

    # 1. 基础信息编码
    df_processed['性别_编码'] = df_processed['2、性别'].map({'男': 1, '女': 0}).fillna(-1)

    # 年龄分段
    age_mapping = {
        '<18岁': 0, '18-25岁': 1, '26-30岁': 2,
        '31-40岁': 3, '41-50岁': 4, '>50岁': 5
    }
    df_processed['年龄_编码'] = df_processed['3、年龄'].map(age_mapping).fillna(-1)

    # 运动等级细化
    sport_level_mapping = {
        '国家级健将': 5, '一级': 4, '二级': 3, '三级': 2, '无等级': 1, '二级以下': 1
    }
    df_processed['运动等级_编码'] = df_processed['8、运动等级'].map(sport_level_mapping).fillna(1)

    # 2. 脚踝相关问题综合评分 - 重点关注
    ankle_questions = [
        '11、踝关节感觉疼痛', '12、踝关节感觉不稳定', '13、急转身时感觉踝关节不稳定',
        '14、下楼梯时踝关节感觉不稳定', '15、单腿站立时踝关节感觉不稳定'
    ]

    # 统一编码这些问题的严重程度（0-4分，0最严重）
    severity_mapping = {
        '从未': 4, '有时在运动中有此感觉': 3, '经常于运动中有此感觉': 2,
        '有时在日常生活中有此感觉': 1, '经常在日常生活中有此感觉': 0,
        '立即': 4, '经常': 3, '偶尔': 2, '从不': 0, '我从未崴过脚': 4
    }

    for col in ankle_questions:
        if col in df_processed.columns:
            encoded_col = f"{col}_严重度"
            df_processed[encoded_col] = df_processed[col].map(severity_mapping).fillna(2)

    # 计算脚踝问题综合得分
    severity_cols = [f"{col}_严重度" for col in ankle_questions if f"{col}_严重度" in df_processed.columns]
    if severity_cols:
        df_processed['脚踝问题综合得分'] = df_processed[severity_cols].mean(axis=1)

    # 3. 扭伤历史特征强化 - 核心特征
    # 扭伤次数编码 - 关键特征
    def encode_sprain_count(x):
        if x == '一次':
            return 1
        elif x == '2-3次':
            return 2
        elif x == '≥4次':
            return 3
        elif x == '(跳过)':
            return 0
        else:
            return 0

    if '20、脚踝扭伤次数' in df_processed.columns:
        df_processed['扭伤次数_编码'] = df_processed['20、脚踝扭伤次数'].apply(encode_sprain_count)

    # 最近扭伤时间编码（时间越近风险越高）- 关键特征
    recent_sprain_mapping = {
        '1个月内': 4, '1–3个月': 3, '3–6个月': 2,
        '半年–1年': 1, '1–2年': 0, '两年以上或无': 0, '(跳过)': 0
    }
    if '21、最近一次脚踝扭伤时间' in df_processed.columns:
        df_processed['最近扭伤时间_编码'] = df_processed['21、最近一次脚踝扭伤时间'].map(recent_sprain_mapping).fillna(0)

    # 是否曾经扭伤过 - 关键特征
    if '19、曾经脚踝扭伤过' in df_processed.columns:
        df_processed['曾经扭伤过_编码'] = df_processed['19、曾经脚踝扭伤过'].map({'是': 1, '否': 0}).fillna(0)

    # 4. 患侧位置编码
    def encode_ankle_position(position):
        if pd.isna(position) or position == '(跳过)':
            return 'unknown'
        position_str = str(position).lower()
        if '左脚' in position_str or '左' in position_str:
            return 'left'
        elif '右脚' in position_str or '右' in position_str:
            return 'right'
        elif '双' in position_str or '两' in position_str or '全踝' in position_str:
            return 'both'
        else:
            return 'unknown'

    if '22、最近一次脚踝扭伤位置' in df_processed.columns:
        df_processed['患侧位置'] = df_processed['22、最近一次脚踝扭伤位置'].apply(encode_ankle_position)
    else:
        df_processed['患侧位置'] = 'unknown'

    # 为左右脚踝创建风险标记
    df_processed['左脚风险标记'] = df_processed['患侧位置'].apply(lambda x: 1 if x in ['left', 'both'] else 0)
    df_processed['右脚风险标记'] = df_processed['患侧位置'].apply(lambda x: 1 if x in ['right', 'both'] else 0)

    # 5. 翻转方向编码
    if '27、最近一次脚踝扭伤时脚踝翻转方向' in df_processed.columns:
        flip_mapping = {
            '内翻': 2,  # 内翻通常风险更高
            '外翻': 1,
            '垂直压缩': 1,
            '其他': 0,
            '(跳过)': 0
        }
        df_processed['翻转方向_编码'] = df_processed['27、最近一次脚踝扭伤时脚踝翻转方向'].map(flip_mapping).fillna(0)

    # 6. 康复情况编码 - 重要特征
    rehab_mapping = {
        '完全恢复': 3, '恢复良好，但偶尔会感觉不适': 2, '恢复一般，存在持续的不适或疼痛': 1,
        '未完全恢复，仍有反复损伤或疼痛': 0, '(跳过)': 2
    }
    if '26、最近一次脚踝扭伤恢复后，脚踝恢复状况' in df_processed.columns:
        df_processed['恢复状况_编码'] = df_processed['26、最近一次脚踝扭伤恢复后，脚踝恢复状况'].map(
            rehab_mapping).fillna(2)

    # 7. 身体指标特征 - 次要特征
    if '4、身高（cm）' in df_processed.columns and '5、体重（kg）' in df_processed.columns:
        df_processed['身高'] = pd.to_numeric(df_processed['4、身高（cm）'], errors='coerce')
        df_processed['体重'] = pd.to_numeric(df_processed['5、体重（kg）'], errors='coerce')
        df_processed['BMI'] = df_processed['体重'] / ((df_processed['身高'] / 100) ** 2)
        df_processed['BMI_分类'] = pd.cut(df_processed['BMI'],
                                          bins=[0, 18.5, 24, 28, 100],
                                          labels=[0, 1, 2, 3])

    # 8. 创建复合特征
    risk_factors = []
    if '扭伤次数_编码' in df_processed.columns:
        risk_factors.append(df_processed['扭伤次数_编码'])
    if '最近扭伤时间_编码' in df_processed.columns:
        risk_factors.append(df_processed['最近扭伤时间_编码'])
    if '脚踝问题综合得分' in df_processed.columns:
        risk_factors.append(4 - df_processed['脚踝问题综合得分'])  # 反转得分，越高越风险
    if '翻转方向_编码' in df_processed.columns:
        risk_factors.append(df_processed['翻转方向_编码'])

    if risk_factors:
        df_processed['风险综合指数'] = sum(risk_factors) / len(risk_factors)

    return df_processed


# ------------------------------------------------------------------------------
# 特征选择优化 - 强制包含关键特征
# ------------------------------------------------------------------------------
def select_best_features(X, y, feature_names, k=15):
    """选择最佳特征 - 强制包含关键扭伤相关特征"""

    # 关键特征列表（必须包含）
    critical_features_keywords = ['扭伤次数', '最近扭伤时间', '曾经扭伤过', '脚踝问题综合得分',
                                  '恢复状况', '翻转方向', '脚踝不稳定', '脚踝疼痛']

    # 找到关键特征的索引
    critical_indices = []
    for i, name in enumerate(feature_names):
        for keyword in critical_features_keywords:
            if keyword in name:
                critical_indices.append(i)
                break

    # 使用梯度提升进行特征重要性排序
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X, y)
    feature_importance = gb.feature_importances_

    # 选择重要性最高的k个特征，但确保包含所有关键特征
    all_indices = list(range(len(feature_names)))

    # 移除关键特征，然后从剩余特征中选择
    remaining_indices = [i for i in all_indices if i not in critical_indices]

    # 从剩余特征中选择重要性最高的
    if len(remaining_indices) > 0:
        remaining_importance = feature_importance[remaining_indices]
        selected_remaining = np.argsort(remaining_importance)[-(k - len(critical_indices)):]
        selected_remaining_indices = [remaining_indices[i] for i in selected_remaining]
    else:
        selected_remaining_indices = []

    # 合并关键特征和选择的其他特征
    selected_indices = list(set(critical_indices + selected_remaining_indices))

    print(f"\n强制包含的关键特征 ({len(critical_indices)}个):")
    for idx in critical_indices:
        print(f"  - {feature_names[idx]} (重要性: {feature_importance[idx]:.4f})")

    print(f"\n特征重要性排序 (前{len(selected_indices)}个):")
    importance_df = pd.DataFrame({
        '特征': [feature_names[i] for i in selected_indices],
        '重要性': feature_importance[selected_indices]
    }).sort_values('重要性', ascending=False)
    print(importance_df.to_string(index=False))

    return selected_indices


# ------------------------------------------------------------------------------
# 特征加权函数
# ------------------------------------------------------------------------------
def apply_feature_weights(X, feature_names, weights_dict):
    """应用特征权重"""
    X_weighted = X.copy()

    print("\n应用特征权重:")
    for feature_pattern, weight in weights_dict.items():
        # 查找匹配的特征列
        matching_indices = [i for i, name in enumerate(feature_names)
                            if feature_pattern in name]

        for idx in matching_indices:
            X_weighted[:, idx] *= weight
            print(f"  - {feature_names[idx]} × {weight}")

    return X_weighted


# ------------------------------------------------------------------------------
# 患侧风险评估函数
# ------------------------------------------------------------------------------
def calculate_side_specific_risk(base_risk_proba, left_marker, right_marker, risk_factors):
    """计算左右脚踝特定风险"""

    # 基础风险调整系数
    left_adjustment = 0.0
    right_adjustment = 0.0

    # 如果有患侧标记，相应侧风险提高
    if left_marker == 1:
        left_adjustment += 0.3  # 患侧风险提高30%
    if right_marker == 1:
        right_adjustment += 0.3

    # 根据风险因素进一步调整
    risk_factor_score = sum(risk_factors) / len(risk_factors) if risk_factors else 0
    additional_adjustment = risk_factor_score * 0.1

    # 计算左右脚踝风险概率
    left_risk_proba = base_risk_proba.copy()
    right_risk_proba = base_risk_proba.copy()

    # 调整高风险概率
    if left_adjustment > 0:
        left_risk_proba[2] = min(1.0, left_risk_proba[2] + left_adjustment + additional_adjustment)
        # 相应调整其他概率
        remaining_prob = 1.0 - left_risk_proba[2]
        left_risk_proba[0] = left_risk_proba[0] * (remaining_prob / (left_risk_proba[0] + left_risk_proba[1]))
        left_risk_proba[1] = left_risk_proba[1] * (remaining_prob / (left_risk_proba[0] + left_risk_proba[1]))

    if right_adjustment > 0:
        right_risk_proba[2] = min(1.0, right_risk_proba[2] + right_adjustment + additional_adjustment)
        remaining_prob = 1.0 - right_risk_proba[2]
        right_risk_proba[0] = right_risk_proba[0] * (remaining_prob / (right_risk_proba[0] + right_risk_proba[1]))
        right_risk_proba[1] = right_risk_proba[1] * (remaining_prob / (right_risk_proba[0] + right_risk_proba[1]))

    # 确定风险等级
    risk_levels = ['低风险', '中风险', '高风险']
    left_risk_level = risk_levels[np.argmax(left_risk_proba)]
    right_risk_level = risk_levels[np.argmax(right_risk_proba)]

    return left_risk_level, right_risk_level, left_risk_proba, right_risk_proba


# ------------------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("脚踝扭伤风险评估模型 - 优化版（专注扭伤特征）")
    print("=" * 60)

    # 1. 数据加载和预处理
    print("1. 加载和预处理数据...")
    df = load_and_preprocess_data(EXCEL_PATH)
    df_processed = enhanced_feature_engineering(df)

    # 2. 准备特征和目标变量
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()

    # 移除排除的列
    feature_cols = [col for col in numeric_cols if col not in EXCLUDE_COLS]

    # 目标变量处理
    if CAIT_LABEL_COL in df_processed.columns:
        cait_mapping = {"低风险": 0, "中风险": 1, "高风险": 2}
        df_processed['CAIT标签_数字'] = df_processed[CAIT_LABEL_COL].map(cait_mapping).fillna(1)

    X = df_processed[feature_cols].values
    y = df_processed['CAIT标签_数字'].values

    print(f"特征矩阵: {X.shape}")
    print(f"目标变量分布: {np.unique(y, return_counts=True)}")

    # 3. 数据预处理
    print("\n2. 数据预处理...")
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # 4. 特征选择和加权
    print("\n3. 特征选择和加权...")
    selected_features = select_best_features(X_scaled, y, feature_cols, k=12)
    X_selected = X_scaled[:, selected_features]
    selected_feature_names = [feature_cols[i] for i in selected_features]

    print(f"\n最终选择的特征 ({len(selected_feature_names)}个):")
    for name in selected_feature_names:
        print(f"  - {name}")

    # 应用特征权重
    X_weighted = apply_feature_weights(X_selected, selected_feature_names, TARGET_FEATURE_WEIGHTS)

    # 5. 只使用梯度提升模型
    print("\n4. 梯度提升模型训练...")

    # 优化梯度提升参数
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42
    )

    # 5折交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(gb_model, X_weighted, y, cv=cv, scoring='accuracy')

    print(f"梯度提升5折交叉验证准确率: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
    print(f"每折准确率: {cv_scores.round(3)}")

    # 6. 最终模型训练和预测
    X_train, X_test, y_train, y_test = train_test_split(
        X_weighted, y, test_size=0.3, random_state=42, stratify=y
    )

    gb_model.fit(X_train, y_train)
    y_pred = gb_model.predict(X_test)

    print(f"\n5. 测试集性能 (梯度提升):")
    print(f"准确率: {accuracy_score(y_test, y_pred):.3f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['低风险', '中风险', '高风险']))

    # 7. 特征重要性分析
    print("\n6. 最终特征重要性排序:")
    importance_df = pd.DataFrame({
        '特征': selected_feature_names,
        '重要性': gb_model.feature_importances_
    }).sort_values('重要性', ascending=False)

    print(importance_df.to_string(index=False))

    # 8. 生成风险评估结果（包含左右脚踝评估）
    print("\n7. 生成最终风险评估...")
    # 使用完整数据训练最终模型
    final_model = gb_model
    final_model.fit(X_weighted, y)

    # 预测所有样本
    all_predictions = final_model.predict(X_weighted)
    prediction_proba = final_model.predict_proba(X_weighted)

    # 为每个受试者计算左右脚踝风险
    side_specific_risks = []
    for i in range(len(df_processed)):
        base_proba = prediction_proba[i]

        # 获取患侧标记
        left_marker = df_processed.iloc[i]['左脚风险标记'] if '左脚风险标记' in df_processed.columns else 0
        right_marker = df_processed.iloc[i]['右脚风险标记'] if '右脚风险标记' in df_processed.columns else 0

        # 获取风险因素
        risk_factors = []
        if '扭伤次数_编码' in df_processed.columns:
            risk_factors.append(df_processed.iloc[i]['扭伤次数_编码'])
        if '最近扭伤时间_编码' in df_processed.columns:
            risk_factors.append(df_processed.iloc[i]['最近扭伤时间_编码'])

        # 计算左右脚踝特定风险
        left_risk, right_risk, left_proba, right_proba = calculate_side_specific_risk(
            base_proba, left_marker, right_marker, risk_factors
        )

        side_specific_risks.append({
            '左脚风险等级': left_risk,
            '右脚风险等级': right_risk,
            '左脚高风险概率': left_proba[2],
            '右脚高风险概率': right_proba[2]
        })

    side_risks_df = pd.DataFrame(side_specific_risks)

    # 创建完整结果表格
    result_df = pd.DataFrame({
        '序号': df_processed['序号'],
        '姓名': df_processed.get('1、姓名', '未知'),
        '原始CAIT风险': df_processed.get(CAIT_LABEL_COL, '未知'),
        '整体预测风险': [['低风险', '中风险', '高风险'][p] for p in all_predictions],
        '左脚风险等级': side_risks_df['左脚风险等级'],
        '右脚风险等级': side_risks_df['右脚风险等级'],
        '左脚高风险概率': side_risks_df['左脚高风险概率'].round(3),
        '右脚高风险概率': side_risks_df['右脚高风险概率'].round(3),
        '患侧位置': df_processed.get('患侧位置', 'unknown'),
        '实际扭伤位置': df_processed.get('22、最近一次脚踝扭伤位置', '未知')
    })

    # 添加整体预测概率
    for i, risk_level in enumerate(['低风险', '中风险', '高风险']):
        result_df[f'整体{risk_level}_概率'] = prediction_proba[:, i].round(3)

    print("\n预测结果样例 (包含左右脚踝风险评估):")
    print(result_df.head(10).to_string(index=False))

    # 9. 保存结果到指定路径
    save_dir = r"D:\项目-脚踝扭伤恢复模型\结果数据"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, '脚踝扭伤风险评估优化结果.xlsx')

    result_df.to_excel(save_path, index=False)
    print(f"\n✅ 结果已保存到: {save_path}")

    # 10. 输出风险统计
    print("\n8. 风险等级统计:")
    print(f"整体高风险人数: {sum(result_df['整体预测风险'] == '高风险')}")
    print(f"左脚高风险人数: {sum(result_df['左脚风险等级'] == '高风险')}")
    print(f"右脚高风险人数: {sum(result_df['右脚风险等级'] == '高风险')}")
    print(f"双侧高风险人数: {sum((result_df['左脚风险等级'] == '高风险') & (result_df['右脚风险等级'] == '高风险'))}")

    return result_df, cv_scores.mean(), accuracy_score(y_test, y_pred)


# ------------------------------------------------------------------------------
# 执行主程序
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    result_df, cv_accuracy, test_accuracy = main()

    print("\n" + "=" * 60)
    print("模型训练完成!")
    print(f"梯度提升交叉验证平均准确率: {cv_accuracy:.3f}")
    print(f"梯度提升测试集准确率: {test_accuracy:.3f}")
    print("=" * 60)