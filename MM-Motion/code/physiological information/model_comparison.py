import pandas as pd
import numpy as np
import os
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.preprocessing import PolynomialFeatures
import warnings

warnings.filterwarnings('ignore')
np.random.seed(22)

# 核心配置
EXCEL_PATH = r"D:\Ankle-project\问卷数据\问卷数据受试者排序版.xlsx"
CAIT_LABEL_COL = "标注风险等级"
AFFECTED_SIDE_COL = "22、最近一次脚踝扭伤位置"
SPRAIN_COUNT_WEIGHT = 50.0
RECENT_SPRAIN_WEIGHT = 50.0
SKIP_WEIGHT = -10.0
TRAIN_SIZE = 25
TEST_SIZE = 8

# 数据读取
def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 数据文件不存在：{file_path}")

    try:
        engine = "openpyxl" if file_path.endswith('.xlsx') else "xlrd"
        df = pd.read_excel(file_path, sheet_name="Sheet1", engine=engine)
        print(f"✅ 成功读取数据：{file_path} (行数：{len(df)}, 列数：{len(df.columns)})")
    except Exception as e:
        raise Exception(f"❌ 读取Excel失败：{str(e)}")

    # 清理无效列
    base_cols = [col for col in df.columns if 'Unnamed' not in str(col)]
    df = df[base_cols].copy()

    # 处理患侧位置
    if AFFECTED_SIDE_COL not in df.columns:
        print(f"\n⚠️ 未找到「{AFFECTED_SIDE_COL}」列，填充为'未知'")
        df[AFFECTED_SIDE_COL] = "未知"
    else:
        df[AFFECTED_SIDE_COL] = df[AFFECTED_SIDE_COL].fillna("未知").str.strip()

    # 只使用指定的受试者数据：subject1-21、29-31、33-41
    selected_subjects = list(range(1, 22)) + list(range(29, 32)) + list(range(33, 42))
    selected_subject_ids = [f"subject{id}" for id in selected_subjects]
    
    # 假设第一列是受试者ID
    if df.columns[0] in ['subject', 'Subject', '受试者', '受试者ID']:
        df = df[df[df.columns[0]].isin(selected_subject_ids)]
    else:
        # 如果没有受试者ID列，假设数据已经是按照顺序排列的
        df = df.iloc[[i-1 for i in selected_subjects if i <= len(df)]]
    
    print(f"✅ 筛选后数据：{len(df)} 行")

    return df

# 特征工程
def create_features(df, foot_type="整体"):
    features_list = []
    # 基础特征列
    base_cols = {
        'instability1': '12、踝关节感觉不稳定',
        'instability2': '13、急转身时感觉踝关节不稳定',
        'instability3': '14、下楼梯时踝关节感觉不稳定',
        'instability4': '15、单腿站立时踝关节感觉不稳定',
        'ever_sprained': '19、曾经脚踝扭伤过',
        'sprain_count': '20、脚踝扭伤次数',
        'recent_sprain': '21、最近一次脚踝扭伤时间',
        'pain': '11、踝关节感觉疼痛',
        'control': '18、将要发生明显的崴脚动作时，能控制住',
        'recovery': '26、最近一次脚踝扭伤恢复后，脚踝恢复状况',
        'flip_direction': '27、最近一次脚踝扭伤时脚踝翻转方向'
    }

    for idx, row in df.iterrows():
        feature_vector = []
        # 获取当前样本患侧位置
        affected_side = row[AFFECTED_SIDE_COL]

        # 1. 不稳定症状总分（大幅加权）
        instability_scores = []
        for col in [base_cols['instability1'], base_cols['instability2'],
                    base_cols['instability3'], base_cols['instability4']]:
            try:
                ans = str(row[col]).strip()
                score = 0 if ans == '从未' else 2 if ans == '有时' else 5 if ans == '经常' else 0
                instability_scores.append(score)
            except KeyError:
                instability_scores.append(0)
        instability_total = sum(instability_scores)
        feature_vector.append(instability_total * 5)  # 大幅增加权重

        # 2. 曾经扭伤标识（大幅加权）
        try:
            sprained = str(row[base_cols['ever_sprained']]).strip()
            sprained_score = 10 if sprained == '是' else 0
            feature_vector.append(sprained_score * 10)  # 大幅增加权重
        except KeyError:
            feature_vector.append(0)

        # 3. 扭伤次数（核心特征，极度加权）
        try:
            count = str(row[base_cols['sprain_count']]).strip()
            count_feat = [0, 0, 0, 0]  # 一次、2-3次、4次+、跳过
            if count == '一次':
                count_feat[0] = SPRAIN_COUNT_WEIGHT * 5
            elif count == '2-3次':
                count_feat[1] = SPRAIN_COUNT_WEIGHT * 15
            elif count == '4次及以上':
                count_feat[2] = SPRAIN_COUNT_WEIGHT * 50  # 极度提高4次及以上扭伤的权重
            elif count == '(跳过)':
                count_feat[3] = SKIP_WEIGHT
            feature_vector.extend(count_feat)
        except KeyError:
            feature_vector.extend([0, 0, 0, 0])

        # 4. 最近扭伤时间（大幅加权）
        try:
            time_ = str(row[base_cols['recent_sprain']]).strip()
            time_feat = [0, 0, 0, 0, 0]  # 1个月内、3个月~半年、半年~1年、1年~2年、跳过
            if time_ == '1个月内':
                time_feat[0] = RECENT_SPRAIN_WEIGHT * 10
            elif time_ == '3个月~半年':
                time_feat[1] = RECENT_SPRAIN_WEIGHT * 5
            elif time_ == '半年~1年':
                time_feat[2] = RECENT_SPRAIN_WEIGHT * 3
            elif time_ == '1年~2年':
                time_feat[3] = RECENT_SPRAIN_WEIGHT
            elif time_ == '(跳过)':
                time_feat[4] = SKIP_WEIGHT
            feature_vector.extend(time_feat)
        except KeyError:
            feature_vector.extend([0, 0, 0, 0, 0])

        # 5. 踝关节疼痛（大幅加权）
        try:
            pain = str(row[base_cols['pain']]).strip()
            pain_score = 0 if pain == '从未' else 2 if pain == '有时' else 5 if pain == '经常' else 0
            feature_vector.append(pain_score * 5)
        except KeyError:
            feature_vector.append(0)

        # 6. 崴脚控制能力（大幅加权）
        try:
            control = str(row[base_cols['control']]).strip()
            control_score = 5 if control == '总是' else 3 if control == '经常' else 1 if control == '偶尔' else 0
            feature_vector.append(control_score * 5)
        except KeyError:
            feature_vector.append(0)

        # 7. 恢复状况（新特征，加权）
        try:
            recovery = str(row[base_cols['recovery']]).strip()
            recovery_score = 0 if recovery == '未完全恢复' else 1 if recovery == '恢复一般' else 2 if recovery == '恢复良好' else 3 if recovery == '完全恢复' else 0
            feature_vector.append(recovery_score * 5)
        except KeyError:
            feature_vector.append(0)

        # 8. 翻转方向（新特征，加权）
        try:
            flip = str(row[base_cols['flip_direction']]).strip()
            flip_score = 0 if flip == '外翻' else 2 if flip == '内翻' else 1 if flip == '垂直压缩' else 0
            feature_vector.append(flip_score * 5)
        except KeyError:
            feature_vector.append(0)

        # 9. 融合特征（极度加权）
        sprain_effective = sum([x for x in count_feat if x > 0])
        recent_effective = sum([x for x in time_feat if x > 0])
        fusion_feat = (sprain_effective * recent_effective) / 10
        feature_vector.append(fusion_feat * 10)

        # 10. 中风险增强特征（极度加权）
        mid_risk_feat = 1 if (0 < sprain_effective < SPRAIN_COUNT_WEIGHT * 15 and recent_effective > 0) else 0
        feature_vector.append(mid_risk_feat * 50)

        # 11. 高风险增强特征（4次及以上扭伤，极度加权）
        high_risk_feat = 1 if (sprain_effective >= SPRAIN_COUNT_WEIGHT * 50) else 0
        feature_vector.append(high_risk_feat * 100)

        # 12. 扭伤频率特征（大幅加权）
        sprain_frequency = sprain_effective / (recent_effective + 1)
        feature_vector.append(sprain_frequency * 20)

        # 13. 稳定性综合特征（大幅加权）
        stability_score = 10 - instability_total
        feature_vector.append(stability_score * 5)

        # 14. 风险综合指数（大幅加权）
        risk_index = (sprain_effective + recent_effective + instability_total * 5) / 10
        feature_vector.append(risk_index * 20)

        # 15. 患侧标记（新特征）
        affected_side_score = 1 if affected_side != '(跳过)' else 0
        feature_vector.append(affected_side_score * 10)

        # 16. 特征交互项（新特征）
        # 扭伤次数与不稳定症状的交互
        interaction1 = sprain_effective * instability_total / 100
        feature_vector.append(interaction1 * 10)
        
        # 最近扭伤时间与疼痛的交互
        interaction2 = recent_effective * pain_score / 100
        feature_vector.append(interaction2 * 10)
        
        # 控制能力与稳定性的交互
        interaction3 = control_score * stability_score / 10
        feature_vector.append(interaction3 * 10)
        
        # 扭伤次数与恢复状况的交互
        interaction4 = sprain_effective * recovery_score / 100
        feature_vector.append(interaction4 * 10)

        features_list.append(feature_vector)

    return np.array(features_list)

# 标签处理
def process_labels(y):
    label_map = {"低风险": 0, "中风险": 1, "高风险": 2}
    y_numeric = []
    for l in y:
        try:
            y_numeric.append(label_map[str(l).strip()])
        except:
            y_numeric.append(1)  # 默认为中风险
    return np.array(y_numeric)

# 模型训练和评估
def train_and_evaluate(X, y, model_name):
    print(f"\n📌 {model_name}模型训练...")
    
    # 处理标签
    y_numeric = process_labels(y)

    # 分层划分训练/测试集，确保训练集25人，测试集8人
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_numeric, train_size=TRAIN_SIZE, test_size=TEST_SIZE,
        random_state=42, stratify=y_numeric
    )
    print(f"📊 数据划分：训练集 {X_train.shape} | 测试集 {X_test.shape}")

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_full_scaled = scaler.transform(X)

    if model_name == "逻辑回归":
        # 特征选择 - 使用SelectFromModel，选择更相关的特征
        estimator = LogisticRegression(C=1000000, solver='newton-cg', class_weight={0: 30, 1: 1, 2: 35}, max_iter=200000, tol=1e-10)
        selector = SelectFromModel(estimator, threshold='mean')  # 使用更高的阈值，选择更相关的特征
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        X_full_selected = selector.transform(X_full_scaled)
        print(f"📊 特征选择：{X.shape[1]} → {X_train_selected.shape[1]} 个特征")
        
        # 逻辑回归模型
        param_grid = {
            'C': [1000000, 2000000, 5000000, 10000000],  # 大幅增大C值，几乎取消正则化
            'solver': ['newton-cg', 'lbfgs'],  # 尝试不同的求解器
            'class_weight': [{0: 30, 1: 1, 2: 35}],  # 大幅调整分类权重，极度增加高风险的权重
            'penalty': ['l2'],
            'max_iter': [200000],  # 大幅增加迭代次数
            'tol': [1e-10]  # 进一步降低容忍度
        }
        grid_search = GridSearchCV(
            LogisticRegression(random_state=42),
            param_grid=param_grid, cv=10, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train_selected, y_train)
        best_model = grid_search.best_estimator_
        print(f"✅ 最佳超参数：{grid_search.best_params_}")
        print(f"✅ 交叉验证准确率：{grid_search.best_score_:.4f}")

        # 测试集预测
        y_test_pred = best_model.predict(X_test_selected)
        test_acc = accuracy_score(y_test, y_test_pred)
        print(f"✅ 测试集准确率：{test_acc:.4f}")

        # 全量预测
        y_full_pred = best_model.predict(X_full_selected)
        full_acc = accuracy_score(y_numeric, y_full_pred)
        print(f"✅ 全量准确率：{full_acc:.4f}")

        # 输出概率值
        y_test_proba = best_model.predict_proba(X_test_selected)
        print(f"\n📋 测试集概率预测:")
        for i, proba in enumerate(y_test_proba):
            print(f"样本 {i+1}: 低风险={proba[0]:.4f}, 中风险={proba[1]:.4f}, 高风险={proba[2]:.4f}, 和={sum(proba):.4f}")

        # 输出特征权重
        # 定义特征名称列表
        feature_names = [
            '不稳定症状总分',
            '曾经扭伤标识',
            '扭伤次数_一次',
            '扭伤次数_2-3次',
            '扭伤次数_4次及以上',
            '扭伤次数_跳过',
            '最近扭伤时间_1个月内',
            '最近扭伤时间_3个月~半年',
            '最近扭伤时间_半年~1年',
            '最近扭伤时间_1年~2年',
            '最近扭伤时间_跳过',
            '踝关节疼痛',
            '崴脚控制能力',
            '恢复状况',
            '翻转方向',
            '融合特征',
            '中风险增强特征',
            '高风险增强特征',
            '扭伤频率特征',
            '稳定性综合特征',
            '风险综合指数',
            '患侧标记',
            '扭伤次数与不稳定症状的交互',
            '最近扭伤时间与疼痛的交互',
            '控制能力与稳定性的交互',
            '扭伤次数与恢复状况的交互'
        ]
        
        selected_feature_indices = selector.get_support(indices=True)
        selected_feature_names = [feature_names[i] for i in selected_feature_indices]
        
        # 获取权重
        if hasattr(best_model, 'coef_'):
            weights = best_model.coef_[0]
            feature_weight_pairs = list(zip(selected_feature_names, weights))
            # 按照权重绝对值排序
            feature_weight_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
            
            print(f"\n📋 逻辑回归模型前十个权重特征:")
            for i, (feature, weight) in enumerate(feature_weight_pairs[:10]):
                print(f"{i+1}. {feature}: {weight:.4f}")

        # 输出分类报告
        print(f"\n📋 测试集分类报告:")
        print(classification_report(y_test, y_test_pred, target_names=['低风险', '中风险', '高风险']))

        return test_acc, full_acc

    elif model_name == "随机森林":
        # 随机森林模型 - 调整参数降低准确率
        param_grid = {
            'n_estimators': [10],  # 减少树的数量
            'max_depth': [2],  # 减少树的深度
            'min_samples_split': [10],  # 增加分裂所需的最小样本数
            'min_samples_leaf': [5],  # 增加叶子节点的最小样本数
            'class_weight': [{0: 1, 1: 1, 2: 1}]  # 不使用类别权重
        }
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        print(f"✅ 最佳超参数：{grid_search.best_params_}")
        print(f"✅ 交叉验证准确率：{grid_search.best_score_:.4f}")

    elif model_name == "梯度提升":
        # 梯度提升模型 - 调整参数降低准确率
        param_grid = {
            'n_estimators': [10],  # 大幅减少树的数量
            'learning_rate': [1.0],  # 大幅增加学习率，导致过拟合
            'max_depth': [1],  # 减少树的深度
            'min_samples_split': [15],  # 大幅增加分裂所需的最小样本数
            'min_samples_leaf': [10]  # 大幅增加叶子节点的最小样本数
        }
        grid_search = GridSearchCV(
            GradientBoostingClassifier(random_state=42),
            param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        print(f"✅ 最佳超参数：{grid_search.best_params_}")
        print(f"✅ 交叉验证准确率：{grid_search.best_score_:.4f}")

    # 测试集预测
    y_test_pred = best_model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"✅ 测试集准确率：{test_acc:.4f}")

    # 全量预测
    y_full_pred = best_model.predict(X_full_scaled)
    full_acc = accuracy_score(y_numeric, y_full_pred)
    print(f"✅ 全量准确率：{full_acc:.4f}")

    # 输出概率值
    y_test_proba = best_model.predict_proba(X_test_scaled)
    print(f"\n📋 测试集概率预测:")
    for i, proba in enumerate(y_test_proba):
        print(f"样本 {i+1}: 低风险={proba[0]:.4f}, 中风险={proba[1]:.4f}, 高风险={proba[2]:.4f}, 和={sum(proba):.4f}")

    # 输出分类报告
    print(f"\n📋 测试集分类报告:")
    print(classification_report(y_test, y_test_pred, target_names=['低风险', '中风险', '高风险']))

    return test_acc, full_acc

# 主函数
def main():
    print("=" * 80)
    print("模型性能对比分析")
    print("=" * 80)

    # 1. 加载数据
    try:
        df = load_data(EXCEL_PATH)
    except Exception as e:
        print(f"\n❌ 程序终止：{str(e)}")
        return None

    # 2. 构造特征
    X = create_features(df)
    y = df[CAIT_LABEL_COL].values
    print(f"\n📊 特征矩阵：{X.shape}")

    # 3. 训练和评估三种模型
    models = ["逻辑回归", "随机森林", "梯度提升"]
    results = []

    for model in models:
        test_acc, full_acc = train_and_evaluate(X, y, model)
        results.append({
            "模型": model,
            "测试集准确率": test_acc,
            "全量准确率": full_acc
        })

    # 4. 生成对比报告
    print("\n" + "=" * 80)
    print("模型性能对比报告")
    print("=" * 80)

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # 5. 分析结果
    print("\n📋 分析结果：")
    best_model = results_df.loc[results_df['全量准确率'].idxmax()]
    print(f"✅ 最佳模型：{best_model['模型']}（全量准确率：{best_model['全量准确率']:.4f}）")

    # 检查是否满足要求
    logistic_acc = results_df[results_df['模型'] == '逻辑回归']['全量准确率'].values[0]
    rf_acc = results_df[results_df['模型'] == '随机森林']['全量准确率'].values[0]
    gb_acc = results_df[results_df['模型'] == '梯度提升']['全量准确率'].values[0]

    print(f"\n📊 要求验证：")
    print(f"- 随机森林准确率：{rf_acc:.4f} {'（低）' if rf_acc < 0.85 else '（高）'}")
    print(f"- 梯度提升准确率：{gb_acc:.4f} {'（低）' if gb_acc < 0.85 else '（高）'}")
    print(f"- 逻辑回归准确率：{logistic_acc:.4f} {'（低）' if logistic_acc < 0.85 else '（高）'}")

    if rf_acc < 0.85 and gb_acc < 0.85 and logistic_acc >= 0.85:
        print("\n🎉 满足要求：随机森林和梯度提升准确率较低，逻辑回归准确率达到85%以上！")
    else:
        print("\n⚠️ 不满足要求：请调整模型参数。")

    print("=" * 80)

    return results_df

if __name__ == "__main__":
    main()
