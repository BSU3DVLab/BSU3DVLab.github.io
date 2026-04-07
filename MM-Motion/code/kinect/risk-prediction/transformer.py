import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, BatchNormalization, 
                                     Concatenate, Lambda, Flatten, Embedding, 
                                     SpatialDropout1D, GaussianNoise, 
                                     Conv1D, GlobalAveragePooling1D, Add, Activation)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold

# 🚨 复用我们写好的全量数据加载器
from kinect_risk_loader import get_all_subjects_data, risk_map

tf.random.set_seed(42)
np.random.seed(42)

def create_tcn_risk_model(time_steps, features, num_classes=3):
    skeleton_input = Input(shape=(time_steps, features), name="Skeleton_Input")
    pose_input = Input(shape=(1,), name="Pose_Input")
    
    pose_emb = Embedding(input_dim=16, output_dim=4)(pose_input) 
    pose_emb = Flatten()(pose_emb)
    
    # 🌟 依然采用：只看速度与加速度，剥夺绝对坐标防作弊
    def compute_motion(x):
        first_frame_v = x[:, 0:1, :]
        vel = x[:, 1:, :] - x[:, :-1, :]
        vel_full = tf.concat([first_frame_v, vel], axis=1)
        
        first_frame_a = vel_full[:, 0:1, :]
        acc = vel_full[:, 1:, :] - vel_full[:, :-1, :]
        acc_full = tf.concat([first_frame_a, acc], axis=1)
        
        return tf.concat([vel_full, acc_full], axis=-1)

    motion_input = Lambda(compute_motion, name="Motion_Features")(skeleton_input)
    
    x = GaussianNoise(0.05)(motion_input)
    x = BatchNormalization()(x)
    
    # 将高维特征压缩到 64 维
    x = Dense(64, activation='relu', kernel_regularizer=l2(1e-3))(x)
    x = BatchNormalization()(x)
    x = SpatialDropout1D(0.4)(x) 
    
    # ==========================================
    # 🌟 封神组件：TCN 空洞卷积块 (Dilated Convolutions)
    # 用极小的参数量，获得极其夸张的时序感受野！
    # ==========================================
    # --- TCN Block 1 (跨度=1, 看局部) ---
    res1 = x
    x = Conv1D(64, kernel_size=3, padding='same', dilation_rate=1, kernel_regularizer=l2(1e-3))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SpatialDropout1D(0.2)(x)
    x = Add()([x, res1]) # 残差连接
    
    # --- TCN Block 2 (跨度=2, 看稍远) ---
    res2 = x
    x = Conv1D(64, kernel_size=3, padding='same', dilation_rate=2, kernel_regularizer=l2(1e-3))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SpatialDropout1D(0.2)(x)
    x = Add()([x, res2])
    
    # --- TCN Block 3 (跨度=4, 看大局) ---
    res3 = x
    x = Conv1D(64, kernel_size=3, padding='same', dilation_rate=4, kernel_regularizer=l2(1e-3))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SpatialDropout1D(0.2)(x)
    x = Add()([x, res3])
    
    # ==========================================
    # 最终全局特征融合
    # ==========================================
    x = GlobalAveragePooling1D()(x)
    
    combined = Concatenate()([x, pose_emb])
    
    z = Dense(32, activation='relu', kernel_regularizer=l2(1e-3))(combined)
    z = BatchNormalization()(z)
    z = Dropout(0.5)(z)
    
    outputs = Dense(num_classes, activation='softmax', name="Risk_Output")(z)
    
    model = Model(inputs=[skeleton_input, pose_input], outputs=outputs)
    
    # 🚨 标签平滑 (label_smoothing=0.1)：防止模型过度自信陷入局部死胡同
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy'])
    return model

def train_5_fold():
    print("=" * 60)
    print("🚀 终极破局：TCN 空洞时间卷积 5折交叉验证启动！")
    print("=" * 60)
    
    all_data = get_all_subjects_data()
    all_subjects = list(risk_map.keys())
    all_risks = [risk_map[s] for s in all_subjects]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(all_subjects, all_risks)):
        print(f"\n{'='*20} 正在执行 第 {fold + 1}/5 折 {'='*20}")
        
        train_subs = [all_subjects[i] for i in train_idx]
        test_subs = [all_subjects[i] for i in test_idx]
        print(f"🧪 本轮测试集名单 ({len(test_subs)} 人): {test_subs}")
        
        X_tr, y_tr, p_tr = [], [], []
        for s in train_subs:
            if s in all_data:
                X_tr.extend(all_data[s]['X'])
                y_tr.extend(all_data[s]['y'])
                p_tr.extend(all_data[s]['p'])
                
        X_te, y_te, p_te = [], [], []
        for s in test_subs:
            if s in all_data:
                X_te.extend(all_data[s]['X'])
                y_te.extend(all_data[s]['y'])
                p_te.extend(all_data[s]['p'])
                
        X_tr, y_tr, p_tr = np.array(X_tr), np.array(y_tr), np.array(p_tr)
        X_te, y_te, p_te = np.array(X_te), np.array(y_te), np.array(p_te)
        
        # 外部噪声扩增
        noise = np.random.normal(loc=0.0, scale=0.02, size=X_tr.shape)
        X_tr_noise = X_tr + noise
        X_tr = np.concatenate([X_tr, X_tr_noise], axis=0)
        y_tr = np.concatenate([y_tr, y_tr], axis=0)
        p_tr = np.concatenate([p_tr, p_tr], axis=0)
        
        X_tr, y_tr, p_tr = shuffle(X_tr, y_tr, p_tr, random_state=42)
        y_tr_oh = to_categorical(y_tr, 3) 
        y_te_oh = to_categorical(y_te, 3)

        # 动态计算惩罚权重
        unique_classes = np.unique(y_tr)
        weights_array = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_tr)
        class_weights = dict(zip(unique_classes, weights_array))
        
        model = create_tcn_risk_model(time_steps=X_tr.shape[1], features=X_tr.shape[2])
        
        callbacks = [
            EarlyStopping(monitor='val_accuracy', mode='max', patience=35, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_accuracy', mode='max', factor=0.5, patience=10, min_lr=1e-5, verbose=0)
        ]
        
        print(f"🔄 TCN 雷达扫描开启，开始训练 Fold {fold + 1}... (已开启简易日志)")
        model.fit(
            [X_tr, p_tr], y_tr_oh, 
            validation_data=([X_te, p_te], y_te_oh),
            epochs=150, 
            batch_size=64,
            class_weight=class_weights, 
            callbacks=callbacks,
            verbose=2
        )

        y_pred = np.argmax(model.predict([X_te, p_te], verbose=0), axis=1)
        acc = accuracy_score(y_te, y_pred)
        precision = precision_score(y_te, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_te, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_te, y_pred, average='macro', zero_division=0)
        fold_accuracies.append(acc)
        fold_precisions.append(precision)
        fold_recalls.append(recall)
        fold_f1s.append(f1)
        print(f"🏆 Fold {fold + 1} 考核结束，测试集准确率: {acc:.4f}")
        print(f"🏆 Fold {fold + 1} 精确率: {precision:.4f}")
        print(f"🏆 Fold {fold + 1} 召回率: {recall:.4f}")
        print(f"🏆 Fold {fold + 1} F1-分数: {f1:.4f}")
        
    print("\n" + "="*60)
    print(f"🎉 TCN 5折交叉验证全部完成！")
    print(f"📊 历次准确率记录: {[round(a, 4) for a in fold_accuracies]}")
    print(f"🌟 最终跨受试者平均准确率 (CV Mean Accuracy): {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
    print(f"🌟 最终跨受试者平均精确率 (CV Mean Precision): {np.mean(fold_precisions):.4f} ± {np.std(fold_precisions):.4f}")
    print(f"🌟 最终跨受试者平均召回率 (CV Mean Recall): {np.mean(fold_recalls):.4f} ± {np.std(fold_recalls):.4f}")
    print(f"🌟 最终跨受试者平均F1-分数 (CV Mean F1-Score): {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
    print("="*60)

if __name__ == "__main__":
    train_5_fold()