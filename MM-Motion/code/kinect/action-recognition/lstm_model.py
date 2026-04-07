import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.utils import shuffle

# 导入你的完美加载器
from data_loader import get_data

tf.random.set_seed(408)
np.random.seed(408)

def create_two_stream_lstm(time_steps, features, num_classes):
    # 【输入1】：双机位拼接后的骨架坐标 (150帧, 192维)
    skeleton_input = Input(shape=(time_steps, features), name="Skeleton_Input")
    
    # 【输入2】：风险等级先验
    risk_input = Input(shape=(1,), name="Risk_Input")
    risk_emb = Embedding(input_dim=3, output_dim=8)(risk_input)
    risk_emb = Flatten()(risk_emb)

    # ==========================================
    # 【核心大招】：物理运动学特征工程 (提取速度流)
    # 速度 = 当前帧坐标 - 上一帧坐标
    # ==========================================
    def compute_velocity(x):
        # 复制第一帧来保持时间步长不变
        first_frame = x[:, 0:1, :]
        # 后一帧减前一帧求导数（速度）
        diff = x[:, 1:, :] - x[:, :-1, :]
        return tf.concat([first_frame, diff], axis=1)

    velocity_input = Lambda(compute_velocity, name="Velocity_Calculation")(skeleton_input)

    # --- 分支 1：静态位置流 (Posture Stream) ---
    x_pos = Dense(128, activation='relu')(skeleton_input)
    x_pos = LayerNormalization()(x_pos)

    # --- 分支 2：动态速度流 (Motion Stream) ---
    x_vel = Dense(128, activation='relu')(velocity_input)
    x_vel = LayerNormalization()(x_vel)

    # --- 时空物理特征融合 ---
    x = Concatenate()([x_pos, x_vel]) 
    
    # 空间注意力机制：自动过滤掉那些不动的关节噪点
    attention = Dense(256, activation='sigmoid')(x)
    x = Multiply()([x, attention])
    
    x = SpatialDropout1D(0.3)(x)

    # --- Bi-LSTM 捕捉时序动态轨迹 ---
    x = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.001)))(x)
    x = MaxPooling1D(pool_size=2)(x) # 时序压缩 150 -> 75
    
    x = Bidirectional(LSTM(64, return_sequences=False, kernel_regularizer=l2(0.001)))(x)

    # --- 多模态知识融合 ---
    combined = Concatenate()([x, risk_emb])
    
    # --- 分类头 ---
    z = Dense(128, activation='relu', kernel_regularizer=l2(0.005))(combined)
    z = BatchNormalization()(z)
    z = Dropout(0.5)(z)
    outputs = Dense(num_classes, activation='softmax')(z)

    model = Model(inputs=[skeleton_input, risk_input], outputs=outputs)
    
    # ==========================================
    # 🛡️ 第二道防线：梯度裁剪 (防爆项圈)
    # clipnorm=1.0 彻底杜绝梯度爆炸导致的 Loss NaN
    # ==========================================
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0), 
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), 
        metrics=['accuracy']
    )
    return model

def train():
    X_tr, X_te, y_tr, y_te, r_tr, r_te = get_data()
    
    # ==========================================
    # 🛡️ 第一道防线：全量数据排毒 (清除 NaN 和 Inf)
    # 把 Kinect 丢失追踪产生的坏点全部归零！
    # ==========================================
    print("🛡️ 正在进行全量数据排毒 (清除 NaN 和 Inf)...")
    X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
    X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)
    
    print("💉 正在对 Kinect 骨架注入空间高斯抖动 (Jittering)...")
    noise = np.random.normal(loc=0.0, scale=0.02, size=X_tr.shape)
    X_tr_noise = X_tr + noise
    
    print("🧬 正在进行骨架缩放 (Scaling)，模拟不同体型受试者...")
    scale_factor = np.random.normal(loc=1.0, scale=0.1, size=(X_tr.shape[0], 1, 1))
    X_tr_scaled = X_tr * scale_factor
    
    # 三份数据合并
    X_tr = np.concatenate([X_tr, X_tr_noise, X_tr_scaled], axis=0)
    y_tr = np.concatenate([y_tr, y_tr, y_tr], axis=0)
    r_tr = np.concatenate([r_tr, r_tr, r_tr], axis=0)
    
    X_tr, y_tr, r_tr = shuffle(X_tr, y_tr, r_tr, random_state=42)
    print(f"✅ 数据极限量级扩增完成！训练集数量暴增至: {X_tr.shape[0]} 条")
    
    y_tr_oh = to_categorical(y_tr, 16)
    y_te_oh = to_categorical(y_te, 16)

    time_steps = X_tr.shape[1]
    features = X_tr.shape[2]
    model = create_two_stream_lstm(time_steps, features, 16)
    
    # 回调函数 (以验证集准确率为绝对监控指标)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', mode='max', factor=0.6, patience=8, min_lr=1e-6, verbose=1)
    early_stop = EarlyStopping(monitor='val_accuracy', mode='max', patience=30, restore_best_weights=True)
    
    print("\n🚀 开始终极冲刺：物理直觉驱动的双流 (位置+速度) Bi-LSTM...")
    history = model.fit(
        [X_tr, r_tr], y_tr_oh,
        validation_data=([X_te, r_te], y_te_oh),
        epochs=150, 
        batch_size=64,
        callbacks=[reduce_lr, early_stop],
        verbose=1
    )

    y_pred = np.argmax(model.predict([X_te, r_te]), axis=1)
    accuracy = accuracy_score(y_te, y_pred)
    precision = precision_score(y_te, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_te, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_te, y_pred, average='macro', zero_division=0)
    
    print(f"\n==========================================")
    print(f"🏆 物理双流 Kinect 骨架模型 验证集准确率: {accuracy:.4f}")
    print(f"🏆 物理双流 Kinect 骨架模型 精确率: {precision:.4f}")
    print(f"🏆 物理双流 Kinect 骨架模型 召回率: {recall:.4f}")
    print(f"🏆 物理双流 Kinect 骨架模型 F1-分数: {f1:.4f}")
    print(f"==========================================\n")
    print(classification_report(y_test=y_te, y_pred=y_pred, labels=range(16), zero_division=0))
    
    model.save('kinect_twostream_lstm_model_fixed.h5')

if __name__ == "__main__":
    train()