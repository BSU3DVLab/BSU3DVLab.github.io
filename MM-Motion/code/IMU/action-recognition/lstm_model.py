import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Bidirectional, BatchNormalization, GlobalAveragePooling1D, Multiply, Reshape
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from data_loader import get_data

# 1. 设置随机种子，保证每次运行结果一致
tf.random.set_seed(42)
np.random.seed(42)

def create_attention_model(input_shape, num_classes):
    """
    创建带有 SE 注意力机制的 1D-CNN + Bi-LSTM 模型
    """
    inputs = Input(shape=input_shape)

    # --- 第一阶段：1D-CNN 局部特征提取 ---
    x = Conv1D(filters=64, kernel_size=7, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # 提取深层特征
    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    
    # ==========================================
    # SE 特征通道注意力机制 (保持不变)
    # 作用：自动放大下肢传感器中，对当前动作最关键的信号
    # ==========================================
    attention = GlobalAveragePooling1D()(x)
    attention = Dense(128 // 8, activation='relu')(attention)
    attention = Dense(128, activation='sigmoid')(attention)
    attention = Reshape((1, 128))(attention)
    x = Multiply()([x, attention])
    # ==========================================
    
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # 【修改点1】：全面调高 Dropout 至 0.4，强行防止模型死记硬背
    x = Dropout(0.4)(x)

    # --- 第二阶段：双向 LSTM 时序理解 ---
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.4)(x)
    x = Bidirectional(LSTM(32, return_sequences=False))(x)
    x = Dropout(0.4)(x)

    # --- 第三阶段：全连接分类层 ---
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_process():
    print("正在调用数据加载器 (使用纯净的下肢专属数据)...")
    X_train, X_test, y_train, y_test = get_data()
    
    num_classes = 16 
    y_train_onehot = to_categorical(y_train, num_classes)
    y_test_onehot = to_categorical(y_test, num_classes)
    
    # 构建模型
    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"✅ 模型输入维度确认: 时间步={input_shape[0]}, 精简后的特征数={input_shape[1]}")
    model = create_attention_model(input_shape, num_classes)
    
    # ==========================================
    # 【修改点2】：给注意力机制充足的对焦时间
    # 把 patience 改为 10，让模型不要太早踩刹车
    # 同时 EarlyStopping 放宽到 20
    # ==========================================
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    
    print("\n🚀 开始训练：去掉强制惩罚、轻装上阵的模型...")
    # 【修改点3】：撤销 class_weight，让模型自由、公平地学习
    history = model.fit(
        X_train, y_train_onehot,
        epochs=100,
        batch_size=64, 
        validation_data=(X_test, y_test_onehot),
        callbacks=[reduce_lr, early_stop],
        verbose=1
    )
    
    # 评估成品
    y_pred = np.argmax(model.predict(X_test), axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    print(f"\n==========================================")
    print(f"🏆 最终净化版模型验证集准确率: {accuracy:.4f}")
    print(f"🏆 最终净化版模型精确率: {precision:.4f}")
    print(f"🏆 最终净化版模型召回率: {recall:.4f}")
    print(f"🏆 最终净化版模型F1-分数: {f1:.4f}")
    print(f"==========================================\n")
    
    print("详细分类报告:")
    print(classification_report(y_test, y_pred, labels=range(16), zero_division=0))
    
    model.save('final_imu_attention_model.h5')
    print("💾 模型已成功保存为 final_imu_attention_model.h5")
    return model

if __name__ == "__main__":
    train_process()