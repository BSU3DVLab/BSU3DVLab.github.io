import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.utils import shuffle

# 导入咱们严格划分好的数据集加载器
from kinect_data_loader import get_data

tf.random.set_seed(42)
np.random.seed(42)

# ==========================================
# 🧱 1. 最经典的常规空间图卷积层 (Standard Graph Conv)
# ==========================================
class SpatialGraphConv(Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        self.nodes = input_shape[2]
        self.in_channels = input_shape[3]
        
        # 经典的权重矩阵 W 和 可学习的邻接矩阵 A
        self.W = self.add_weight(shape=(self.in_channels, self.filters), 
                                 initializer='he_normal', 
                                 regularizer=l2(1e-4), name='W')
        # 初始化为单位矩阵，避免初期连线混乱
        self.A = self.add_weight(shape=(self.nodes, self.nodes), 
                                 initializer=tf.keras.initializers.Identity(), 
                                 regularizer=l2(1e-4), name='A')

    def call(self, x):
        # 正常的图卷积操作: A * X * W
        xw = tf.matmul(x, self.W)
        out = tf.einsum('vw,btwf->btvf', self.A, xw)
        return out

# ==========================================
# 🧱 2. 最经典的时空图卷积块 (Standard ST-GCN Block)
# ==========================================
def stgcn_block(x, filters, stride=1, dropout=0.5):
    residual = x
    
    # --- 空间维度提取 (Spatial) ---
    x = SpatialGraphConv(filters)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # --- 时间维度提取 (Temporal) ---
    # 经典的 9x1 长条形时间卷积核
    x = Conv2D(filters, (9, 1), strides=(stride, 1), padding='same', 
               use_bias=False, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    
    # --- 残差连接 (Residual) ---
    if stride != 1 or residual.shape[-1] != filters:
        residual = Conv2D(filters, (1, 1), strides=(stride, 1), padding='same', use_bias=False)(residual)
        
    x = Add()([x, residual])
    x = Activation('relu')(x)
    x = Dropout(dropout)(x)
    return x

# ==========================================
# 🧱 3. 常规版 ST-GCN 网络主架构
# ==========================================
def create_standard_stgcn(input_shape, num_classes=16):
    skeleton_input = Input(shape=input_shape, name="Skeleton_Input")
    risk_input = Input(shape=(1,), name="Risk_Input")
    
    risk_emb = Embedding(input_dim=3, output_dim=8)(risk_input)
    risk_emb = Flatten()(risk_emb)
    
    # 还原到 (Batch, 150帧, 64节点, 3坐标)
    x = Reshape((input_shape[0], 64, 3))(skeleton_input)
    
    # 先用一个 1x1 卷积把 3 维坐标升维，让模型更好提取特征
    x = Conv2D(64, (1, 1), padding='same', use_bias=False)(x)
    
    # 经典的 ST-GCN 三层堆叠 (利用 stride 自然浓缩时间帧)
    x = stgcn_block(x, filters=64, stride=1, dropout=0.3)  # 输出时间帧: 150
    x = stgcn_block(x, filters=128, stride=2, dropout=0.4) # 输出时间帧: 75
    x = stgcn_block(x, filters=256, stride=2, dropout=0.5) # 输出时间帧: 38
    
    # 经典收尾：全局平均池化，彻底消灭全连接层带来的百万级冗余参数
    x = GlobalAveragePooling2D()(x)
    
    combined = Concatenate()([x, risk_emb])
    
    z = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(combined)
    z = BatchNormalization()(z)
    z = Dropout(0.5)(z)
    
    outputs = Dense(num_classes, activation='softmax')(z)
    
    model = Model(inputs=[skeleton_input, risk_input], outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train():
    print("=" * 60)
    print("🚀 返璞归真：标准经典版 ST-GCN 启动！")
    print("=" * 60)
    
    X_tr, X_te, y_tr, y_te, r_tr, r_te = get_data()
    
    X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
    X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)
    
    print("💉 注入标准空间抖动...")
    noise = np.random.normal(loc=0.0, scale=0.02, size=X_tr.shape)
    X_tr_noise = X_tr + noise
    
    X_tr = np.concatenate([X_tr, X_tr_noise], axis=0)
    y_tr = np.concatenate([y_tr, y_tr], axis=0)
    r_tr = np.concatenate([r_tr, r_tr], axis=0)
    
    X_tr, y_tr, r_tr = shuffle(X_tr, y_tr, r_tr, random_state=42)
    
    y_tr_oh = to_categorical(y_tr, 16)
    y_te_oh = to_categorical(y_te, 16)
    
    model = create_standard_stgcn(X_tr.shape[1:], 16)
    
    # 增加 patience，给标准模型更多的时间去收敛
    callbacks = [
        EarlyStopping(monitor='val_accuracy', mode='max', patience=35, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_accuracy', mode='max', factor=0.5, patience=8, min_lr=1e-6, verbose=1)
    ]
    
    print("\n🔄 开始训练，享受经典网络带来的稳健提升...")
    model.fit(
        [X_tr, r_tr], y_tr_oh,
        validation_data=([X_te, r_te], y_te_oh),
        epochs=150,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    y_pred = np.argmax(model.predict([X_te, r_te]), axis=1)
    accuracy = accuracy_score(y_te, y_pred)
    precision = precision_score(y_te, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_te, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_te, y_pred, average='macro', zero_division=0)
    
    print(f"\n{'=' * 60}")
    print(f"🏆 标准版 ST-GCN 测试集准确率: {accuracy:.4f}")
    print(f"🏆 标准版 ST-GCN 精确率: {precision:.4f}")
    print(f"🏆 标准版 ST-GCN 召回率: {recall:.4f}")
    print(f"🏆 标准版 ST-GCN F1-分数: {f1:.4f}")
    print(f"{'=' * 60}\n")
    print(classification_report(y_test, y_pred, labels=range(16), zero_division=0))
    
    model.save('kinect_standard_stgcn_model3.h5')

if __name__ == "__main__":
    train()