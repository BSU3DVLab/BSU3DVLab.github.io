import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, GlobalAveragePooling2D, Layer, Add, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from data_loader import get_data

tf.random.set_seed(42)
np.random.seed(42)

# ==========================================
# 1. 定义下半身骨架拓扑图
# ==========================================
num_nodes = 7
A = np.zeros((num_nodes, num_nodes), dtype=np.float32)

edges = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6)]
for i, j in edges:
    A[i, j] = 1.0
    A[j, i] = 1.0
for i in range(num_nodes):
    A[i, i] = 1.0

row_sum = np.sum(A, axis=1)
D_inv = np.diag(1.0 / row_sum)
A_norm = np.dot(D_inv, A)

# ==========================================
# 2. 增强版：带可学习权重的空间图卷积层
# ==========================================
class SpatialGraphConv(Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.A = tf.constant(A_norm, dtype=tf.float32)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.filters),
                                 initializer='glorot_uniform', trainable=True, name='W')
        # 【关键增强】：可学习的边权重掩码 (Learnable Edge Mask)
        self.edge_weight = self.add_weight(shape=(num_nodes, num_nodes),
                                           initializer='ones', trainable=True, name='edge_weight')

    def call(self, inputs):
        # 1. 特征映射
        x = tf.matmul(inputs, self.W)
        # 2. 赋予骨架灵活性：A * Mask
        A_masked = tf.multiply(self.A, self.edge_weight)
        # 3. 空间聚合
        output = tf.einsum('vw,btwf->btvf', A_masked, x)
        return output

# ==========================================
# 3. 封装标准 ST-GCN 块 (带残差连接)
# ==========================================
def stgcn_block(x, filters, kernel_size=9, dropout_rate=0.3):
    # 记录输入，用于残差相加
    residual = x
    
    # --- 空间维度的 GCN ---
    x = SpatialGraphConv(filters)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    
    # --- 时间维度的 CNN ---
    x = Conv2D(filters, kernel_size=(kernel_size, 1), padding='same')(x)
    x = BatchNormalization()(x)
    
    # 【关键增强】：残差连接 (Skip Connection)
    if residual.shape[-1] != filters:
        # 如果通道数不匹配，用 1x1 卷积对齐
        residual = Conv2D(filters, kernel_size=(1, 1), padding='same')(residual)
        
    x = Add()([x, residual]) # 直达电梯！
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    
    return x

# ==========================================
# 4. 搭建完全体架构
# ==========================================
def create_real_stgcn_model(time_steps, num_nodes, num_features, num_classes):
    inputs = Input(shape=(time_steps, num_nodes, num_features))
    
    # 通过 3 个带有残差连接的 ST-GCN 块
    x = stgcn_block(inputs, filters=64, kernel_size=9, dropout_rate=0.3)
    x = stgcn_block(x, filters=128, kernel_size=5, dropout_rate=0.4)
    x = stgcn_block(x, filters=256, kernel_size=5, dropout_rate=0.5) # 深层加大 Dropout 惩罚
    
    # 全局池化
    x = GlobalAveragePooling2D()(x)
    
    # 强力防过拟合的分类层
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    # 加上了 weight_decay (L2正则化) 防止死记硬背
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, weight_decay=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_stgcn():
    print("正在加载下肢专属数据...")
    X_train, X_test, y_train, y_test = get_data()
    
    time_steps = X_train.shape[1]
    num_nodes = 7
    num_features = 21 
    
    X_train_graph = X_train.reshape(-1, time_steps, num_nodes, num_features)
    X_test_graph = X_test.reshape(-1, time_steps, num_nodes, num_features)
    print(f"✅ 图结构数据重组成功！输入形状变更为: {X_train_graph.shape}")
    
    num_classes = 16
    y_train_onehot = to_categorical(y_train, num_classes)
    y_test_onehot = to_categorical(y_test, num_classes)
    
    model = create_real_stgcn_model(time_steps, num_nodes, num_features, num_classes)
    
    # 学习率策略稍微平缓一些
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
    
    print("\n🚀 开始训练：带残差连接的 完全体 ST-GCN 模型...")
    history = model.fit(
        X_train_graph, y_train_onehot,
        epochs=120, # 给定更多的训练轮数
        batch_size=64, 
        validation_data=(X_test_graph, y_test_onehot),
        callbacks=[reduce_lr, early_stop],
        verbose=1
    )
    
    y_pred = np.argmax(model.predict(X_test_graph), axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    print(f"\n==========================================")
    print(f"🏆 ST-GCN 完全体最终准确率: {accuracy:.4f}")
    print(f"🏆 ST-GCN 完全体精确率: {precision:.4f}")
    print(f"🏆 ST-GCN 完全体召回率: {recall:.4f}")
    print(f"🏆 ST-GCN 完全体F1-分数: {f1:.4f}")
    print(f"==========================================\n")
    print("详细分类报告:")
    print(classification_report(y_test, y_pred, labels=range(16), zero_division=0))
    
    model.save('real_stgcn_model.h5')
    return model

if __name__ == "__main__":
    train_stgcn()