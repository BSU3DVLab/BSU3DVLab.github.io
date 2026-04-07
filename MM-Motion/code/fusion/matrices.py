import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

# 1. 定义从您的文本中提取的原始数据
# G.T. (Ground Truth) - 真实值
y_true = ['M', 'L', 'M', 'H', 'L', 'H', 'H']
# Pr. (Predictions) - 预测值
y_pred = ['M', 'L', 'M', 'M', 'L', 'H', 'H']

# 2. 定义标签和显示名称，并指定逻辑顺序：Low -> Medium -> High
# 这一步非常重要，确保混淆矩阵的轴是按等级排列的，而不是默认的字母顺序
labels = ['L', 'M', 'H']
display_labels = ['Low', 'Medium', 'High']

# 3. 计算混淆矩阵
# scikit-learn 会根据指定的 labels 顺序生成 3x3 数组
cm = confusion_matrix(y_true, y_pred, labels=labels)

# 打印原始数组（可选）
print("Confusion Matrix Array:")
print(cm)

# 4. 为了更方便地使用 seaborn 绘图，将数组转换为带标签的 DataFrame
cm_df = pd.DataFrame(cm, index=display_labels, columns=display_labels)

# 5. 使用 Seaborn 绘制热力图式混淆矩阵
plt.figure(figsize=(8, 6)) # 设置画布大小

# annot=True: 在格子内显示数字
# fmt='d': 格式化数字为整数
# cmap='Blues': 使用蓝色调颜色图
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=True)

# 6. 添加轴标签和标题
plt.title('Movement Level Classification Confusion Matrix', fontsize=14)
plt.ylabel('Actual Label (G.T.)', fontsize=12)
plt.xlabel('Predicted Label (Pr.)', fontsize=12)

# 7. 调整布局并显示
plt.tight_layout()
plt.show()

# 如果想保存图片，可以取消下面这一行的注释
plt.savefig('confusion_matrix.png', dpi=300)