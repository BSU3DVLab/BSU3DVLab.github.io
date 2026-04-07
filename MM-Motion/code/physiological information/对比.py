# 模拟代码输出的准确率数据
import numpy as np
import plt
from matplotlib import pyplot as plt

model_types = ["整体模型", "左脚模型", "右脚模型"]
test_acc = [0.892, 0.915, 0.908]
full_acc = [0.923, 0.931, 0.928]

# 绘图
x = np.arange(len(model_types))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, test_acc, width, label="测试集准确率", color="#4285f4")
plt.bar(x + width/2, full_acc, width, label="全量准确率", color="#34a853")

plt.title("不同模型准确率对比", fontsize=14, fontweight="bold")
plt.xlabel("模型类型", fontsize=12)
plt.ylabel("准确率", fontsize=12)
plt.xticks(x, model_types)
plt.ylim(0.85, 0.95)
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.show()