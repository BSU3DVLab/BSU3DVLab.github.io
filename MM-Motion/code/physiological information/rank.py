import matplotlib.pyplot as plt
import seaborn as sns

# 提取整体特征的评分（模拟代码输出的top5数据）
top5_names = [
    "整体扭伤次数_4次及以上(有效)",
    "整体最近扭伤_1个月内(有效)",
    "整体不稳定症状总分（加权）",
    "整体有效扭伤×有效时间（融合）",
    "整体曾经扭伤（加权）"
]
top5_scores = [120.5678, 118.4567, 89.2345, 85.1234, 78.9876]

# 绘图
plt.figure(figsize=(10, 6))
sns.barplot(x=top5_scores, y=top5_names, palette="Blues_r")
plt.title("特征重要性排名（Top5）", fontsize=14, fontweight="bold")
plt.xlabel("互信息加权得分", fontsize=12)
plt.ylabel("特征名称", fontsize=12)
plt.grid(axis="x", alpha=0.3)
plt.show()