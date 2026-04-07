import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

# nuoyiteng数据集有20个关节点
num_node = 20
self_link = [(i, i) for i in range(num_node)]

# 定义关节点之间的连接关系（基于常见人体骨架结构）
# 假设关节点顺序：0-头部, 1-颈部, 2-右肩, 3-右肘, 4-右手腕,
# 5-左肩, 6-左肘, 7-左手腕, 8-右髋, 9-右膝, 10-右脚踝,
# 11-左髋, 12-左膝, 13-左脚踝, 14-右胸, 15-左胸,
# 16-右腰, 17-左腰, 18-右臀, 19-左臀
inward_ori_index = [
    (1, 0),   # 颈部-头部
    (2, 1),   # 右肩-颈部
    (3, 2),   # 右肘-右肩
    (4, 3),   # 右手腕-右肘
    (5, 1),   # 左肩-颈部
    (6, 5),   # 左肘-左肩
    (7, 6),   # 左手腕-左肘
    (8, 14),  # 右髋-右胸
    (9, 8),   # 右膝-右髋
    (10, 9),  # 右脚踝-右膝
    (11, 15), # 左髋-左胸
    (12, 11), # 左膝-左髋
    (13, 12), # 左脚踝-左膝
    (14, 1),  # 右胸-颈部
    (15, 1),  # 左胸-颈部
    (16, 14), # 右腰-右胸
    (17, 15), # 左腰-左胸
    (18, 16), # 右臀-右腰
    (19, 17), # 左臀-左腰
    (16, 8),  # 右腰-右髋
    (17, 11), # 左腰-左髋
]

inward = [(i, j) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
