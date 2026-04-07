import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

# IMU Skeleton with 21 joints
# Joint indices (0-20):
# 0: Head
# 1: Neck
# 2: Right Shoulder
# 3: Right Elbow
# 4: Right Wrist
# 5: Left Shoulder
# 6: Left Elbow
# 7: Left Wrist
# 8: Spine
# 9: Right Hip
# 10: Right Knee
# 11: Right Ankle
# 12: Left Hip
# 13: Left Knee
# 14: Left Ankle
# 15: Pelvis
# 16: Right Hand (additional)
# 17: Left Hand (additional)
# 18: Right Foot (additional)
# 19: Left Foot (additional)
# 20: Chest (additional)

num_node = 21
self_link = [(i, i) for i in range(num_node)]

# Define inward edges based on body structure
# Using standard skeleton topology
inward_ori_index = [
    (0, 1),    # Head -> Neck
    (1, 2),    # Neck -> Right Shoulder
    (2, 3),    # Right Shoulder -> Right Elbow
    (3, 4),    # Right Elbow -> Right Wrist
    (1, 5),    # Neck -> Left Shoulder
    (5, 6),    # Left Shoulder -> Left Elbow
    (6, 7),    # Left Elbow -> Left Wrist
    (1, 8),    # Neck -> Spine
    (8, 9),    # Spine -> Right Hip
    (9, 10),   # Right Hip -> Right Knee
    (10, 11),  # Right Knee -> Right Ankle
    (8, 12),   # Spine -> Left Hip
    (12, 13),  # Left Hip -> Left Knee
    (13, 14),  # Left Knee -> Left Ankle
    (8, 15),   # Spine -> Pelvis
    (4, 16),   # Right Wrist -> Right Hand
    (7, 17),   # Left Wrist -> Left Hand
    (11, 18),  # Right Ankle -> Right Foot
    (14, 19),  # Left Ankle -> Left Foot
    (1, 20),   # Neck -> Chest
]

# Convert to 0-indexed
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
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
