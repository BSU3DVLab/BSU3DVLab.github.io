"""
Subject-level classifier using original ST-GCN as backbone
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.stgcn import Model as STGCN, TCN_GCN_unit, import_class, bn_init


class SubjectClassifierWithSTGCN(nn.Module):
    """
    Subject-level classifier using ST-GCN backbone.
    
    For each subject, we aggregate multiple pose samples, then use ST-GCN
    to extract features from each sample, and finally aggregate to a 
    subject-level representation for classification.
    """
    
    def __init__(self, num_class=3, num_point=25, num_frame=64, num_person=1, 
                 graph='graph.ntu_rgb_d.AdjMatrixGraph', graph_args=dict(), 
                 in_channels=3, drop_out=0, adaptive=True, num_set=3,
                 hidden_dim=256, aggregation='mean'):
        """
        Args:
            num_class: number of subject classes
            num_point: number of skeleton joints
            num_frame: number of frames per sample
            num_person: number of persons (set to 1 for subject aggregation)
            graph: graph class for ST-GCN
            graph_args: arguments for graph
            in_channels: input channels (3 for xyz)
            drop_out: dropout rate
            adaptive: use adaptive graph or not
            num_set: number of graph adjacency matrix sets
            hidden_dim: hidden dimension for classifier
            aggregation: how to aggregate pose samples ('mean', 'max', 'concat')
        """
        super(SubjectClassifierWithSTGCN, self).__init__()
        
        self.num_class = num_class
        self.num_point = num_point
        self.num_frame = num_frame
        self.num_person = num_person
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation
        
        # Initialize ST-GCN backbone
        if graph is None:
            raise ValueError("Graph must be specified for ST-GCN backbone")
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
        
        A = np.stack([np.eye(num_point)] * num_set, axis=0)
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        
        # ST-GCN backbone layers (same as original ST-GCN)
        self.stgcn_l1 = TCN_GCN_unit(3, 64, A, residual=False, adaptive=adaptive)
        self.stgcn_l2 = TCN_GCN_unit(64, 64, A, adaptive=adaptive)
        self.stgcn_l3 = TCN_GCN_unit(64, 64, A, adaptive=adaptive)
        self.stgcn_l4 = TCN_GCN_unit(64, 64, A, adaptive=adaptive)
        self.stgcn_l5 = TCN_GCN_unit(64, 128, A, stride=2, adaptive=adaptive)
        self.stgcn_l6 = TCN_GCN_unit(128, 128, A, adaptive=adaptive)
        self.stgcn_l7 = TCN_GCN_unit(128, 128, A, adaptive=adaptive)
        self.stgcn_l8 = TCN_GCN_unit(128, 256, A, stride=2, adaptive=adaptive)
        self.stgcn_l9 = TCN_GCN_unit(256, 256, A, adaptive=adaptive)
        self.stgcn_l10 = TCN_GCN_unit(256, 256, A, adaptive=adaptive)
        
        # Feature dimension after ST-GCN backbone
        self.stgcn_output_dim = 256
        
        # Subject-level aggregation layers
        if aggregation == 'concat':
            # Concatenate all pose features
            self.agg_fc = nn.Linear(self.stgcn_output_dim * num_frame, hidden_dim)
            self.feature_dim = hidden_dim
        elif aggregation == 'mean':
            # Mean pooling of pose features
            self.agg_fc = nn.Linear(self.stgcn_output_dim, hidden_dim)
            self.feature_dim = hidden_dim
        elif aggregation == 'max':
            self.agg_fc = nn.Linear(self.stgcn_output_dim, hidden_dim)
            self.feature_dim = hidden_dim
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        # Final classifier
        self.fc = nn.Linear(self.feature_dim, num_class)
        
        # Dropout
        self.drop_out = nn.Dropout(drop_out) if drop_out > 0 else nn.Identity()
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        bn_init(self.data_bn, 1)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / self.num_class))
        
    def forward(self, x, get_subject_feat=False):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [N, C, T, V, M] where
               N = num_subjects * num_samples_per_subject
               C = in_channels (3 for xyz)
               T = num_frames
               V = num_points (joints)
               M = num_persons
            get_subject_feat: if True, return subject-level features instead of logits
            
        Returns:
            logits or features depending on get_subject_feat
        """
        N, C, T, V, M = x.size()
        
        # Process through ST-GCN backbone
        # Shape: [N, M, V, C, T] -> [N, M*V*C, T] -> BN -> [N, M, V, C, T]
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        
        # ST-GCN layers
        x = self.stgcn_l1(x)
        x = self.stgcn_l2(x)
        x = self.stgcn_l3(x)
        x = self.stgcn_l4(x)
        x = self.stgcn_l5(x)
        x = self.stgcn_l6(x)
        x = self.stgcn_l7(x)
        x = self.stgcn_l8(x)
        x = self.stgcn_l9(x)
        x = self.stgcn_l10(x)
        
        # Output shape: [N*M, C, T, V]
        c_new = x.size(1)
        
        if self.aggregation == 'mean':
            # Average over time and vertices: [N*M, C, T, V] -> [N*M, C]
            x = x.view(N, M, c_new, -1).mean(3).mean(1)
        elif self.aggregation == 'max':
            # Max over time and vertices: [N*M, C, T, V] -> [N*M, C]
            x = x.view(N, M, c_new, -1).max(3)[0].max(1)[0]
        elif self.aggregation == 'concat':
            # Flatten: [N*M, C, T, V] -> [N, C*T*V] (assuming M=1)
            x = x.view(N, M, c_new, T, V)
            if M == 1:
                x = x.squeeze(1).view(N, -1)
            else:
                x = x.view(N, -1)
        
        # Subject-level feature processing
        x = self.agg_fc(x)
        x = F.relu(x)
        x = self.drop_out(x)
        
        if get_subject_feat:
            return x
            
        logits = self.fc(x)
        
        return logits
    
    def get_stgcn_features(self, x):
        """Extract features from ST-GCN backbone before pooling."""
        N, C, T, V, M = x.size()
        
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        
        x = self.stgcn_l1(x)
        x = self.stgcn_l2(x)
        x = self.stgcn_l3(x)
        x = self.stgcn_l4(x)
        x = self.stgcn_l5(x)
        x = self.stgcn_l6(x)
        x = self.stgcn_l7(x)
        x = self.stgcn_l8(x)
        x = self.stgcn_l9(x)
        x = self.stgcn_l10(x)
        
        return x  # [N*M, C, T, V]


# Keep the original simple classifier for comparison
class SimpleSubjectClassifier(nn.Module):
    """
    Simple MLP classifier for subject-level features.
    Kept for comparison purposes.
    """
    
    def __init__(self, input_dim=567, hidden_dim=128, num_class=3, drop_out=0.5):
        """
        Args:
            input_dim: dimension of input features
            hidden_dim: hidden layer dimension
            num_class: number of classes
            drop_out: dropout rate
        """
        super(SimpleSubjectClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(drop_out)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_class)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
