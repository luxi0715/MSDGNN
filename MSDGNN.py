# -*- coding:utf-8 -*-
"""
MSDGNN: Multi-Scale Dynamic Graph Neural Network for PM2.5 Forecasting

This module implements the MSDGNN architecture, which combines:
- Multi-head spatial-temporal attention mechanisms
- Dynamic graph convolution with adaptive node grouping
- Multi-scale temporal modeling (hour, day, week)
- Chebyshev graph convolution with spatial attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import scaled_Laplacian, cheb_polynomial
import pickle
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean

class ScaledDotProductAttention(nn.Module):
    """
    Implements scaled dot-product attention mechanism.
    Computes attention weights using query, key, and value matrices.
    """
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        """
        Args:
            Q: Query tensor of shape [B, n_heads, len1, len2, d_k]
            K: Key tensor of shape [B, n_heads, len1, len2, d_k]
            V: Value tensor of shape [B, n_heads, len1, len2, d_k]
        Returns:
            context: Attention-weighted value tensor [B, n_heads, len1, len2, d_k]
        """
        B, n_heads, len1, len2, d_k = Q.shape
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)

        attn = nn.Softmax(dim=-1)(scores)
        # Output shape: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        context = torch.matmul(attn, V)
        return context


class SMultiHeadAttention(nn.Module):
    """
    Spatial Multi-Head Attention module for capturing spatial dependencies.
    Applies multiple attention heads to learn diverse spatial patterns.
    """
    def __init__(self, embed_size, heads):
        super(SMultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, input_Q, input_K, input_V):
        """
        Args:
            input_Q: Query tensor [batch_size, N, T, C]
            input_K: Key tensor [batch_size, N, T, C]
            input_V: Value tensor [batch_size, N, T, C]
        Returns:
            output: Multi-head attention output [batch_size, N, T, C]
        """
        B, N, T, C = input_Q.shape
        # Transform: [B, N, T, C] -> [B, N, T, h * d_k] -> [B, N, T, h, d_k] -> [B, h, T, N, d_k]
        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # Q: [B, h, T, N, d_k]
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # K: [B, h, T, N, d_k]
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # V: [B, h, T, N, d_k]
        context = ScaledDotProductAttention()(Q, K, V)  # [B, h, T, N, d_k]
        context = context.permute(0, 3, 2, 1, 4)  # [B, N, T, h, d_k]
        context = context.reshape(B, N, T, self.heads * self.head_dim)  # [B, N, T, C]
        output = self.fc_out(context)  # [batch_size, N, T, embed_size]
        return output

class Spatial_Attention_layer(nn.Module):
    """
    Computes spatial attention scores to capture spatial dependencies between nodes.
    Uses learnable parameters to generate adaptive spatial attention matrix.
    """
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps).to(DEVICE))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps).to(DEVICE))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices).to(DEVICE))
        self.Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(DEVICE))


    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, N, F_in, T]
        Returns:
            S_normalized: Normalized spatial attention matrix [B, N, N]
        """

        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)

        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (F)(b,N,F,T)->(b,N,T)->(b,T,N)

        product = torch.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)

        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (N,N)(B, N, N)->(B,N,N)

        S_normalized = F.softmax(S, dim=1)

        return S_normalized


def load_pickle(pickle_file):
    """Load pickle file with multiple encoding fallbacks."""
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data', pickle_file, ':', e)
        raise
    return pickle_data

def load_graph_data(pkl_filename):
    """Load adjacency matrix from pickle file."""
    adj_mx = load_pickle(pkl_filename)
    return adj_mx



class cheb_conv_withSAt(nn.Module):
    """
    K-order Chebyshev graph convolution with spatial attention.
    Integrates learned node embeddings to generate adaptive graph structure.
    """

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        """
        Args:
            K: Order of Chebyshev polynomial
            cheb_polynomials: Precomputed Chebyshev polynomials
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super(cheb_conv_withSAt, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

        # OPTIMIZATION: Removed redundant parameter definitions, keeping only one set of node_embeddings
        # Original code lines 126-133 had duplicate definitions and unused parameters
        self.node_embeddings = nn.Parameter(torch.randn(22, 22), requires_grad=True)


    def forward(self, x, spatial_attention):
        """
        Performs Chebyshev graph convolution with spatial attention.
        Args:
            x: Input tensor [batch_size, N, F_in, T]
            spatial_attention: Spatial attention matrix [batch_size, N, N]
        Returns:
            Output tensor [batch_size, N, F_out, T]
        """
        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            node_num = self.node_embeddings.shape[0]
            # Compute initial support matrix S using learned node embeddings
            supports = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
            support_set = [torch.eye(node_num).to(supports.device), supports]
            # Generate Chebyshev polynomials (default cheb_k = 3)
            for k in range(2, self.K):
                new_support = torch.matmul(2 * supports, support_set[-1]) - support_set[-2]
                support_set.append(new_support)

            for k in range(self.K):

                T_k = support_set[k]

                T_k_with_at = T_k.mul(spatial_attention)

                theta_k = self.Theta[k]

                # Left multiplication: (N, N)(b, N, F_in) = (b, N, F_in)
                # Row sum = 1 becomes column sum = 1 after permute for left multiplication
                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)

                output = output + rhs.matmul(theta_k)  # (b, N, F_in)(F_in, F_out) = (b, N, F_out)

            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)

        return F.relu(torch.cat(outputs, dim=-1))


class Temporal_Attention_layer(nn.Module):
    """
    Computes temporal attention scores to capture temporal dependencies.
    Generates adaptive attention matrix across time steps.
    """
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Temporal_Attention_layer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices).to(DEVICE))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices).to(DEVICE))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).to(DEVICE))
        self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(DEVICE))

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, N, F_in, T]
        Returns:
            E_normalized: Normalized temporal attention matrix [B, T, T]
        """
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        # Transform: x [B, N, F_in, T] -> [B, T, F_in, N]
        # Then: [B, T, F_in, N](N) -> [B,T,F_in]
        # Finally: [B,T,F_in](F_in,N) -> [B,T,N]

        rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T) -> (B, N, T)

        product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T) -> (B,T,T)

        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)

        E_normalized = F.softmax(E, dim=1)

        return E_normalized


class cheb_conv(nn.Module):
    """
    Standard K-order Chebyshev graph convolution without spatial attention.
    Uses precomputed Chebyshev polynomials for efficient graph filtering.
    """

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        """
        Args:
            K: Order of Chebyshev polynomial
            cheb_polynomials: Precomputed Chebyshev polynomials
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super(cheb_conv, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x):
        """
        Performs Chebyshev graph convolution.
        Args:
            x: Input tensor [batch_size, N, F_in, T]
        Returns:
            Output tensor [batch_size, N, F_out, T]
        """

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):

                T_k = self.cheb_polynomials[k]  # (N,N)

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)

                output = output + rhs.matmul(theta_k)

            outputs.append(output.unsqueeze(-1))

        return F.relu(torch.cat(outputs, dim=-1))


class MSDGNN_block(nn.Module):
    """
    MSDGNN block: Adaptive Spatial-Temporal Graph Convolution Block with Dynamic Node Grouping.

    Core component of MSDGNN that integrates:
    - Spatial and temporal attention mechanisms
    - Multi-scale dynamic graph neural networks (group-level and global-level)
    - Chebyshev graph convolution with learned spatial attention
    - Temporal convolution with residual connections
    """

    def __init__(self, DEVICE, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials, num_of_vertices, num_of_timesteps):
        super(MSDGNN_block, self).__init__()

        self.SAT = SMultiHeadAttention(32, 4)
        self.pos_embed = nn.Parameter(torch.zeros(1, 22, in_channels, 32), requires_grad=True)

        self.TAt = Temporal_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.SAt = Spatial_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.cheb_conv_SAt = cheb_conv_withSAt(K, cheb_polynomials, in_channels, nb_chev_filter)
        self.time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(nb_time_filter)  # Requires channel dimension as last dimension

        self.DEVICE = DEVICE

        self.gnn_layer = 2
        # Node grouping configuration
        self.group_num = 8
        # Partition 22 cities into 8 groups with learnable soft assignment
        self.w = Parameter(torch.randn(22, self.group_num).to(self.DEVICE, non_blocking=True), requires_grad=True)

        # Temporal embedding layers for multi-scale time features
        self.u_embed1 = nn.Embedding(12, 4)  # Month embedding
        self.u_embed2 = nn.Embedding(7, 4)   # Day of week embedding
        self.u_embed3 = nn.Embedding(24, 4)  # Hour embedding

        # Edge information encoder: input_dim = 7*num_of_timesteps*2 + 12, output_dim = 12
        self.edge_inf = Seq(Lin(7 * num_of_timesteps * 2 + 12, 12), ReLU(inplace=True))

        # OPTIMIZATION: Group-level GNN layers for hierarchical feature aggregation
        # x_em = 32 loc_em = 12 edge_h = 12 gnn_h = 32 gnn_layer = 2
        self.group_gnn = nn.ModuleList([NodeModel(7 * num_of_timesteps, 12, 7 * num_of_timesteps)])
        for i in range(self.gnn_layer - 1):
            self.group_gnn.append(NodeModel(7 * num_of_timesteps, 12, 7 * num_of_timesteps))

        # OPTIMIZATION: Global-level GNN layers with optimized parameters
        self.global_gnn = nn.ModuleList([NodeModel(num_of_timesteps*2, 1, num_of_timesteps)])
        for i in range(self.gnn_layer - 1):
            self.global_gnn.append(NodeModel(num_of_timesteps, 1, num_of_timesteps))

        self.change = Seq(Lin(num_of_timesteps * 2, num_of_timesteps), ReLU(inplace=True))

        # Channel transformation layers
        self.c1 = nn.Conv2d(64, 7, kernel_size=1)
        self.c2 = nn.Conv2d(7, 64, kernel_size=1)

        self.predMLP = Seq(Lin(32, 16), ReLU(inplace=True), Lin(16, 1), ReLU(inplace=True))

        # OPTIMIZATION: Load precomputed edge indices and weights, fix on GPU
        # Use relative paths to automatically locate project root directory
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        edge_index_path = os.path.join(project_root, 'new_test08_index_and_weight', 'edge_index.npy')
        edge_w_path = os.path.join(project_root, 'new_test08_index_and_weight', 'edge_w.npy')

        self.index = torch.from_numpy(np.load(edge_index_path, allow_pickle=True)).to(DEVICE)
        self.weight = torch.from_numpy(np.load(edge_w_path, allow_pickle=True)).type(torch.FloatTensor).to(DEVICE)

        # OPTIMIZATION: Precompute inter-group edge indices to avoid repeated computation in forward pass
        # Generate inter-group edge index template: (group_num * (group_num - 1), 2)
        edge_list = []
        for i in range(self.group_num):
            for j in range(self.group_num):
                if i != j:
                    edge_list.append([i, j])
        self.group_edge_index_template = torch.tensor(edge_list, dtype=torch.long).to(DEVICE)
        self.num_group_edges = len(edge_list)


    def batchInput(self, x, edge_w, edge_index):
        """
        OPTIMIZATION: Optimized batchInput method for batch graph processing.
        Uses vectorized operations instead of loops for improved efficiency.

        Args:
            x: Node features [batch_size, num_nodes, feature_dim]
            edge_w: Edge weights [batch_size, num_edges, edge_dim]
            edge_index: Edge indices [batch_size, 2, num_edges]
        Returns:
            x: Flattened node features
            edge_w: Flattened edge weights
            edge_index: Adjusted edge indices for batched processing
        """
        batch_size = x.size(0)
        sta_num = x.shape[1]

        # Reshape node feature tensor, merge batch and node dimensions
        x = x.reshape(-1, x.shape[-1])

        # Reshape edge weight tensor
        edge_w = edge_w.reshape(-1, edge_w.shape[-1])

        # OPTIMIZATION: Update edge indices using vectorized operations
        # Create offset tensor to avoid loops
        offset = torch.arange(batch_size, device=edge_index.device).view(-1, 1, 1) * sta_num
        edge_index = edge_index + offset  # Broadcasting

        # Transpose and reshape edge index tensor
        edge_index = edge_index.transpose(1, 2).reshape(2, -1)

        return x, edge_w, edge_index

    def forward(self, x, u):
        """
        Forward pass of MSDGNN block.

        Args:
            x: Input features [batch_size, N, F_in, T]
            u: Temporal information [batch_size, 3] containing (month, day_of_week, hour)
        Returns:
            x_residual: Output features [batch_size, N, nb_time_filter, T]
        """

        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        # ---------------------------------------------------------------------------
        # Group-level GNN processing
        x_T = x
        a,b,c,d = x_T.shape
        # Dimension reduction if necessary: convert 64 channels to 7 channels
        if c == 64:
          x_T = x_T.permute(0, 2, 1, 3)  # Transform to (batch_size, 64, 22, 32)
          x_T = self.c1(x_T)
          x_T = x_T.permute(0, 2, 1, 3)  # Transform to (batch_size, 22, 7, 32)

        a1,b1,c1_dim,d1 = x_T.shape

        # Compute soft group assignment weights
        w = F.softmax(self.w, dim=1)  # Softmax over group dimension
        w1 = w.transpose(0, 1).unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, group_num, 22)

        # Aggregate node features into group features: flatten feature and time dimensions
        new_x = x_T.reshape(batch_size, num_of_vertices, -1)  # (batch_size, 22, c1*d1)
        g_x = torch.bmm(w1, new_x)  # (batch_size, group_num, c1*d1)

        # Temporal embeddings for multi-scale modeling
        u_em1 = self.u_embed1(u[:, 0])
        u_em2 = self.u_embed2(u[:, 1])
        u_em3 = self.u_embed3(u[:, 2])
        u_em = torch.cat([u_em1, u_em2, u_em3], dim=-1)  # (batch_size, 12)

        # OPTIMIZATION: Optimized inter-group relationship computation
        # Use precomputed edge index template to avoid nested loops
        num_edges = self.num_group_edges

        # Build input features for all edge pairs
        src_idx = self.group_edge_index_template[:, 0]  # (num_edges,)
        dst_idx = self.group_edge_index_template[:, 1]  # (num_edges,)

        # Retrieve source and destination node features using gather to avoid advanced indexing
        batch_size_val = g_x.size(0)
        feature_dim = g_x.size(2)

        # Expand indices to all batch and feature dimensions
        src_idx_expanded = src_idx.unsqueeze(0).unsqueeze(2).expand(batch_size_val, num_edges, feature_dim)
        dst_idx_expanded = dst_idx.unsqueeze(0).unsqueeze(2).expand(batch_size_val, num_edges, feature_dim)

        # Use gather for indexing
        g_x_src = torch.gather(g_x, 1, src_idx_expanded)  # (batch_size, num_edges, feature_dim)
        g_x_dst = torch.gather(g_x, 1, dst_idx_expanded)  # (batch_size, num_edges, feature_dim)

        # Expand temporal embeddings to match number of edges
        u_em_expanded = u_em.unsqueeze(1).expand(batch_size, num_edges, -1)  # (batch_size, num_edges, 12)

        # Concatenate features for edge information encoding
        g_edge_input = torch.cat([g_x_src, g_x_dst, u_em_expanded], dim=-1)  # (batch_size, num_edges, 7*32*2+12)

        # Compute edge weights through MLP
        g_edge_w = self.edge_inf(g_edge_input)  # (batch_size, num_edges, 12)

        # Prepare edge indices for graph convolution
        g_edge_index = self.group_edge_index_template.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, num_edges, 2)
        g_edge_index = g_edge_index.transpose(1, 2)  # (batch_size, 2, num_edges)

        # Batch input processing for GNN
        g_x, g_edge_w, g_edge_index = self.batchInput(g_x, g_edge_w, g_edge_index)

        # Group-level GNN message passing
        for i in range(self.gnn_layer):
            g_x = self.group_gnn[i](g_x, g_edge_index, g_edge_w)

        # Reshape back to original batch structure
        g_x = g_x.reshape(batch_size, self.group_num, -1)  # (batch_size, group_num, 7*32)

        # Aggregate back to node level using soft assignment weights
        w2 = w.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, 22, group_num)
        new_x = torch.bmm(w2, g_x)  # (batch_size, 22, c1_dim*d1)

        # Reshape to original dimensions
        new_x = new_x.reshape(batch_size, num_of_vertices, c1_dim, -1)  # (batch_size, 22, c1_dim, d1)
        new_x = torch.cat([x_T, new_x], dim=-1)  # Concatenate with original: (batch_size, 22, c1_dim, 2*d1)

        # ------------------------------------------------------------------------------- #
        # Global-level GNN processing
        index_ = self.index.unsqueeze(0).expand(batch_size, -1, -1)
        weight_ = self.weight.unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)

        new_x, edge_w, edge_index = self.batchInput(new_x, weight_, index_)

        # Global-level GNN message passing
        for i in range(self.gnn_layer):
            new_x = self.global_gnn[i](new_x, edge_index, edge_w)

        new_x = new_x.reshape(batch_size, num_of_vertices, c1_dim, -1)  # (batch_size, 22, c1_dim, d1)

        # Dimension expansion if necessary: convert 7 channels back to 64 channels
        if c == 64:
            new_x = new_x.permute(0, 2, 1, 3)  # Transform to (batch_size, 7, 22, 32)
            new_x = self.c2(new_x)
            new_x = new_x.permute(0, 2, 1, 3)  # Transform to (batch_size, 22, 64, 32)

        # ---------------------------------------------------------------------------
        # Spatial-Temporal Attention Mechanism
        input = x + self.pos_embed

        x_T_Q = input
        x_T_K = input
        x_T_V = input

        x_T_ = self.SAT(x_T_Q, x_T_K, x_T_V)

        # Temporal Attention
        temporal_At = self.TAt(x_T_)  # (b, T, T)

        x_TAt = torch.matmul(x.reshape(batch_size, -1, num_of_timesteps), temporal_At).reshape(batch_size,
                                                                                               num_of_vertices,
                                                                                               num_of_features,
                                                                                               num_of_timesteps)
        # Spatial Attention
        spatial_At = self.SAt(x_TAt)

        # Chebyshev graph convolution with spatial attention
        spatial_gcn = self.cheb_conv_SAt(new_x, spatial_At)  # (b,N,F,T)

        # Temporal convolution along time axis
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))  # (b,N,F,T)->(b,F,N,T)

        # Residual shortcut connection
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))  # (b,N,F,T)->(b,F,N,T)

        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)

        return x_residual


class MSDGNN_submodule(nn.Module):
    """
    MSDGNN submodule for single temporal scale (hour/day/week).

    """

    def __init__(self, DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials, num_for_predict, len_input, num_of_vertices):
        """
        Args:
            nb_block: Number of MSDGCN blocks
            in_channels: Number of input channels
            K: Order of Chebyshev polynomial
            nb_chev_filter: Number of Chebyshev filters
            nb_time_filter: Number of temporal filters
            time_strides: Stride for temporal convolution
            cheb_polynomials: Precomputed Chebyshev polynomials
            num_for_predict: Number of time steps to predict
            len_input: Length of input sequence
            num_of_vertices: Number of graph nodes
        """

        super(ASTGCN_submodule, self).__init__()

        self.BlockList = nn.ModuleList([ASTGCN_block(DEVICE, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials, num_of_vertices, len_input)])

        self.BlockList.extend([ASTGCN_block(DEVICE, nb_time_filter, K, nb_chev_filter, nb_time_filter, 1, cheb_polynomials, num_of_vertices, len_input//time_strides) for _ in range(nb_block-1)])

        self.final_conv = nn.Conv2d(int(len_input/time_strides), num_for_predict, kernel_size=(1, nb_time_filter))

        self.DEVICE = DEVICE

        self.to(DEVICE)


    def forward(self, x, u):
        """
        Args:
            x: Input features [B, N_nodes, F_in, T_in]
            u: Temporal information [B, 3]
        Returns:
            output: Predictions [B, N_nodes, T_out]
        """

        for block in self.BlockList:
            x = block(x,u)

        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        # Transform: (b,N,F,T)->(b,T,N,F)->[1,F] conv->(b,T_out,N,1)->(b,T_out,N)->(b,N,T_out)

        return output

class MSDGNN_full_submodule(nn.Module):
    """
    Full MSDGNN model with multi-scale temporal modeling.
    Combines predictions from hour, day, and week components with learnable fusion weights.
    """
    def __init__(   self, DEVICE, nb_block, in_channels, K,
                    nb_chev_filter, nb_time_filter,
                    time_strides, adj_mx, num_for_predict,
                    len_input, num_of_vertices,
                    L_tilde, cheb_polynomials):
        super().__init__()
        self.h_model = MSDGNN_submodule(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials, num_for_predict, len_input, num_of_vertices)
        self.d_model = MSDGNN_submodule(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials, num_for_predict, len_input, num_of_vertices)
        self.w_model = MSDGNN_submodule(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials, num_for_predict, len_input, num_of_vertices)

        self.W_h = torch.zeros(num_for_predict, requires_grad=True, device=DEVICE)
        self.W_d = torch.zeros(num_for_predict, requires_grad=True, device=DEVICE)
        self.W_w = torch.zeros(num_for_predict, requires_grad=True, device=DEVICE)

        nn.init.uniform_(self.W_h)
        nn.init.uniform_(self.W_d)
        nn.init.uniform_(self.W_w)

        self.to(DEVICE)


    def forward(self, x_h, x_d, x_w, u):
        """
        Args:
            x_h: Hourly input [B, N_nodes, F_in, T_in]
            x_d: Daily input [B, N_nodes, F_in, T_in]
            x_w: Weekly input [B, N_nodes, F_in, T_in]
            u: Temporal information [B, 3]
        Returns:
            Fused prediction from multi-scale temporal components
        """
        h_pred = self.h_model(x_h, u) # (B, N_nodes, T_out)
        d_pred = self.d_model(x_d, u) # (B, N_nodes, T_out)
        w_pred = self.w_model(x_w, u) # (B, N_nodes, T_out)

        return self.W_h*h_pred + self.W_d*d_pred + self.W_w*w_pred


def make_model(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, adj_mx, num_for_predict, len_input, num_of_vertices):
    """
    Factory function to create MSDGNN model.

    Args:
        DEVICE: Target device (CPU/GPU)
        nb_block: Number of MSDGNN blocks
        in_channels: Number of input channels
        K: Order of Chebyshev polynomial
        nb_chev_filter: Number of Chebyshev filters
        nb_time_filter: Number of temporal filters
        time_strides: Stride for temporal convolution
        adj_mx: Adjacency matrix
        num_for_predict: Number of prediction steps
        len_input: Length of input sequence
        num_of_vertices: Number of graph nodes
    Returns:
        model: Initialized MSDGNN model
    """
    L_tilde = scaled_Laplacian(adj_mx)
    cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE) for i in cheb_polynomial(L_tilde, K)]
    model = MSDGNN_full_submodule(  DEVICE, nb_block, in_channels,
                                    K, nb_chev_filter, nb_time_filter,
                                    time_strides, cheb_polynomials,
                                    num_for_predict, len_input, num_of_vertices,
                                    L_tilde, cheb_polynomials)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model

class NodeModel(torch.nn.Module):
    """
    Graph neural network node update model.
    Updates node representations by aggregating information from neighboring edges and nodes.
    """
    def __init__(self,node_h,edge_h,gnn_h):
        super(NodeModel, self).__init__()
        self.node_mlp_1 = Seq(Lin(node_h+edge_h,gnn_h), ReLU(inplace=True))
        self.node_mlp_2 = Seq(Lin(node_h+gnn_h,gnn_h), ReLU(inplace=True))

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: Node features [N, F_x], where N is the number of nodes
            edge_index: Edge connectivity [2, E] with max entry N-1
            edge_attr: Edge attributes [E, F_e]
        Returns:
            Updated node representations
        """
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)

        # Encode edge representations
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)
