import torch
import torch.nn as nn
import numpy as np


def compute_affine_lambda_batch(nodes, cells):
    
    nodes_tensor = torch.tensor(nodes, dtype=torch.float64)
    
    # 使用索引收集，避免多次索引操作
    A = nodes_tensor[cells[:, 0]]
    B = nodes_tensor[cells[:, 1]] 
    C = nodes_tensor[cells[:, 2]]
    
    M1 = B - A
    M2 = C - A
    a, c = M1[:, 0], M1[:, 1]
    b, d = M2[:, 0], M2[:, 1]
    det = a * d - b * c
    inv_det = 1.0 / det
    
    if torch.any(torch.abs(det) < 1e-12):
        print(f"警告：发现 {torch.sum(torch.abs(det) < 1e-12)} 个接近退化的三角形")
        print("但Delaunay三角剖分理论上不应该产生退化三角形")
    
    # 直接在GPU上创建张量
    W_all = torch.zeros((len(cells), 3, 2), dtype=torch.float64)  
    W_all[:, 1, 0] =  d * inv_det
    W_all[:, 1, 1] = -b * inv_det
    W_all[:, 2, 0] = -c * inv_det
    W_all[:, 2, 1] =  a * inv_det
    W_all[:, 0, :] = -W_all[:, 1, :] - W_all[:, 2, :]

    b_all = torch.zeros((len(cells), 3), dtype=torch.float64)
    b23_0 = -(W_all[:, 1, 0] * A[:, 0] + W_all[:, 1, 1] * A[:, 1])
    b23_1 = -(W_all[:, 2, 0] * A[:, 0] + W_all[:, 2, 1] * A[:, 1])
    
    b_all[:, 1] = b23_0
    b_all[:, 2] = b23_1  
    b_all[:, 0] = 1 - (b23_0 + b23_1)
    
    return W_all, b_all


class HardGateFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        eps = 1e-10
        gate = (x <= 1.0 + eps).to(x.dtype)   # <=1 → 1,  >1 → 0
        
        gate_sum = gate.sum(dim=-1, keepdim=True)
        normalized_gate = gate / gate_sum
        return normalized_gate

    @staticmethod
    def backward(ctx, grad_output):
        return torch.zeros_like(grad_output)

class CustomReluFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.relu(x)  # 前向传播还是标准的ReLU
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output
        grad_input[x < 0] = 0.0  # 负值区域梯度为0

        return grad_input

class HardGate(nn.Module):
    def forward(self, x): return HardGateFn.apply(x)
    
class CustomRelu(nn.Module):
    def forward(self, x): return CustomReluFn.apply(x)

class ReLU2(nn.Module):
    def forward(self, x): return torch.relu(x) ** 2

class FiniteElementNetp1(nn.Module):
    def __init__(self, cells_weights, cells_biases, cells_values):
        super().__init__()
        
        # 基函数权重
        self.register_buffer('cells_weights', cells_weights)                              # (N, 3, 2)
        self.register_buffer('cells_biases', cells_biases)                                # (N, 3)
        self.cells_act = CustomRelu()
        
        # 门控
        self.register_buffer('cells_values', cells_values)                                # (N, 3)
        self.register_buffer('cells_gates', torch.ones(3, 1, dtype=torch.float64))       # (3, 1)
        self.gate_act = HardGate()
        
        # 用于实现乘法运算u*g
        self.register_buffer('multi_weights', torch.tensor([[0.5, -0.5, 0.5, -0.5], [0.5, -0.5, -0.5, 0.5]], dtype=torch.float64)) # (2, 4)
        self.relu2_act = ReLU2()
        
        # 合并权重
        self.register_buffer('combi_weights', torch.tensor([1.0, 1.0, -1.0, -1.0], dtype=torch.float64))                   # (4,)
        
    def forward(self, coordinates):
        
        # 1. 计算重心坐标基函数值
        bar_coords = torch.einsum('njc,...c->...nj', self.cells_weights, coordinates) + self.cells_biases
        bar_coords = self.cells_act(bar_coords)                       # (B, N, 3)
        
        # 2. 计算三角形贡献和门控信号
        u = torch.einsum('...nj,nj->...n', bar_coords, self.cells_values)
        gate = torch.einsum('...nj,jk->...n', bar_coords, self.cells_gates)
        gate = self.gate_act(gate)                                  # (B, N)
        
        # 3. 通过relu2实现乘法运算
        ug_pair = torch.stack([u, gate], dim=-1)                    # (B, N, 2)
        unfold = torch.einsum('...nj,jk->...nk', ug_pair, self.multi_weights)
        unfold = self.relu2_act(unfold)                              # (B, N, 4)
        
        # 4. 合并得到乘法结果并求和
        # U = torch.einsum('...nk,k->...', unfold, self.combi_weights) # (B,)
        U = torch.einsum('...n->...', torch.einsum('...nk,k->...n', unfold, self.combi_weights))      # 此做法无精度损失，上述做法最后的爱因斯坦求和有损
        return U
        
        U = torch.einsum('...n,...n->...', u, gate)               # (B,)  
        return U
    
# %% 第二层操作用两个爱因斯坦求和分别操作     
class FiniteElementNetp2(nn.Module):
    def __init__(self, cells_weights, cells_biases, cells_values):
        super().__init__()
        
        # 基函数权重
        self.register_buffer('cells_weights', cells_weights)                              # (N, 3, 2)
        self.register_buffer('cells_biases', cells_biases)                                # (N, 3)
        self.cells_act = CustomRelu()
        self.cells_act = nn.ReLU()
        # 门控
        self.register_buffer('cells_values', cells_values)                                # (N, 6)
        self.register_buffer('cells_gates', torch.ones(6, 1, dtype=torch.float64))       # (6, 1)
        self.gate_act = HardGate()
        
        # 用于实现乘法运算u*g
        self.register_buffer('multi_weights', torch.tensor([[0.5, -0.5, 0.5, -0.5], [0.5, -0.5, -0.5, 0.5]], dtype=torch.float64)) # (2, 4)
        self.relu2_act = ReLU2()
        
        # 合并权重
        self.register_buffer('combi_weights', torch.tensor([1.0, 1.0, -1.0, -1.0], dtype=torch.float64))                   # (4,)
        
    def forward(self, coordinates):
        
        # 1. 计算重心坐标基函数值
        bar_coords = torch.einsum('njc,...c->...nj', self.cells_weights, coordinates) + self.cells_biases
        bar_coords = self.cells_act(bar_coords)                                      # (B, N, 3)
        
        l1, l2, l3 = bar_coords[..., 0], bar_coords[..., 1], bar_coords[..., 2]
        
        # 顶点基函数
        N1 = l1 * (2*l1 - 1)
        N2 = l2 * (2*l2 - 1)
        N3 = l3 * (2*l3 - 1)
        
        # 边中点基函数
        N4 = 4 * l1 * l2
        N5 = 4 * l2 * l3
        N6 = 4 * l3 * l1
        phi = torch.stack([N1, N4, N6, N2, N5, N3], dim=-1)                         # (B, N, 6)
        
        # 2. 计算门控信号
        gate = torch.einsum('...cn,nj->...c', phi, self.cells_gates)
        gate = self.gate_act(gate)                                                   # (B, N)
        
        # 标量值函数
        if self.cells_values.ndim == 2:
            u = torch.einsum('...cn,cn->...c', phi, self.cells_values)                # (B, N)
            
            # 3. 通过 ReLU2 实现乘法
            ug_pair = torch.stack([u, gate], dim=-1)                                 # (B, N, 2)
            unfold = torch.einsum('bnj,jk->bnk', ug_pair, self.multi_weights)
            relu2 = self.relu2_act(unfold)                                           # (B, N, 4)
            
            # 4. 合并得到乘法结果并求和
            # U = torch.einsum('bnk,k->b', relu2, self.combi_weights)
            # return U
            
            U = torch.einsum('bn,bn->b', u, gate)
            return U
        
        # 二维向量值函数
        u1 = torch.einsum('...cn,cn->...c', phi, self.cells_values[..., 0])           # (B, N)
        u2 = torch.einsum('...cn,cn->...c', phi, self.cells_values[..., 1])           # (B, N)
        
        # 3. 通过 ReLU2 实现乘法
        u1g_pair = torch.stack([u1, gate], dim=-1)                                   # (B, N, 2)
        u2g_pair = torch.stack([u2, gate], dim=-1)                                   # (B, N, 2)
        unfold1 = torch.einsum('bnj,jk->bnk', u1g_pair, self.multi_weights)
        unfold2 = torch.einsum('bnj,jk->bnk', u2g_pair, self.multi_weights)
        relu2_1 = self.relu2_act(unfold1)                                             # (B, N, 4)
        relu2_2 = self.relu2_act(unfold2)                                             # (B, N, 4)
        
        # 4. 合并得到乘法结果并求和
        # U1 = torch.einsum('bnk,k->b', relu2_1, self.combi_weights)
        # U2 = torch.einsum('bnk,k->b', relu2_2, self.combi_weights)
        # return torch.stack([U1, U2], dim=-1)
        
        U1 = torch.einsum('bn,bn->b', u1, gate)
        U2 = torch.einsum('bn,bn->b', u2, gate)
        return torch.stack([U1, U2], dim=-1)    


def build_p1fem_network(oldSpace, uh, device='cpu'):
    """根据 oldmesh 和节点值 uh 构造有限元插值器。"""

    node = torch.as_tensor(oldSpace.mesh.entity('node'), dtype=torch.float64, device=device)
    cell = torch.as_tensor(oldSpace.mesh.entity('cell'), dtype=torch.long, device=device)
    cellDOF = torch.as_tensor(oldSpace.cell_to_dof(), dtype=torch.long, device=device)
    uh = torch.as_tensor(np.asarray(uh[:]), dtype=torch.float64, device=device)

    W_all, b_all = compute_affine_lambda_batch(node, cell)
    model = FiniteElementNetp1(W_all, b_all, uh[cellDOF]).to(device)

    def predict(coordinates):
        """预测 smesh 所有节点处的函数值。"""
        test_points = torch.as_tensor(coordinates, dtype=torch.float64, device=device)
        return model(test_points)

    return predict

    
def build_p2fem_network(oldSpace, uh, device='cpu'):
    """根据 oldmesh 和节点值 uh 构造有限元插值器。"""

    node = torch.as_tensor(oldSpace.mesh.entity('node'), dtype=torch.float64, device=device)
    cell = torch.as_tensor(oldSpace.mesh.entity('cell'), dtype=torch.long, device=device)
    cellDOF = torch.as_tensor(oldSpace.cell_to_dof(), dtype=torch.long, device=device)
    uh = torch.as_tensor(np.asarray(uh[:]), dtype=torch.float64, device=device)

    W_all, b_all = compute_affine_lambda_batch(node, cell)
    model = FiniteElementNetp2(W_all, b_all, uh[cellDOF]).to(device)

    def predict(coordinates):
        """预测 smesh 所有节点处的函数值。"""
        test_points = torch.as_tensor(coordinates, dtype=torch.float64, device=device)
        return model(test_points)

    return predict

