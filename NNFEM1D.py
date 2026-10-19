import torch
import torch.nn as nn
import numpy as np
import time
from scipy.interpolate import interp1d


def generate_mesh(n):
    nodes = np.linspace(0.0, 1.0, n + 1).reshape(-1, 1)  # (n+1, 1)
    
    # 无循环版本 - 向量化操作，速度很快
    cells = np.column_stack([np.arange(n), np.arange(1, n + 1)])

    return nodes, cells

def compute_affine_lambda_batch(nodes, cells):
    nodes_tensor = torch.tensor(nodes, dtype=torch.float64)
    
    A = nodes_tensor[cells[:, 0]]
    B = nodes_tensor[cells[:, 1]] 
    
    M1 = B - A
    a = M1[:, 0]  # 1D情况只有x分量
    det = a  # 1D的"行列式"就是线段长度
    
    inv_det = 1.0 / det
    
    if torch.any(torch.abs(det) < 1e-12):
        print(f"警告：发现 {torch.sum(torch.abs(det) < 1e-12)} 个接近退化的线段")
    
    # 直接在GPU上创建张量
    W_all = torch.zeros((len(cells), 2, 1), dtype=torch.float64)  
    W_all[:, 1, 0] = inv_det  # λ₁的权重
    W_all[:, 0, 0] = -inv_det  # λ₀的权重 = -λ₁的权重

    b_all = torch.zeros((len(cells), 2), dtype=torch.float64)
    b1 = -(W_all[:, 1, 0] * A[:, 0])  # λ₁的偏置
    
    b_all[:, 1] = b1
    b_all[:, 0] = 1 - b1  # λ₀的偏置 = 1 - λ₁的偏置
    
    return W_all, b_all

def target_function(x):
    """目标函数：定义在计算域上的函数"""
    # return x                # f(x,y) = x
    # return x**2             # f(x,y) = x^2
    return np.sin(x)          # f(x,y) = sin(x)

def create_cells_nodeValues(nodes, cells):
    """创建1D情况下的节点函数值"""
    # 计算所有节点上的函数值
    point_values = target_function(nodes[:, 0])
    
    # 创建线段节点函数值数组 (N_seg, 2)
    N_seg = len(cells)
    node_values = np.zeros((N_seg, 2), dtype=np.float64)
    
    # 将点上的函数值赋给对应的线段端点
    node_values = point_values[cells]
    
    return torch.tensor(node_values, dtype=torch.float64)

def Piecewise_interpolator(nodes, cells, verbose=False):
    """创建1D分段线性插值器"""
    
    # 计算节点函数值
    point_values = target_function(nodes)
    
    # 创建分段线性插值器
    interpolator = interp1d(nodes, point_values, kind='linear', 
                          bounds_error=False, fill_value="extrapolate")
    
    # 根据verbose参数决定是否打印每个线段的插值函数
    if verbose:
        for i, seg_indices in enumerate(cells):
            # 取出线段端点坐标和函数值
            x_seg = nodes[seg_indices]
            f_seg = point_values[seg_indices]
            
            # 1D线性插值：U(x) = a*x + b
            # 解线性系统：
            # a*x0 + b = f0
            # a*x1 + b = f1
            A = np.column_stack([x_seg, np.ones(2)])
            coeff = np.linalg.solve(A, f_seg)
            
            a, b = coeff
            print(f"线段 {i}: 端点索引 {seg_indices} | U(x) = {a:.6f} * x + {b:.6f}")

    return interpolator

class HardGateFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        eps = 1e-12
        gate = (x <= 1.0 + eps).to(x.dtype)   # <=1 → 1,  >1 → 0
        
        gate_sum = gate.sum(dim=1, keepdim=True)
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
        grad_input = grad_output.clone()
        grad_input[x < 0] = 0.0  # 负值区域梯度为0

        return grad_input

class HardGate(nn.Module):
    def forward(self, x): return HardGateFn.apply(x)
    
class CustomRelu(nn.Module):
    def forward(self, x): return CustomReluFn.apply(x)

class ReLU2(nn.Module):
    def forward(self, x): return torch.relu(x) ** 2

class FiniteElementNet(nn.Module):
    def __init__(self, cells_weights, cells_biases, cells_values):
        super().__init__()

        # 基函数权重: (N, 2, 1)
        self.cells_weights = cells_weights                              # (N, 2, 1)
        self.cells_biases = cells_biases                                # (N, 2)
        self.cells_values = cells_values                                # (N, 2)
        self.cells_act = CustomRelu()
        
        # 门控: 2个基函数
        self.gate_weights = torch.ones(2, 1, dtype=torch.float64)   # (2, 1)
        self.gate_act = HardGate()
        
        # 用于实现乘法运算u*g
        self.multiply_weights = torch.tensor([
            [0.5, -0.5, 0.5, -0.5], [0.5, -0.5, -0.5, 0.5]
        ], dtype=torch.float64)                                     # (2, 4)
        self.relu2_act = ReLU2()
        
        # 合并权重
        self.combined_weights = torch.tensor([
            [1.0], [1.0], [-1.0], [-1.0]
        ], dtype=torch.float64)                                     # (4, 1)
        
    def forward(self, coordinates):
        # coordinates: (batch_size, 1) - 修改为1D输入坐标点
        
        # 1. 计算线性坐标基函数值
        bar_coords = torch.einsum('cbd,sd->scb', self.cells_weights, coordinates) + self.cells_biases
        bar_coords = self.cells_act(bar_coords)                       # (B, N, 2)
        
        # 2. 计算线段贡献和门控信号
        u = torch.einsum('snj,nj->sn', bar_coords, self.cells_values)
        gate = torch.einsum('snj,jk->n', bar_coords, self.gate_weights)
        gate = self.gate_act(gate)                                  # (B, N)
        
        # 3. 通过relu2实现乘法运算
        ug_pair = torch.stack([u, gate], dim=-1)                    # (B, N, 2)
        unfold = torch.einsum('bnj,jk->bnk', ug_pair, self.multiply_weights)
        relu2 = self.relu2_act(unfold)                              # (B, N, 4)
        
        # 4. 合并得到乘法结果并求和
        # U = torch.einsum('bnk,km->b', relu2, self.combined_weights) # (B,)   
        # return U

        U = torch.einsum('bn,bn->b', u, gate)               # (B,)  此做法无精度损失，上述做法最后的爱因斯坦求和有损
        return U

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    n = 5000  # 增加网格点数以获得更多三角形

    # 1. 生成三角形网格、节点函数值和插值器
    nodes_np, cells_np = generate_mesh(n)

    nodes = torch.as_tensor(nodes_np, dtype=torch.float64, device=device).reshape(-1, 1)
    cells = torch.as_tensor(cells_np, dtype=torch.long, device=device)
    
    # 2. 计算仿射变换参数
    W_all, b_all = compute_affine_lambda_batch(nodes, cells)

    # 3. 创建三角形节点函数值
    node_values = create_cells_nodeValues(nodes, cells)
    
    # 4. 初始化模型
    model = FiniteElementNet(
        W_all.to(device), 
        b_all.to(device), 
        node_values.to(device)
    ).to(device)
    
    # 5. 测试输入
    # ============================================================
    # testpoints1：所有网格节点
    # ============================================================

    testpoints1 = nodes


    # ============================================================
    # testpoints2：所有一维单元的中点
    # ============================================================

    left_nodes = nodes[cells[:, 0]]
    right_nodes = nodes[cells[:, 1]]

    testpoints2 = 0.5 * (
        left_nodes + right_nodes
    )


    # ============================================================
    # testpoints3：每个一维单元上的两个 Gauss 点
    # ============================================================

    # 参考区间 [-1, 1] 上的二点 Gauss 坐标
    gauss_reference = nodes.new_tensor(
        [
            -1.0 / np.sqrt(3.0),
            1.0 / np.sqrt(3.0),
        ]
    )

    cell_centers = 0.5 * (
        left_nodes + right_nodes
    )

    cell_half_lengths = 0.5 * (
        right_nodes - left_nodes
    )

    testpoints3 = (
        cell_centers[:, None, :]
        + cell_half_lengths[:, None, :]
        * gauss_reference[None, :, None]
    ).reshape(-1, 1)


    print("testpoints1:", testpoints1.shape)
    print("testpoints2:", testpoints2.shape)
    print("testpoints3:", testpoints3.shape)
        
    test_points_np = testpoints3
    test_points = torch.tensor(test_points_np, dtype=torch.float64).to(device)  # (B, 1)
    # print(f"测试点坐标为: {test_points.squeeze()}")
    
    # 7. 验证梯度
    start_time = time.time()
    test_points.requires_grad_(True)
    output = model(test_points)
    model_time = time.time() - start_time 
    print(f"模型计算时间: {model_time:.6f} 秒")
    print(f"输出值: {output.squeeze()}")

    # # 计算导数
    # grad = torch.autograd.grad(
    #     outputs=output, 
    #     inputs=test_points, 
    #     grad_outputs=torch.ones_like(output),
    #     retain_graph=False, 
    #     create_graph=False
    # )[0]

    # print(f"导数值: {grad.T}")
    
    #############################################################################################
    #############################################################################################
    #############################################################################################
          
    # 使用 scipy 插值器计算（用于验证）
    interpolator = Piecewise_interpolator(nodes[:,0], cells)
    scipy_output = interpolator(test_points_np[:,0])
    print(f"scipy 输出值: {scipy_output}")
    
    # 验证函数值差异
    output_np = output.detach().cpu().numpy()
    func_diff = np.abs(output_np - scipy_output)

    print(f"\n=== 函数值差异验证 ===")
    print(f"最大差异: {np.max(func_diff):.6e}")
    print(f"平均差异: {np.mean(func_diff):.6e}")
