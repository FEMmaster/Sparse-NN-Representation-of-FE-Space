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
    # return np.ones_like(x)

def create_cells_nodeValues(nodes, cells):
    """
    创建一维 P2 单元的局部自由度函数值。

    局部编号顺序必须与 phi 的输出顺序一致：
        0: cells[:, 0] 对应的端点
        1: 单元中点
        2: cells[:, 1] 对应的端点

    返回：
        node_values.shape = (N_cells, 3)
    """

    # 兼容 nodes.shape 为 (N,) 或 (N, 1)
    if nodes.ndim == 1:
        x = nodes
    else:
        x = nodes[:, 0]

    # 每个单元的两个端点坐标
    x0 = x[cells[:, 0]]
    x1 = x[cells[:, 1]]

    # 每个单元的中点坐标
    xm = 0.5 * (x0 + x1)

    # 按 phi 的顺序排列：端点0、中点、端点1
    cell_values = np.stack(
        [
            target_function(x0),
            target_function(xm),
            target_function(x1),
        ],
        axis=1,
    )

    return torch.tensor(cell_values, dtype=torch.float64)

def Piecewise_interpolator(nodes, cells, verbose=False):
    """
    创建一维分片二次 P2 插值器。

    每个单元的局部自由度顺序为：
        0: 左端点 x0
        1: 中点   xm
        2: 右端点 x1

    该顺序与 create_cells_nodeValues 和网络中的 phi 一致。
    """

    # nodes 可以是 (N,) 或 (N, 1)
    x_nodes = np.asarray(nodes, dtype=np.float64).reshape(-1)
    cells = np.asarray(cells, dtype=np.int64)

    # 每个单元的三个局部插值点
    x0 = x_nodes[cells[:, 0]]
    x1 = x_nodes[cells[:, 1]]
    xm = 0.5 * (x0 + x1)

    # 检查单元方向
    if np.any(x1 <= x0):
        raise ValueError("当前实现要求每个单元满足 x1 > x0。")

    # 三个局部自由度上的函数值
    f0 = target_function(x0)
    fm = target_function(xm)
    f1 = target_function(x1)

    # 按单元左端点排序，便于查找测试点所属单元
    order = np.argsort(x0)

    x0 = x0[order]
    xm = xm[order]
    x1 = x1[order]

    f0 = f0[order]
    fm = fm[order]
    f1 = f1[order]

    if verbose:
        for c in range(len(cells)):
            # 仅用于打印标准多项式：
            # U(x) = a*x^2 + b*x + d
            coefficients = np.polyfit(
                [x0[c], xm[c], x1[c]],
                [f0[c], fm[c], f1[c]],
                deg=2,
            )

            a, b, d = coefficients

            print(
                f"单元 {c}: "
                f"[{x0[c]:.6f}, {x1[c]:.6f}] | "
                f"U(x) = {a:.12f} * x^2 "
                f"+ {b:.12f} * x "
                f"+ {d:.12f}"
            )

    def interpolator(points):
        """
        计算测试点处的分片 P2 插值值。
        """

        points = np.asarray(points, dtype=np.float64)
        original_shape = points.shape
        points_flat = points.reshape(-1)

        # 查找测试点所属单元
        #
        # 例如右端点数组：
        # [0.5, 1.0]
        #
        # x=0.2 -> cell_id=0
        # x=0.5 -> cell_id=0
        # x=0.6 -> cell_id=1
        cell_id = np.searchsorted(
            x1,
            points_flat,
            side="left",
        )

        # 与原来的 fill_value="extrapolate" 类似：
        # 左侧使用第一个单元外推；
        # 右侧使用最后一个单元外推。
        cell_id = np.clip(cell_id, 0, len(x1) - 1)

        xa = x0[cell_id]
        xb = x1[cell_id]
        h = xb - xa

        # 线性重心坐标
        lambda0 = (xb - points_flat) / h
        lambda1 = (points_flat - xa) / h

        # 一维 P2 拉格朗日基函数
        phi0 = lambda0 * (2.0 * lambda0 - 1.0)
        phim = 4.0 * lambda0 * lambda1
        phi1 = lambda1 * (2.0 * lambda1 - 1.0)

        # 分片二次插值
        values = (
            f0[cell_id] * phi0
            + fm[cell_id] * phim
            + f1[cell_id] * phi1
        )

        return values.reshape(original_shape)

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

        # 基函数权重
        self.cells_weights = cells_weights                              # (N, 2, 1)
        self.cells_biases = cells_biases                                # (N, 2)
        self.cells_values = cells_values                                # (N, 3)
        self.cells_act = CustomRelu()
        
        # 第二层：从线性元构造二次元扩展特征（2 → 12）
        self.expand_weights = torch.tensor([
            [ 1.5, -1.5, -0.5,  0.5,  1.0, -1.0,  1.0, -1.0,  0.0,  0.0,  0.0,  0.0],
            [ 0.0,  0.0,  0.0,  0.0,  1.0, -1.0, -1.0,  1.0,  1.5, -1.5, -0.5,  0.5]
        ], dtype=torch.float64)  
        self.expand_bias = torch.tensor([
            -0.5,  0.5,  0.5, -0.5,  0.0,  0.0,  0.0,  0.0, -0.5,  0.5,  0.5, -0.5
        ], dtype=torch.float64)        
        self.relu2_act = ReLU2()
        
        # 第三层             
        self.merge_weights = nn.Parameter(torch.tensor([
            [ 1.0,  1.0, -1.0, -1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [ 0.0,  0.0,  0.0,  0.0,  1.0,  1.0, -1.0, -1.0,  0.0,  0.0,  0.0,  0.0],
            [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  1.0, -1.0, -1.0]
        ], dtype=torch.float64))
        
        # 门控: 3个基函数
        self.gate_weights = torch.ones(3, 1, dtype=torch.float64)   # (3, 1)
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
        # S: number of samples
        # C: number of cells
        # D: dimension
        # B: number of baryCoords
        # N: number of basisFuncs
        # J: temporary
        # K: temporary
        
        # 1. 计算线性坐标基函数值
        bar_coords = torch.einsum('cbd,sd->scb', self.cells_weights, coordinates) + self.cells_biases
        bar_coords = self.cells_act(bar_coords)                       # (samples, cells, baryCoords)
        
        temp = torch.einsum('scb,bj->scj', bar_coords, self.expand_weights) + self.expand_bias
        temp = self.relu2_act(temp)                                 # (samples, cells, 4*)
        
        phi = torch.einsum('scj,nj->scn', temp, self.merge_weights)
        
        # 2. 计算线段贡献和门控信号
        u = torch.einsum('scn,cn->sc', phi, self.cells_values)
        gate = torch.einsum('scn,nj->sc', phi, self.gate_weights)
        gate = self.gate_act(gate)                                  # (B, N)
        
        # 3. 通过relu2实现乘法运算
        ug_pair = torch.stack([u, gate], dim=-1)                    # (B, N, 2)
        unfold = torch.einsum('bnj,jk->bnk', ug_pair, self.multiply_weights)
        relu2 = self.relu2_act(unfold)                              # (B, N, 4)
        
        # # 4. 合并得到乘法结果并求和
        # U = torch.einsum('bnk,km->b', relu2, self.combined_weights) # (B,)   
        # return U

        U = torch.einsum('bn,bn->b', u, gate)               # (B,)  此做法无精度损失，上述做法最后的爱因斯坦求和有损
        return U

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n = 200  # 增加网格点数以获得更多三角形

    # 1. 生成三角形网格、节点函数值和插值器
    nodes, cells = generate_mesh(n)
    
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
    # 创建测试点，形状 (B, 1)
    test_points_np = np.random.rand(3000, 1).astype(np.float64)
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
