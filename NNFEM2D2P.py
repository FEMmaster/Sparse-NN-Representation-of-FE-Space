import torch
import torch.nn as nn
import numpy as np
import time
from scipy.spatial import Delaunay
from torch.profiler import profile, record_function, ProfilerActivity
import matplotlib.tri as mtri

def Piecewise_interpolator(nodes, cells, verbose=False):
    """
    在每个三角形上，用三个顶点和三个边中点拟合二次多项式

        P(x, y) = a0 + a1*x + a2*y + a3*x^2 + a4*x*y + a5*y^2

    调用方式：
        interpolator = Piecewise_interpolator(nodes, cells)
        values = interpolator(x, y)
    """

    cell_nodes = nodes[cells]
    v0, v1, v2 = cell_nodes[:, 0], cell_nodes[:, 1], cell_nodes[:, 2]

    # 六个插值点：[v0, e01, e02, v1, e12, v2]
    cell_points = np.stack([
        v0,
        0.5 * (v0 + v1),
        0.5 * (v0 + v2),
        v1,
        0.5 * (v1 + v2),
        v2
    ], axis=1)

    px, py = cell_points[..., 0], cell_points[..., 1]
    cell_values = target_function(px, py)

    # 每个单元的 Vandermonde 矩阵
    V = np.stack([
        np.ones_like(px),
        px,
        py,
        px**2,
        px * py,
        py**2
    ], axis=-1)

    # 每个单元上的二次多项式系数
    coefficients = np.linalg.solve(V, cell_values[..., None])[..., 0]

    if verbose:
        for c, a in enumerate(coefficients):
            print(
                f"单元 {c}: P(x,y) = {a[0]:.12e} + ({a[1]:.12e})x "
                f"+ ({a[2]:.12e})y + ({a[3]:.12e})x² "
                f"+ ({a[4]:.12e})xy + ({a[5]:.12e})y²"
            )

    triangle_finder = mtri.Triangulation(
        nodes[:, 0], nodes[:, 1], triangles=cells
    ).get_trifinder()

    def interpolator(x, y):
        x, y = np.broadcast_arrays(
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64)
        )

        original_shape = x.shape
        x, y = x.reshape(-1), y.reshape(-1)
        cell_id = np.asarray(triangle_finder(x, y), dtype=np.int64)

        values = np.full(x.shape, np.nan, dtype=np.float64)
        inside = cell_id >= 0

        xi, yi = x[inside], y[inside]
        monomials = np.stack([
            np.ones_like(xi),
            xi,
            yi,
            xi**2,
            xi * yi,
            yi**2
        ], axis=-1)

        values[inside] = np.einsum(
            'ij,ij->i',
            monomials,
            coefficients[cell_id[inside]]
        )

        return values.reshape(original_shape)

    return interpolator

def generate_mesh(n):

    # 在单位正方形内生成均匀网格点
    x = np.linspace(0, 1, n+1, dtype=np.float64)
    y = np.linspace(0, 1, n+1, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    nodes = np.stack([xx.ravel(), yy.ravel()], axis=1)

    # 进行Delaunay三角剖分
    tet = Delaunay(nodes)

    return nodes, tet.simplices

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

def target_function(x, y):
    """目标函数：定义在计算域上的函数"""
    # return x + 2*y                    # f(x,y) = x + y
    # return x * y                   # f(x,y) = x*y
    # return x**2 + y**2             # f(x,y) = x^2 + y^2
    return np.sin(x) * np.cos(y)   # f(x,y) = sin(x)*cos(y)
    
def create_cells_nodeValues(nodes, cells):
    """
    创建二维三角形 P2 单元的局部自由度值。

    输入保持不变：
        nodes: (N_node, 2)
        cells: (N_cell, 3)

    输出：
        node_values: (N_cell, 6)

    局部顺序与当前 P2 基函数一致：
        [v0, e01, e02, v1, e12, v2]
    """

    # 每个三角形的三个顶点坐标
    # shape: (N_cell, 3, 2)
    cell_nodes = nodes[cells]

    v0 = cell_nodes[:, 0, :]
    v1 = cell_nodes[:, 1, :]
    v2 = cell_nodes[:, 2, :]

    # 三条边的中点
    e01 = 0.5 * (v0 + v1)
    e02 = 0.5 * (v0 + v2)
    e12 = 0.5 * (v1 + v2)

    # 按照基函数对应顺序排列：
    # [v0, e01, e02, v1, e12, v2]
    cell_points = np.stack(
        [
            v0,
            e01,
            e02,
            v1,
            e12,
            v2,
        ],
        axis=1,
    )
    # shape: (N_cell, 6, 2)

    # 在六个 P2 插值点上计算函数值
    node_values = target_function(
        cell_points[:, :, 0],
        cell_points[:, :, 1],
    )
    # shape: (N_cell, 6)

    return torch.tensor(node_values, dtype=torch.float64)
       
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


# %% 第二层操作用两个爱因斯坦求和分别操作     
class FiniteElementNet2(nn.Module):
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
        bar_coords = self.cells_act(bar_coords)                       # (B, N, 3)
        
        l1, l2, l3 = bar_coords[..., 0], bar_coords[..., 1], bar_coords[..., 2]
        
        # 顶点基函数
        N1 = l1 * (2*l1 - 1)   # λ1(2λ1-1)
        N2 = l2 * (2*l2 - 1)   # λ2(2λ2-1)
        N3 = l3 * (2*l3 - 1)   # λ3(2λ3-1)
        
        # 边中点基函数
        N4 = 4 * l1 * l2        # 边1-2中点
        N5 = 4 * l2 * l3        # 边2-3中点
        N6 = 4 * l3 * l1        # 边3-1中点
        phi = torch.stack([N1, N4, N6, N2, N5, N3], dim=-1)
        
        # 2. 计算线段贡献和门控信号     
        u = torch.einsum('scn,cn->sc', phi, self.cells_values)
        gate = torch.einsum('scn,nj->sc', phi, self.cells_gates)
        gate = self.gate_act(gate)                                  # (B, N)
        
        # 3. 通过relu2实现乘法运算
        ug_pair = torch.stack([u, gate], dim=-1)                    # (B, N, 2)
        unfold = torch.einsum('bnj,jk->bnk', ug_pair, self.multi_weights)
        relu2 = self.relu2_act(unfold)                              # (B, N, 4)
        
        # # 4. 合并得到乘法结果并求和
        # U = torch.einsum('bnk,km->b', relu2, self.combined_weights) # (B,)   
        # return U

        U = torch.einsum('bn,bn->b', u, gate)               # (B,)  此做法无精度损失，上述做法最后的爱因斯坦求和有损
        return U
 
        
#%% 主入口
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n = 30  # 增加网格点数以获得更多三角形

    # 1. 生成三角形网格、节点函数值和插值器
    nodes, cells = generate_mesh(n)
    
    # nodes = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    
    # cells = np.array([[3, 0, 1]], dtype=np.int32)
 
    # 2. 计算仿射变换参数
    W_all, b_all = compute_affine_lambda_batch(nodes, cells)

    # 3. 创建三角形节点函数值
    node_values = create_cells_nodeValues(nodes, cells)
    
    # 4. 初始化模型
    Model_time = time.time()
    model = FiniteElementNet2(W_all, b_all, node_values).to(device)
    Model_time = time.time() - Model_time
    
    # 5. 测试输入
    test_points_np = np.random.rand(500, 2).astype(np.float64)
    # test_points_np = torch.tensor([[0.50, 0.50], [1.0, 1.0]], dtype=torch.float64, device=device)
    test_points = torch.tensor(test_points_np, dtype=torch.float64).to(device)  # (B, 2)
    warm_points = torch.tensor([[0.50, 0.50], [0.20, 0.35]], dtype=torch.float64, device=device)
    
    # 6. 网络预热
    print("开始预热...")
    with torch.no_grad():
        for _ in range(5):
            _ = model(warm_points)
    
    # 7. 测试点评估
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    output = model(test_points.requires_grad_(True))
    with torch.no_grad(): output = model(test_points)
    end_event.record()
    torch.cuda.synchronize()
    model_time = start_event.elapsed_time(end_event)
    print(f"模型计算时间: {model_time:.6f} 毫秒")
    
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        output = model(test_points)
    print(prof.key_averages().table(sort_by="cuda_time_total"))

    # 对比 scipy 插值
    interpolator = Piecewise_interpolator(nodes, cells)
    start_time = time.time()
    scipy_output = interpolator(test_points_np[:, 0], test_points_np[:, 1])
    scipy_time = time.time() - start_time 
    print(f"scipy计算时间: {scipy_time*1000:.6f} 毫秒")

    # 验证函数值差异
    output_np = output.detach().cpu().numpy()
    func_diff = np.abs(output_np - scipy_output)

    print(f"\n=== 函数值差异验证 ===")
    print(f"平均差异: {np.mean(func_diff):.4e}")
    print(f"最大差异: {np.max(func_diff):.4e}")

    # 验证梯度差异
    if not test_points.requires_grad:
        test_points.requires_grad_(True)
    