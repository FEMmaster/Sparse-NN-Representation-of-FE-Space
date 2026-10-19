import torch
import torch.nn as nn
import numpy as np
import time
from scipy.spatial import Delaunay
from matplotlib.tri import LinearTriInterpolator, Triangulation
from torch.profiler import profile, record_function, ProfilerActivity


def Piecewise_interpolator(nodes, cells, verbose=False):
    """
    创建三角形线性插值器
    
    Args:
        nodes: (N_nodes, 2) 节点坐标数组
        cells: (N_tri, 3) 三角形单元顶点索引
        verbose: 是否打印三角形方程信息
    
    Returns:
        interpolator: LinearTriInterpolator 对象
    """
    # 创建 matplotlib Triangulation 对象
    triangulation = Triangulation(nodes[:, 0], nodes[:, 1], triangles=cells)
    
    # 计算节点函数值
    point_values = target_function(nodes[:, 0], nodes[:, 1])
    
    # 创建 LinearTriInterpolator
    interpolator = LinearTriInterpolator(triangulation, point_values)
    
    # 根据verbose参数决定是否打印
    if verbose:
        for i, cells_indices in enumerate(triangulation.triangles):
            # 取出三角形顶点坐标
            x_tri = nodes[cells_indices, 0]
            y_tri = nodes[cells_indices, 1]
            f_tri = point_values[cells_indices]

            # 组装线性系统 A * [a,b,c] = f
            A = np.column_stack([x_tri, y_tri, np.ones(3)])
            coeff = np.linalg.solve(A, f_tri)

            a, b, c = coeff
            print(f"三角形 {i}: 顶点索引 {cells_indices} | U(x,y) = {a:.6f} * x + {b:.6f} * y + {c:.6f}")

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

def target_function(x, y,  eps=0.1):
    """目标函数：定义在计算域上的函数"""
    # return x + 2*y                    # f(x,y) = x + y
    # return x * y                   # f(x,y) = x*y
    # return x**2 + y**2             # f(x,y) = x^2 + y^2
    return np.sin(x) * np.cos(y)   # f(x,y) = sin(x)*cos(y)
    # return np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / eps**2)
    
def create_cells_nodeValues(nodes, cells):
   
    # 计算所有节点上的函数值
    point_values = target_function(nodes[:, 0], nodes[:, 1])
    
    # 创建节点函数值数组 (N_tri, 3)
    N_tri = len(cells)
    node_values = np.zeros((N_tri, 3), dtype=np.float64)
    
    # 将点上的函数值赋给对应的三角形顶点
    node_values = point_values[cells]
    
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

#%% 第二层操作用一个爱因斯坦求和搞定
class FiniteElementNet1(nn.Module):
    def __init__(self, cells_weights, cells_biases, cells_values):
        super().__init__()

        # 基函数权重
        self.register_buffer('cells_weights', cells_weights)
        self.register_buffer('cells_biases', cells_biases)
        self.cells_act = CustomRelu()
        
        # 门控
        self.register_buffer('cells_vg', torch.stack([cells_values, torch.ones(cells_values.shape[0], 3, dtype=torch.float64)], dim=-1))
        self.gate_act = HardGate()
        
        # 用于实现乘法运算u*g
        self.register_buffer('multi_weights', torch.tensor([[0.5, -0.5, 0.5, -0.5], [0.5, -0.5, -0.5, 0.5]], dtype=torch.float64))
        self.relu2_act = ReLU2()
        
        # 合并权重
        self.register_buffer('combi_weights', torch.tensor([1.0, 1.0, -1.0, -1.0], dtype=torch.float64))
 
    def forward(self, coordinates):
        
        # 1. 计算重心坐标基函数值
        bar_coords = torch.einsum('...c,njc->...nj', coordinates, self.cells_weights) + self.cells_biases
        bar_coords = self.cells_act(bar_coords)                       # (B, N, 3)
        
        # 2. 计算三角形贡献和门控信号
        ug = torch.einsum('...nj,njk->...nk', bar_coords, self.cells_vg) 
        gate = self.gate_act(ug[...,1])
        ug = torch.stack([ug[...,0], gate], dim=-1)
        
        # 3. 通过relu2实现乘法运算
        unfold = torch.einsum('...nj,jk->...nk', ug, self.multi_weights)
        unfold = self.relu2_act(unfold)                              # (B, N, 4)
        
        # 4. 合并得到乘法结果并求和
        # U = torch.einsum('...nk,k->...', unfold, self.combi_weights) # (B,)
        # return U
        
        U = torch.einsum('...n->...', torch.einsum('...nk,k->...n', unfold, self.combi_weights))      
        return U    # 此做法无精度损失，上述做法最后的爱因斯坦求和有损
        
        U = torch.einsum('...n,...n->...', ug[...,0], ug[...,1])               # (B,)  
        return U    # 直接相乘，无精度损失

# %% 第二层操作用两个爱因斯坦求和分别操作     
class FiniteElementNet2(nn.Module):
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
 
#%% 用matmul做矩阵乘法    
class FiniteElementNet3(nn.Module):
    def __init__(self, cells_weights, cells_biases, cells_values):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 基函数权重 - 在init中预处理所有维度
        self.cells_weights = cells_weights.unsqueeze(0)              # (1, N, 3, 2)
        self.cells_biases = cells_biases.unsqueeze(0).unsqueeze(-1)  # (1, N, 3, 1)
        self.cells_values = cells_values.unsqueeze(0).unsqueeze(2)  # (1, N, 1, 3)
        self.cells_act = CustomRelu()
        # self.cells_act =torch.relu
        
        # 门控
        self.cells_gates = torch.ones(1, 1, 1, 3, dtype=torch.float64, device=self.device)   # (1, 3)
        self.gate_act = HardGate()
        
    def forward(self, coordinates):

        # 1. 计算重心坐标基函数值
        coords_expanded = coordinates.unsqueeze(1)                   # (B, 1, 2, 1)
        bar_coords = self.cells_act(torch.matmul(self.cells_weights, coords_expanded) + self.cells_biases)   # (B, N, 3, 1)
        
        # 2. 计算三角形贡献和门控信号
        # u计算保持不变
        u = torch.matmul(self.cells_values, bar_coords)              # (1, N, 1, 3) @ (B, N, 3, 1) -> (B, N, 1, 1)
        u = u.squeeze(-1).squeeze(-1)                               # (B, N)
        
        # gate计算优化：避免转置
        # bar_coords: (B, N, 3, 1), cells_gates: (1, 1, 1, 3)
        # 使用逐元素乘法和求和
        gate = torch.matmul(self.cells_gates, bar_coords)  # (1, 1, 3, 1) @ (B, N, 3, 1) -> (B, N, 1, 1)
        gate = gate.squeeze(-1).squeeze(-1)                # (B, N)
        gate = self.gate_act(gate)                                   # (B, N)
        
        # 3. 最终计算
        U = torch.sum(u * gate, dim=1)                              # (B,)
        return U
        
#%% 主入口
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n = 50  # 增加网格点数以获得更多三角形

    # 1. 生成三角形网格、节点函数值和插值器
    nodes_np, cells_np = generate_mesh(n)

    nodes = torch.tensor(nodes_np, dtype=torch.float64, device=device)
    cells = torch.tensor(cells_np, dtype=torch.long, device=device)
    
    # 2. 计算仿射变换参数
    W_all, b_all = compute_affine_lambda_batch(nodes, cells)

    # 3. 创建三角形节点函数值
    node_values = create_cells_nodeValues(nodes, cells)
    
    # 4. 初始化模型
    Model_time = time.time()
    model = FiniteElementNet2(W_all, b_all, node_values).to(device)
    Model_time = time.time() - Model_time
    
    # 5. 测试输入
    # ============================================================
    # 第一类测试点：所有网格节点
    # ============================================================

    testpoints1 = nodes

    # ============================================================
    # 第二类测试点：所有不重复网格边的中点
    # ============================================================

    # 提取每个三角形的三条边
    edges = torch.cat(
        (
            cells[:, [0, 1]],
            cells[:, [1, 2]],
            cells[:, [2, 0]],
        ),
        dim=0,
    )

    # 将每条边的两个节点编号按升序排列，
    # 使共享边具有完全相同的表示
    edges = torch.sort(
        edges,
        dim=1,
    ).values

    # 删除相邻三角形产生的重复边
    edges = torch.unique(
        edges,
        dim=0,
    )

    # 计算每条边的中点
    testpoints2 = 0.5 * (
        nodes[edges[:, 0]]
        + nodes[edges[:, 1]]
    )


    # ============================================================
    # 第三类测试点：每个三角形内的三个高斯点
    # ============================================================

    # 三点高斯积分的重心坐标
    gauss_barycentric = nodes.new_tensor(
        [
            [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
        ]
    )

    # 每个三角形三个顶点的坐标，形状为
    # (三角形数量, 3, 2)
    triangle_nodes = nodes[cells]

    # 每个三角形生成三个高斯点，随后展开为
    # (3 × 三角形数量, 2)
    testpoints3 = torch.einsum(
        "gi,tic->tgc",
        gauss_barycentric,
        triangle_nodes,
    ).reshape(-1, 2)


    print("Number of node points:", testpoints1.shape[0])
    print("Number of edge-midpoint points:", testpoints2.shape[0])
    print("Number of Gaussian points:", testpoints3.shape[0])
    
    test_points_np = testpoints3
    test_points = torch.tensor(test_points_np, dtype=torch.float64).to(device)  # (B, 2)
    warm_points = torch.tensor([[0.50, 0.50], [0.60, 0.55]], dtype=torch.float64, device=device)
    
    # 6. 网络预热
    print("开始预热...")
    with torch.no_grad():
        for _ in range(5):
            _ = model(warm_points)
    
    # 7. 测试点评估
    # start_event = torch.cuda.Event(enable_timing=True)
    # end_event = torch.cuda.Event(enable_timing=True)
    # start_event.record()
    output = model(test_points.requires_grad_(True))
    # with torch.no_grad(): output = model(test_points)
    # end_event.record()
    # torch.cuda.synchronize()
    # model_time = start_event.elapsed_time(end_event)
    # print(f"模型计算时间: {model_time:.6f} 毫秒")
    
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    #     output = model(test_points)
    # print(prof.key_averages().table(sort_by="cuda_time_total"))

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
    print(f"平均差异: {np.mean(func_diff):.6e}")
    print(f"最大差异: {np.max(func_diff):.6e}")
    print(f"最小差异: {np.min(func_diff):.4e}")

    # 验证梯度差异
    grad = torch.autograd.grad(
        outputs=output, 
        inputs=test_points, 
        grad_outputs=torch.ones_like(output),
        retain_graph=False, 
        create_graph=False
    )[0]
    
    scipy_grad = interpolator.gradient(test_points_np[:, 0], test_points_np[:, 1])
    
    grad_np = grad.detach().cpu().numpy()  # 你的模型梯度 (n, 2)
    scipy_grad_np = np.column_stack([scipy_grad[0], scipy_grad[1]])  # 组合成 (n, 2)
    grad_diff = np.abs(grad_np - scipy_grad_np)

    print(f"\n=== 梯度差异验证 ===")
    print(f"梯度最大差异: {np.max(grad_diff):.6e}")
    print(f"梯度平均差异: {np.mean(grad_diff):.6e}")
    