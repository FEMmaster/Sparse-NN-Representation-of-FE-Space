import torch
import torch.nn as nn
import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import KDTree


def generate_mesh(n):
    """
    在单位立方体 [0,1]^3 上生成规则四面体网格。

    每个小立方体使用固定的 6-tet 剖分：
        [5,4,2,0]
        [5,1,0,2]
        [5,3,2,7]
        [5,6,7,2]
        [5,6,2,4]
        [5,3,1,2]

    该局部编号方式与下面的节点排列一致：
        0 = (i,   j,   k)
        1 = (i,   j,   k+1)
        2 = (i+1, j,   k)
        3 = (i+1, j,   k+1)
        4 = (i,   j+1, k)
        5 = (i,   j+1, k+1)
        6 = (i+1, j+1, k)
        7 = (i+1, j+1, k+1)

    参数
    ----
    n : int
        每个坐标方向的区间数。

    返回
    ----
    nodes : ndarray, shape ((n+1)^3, 3)
        节点坐标。

    cells : ndarray, shape (6*n^3, 4)
        四面体顶点编号。
    """
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n 必须是正整数")

    # --------------------------------------------------
    # 1. 生成节点
    # --------------------------------------------------
    x = np.linspace(0.0, 1.0, n + 1, dtype=np.float64)
    y = np.linspace(0.0, 1.0, n + 1, dtype=np.float64)
    z = np.linspace(0.0, 1.0, n + 1, dtype=np.float64)

    # 使用默认 indexing='xy'，保持与你原来的节点编号顺序一致
    xx, yy, zz = np.meshgrid(x, y, z)

    nodes = np.stack(
        [
            xx.reshape(-1),
            yy.reshape(-1),
            zz.reshape(-1),
        ],
        axis=1,
    )

    # --------------------------------------------------
    # 2. 节点编号函数
    # --------------------------------------------------
    #
    # 对于默认 np.meshgrid(x, y, z)，reshape 后：
    #
    #   z 最快变化
    #   x 次快变化
    #   y 最慢变化
    #
    # 因此全局节点编号为：
    #
    #   index(i,j,k) = j*(n+1)^2 + i*(n+1) + k
    #
    def node_index(i, j, k):
        return j * (n + 1) * (n + 1) + i * (n + 1) + k

    # 每个立方体 6 个四面体
    cells = np.empty((6 * n**3, 4), dtype=np.int64)

    cell_counter = 0

    # --------------------------------------------------
    # 3. 遍历所有小立方体
    # --------------------------------------------------
    for j in range(n):
        for i in range(n):
            for k in range(n):

                # 当前小立方体的8个顶点
                v0 = node_index(i,     j,     k)
                v1 = node_index(i,     j,     k + 1)
                v2 = node_index(i + 1, j,     k)
                v3 = node_index(i + 1, j,     k + 1)
                v4 = node_index(i,     j + 1, k)
                v5 = node_index(i,     j + 1, k + 1)
                v6 = node_index(i + 1, j + 1, k)
                v7 = node_index(i + 1, j + 1, k + 1)

                # 固定的6-tet剖分
                cells[cell_counter + 0] = [v5, v4, v2, v0]
                cells[cell_counter + 1] = [v5, v1, v0, v2]
                cells[cell_counter + 2] = [v5, v3, v2, v7]
                cells[cell_counter + 3] = [v5, v6, v7, v2]
                cells[cell_counter + 4] = [v5, v6, v2, v4]
                cells[cell_counter + 5] = [v5, v3, v1, v2]

                cell_counter += 6

    return nodes, cells

def compute_affine_lambda_batch(nodes, cells):
    """
    3D四面体网格的仿射变换参数计算
    """
    nodes_tensor = torch.tensor(nodes, dtype=torch.float64)
    
    # 提取四面体顶点 [n_cells, 4, 3]
    A = nodes_tensor[cells[:, 0]]  # 顶点0
    B = nodes_tensor[cells[:, 1]]  # 顶点1  
    C = nodes_tensor[cells[:, 2]]  # 顶点2
    D = nodes_tensor[cells[:, 3]]  # 顶点3
    
    # 构建变换矩阵 M = [B-A, C-A, D-A]^T
    M1 = B - A  # [n_cells, 3]
    M2 = C - A  # [n_cells, 3]
    M3 = D - A  # [n_cells, 3]
    
    # 计算行列式 (6倍体积)
    det = (M1[:,0] * (M2[:,1]*M3[:,2] - M2[:,2]*M3[:,1]) -
           M1[:,1] * (M2[:,0]*M3[:,2] - M2[:,2]*M3[:,0]) +
           M1[:,2] * (M2[:,0]*M3[:,1] - M2[:,1]*M3[:,0]))
    
    # 检查退化四面体
    if torch.any(torch.abs(det) < 1e-12):
        degenerate_count = torch.sum(torch.abs(det) < 1e-12)
        print(f"警告：发现 {degenerate_count} 个接近退化的四面体")
    
    inv_det = 1.0 / det
    
    # 构建权重张量 [n_cells, 4, 3]
    W_all = torch.zeros((len(cells), 4, 3), dtype=torch.float64)
    
    # 计算逆变换矩阵的系数
    W_all[:, 1, 0] = (M2[:,1]*M3[:,2] - M2[:,2]*M3[:,1]) * inv_det  # ∂λ₁/∂x
    W_all[:, 1, 1] = (M2[:,2]*M3[:,0] - M2[:,0]*M3[:,2]) * inv_det  # ∂λ₁/∂y  
    W_all[:, 1, 2] = (M2[:,0]*M3[:,1] - M2[:,1]*M3[:,0]) * inv_det  # ∂λ₁/∂z
    
    W_all[:, 2, 0] = (M1[:,2]*M3[:,1] - M1[:,1]*M3[:,2]) * inv_det  # ∂λ₂/∂x
    W_all[:, 2, 1] = (M1[:,0]*M3[:,2] - M1[:,2]*M3[:,0]) * inv_det  # ∂λ₂/∂y
    W_all[:, 2, 2] = (M1[:,1]*M3[:,0] - M1[:,0]*M3[:,1]) * inv_det  # ∂λ₂/∂z
    
    W_all[:, 3, 0] = (M1[:,1]*M2[:,2] - M1[:,2]*M2[:,1]) * inv_det  # ∂λ₃/∂x
    W_all[:, 3, 1] = (M1[:,2]*M2[:,0] - M1[:,0]*M2[:,2]) * inv_det  # ∂λ₃/∂y
    W_all[:, 3, 2] = (M1[:,0]*M2[:,1] - M1[:,1]*M2[:,0]) * inv_det  # ∂λ₃/∂z
    
    # λ₀ = 1 - λ₁ - λ₂ - λ₃
    W_all[:, 0, :] = -W_all[:, 1, :] - W_all[:, 2, :] - W_all[:, 3, :]
    
    # 构建偏置项 [n_cells, 4]
    b_all = torch.zeros((len(cells), 4), dtype=torch.float64)
    
    # bᵢ = -Wᵢ·A
    b_all[:, 1] = -(W_all[:, 1, 0]*A[:,0] + W_all[:, 1, 1]*A[:,1] + W_all[:, 1, 2]*A[:,2])
    b_all[:, 2] = -(W_all[:, 2, 0]*A[:,0] + W_all[:, 2, 1]*A[:,1] + W_all[:, 2, 2]*A[:,2])
    b_all[:, 3] = -(W_all[:, 3, 0]*A[:,0] + W_all[:, 3, 1]*A[:,1] + W_all[:, 3, 2]*A[:,2])
    b_all[:, 0] = 1.0 - b_all[:, 1] - b_all[:, 2] - b_all[:, 3]
    
    return W_all, b_all

def target_function_3d(x, y, z):
    """3D目标函数：定义在计算域上的函数"""
    # return x + 2*y + 3*z               # f(x,y,z) = x + 2y + 3z
    # return x * y * z                   # f(x,y,z) = x*y*z  
    # return x**2 + y**2 + z**2         # f(x,y,z) = x^2 + y^2 + z^2
    return np.sin(x) * np.cos(y) * np.exp(z)  # f(x,y,z) = sin(x)*cos(y)*exp(z)

def create_cells_nodeValues(nodes, cells):
    """创建3D情况下的节点函数值"""
    # 计算所有节点上的函数值
    point_values = target_function_3d(nodes[:, 0], nodes[:, 1], nodes[:, 2])
    
    # 创建节点函数值数组 (N_tet, 4)
    N_tet = len(cells)
    node_values = np.zeros((N_tet, 4), dtype=np.float64)
    
    # 将点上的函数值赋给对应的四面体顶点
    node_values = point_values[cells]
    
    return torch.tensor(node_values, dtype=torch.float64)

def Piecewise_interpolator(nodes, cells, points):
    """
    在给定四面体网格上进行三维分片线性插值。

    Parameters
    ----------
    nodes : (NN, 3)
        节点坐标。
    cells : (NC, 4)
        四面体节点编号。
    points : (NP, 3)
        待插值点。
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.int64)
    points = np.asarray(points, dtype=np.float64)

    NC = cells.shape[0]

    # 节点函数值
    node_values = target_function_3d(
        nodes[:, 0], nodes[:, 1], nodes[:, 2]
    )

    # ---------- 构造四面体邻接关系 ----------
    localFace = np.array([
        [1, 2, 3],
        [0, 2, 3],
        [0, 1, 3],
        [0, 1, 2],
    ])

    totalFace = cells[:, localFace].reshape(-1, 3)

    _, i0, j = np.unique(
        np.sort(totalFace, axis=1),
        return_index=True,
        return_inverse=True,
        axis=0,
    )

    i1 = np.zeros(len(i0), dtype=np.int64)
    i1[j] = np.arange(4 * NC)

    face2cell = np.column_stack([
        i0 // 4,
        i1 // 4,
        i0 % 4,
        i1 % 4,
    ])

    cell2cell = np.zeros((NC, 4), dtype=np.int64)

    cell2cell[
        face2cell[:, 0], face2cell[:, 2]
    ] = face2cell[:, 1]

    cell2cell[
        face2cell[:, 1], face2cell[:, 3]
    ] = face2cell[:, 0]

    # ---------- 每个节点选择一个关联单元 ----------
    start_cell = np.zeros(len(nodes), dtype=np.int64)

    for i in range(4):
        start_cell[cells[:, i]] = np.arange(NC)

    # ---------- KDTree 给出初始单元 ----------
    tree = KDTree(nodes)
    _, nearest_nodes = tree.query(points)
    start = start_cell[nearest_nodes].copy()

    NP = points.shape[0]
    barycentric = np.zeros((NP, 4))
    isNotOK = np.ones(NP, dtype=bool)

    # ---------- 沿邻接四面体寻找查询点 ----------
    while np.any(isNotOK):
        idx0 = np.where(isNotOK)[0]
        idx = start[idx0]

        tet = nodes[cells[idx]]

        J = np.stack([
            tet[:, 1] - tet[:, 0],
            tet[:, 2] - tet[:, 0],
            tet[:, 3] - tet[:, 0],
        ], axis=2)

        b123 = np.linalg.solve(
            J,
            points[idx0] - tet[:, 0],
        )

        b = np.column_stack([
            1.0 - b123.sum(axis=1),
            b123,
        ])

        barycentric[idx0] = b
        isOutCell = b.min(axis=1) < -1.0e-10

        if np.any(isOutCell):
            old_cell = idx[isOutCell]
            local_face = b[isOutCell].argmin(axis=1)
            new_cell = cell2cell[old_cell, local_face]

            if np.any(new_cell == old_cell):
                raise ValueError("存在位于网格区域外的查询点")

            start[idx0[isOutCell]] = new_cell

        isNotOK[idx0] = isOutCell

    # ---------- 找到单元后直接插值 ----------
    cell_nodes = cells[start]
    node_vals = node_values[cell_nodes]

    return np.einsum("ni,ni...->n...", barycentric, node_vals)

class HardGateFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        eps = 1e-10
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

        # 基函数权重 - 修改为3D: (N, 4, 3)
        self.cells_weights = cells_weights                              # (N, 4, 3)
        self.cells_biases = cells_biases                                # (N, 4)
        self.cells_values = cells_values                                # (N, 4)
        self.cells_act = CustomRelu()
        
        # 门控 - 修改为3D: 4个基函数
        self.gate_weights = torch.ones(4, 1, dtype=torch.float64)   # (4, 1)
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
        # coordinates: (batch_size, 3) - 3D输入坐标点
        
        # 1. 计算重心坐标基函数值
        bar_coords = torch.einsum('njc,bc->bnj', self.cells_weights, coordinates) + self.cells_biases
        bar_coords = self.cells_act(bar_coords)                       # (B, N, 4)
        
        # 2. 计算四面体贡献和门控信号
        u = torch.einsum('bnj,nj->bn', bar_coords, self.cells_values)
        gate = torch.einsum('bnj,jk->bn', bar_coords, self.gate_weights)
        gate = self.gate_act(gate)                                  # (B, N)
        
        # 3. 通过relu2实现乘法运算
        ug_pair = torch.stack([u, gate], dim=-1)                    # (B, N, 2)
        unfold = torch.einsum('bnj,jk->bnk', ug_pair, self.multiply_weights)
        relu2 = self.relu2_act(unfold)                              # (B, N, 8)
        
        # 4. 合并得到乘法结果并求和
        # U = torch.einsum('bnk,km->b', relu2, self.combined_weights) # (B,)
        # return U
        
        U = torch.einsum('bn,bn->b', u, gate)               # (B,)  此做法无精度损失，上述做法最后的爱因斯坦求和有损
        return U

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    n = 10  # 增加网格点数以获得更多三角形

    # 1. 生成三角形网格、节点函数值和插值器
    nodes_np, cells_np = generate_mesh(n)

    # 转换为 PyTorch Tensorg
    nodes = torch.as_tensor(
        nodes_np,
        dtype=torch.float64,
        device=device,
    )

    cells = torch.as_tensor(
        cells_np,
        dtype=torch.long,
        device=device,
    )

    # 计算仿射变换参数
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
    # testpoints2：所有不重复边的中点
    # ============================================================

    # 每个四面体共有 6 条边
    edges = torch.cat(
        (
            cells[:, [0, 1]],
            cells[:, [0, 2]],
            cells[:, [0, 3]],
            cells[:, [1, 2]],
            cells[:, [1, 3]],
            cells[:, [2, 3]],
        ),
        dim=0,
    )

    # 对每条边的两个节点编号排序
    edges = torch.sort(
        edges,
        dim=1,
    ).values

    # 删除相邻四面体产生的重复边
    edges = torch.unique(
        edges,
        dim=0,
    )

    # 计算边中点
    testpoints2 = 0.5 * (
        nodes[edges[:, 0]]
        + nodes[edges[:, 1]]
    )


    # ============================================================
    # testpoints3：每个四面体内的 4 个 Gauss 点
    # ============================================================

    # 四面体二阶 4 点 Gauss 公式的重心坐标
    a = 0.5854101966249685
    b = 0.1381966011250105

    gauss_barycentric = nodes.new_tensor(
        [
            [a, b, b, b],
            [b, a, b, b],
            [b, b, a, b],
            [b, b, b, a],
        ]
    )

    # 每个四面体的 4 个顶点坐标
    # 形状：(四面体数量, 4, 3)
    tetrahedron_nodes = nodes[cells]

    # 重心坐标转换到实际四面体
    # 结果形状：(四面体数量, 4, 3)
    gauss_points = torch.einsum(
        "gi,tic->tgc",
        gauss_barycentric,
        tetrahedron_nodes,
    )

    # 展开为：(4 × 四面体数量, 3)
    testpoints3 = gauss_points.reshape(-1, 3)


    print("testpoints1:", testpoints1.shape)
    print("testpoints2:", testpoints2.shape)
    print("testpoints3:", testpoints3.shape)
        
    test_points_np = testpoints3
    test_points = torch.tensor(test_points_np, dtype=torch.float64)  # (B, 3)
    
    # 7. 验证梯度
    start_time = time.time()
    test_points.requires_grad_(True)
    output = model(test_points)
    model_time = time.time() - start_time 
    print(f"模型计算时间: {model_time:.6f} 秒")
    # print(f"输出值: {output.squeeze()}")

    # # 计算导数
    # grad = torch.autograd.grad(
    #     outputs=output, 
    #     inputs=test_points, 
    #     grad_outputs=torch.ones_like(output),
    #     retain_graph=False, 
    #     create_graph=False
    # )[0]

    # print(grad.T)
    
    #############################################################################################
    #############################################################################################
    #############################################################################################
          
    # 使用 scipy 插值器计算（用于验证）
    scipy_output = Piecewise_interpolator(nodes, cells, test_points_np)

    # 验证函数值差异
    output_np = output.detach().cpu().numpy()
    func_diff = np.abs(output_np - scipy_output)

    print(f"\n=== 函数值差异验证 ===")
    print(f"最大差异: {np.max(func_diff):.6e}")
    print(f"平均差异: {np.mean(func_diff):.6e}")