import numpy as np
import gmsh
import meshio
from fealpy.mesh import TriangleMesh, TetrahedronMesh

def generate_mesh_ex1(mesh, eta, flag=1):
    # Precompute frequently used values
    cells = mesh.entity('cell')
    node2cell = mesh.ds.node_to_cell()
    volume = mesh.cell_area()
    
    # Compute average edge lengths for each cell
    edge_lengths = np.sqrt((4.0 * volume) / np.sqrt(3))  # 根据面积公式求平均边长, NC*1
    # edge_lengths2 = mesh.edge_length()                    # 精确的每个边的长度, NE*1
    
    # Cache matrix-vector products
    node2cell_sum = node2cell @ np.ones_like(eta)
    edge_lengths_sum = node2cell @ edge_lengths
    
    # Compute node-based error and edge length metrics
    node_error = (node2cell @ eta) / node2cell_sum
    node_edge_length = edge_lengths_sum / node2cell_sum
    rho = node_error / node_edge_length
    
    # Select top error nodes covering 99% of total error
    sorted_indices = np.argsort(-rho)  # Descending order
    sorted_rho = rho[sorted_indices]
    threshold = 0.9 * np.sum(rho**2)
    N = np.searchsorted(np.cumsum(sorted_rho**2), threshold, side='left') + 1
    
    print(f"Marked {N} points for refinement ({(N/len(rho))*100:.1f}% of nodes)")
    
    # Compute refinement factor
    NN = mesh.number_of_nodes()
    value = np.sqrt(0.5) ** (np.log(NN/N + 1) / np.log(2.0))

    labels = np.full_like(rho, 1.0)
    # labels[sorted_indices[:N]] = value**flag
    labels[sorted_indices[:N]] = max(value**flag, 0.5)
    new_size = node_edge_length * labels

    # 初始化 gmsh
    gmsh.initialize()
    gmsh.model.add("gntest2")
    
    # 禁用gmsh的终端输出
    gmsh.option.setNumber("General.NumThreads", 4)
    gmsh.option.setNumber("General.Terminal", 0)    
        
    gmsh.model.occ.addRectangle(0, 0, 0, 2.2, 0.41, 1)
    gmsh.model.occ.addDisk(0.2, 0.2, 0, 0.05, 0.05, 2)
    gmsh.model.occ.cut([(2, 1)], [(2, 2)], 3)
    gmsh.model.occ.synchronize()
    
    # # 设置优化参数
    # gmsh.option.setNumber("Mesh.Algorithm", 6)
    # gmsh.option.setNumber("Mesh.Optimize", 1)
    
    # gmsh.model.mesh.setSizeCallback(size_field)
    # gmsh.model.mesh.generate(2)
      
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
    node = mesh.entity('node')
    # === 写入 Gmsh .pos 文件 ===
    pos_filename = "background_field.pos"
    with open("background_field.pos", "w") as f:
        f.write('View "meshSize" {\n')
        for (x, y), s in zip(node, new_size):
            f.write(f'SP({x}, {y}, 0){{{s}}};\n')
        f.write("};\n")
    
    gmsh.merge(pos_filename)  # 正确加载 .pos 文件作为 PostView 视图
    gmsh.model.mesh.field.add("PostView", 1)
    gmsh.model.mesh.field.setNumber(1, "ViewIndex", 0)  # 第一个被加载的视图，索引为 0
    gmsh.model.mesh.field.setAsBackgroundMesh(1)

    gmsh.model.mesh.generate(2)
        
    # Write and read mesh
    gmsh.write("temp_mesh.vtk")
    gmsh.finalize()   

    # 使用 meshio 读取生成的 .msh 文件
    mesh = meshio.read("temp_mesh.vtk")
    node = mesh.points[:, :2]
    cell = mesh.cells_dict["triangle"].astype(np.int64)
    return TriangleMesh(node,cell)  

def generate_mesh_ex2(mesh, eta):
    # Precompute frequently used values
    cells = mesh.entity('cell')
    node2cell = mesh.ds.node_to_cell()
    volume = mesh.cell_area()
    
    # Compute average edge lengths for each cell
    edge_lengths = np.sqrt((4.0 * volume) / np.sqrt(3))  # 根据面积公式求平均边长, NC*1
    # edge_lengths2 = mesh.edge_length()                    # 精确的每个边的长度, NE*1
    
    # Cache matrix-vector products
    node2cell_sum = node2cell @ np.ones_like(eta)
    edge_lengths_sum = node2cell @ edge_lengths
    
    # Compute node-based error and edge length metrics
    node_error = (node2cell @ eta) / node2cell_sum
    node_edge_length = edge_lengths_sum / node2cell_sum
    rho = node_error / node_edge_length
    
    # Select top error nodes covering 99% of total error
    sorted_indices = np.argsort(-rho)  # Descending order
    sorted_rho = rho[sorted_indices]
    threshold = 0.9 * np.sum(rho**2)
    N = np.searchsorted(np.cumsum(sorted_rho**2), threshold, side='right')
    
    print(f"    Marked {N} points for refinement ({(N/len(rho))*100:.1f}% of nodes)")
    
    # Compute refinement factor
    NN = mesh.number_of_nodes()
    value = (np.sqrt(0.5)) ** (np.log(NN / max(N, 1) + 1) / np.log(2.0))
    
    # Create size field for refinement
    labels = np.ones_like(rho) * 1.1
    labels[sorted_indices[:N]] = 3.0 * value / 4.0
    new_size = node_edge_length * labels 

    # 初始化 gmsh
    gmsh.initialize()
    gmsh.model.add("gntest2")
    
    # 禁用gmsh的终端输出
    gmsh.option.setNumber("General.NumThreads", 4)
    gmsh.option.setNumber("General.Terminal", 0)    
        
    gmsh.model.occ.addRectangle(0, 0, 0, 1.0, 1.0, 1)
    gmsh.model.occ.synchronize()
    
    # # 设置优化参数
    # gmsh.option.setNumber("Mesh.Algorithm", 6)
    # gmsh.option.setNumber("Mesh.Optimize", 1)
    
    # gmsh.model.mesh.setSizeCallback(size_field)
    # gmsh.model.mesh.generate(2)
      
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
    node = mesh.entity('node')
    # === 写入 Gmsh .pos 文件 ===
    pos_filename = "background_field.pos"
    with open("background_field.pos", "w") as f:
        f.write('View "meshSize" {\n')
        for (x, y), s in zip(node, new_size):
            f.write(f'SP({x}, {y}, 0){{{s}}};\n')
        f.write("};\n")
    
    gmsh.merge(pos_filename)  # 正确加载 .pos 文件作为 PostView 视图
    gmsh.model.mesh.field.add("PostView", 1)
    gmsh.model.mesh.field.setNumber(1, "ViewIndex", 0)  # 第一个被加载的视图，索引为 0
    gmsh.model.mesh.field.setAsBackgroundMesh(1)

    gmsh.model.mesh.generate(2)
        
    # Write and read mesh
    gmsh.write("temp_mesh.vtk")
    gmsh.finalize()   

    # 使用 meshio 读取生成的 .msh 文件
    mesh = meshio.read("temp_mesh.vtk")
    node = mesh.points[:, :2]
    cell = mesh.cells_dict["triangle"].astype(np.int64)
    return TriangleMesh(node,cell)

def generate_mesh_heat_3D(mesh, eta, flag=1):
    # 构建point2cell矩阵
    NC = mesh.number_of_cells()
    NN = mesh.number_of_nodes()

    cells = mesh.entity('cell')
    node2cell = mesh.ds.node_to_cell()
    volume = mesh.cell_volume()
    edge_lengths1 = np.cbrt(6 * np.sqrt(2) * volume)  # 根据面积公式求平均边长, NC*1
    # edge_lengths2 = mesh.edge_length()                    # 精确的每个边的长度, NE*1

    # 根据单元误差 eta 转化为节点误差
    nodeError = (node2cell @ eta) / (node2cell @ np.ones(NC))
    nodeEdgeLength = (node2cell @ edge_lengths1) / (node2cell @ np.ones(NC))
    rho = nodeError / nodeEdgeLength**1.5
    sorted_indices = np.argsort(-rho)  # -rho 表示降序排序
    sorted_rho = rho[sorted_indices]

    threshold = 0.99 * np.sum(rho ** 2)
    cumulative_error_squared = np.cumsum(sorted_rho ** 2)
    N = np.searchsorted(cumulative_error_squared, threshold, side='right')
    # print(f"      Marked {N} points that will be refined.")
    value = (np.cbrt(0.5)) ** (np.log(NN / N + 1) / np.log(2.0))

    labels = np.full_like(rho, 1.0)
    labels[sorted_indices[:N]] = value ** flag
    new_size = nodeEdgeLength * labels
    
    gmsh.initialize()
    gmsh.model.add("gntest3D")
    
    # 禁用gmsh的终端输出
    gmsh.option.setNumber("General.NumThreads", 4)
    gmsh.option.setNumber("General.Terminal", 0)    
        
    # 创建几何模型
    gmsh.model.occ.addBox(-1.0, -1.0, -1.0, 2.0, 2.0, 2.0, 1)
    # gmsh.model.occ.addBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1)
    gmsh.model.occ.synchronize()
    
    # 设置网格生成选项
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
    node = mesh.entity('node')
    # 写入 Gmsh .pos 文件
    pos_filename = "background_field.pos"
    with open(pos_filename, "w") as f:
        f.write('View "meshSize" {\n')
        for (x, y, z), s in zip(node, new_size):
            f.write(f'SP({x}, {y}, {z}){{{s}}};\n')
        f.write("};\n")
    
    # 加载背景网格尺寸场
    gmsh.merge(pos_filename)
    gmsh.model.mesh.field.add("PostView", 1)
    gmsh.model.mesh.field.setNumber(1, "ViewIndex", 0)
    gmsh.model.mesh.field.setAsBackgroundMesh(1)

    # 生成网格
    gmsh.model.mesh.generate(3)
    
    # 写入临时文件
    gmsh.write("temp_mesh.vtk")
    gmsh.finalize()   

    # 使用 meshio 读取生成的网格
    mesh = meshio.read("temp_mesh.vtk")
    node = mesh.points
    cell = mesh.cells_dict["tetra"].astype(np.int64)
    return TetrahedronMesh(node, cell)  


def generate_mesh_heat_lshape(mesh, eta, flag=1):
    # 构建point2cell矩阵
    NC = mesh.number_of_cells()
    NN = mesh.number_of_nodes()

    cells = mesh.entity('cell')
    node2cell = mesh.ds.node_to_cell()
    volume = mesh.cell_area()
    edge_lengths1 = np.sqrt((4.0 * volume) / np.sqrt(3))  # 根据面积公式求平均边长, NC*1
    # edge_lengths2 = mesh.edge_length()                    # 精确的每个边的长度, NE*1

    # 根据单元误差 eta 转化为节点误差
    nodeError = (node2cell @ eta) / (node2cell @ np.ones(NC))
    nodeEdgeLength = (node2cell @ edge_lengths1) / (node2cell @ np.ones(NC))
    rho = nodeError / nodeEdgeLength
    sorted_indices = np.argsort(-rho)  # -rho 表示降序排序
    sorted_rho = rho[sorted_indices]

    threshold = 0.99 * np.sum(rho ** 2)
    cumulative_error_squared = np.cumsum(sorted_rho ** 2)
    N = np.searchsorted(cumulative_error_squared, threshold, side='right')
    # print(f"      Marked {N} points that will be refined.")
    value = (np.sqrt(0.5)) ** (np.log(NN / N + 1) / np.log(2.0))

    labels = np.full_like(rho, 1.0)
    labels[sorted_indices[:N]] = value ** flag
    new_size = nodeEdgeLength * labels
    
    gmsh.initialize()
    gmsh.model.add("gntest2")
    
    # 禁用gmsh的终端输出
    gmsh.option.setNumber("General.NumThreads", 4)
    gmsh.option.setNumber("General.Terminal", 0)    
        
    # 创建几何模型：L-shaped domain
    # 区域为 [-1, 1]^2 去掉右下角 [0, 1] x [-1, 0]
    # 边界点顺序：
    # (-1,-1) -> (0,-1) -> (0,0) -> (1,0) -> (1,1) -> (-1,1)
    p0 = gmsh.model.occ.addPoint(-1.0, -1.0, 0.0)
    p1 = gmsh.model.occ.addPoint( 0.0, -1.0, 0.0)
    p2 = gmsh.model.occ.addPoint( 0.0,  0.0, 0.0)
    p3 = gmsh.model.occ.addPoint( 1.0,  0.0, 0.0)
    p4 = gmsh.model.occ.addPoint( 1.0,  1.0, 0.0)
    p5 = gmsh.model.occ.addPoint(-1.0,  1.0, 0.0)

    l0 = gmsh.model.occ.addLine(p0, p1)
    l1 = gmsh.model.occ.addLine(p1, p2)
    l2 = gmsh.model.occ.addLine(p2, p3)
    l3 = gmsh.model.occ.addLine(p3, p4)
    l4 = gmsh.model.occ.addLine(p4, p5)
    l5 = gmsh.model.occ.addLine(p5, p0)

    cl = gmsh.model.occ.addCurveLoop([l0, l1, l2, l3, l4, l5])
    sf = gmsh.model.occ.addPlaneSurface([cl])

    gmsh.model.occ.synchronize()
    
    # 设置网格生成选项
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)

    node = mesh.entity('node')

    # 写入 Gmsh .pos 文件
    pos_filename = "background_field.pos"
    with open(pos_filename, "w") as f:
        f.write('View "meshSize" {\n')
        for (x, y), s in zip(node, new_size):
            f.write(f'SP({x}, {y}, 0){{{s}}};\n')
        f.write("};\n")
    
    # 加载背景网格尺寸场
    gmsh.merge(pos_filename)
    gmsh.model.mesh.field.add("PostView", 1)
    gmsh.model.mesh.field.setNumber(1, "ViewIndex", 0)
    gmsh.model.mesh.field.setAsBackgroundMesh(1)

    # 生成网格
    gmsh.model.mesh.generate(2)
    
    # 写入临时文件
    gmsh.write("temp_mesh.vtk")
    gmsh.finalize()   

    # 使用 meshio 读取生成的网格
    mesh = meshio.read("temp_mesh.vtk")
    node = mesh.points[:, :2]
    cell = mesh.cells_dict["triangle"].astype(np.int64)

    return TriangleMesh(node, cell)


def generate_mesh_heat(mesh, eta, flag=1):
    # 构建point2cell矩阵
    NC = mesh.number_of_cells()
    NN = mesh.number_of_nodes()

    cells = mesh.entity('cell')
    node2cell = mesh.ds.node_to_cell()
    volume = mesh.cell_area()
    edge_lengths1 = np.sqrt((4.0 * volume) / np.sqrt(3))  # 根据面积公式求平均边长, NC*1
    # edge_lengths2 = mesh.edge_length()                    # 精确的每个边的长度, NE*1

    # 根据单元误差 eta 转化为节点误差
    nodeError = (node2cell @ eta) / (node2cell @ np.ones(NC))
    nodeEdgeLength = (node2cell @ edge_lengths1) / (node2cell @ np.ones(NC))
    rho = nodeError / nodeEdgeLength
    sorted_indices = np.argsort(-rho)  # -rho 表示降序排序
    sorted_rho = rho[sorted_indices]

    threshold = 0.98 * np.sum(rho ** 2)
    cumulative_error_squared = np.cumsum(sorted_rho ** 2)
    N = np.searchsorted(cumulative_error_squared, threshold, side='right')
    # print(f"      Marked {N} points that will be refined.")
    value = (np.sqrt(0.5)) ** (np.log(NN / N + 1) / np.log(2.0))

    labels = np.full_like(rho, 1.0)
    labels[sorted_indices[:N]] = value ** flag
    new_size = nodeEdgeLength * labels
    
    gmsh.initialize()
    gmsh.model.add("gntest2")
    
    # 禁用gmsh的终端输出
    gmsh.option.setNumber("General.NumThreads", 4)
    gmsh.option.setNumber("General.Terminal", 0)    
        
    # 创建几何模型
    gmsh.model.occ.addRectangle(-1.0, -1.0, -1.0, 2.0, 2.0, 1)
    # gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, 1.0, 1.0, 1)
    gmsh.model.occ.synchronize()
    
    # 设置网格生成选项
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
    node = mesh.entity('node')
    # 写入 Gmsh .pos 文件
    pos_filename = "background_field.pos"
    with open(pos_filename, "w") as f:
        f.write('View "meshSize" {\n')
        for (x, y), s in zip(node, new_size):
            f.write(f'SP({x}, {y}, 0){{{s}}};\n')
        f.write("};\n")
    
    # 加载背景网格尺寸场
    gmsh.merge(pos_filename)
    gmsh.model.mesh.field.add("PostView", 1)
    gmsh.model.mesh.field.setNumber(1, "ViewIndex", 0)
    gmsh.model.mesh.field.setAsBackgroundMesh(1)

    # 生成网格
    gmsh.model.mesh.generate(2)
    
    # 写入临时文件
    gmsh.write("temp_mesh.vtk")
    gmsh.finalize()   

    # 使用 meshio 读取生成的网格
    mesh = meshio.read("temp_mesh.vtk")
    node = mesh.points[:, :2]
    cell = mesh.cells_dict["triangle"].astype(np.int64)
    return TriangleMesh(node, cell)  

def generate_mesh_possion(mesh, eta, flag=1):
    # 构建point2cell矩阵
    NC = mesh.number_of_cells()
    NN = mesh.number_of_nodes()

    cells = mesh.entity('cell')
    node2cell = mesh.ds.node_to_cell()
    volume = mesh.cell_area()
    edge_lengths1 = np.sqrt((4.0 * volume) / np.sqrt(3))  # 根据面积公式求平均边长, NC*1
    # edge_lengths2 = mesh.edge_length()                    # 精确的每个边的长度, NE*1

    # 根据单元误差 eta 转化为节点误差
    nodeError = (node2cell @ eta) / (node2cell @ np.ones(NC))
    nodeEdgeLength = (node2cell @ edge_lengths1) / (node2cell @ np.ones(NC))
    rho = nodeError / nodeEdgeLength
    sorted_indices = np.argsort(-rho)  # -rho 表示降序排序
    sorted_rho = rho[sorted_indices]

    threshold = 0.94 * np.sum(rho ** 2)
    cumulative_error_squared = np.cumsum(sorted_rho ** 2)
    N = np.searchsorted(cumulative_error_squared, threshold, side='right')
    print(f"  Marked {N} points that will be refined.")
    value = (np.sqrt(0.5)) ** (np.log(NN / N + 1) / np.log(2.0))

    labels = np.full_like(rho, 1.1)
    labels[sorted_indices[:N]] = value ** flag
    new_size = nodeEdgeLength * labels
    
    gmsh.initialize()
    gmsh.model.add("gntest2")
    
    # 禁用gmsh的终端输出
    gmsh.option.setNumber("General.NumThreads", 4)
    gmsh.option.setNumber("General.Terminal", 0)    
        
    # 创建一个大矩形区域
    rectangle1 = gmsh.model.occ.addRectangle(-1.0, -1.0, 0.0, 2.0, 2.0, 1)  # 外矩形
    
    # 创建一个小矩形区域（去除的部分）
    rectangle2 = gmsh.model.occ.addRectangle(0.0, -1.0, 0.0, 1.0, 1.0, 2)  # 内矩形，切除
    
    # 同步并执行布尔操作
    gmsh.model.occ.synchronize()
    gmsh.model.occ.cut([(2, rectangle1)], [(2, rectangle2)])
    gmsh.model.occ.synchronize()
    
    # 设置网格生成选项
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
    node = mesh.entity('node')
    # 写入 Gmsh .pos 文件
    pos_filename = "background_field.pos"
    with open(pos_filename, "w") as f:
        f.write('View "meshSize" {\n')
        for (x, y), s in zip(node, new_size):
            f.write(f'SP({x}, {y}, 0){{{s}}};\n')
        f.write("};\n")
    
    # 加载背景网格尺寸场
    gmsh.merge(pos_filename)
    gmsh.model.mesh.field.add("PostView", 1)
    gmsh.model.mesh.field.setNumber(1, "ViewIndex", 0)
    gmsh.model.mesh.field.setAsBackgroundMesh(1)

    # 生成网格
    gmsh.model.mesh.generate(2)
    
    # 写入临时文件
    gmsh.write("temp_mesh.vtk")
    gmsh.finalize()   

    # 使用 meshio 读取生成的网格
    mesh = meshio.read("temp_mesh.vtk")
    node = mesh.points[:, :2]
    cell = mesh.cells_dict["triangle"].astype(np.int64)
    return TriangleMesh(node, cell)  