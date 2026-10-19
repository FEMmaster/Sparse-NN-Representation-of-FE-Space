#!/usr/bin/env python3
# 
import argparse
import os
import time
import torch
import numpy as np
import scipy.io as sio
from matplotlib import rc
rc('text', usetex=True)

from fealpy.decorator import cartesian
from fealpy.mesh import MeshFactory as MF
from fealpy.timeintegratoralg import UniformTimeLine
from fealpy.pde.heatequation_model_2d import Rotation, Diffusion, Splitting
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC
from scipy.sparse.linalg import spsolve
from fealpy.mesh.quality import TriRadiusRatio

from NN4FEM import  build_p2fem_network
from mesh_regenerator import generate_mesh_heat


## 参数解析
parser = argparse.ArgumentParser(description = "hr自适应有限元求解HEAT方程")
parser.add_argument('--ns', default=8, type=int, help='空间各个方向剖分段数, 默认剖分 2 次.')
parser.add_argument('--nt', default=1000, type=int, help='时间剖分段数, 默认剖分 100 段.')
parser.add_argument('--tol', default=0.006, type=float, help='自适应加密停止阈值, 默认设定为 0.05.')
parser.add_argument('--output', default='./AFEM-hr-heat', type=str, help='结果输出目录, 默认为 ./')

args = parser.parse_args()      # 解析命令行参数
ns = args.ns
nt = args.nt
tol = args.tol
output = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
if not os.path.exists(output):
    os.makedirs(output)

# 待求解问题设定    
pde = Rotation()
domain = pde.domain()
c = pde.diffusionCoefficient

# 网格设定 (hr自适应空间网格剖分在循环内部)
tmesh = UniformTimeLine(0, 1, nt) # 均匀时间剖分

# 开始时间迭代
results = []

total_start_time = time.time()
for j in range(0, nt+1):
    time_step_start = time.time()
    smesh = MF.boxmesh2d(domain, nx=ns, ny=ns, meshtype='tri')
    
    t = tmesh.current_time_level()
    print("当前时刻是: ", t)
    meshQuality = TriRadiusRatio()

    lsFit = np.zeros((2, 7))
    for i in range(0, 7):
        # meshQuality.show(meshQuality.quality(mesh=smesh))
        space = LagrangeFiniteElementSpace(smesh, p=2)
        
        # print('    网格自由度为: ', space.number_of_global_dofs())
        
        if j == 0:
            uh1 = space.interpolation(pde.init_value)
        else:                           
            # 下一层时间步的有限元解
            uh1 = space.function()
            A = c*space.stiff_matrix() # 刚度矩阵
            M = space.mass_matrix() # 质量矩阵
            dt = tmesh.current_time_step_length() # 时间步长
            G = M + dt*A # 隐式迭代矩阵

            # 当前时间层的右端项
            @cartesian
            def source(p):
                return pde.source(p, t)
            F = space.source_vector(source)
            F *= dt
            uh0 = network(space.interpolation_points()).detach().cpu().numpy().flatten()
            F += M@uh0

            # 当前时间层的 Dirichlet 边界条件处理
            @cartesian
            def dirichlet(p):
                return pde.dirichlet(p, t)
            bc = DirichletBC(space, dirichlet)
            GD, F = bc.apply(G, F, uh1)

            # 代数系统求解
            uh1[:] = spsolve(GD, F)
        
        eta = space.recovery_estimate_2nd_element(uh1, method='area_harmonic')
        err = np.sqrt(np.sum(eta**2))
        print('    DOF= ', space.number_of_global_dofs())
        print('    误差估计子: ', err)
        
        # t 时间层的误差
        @cartesian
        def solution(p):
            return pde.solution(p, t)
        @cartesian
        def grad_solution(p):
            return np.stack([pde.u_x(p, t), pde.u_y(p, t)], axis=-1)
        error_L2 = space.integralalg.error(solution, uh1)
        print('    L2 error = ', error_L2)
        error_H1_semi = space.integralalg.error(grad_solution, uh1.grad_value)
        error_H1 = np.sqrt(error_L2**2 + error_H1_semi**2)
    
        # 保存网格和数值解信息
        if abs(t % 0.01) < 1e-12:
            fname = os.path.join(output, 'test_' +str(j).zfill(5) + str(i).zfill(3) + '.vtu')
            smesh.nodedata['uh'] = uh1
            smesh.nodedata['err_ZZ'] = eta
            smesh.to_vtk(fname=fname)
        lsFit[0, i] = space.number_of_global_dofs()
        lsFit[1, i] = err

        results.append([
            t,
            i,
            space.number_of_global_dofs(),
            error_L2,
            error_H1_semi,
            err
        ])

        if err < tol or i > 5:
            break

        if i == 4:
            A = np.ones((3, 2))
            b = np.log(lsFit[1, 2:5])
            A[:, 1] = np.log(lsFit[0, 2:5])
            p = np.linalg.lstsq(A, b, rcond=None)[0]
            ideaN = int(np.floor(np.exp((np.log(tol) - p[0]) / p[1])))
            flag = max(int(np.floor(np.log(ideaN / lsFit[0, i]) / np.log(2))), 1)
            # print("  aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", flag)
        else:
            flag = 1

        smesh = generate_mesh_heat(smesh, eta, flag)
        
    if j <= nt: 
        network = build_p2fem_network(space, uh1)
  
    tmesh.advance() # 时间步进一层
    
total_time =  time.time() - total_start_time
print(f"总耗时: {total_time:.2f}秒")

results = np.asarray(results, dtype=np.float64)

sio.savemat(os.path.join(output, 'results.mat'), {'results': results})