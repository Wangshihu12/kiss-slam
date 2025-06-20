# MIT License
#
# Copyright (c) 2025 Tiziano Guadagnino, Benedikt Mersch, Saurabh Gupta, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import numpy as np
import open3d as o3d
from map_closures.map_closures import MapClosures

from kiss_slam.config import LoopCloserConfig
from kiss_slam.local_map_graph import LocalMapGraph
from kiss_slam.voxel_map import VoxelMap


class LoopCloser:
    """
    闭环检测器类：负责检测机器人是否回到之前访问过的位置
    通过比较当前局部地图与历史局部地图来识别闭环
    """
    def __init__(self, config: LoopCloserConfig):
        """
        初始化闭环检测器
        
        Args:
            config: 闭环检测器配置对象
        """
        self.config = config
        
        # 初始化地图闭环检测器，用于快速筛选候选闭环
        self.detector = MapClosures(config.detector)
        
        # 设置局部地图体素大小，与密度地图分辨率相同
        self.local_map_voxel_size = config.detector.density_map_resolution
        
        # 设置ICP（迭代最近点算法）距离阈值
        # sqrt(3) * voxel_size 是体素对角线长度，用作合理的匹配距离
        self.icp_threshold = np.sqrt(3) * self.local_map_voxel_size
        
        # 设置ICP算法类型：点到平面的变换估计
        # 这种方法比点到点匹配更稳定，特别适合平面结构较多的环境
        self.icp_algorithm = o3d.t.pipelines.registration.TransformationEstimationPointToPlane()
        
        # 设置ICP收敛判断标准
        # relative_rmse=1e-4 表示当相对均方根误差变化小于0.0001时认为收敛
        self.termination_criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(
            relative_rmse=1e-4
        )
        
        # 设置重叠度阈值，用于验证闭环质量
        # 只有当两个局部地图的重叠度超过此阈值时，闭环才被接受
        self.overlap_threshold = config.overlap_threshold

    def compute(self, query_id, points, local_map_graph: LocalMapGraph):
        """
        计算并检测闭环
        
        Args:
            query_id: 查询局部地图的ID（当前要检测闭环的局部地图）
            points: 查询局部地图的点云数据
            local_map_graph: 局部地图图结构，包含所有历史局部地图
            
        Returns:
            tuple: (是否检测到有效闭环, 参考地图ID, 查询地图ID, 姿态约束)
        """
        # 使用检测器找到最佳候选闭环
        # 这一步通常使用描述符匹配等快速方法进行粗筛选
        closure = self.detector.get_best_closure(query_id, points)
        
        # 初始化返回值
        is_good = False          # 闭环是否有效
        ref_id = -1             # 参考局部地图ID
        pose_constraint = np.eye(4)  # 姿态约束（4x4变换矩阵）
        
        # 检查候选闭环的内点数量是否满足阈值要求
        if closure.number_of_inliers >= self.config.detector.inliers_threshold:
            # 获取参考局部地图ID
            ref_id = closure.source_id
            
            # 从局部地图图中获取参考点云和查询点云
            source = local_map_graph[ref_id].pcd      # 参考局部地图点云
            target = local_map_graph[query_id].pcd    # 查询局部地图点云
            
            print("\nKissSLAM| Closure Detected")
            
            # 验证闭环的质量：使用精确的ICP算法和重叠度计算
            is_good, pose_constraint = self.validate_closure(source, target, closure.pose)
        
        return is_good, ref_id, query_id, pose_constraint

    def validate_closure(self, source, target, initial_guess):
        """
        验证闭环的质量
        这是计算量最大的部分，使用ICP精配准和重叠度计算来验证闭环
        
        Args:
            source: 参考局部地图点云
            target: 查询局部地图点云  
            initial_guess: 初始姿态估计（来自粗检测）
            
        Returns:
            tuple: (闭环是否被接受, 精确的姿态变换矩阵)
        """
        # === ICP精配准阶段 ===
        # 使用ICP算法进行精确的点云配准
        registration_result = o3d.t.pipelines.registration.icp(
            source,              # 源点云（参考）
            target,              # 目标点云（查询）
            self.icp_threshold,  # 最大对应点距离
            initial_guess,       # 初始变换猜测
            self.icp_algorithm,  # 使用点到平面ICP
            self.termination_criteria,  # 收敛标准
        )
        
        # === 重叠度计算阶段 ===
        # 创建联合体素地图用于计算重叠度
        union_map = VoxelMap(self.local_map_voxel_size)
        
        # 提取点云数据并转换为numpy数组
        source_pts = source.point.positions.numpy().astype(np.float64)
        target_pts = target.point.positions.numpy().astype(np.float64)
        
        # 获取ICP配准后的变换矩阵
        pose = registration_result.transformation.numpy()
        
        # 将变换后的源点云添加到联合地图中
        union_map.integrate_frame(source_pts, pose)
        num_source_voxels = union_map.num_voxels()  # 源点云的体素数量
        
        # 目标点云的点数（用作体素数量的近似）
        num_target_voxels = len(target_pts)
        
        # 将目标点云也添加到联合地图中
        union_map.add_points(target_pts)
        union = union_map.num_voxels()  # 联合后的总体素数量
        
        # 使用集合论计算交集大小
        # 交集 = 源体素数 + 目标体素数 - 联合体素数
        intersection = num_source_voxels + num_target_voxels - union
        
        # 计算重叠度：交集与较小集合的比值
        # 这表示两个点云的重叠程度
        overlap = intersection / np.min([num_source_voxels, num_target_voxels])
        
        # 判断闭环是否被接受：重叠度必须超过设定阈值
        closure_is_accepted = overlap > self.overlap_threshold
        
        # 输出调试信息
        print(f"KissSLAM| LocalMaps Overlap: {overlap}")
        if closure_is_accepted:
            print("KissSLAM| Closure Accepted")
        else:
            print(f"KissSLAM| Closure rejected for low overlap.")
        
        return closure_is_accepted, pose
