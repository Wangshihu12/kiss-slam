# MIT License

# Copyright (c) 2025 Tiziano Guadagnino, Benedikt Mersch, Saurabh Gupta, Cyrill
# Stachniss.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import numpy as np
from kiss_icp.kiss_icp import KissICP
from kiss_icp.voxelization import voxel_down_sample

from kiss_slam.config import KissSLAMConfig
from kiss_slam.local_map_graph import LocalMapGraph
from kiss_slam.loop_closer import LoopCloser
from kiss_slam.pose_graph_optimizer import PoseGraphOptimizer
from kiss_slam.voxel_map import VoxelMap


def transform_points(pcd, T):
    R = T[:3, :3]
    t = T[:3, -1]
    return pcd @ R.T + t


class KissSLAM:
    """
    KissSLAM主类：实现基于局部地图的SLAM系统
    该系统结合了里程计、闭环检测和姿态图优化
    """
    def __init__(self, config: KissSLAMConfig):
        """
        初始化KissSLAM系统
        
        Args:
            config: KissSLAM配置对象，包含各模块的参数设置
        """
        self.config = config
        
        # 初始化里程计模块（KissICP），用于帧间运动估计
        self.odometry = KissICP(config.kiss_icp_config())
        
        # 初始化闭环检测器，用于识别机器人是否回到之前访问过的位置
        self.closer = LoopCloser(config.loop_closer)
        
        # 获取局部地图配置参数
        local_map_config = self.config.local_mapper
        
        # 设置局部地图的体素大小，用于点云降采样
        self.local_map_voxel_size = local_map_config.voxel_size
        
        # 初始化体素地图，用于存储和管理点云数据
        self.voxel_grid = VoxelMap(self.local_map_voxel_size)
        
        # 初始化局部地图图结构，管理多个局部地图节点
        self.local_map_graph = LocalMapGraph()
        
        # 设置局部地图分割距离阈值，超过此距离将创建新的局部地图节点
        self.local_map_splitting_distance = local_map_config.splitting_distance
        
        # 初始化姿态图优化器，用于全局优化机器人轨迹
        self.optimizer = PoseGraphOptimizer(config.pose_graph_optimizer)
        
        # 将第一个局部地图节点添加到优化器中
        self.optimizer.add_variable(self.local_map_graph.last_id, self.local_map_graph.last_keypose)
        
        # 固定第一个节点的姿态作为参考坐标系
        self.optimizer.fix_variable(self.local_map_graph.last_id)
        
        # 存储检测到的闭环信息
        self.closures = []

    def get_closures(self):
        """
        获取所有检测到的闭环信息
        
        Returns:
            list: 闭环对列表，每个元素为(source_id, target_id)
        """
        return self.closures

    def get_keyposes(self):
        """
        获取所有关键姿态（局部地图节点的姿态）
        
        Returns:
            list: 关键姿态列表
        """
        return list(self.local_map_graph.keyposes())

    def process_scan(self, frame, timestamps):
        """
        处理单帧激光雷达扫描数据
        
        Args:
            frame: 点云数据
            timestamps: 时间戳信息
        """
        # 使用里程计模块处理当前帧，获得去畸变后的点云
        deskewed_frame, _ = self.odometry.register_frame(frame, timestamps)
        
        # 获取当前帧的估计姿态
        current_pose = self.odometry.last_pose
        
        # 对去畸变后的点云进行体素降采样，减少数据量
        mapping_frame = voxel_down_sample(deskewed_frame, self.local_map_voxel_size)
        
        # 将降采样后的点云整合到体素地图中
        self.voxel_grid.integrate_frame(mapping_frame, current_pose)
        
        # 将当前姿态添加到局部轨迹中
        self.local_map_graph.last_local_map.local_trajectory.append(current_pose)
        
        # 计算从局部地图起点到当前位置的移动距离
        traveled_distance = np.linalg.norm(current_pose[:3, -1])
        
        # 如果移动距离超过阈值，则创建新的局部地图节点
        if traveled_distance > self.local_map_splitting_distance:
            self.generate_new_node()

    def compute_closures(self, query_id, query):
        """
        计算闭环检测
        
        Args:
            query_id: 查询局部地图的ID
            query: 查询点云数据
        """
        # 使用闭环检测器计算是否存在闭环
        is_good, source_id, target_id, pose_constraint = self.closer.compute(
            query_id, query, self.local_map_graph
        )
        
        # 如果检测到有效闭环
        if is_good:
            # 记录闭环信息
            self.closures.append((source_id, target_id))
            
            # 向姿态图优化器添加闭环约束
            # np.eye(6)是6x6的单位矩阵，表示约束的信息矩阵（权重）
            self.optimizer.add_factor(source_id, target_id, pose_constraint, np.eye(6))
            
            # 执行姿态图优化
            self.optimize_pose_graph()

    def optimize_pose_graph(self):
        """
        执行姿态图优化，更新所有局部地图节点的姿态
        """
        # 运行优化算法
        self.optimizer.optimize()
        
        # 获取优化后的姿态估计
        estimates = self.optimizer.estimates()
        
        # 更新局部地图图中每个节点的关键姿态
        for id_, pose in estimates.items():
            self.local_map_graph[id_].keypose = np.copy(pose)

    def generate_new_node(self):
        """
        生成新的局部地图节点
        这个过程包括：
        1. 保存当前局部地图
        2. 重置里程计
        3. 创建新的局部地图节点
        4. 执行闭环检测
        """
        # 获取当前里程计维护的局部地图点云
        points = self.odometry.local_map.point_cloud()
        
        # === 重置里程计部分 ===
        # 获取当前局部地图的最后一个轨迹点
        last_local_map = self.local_map_graph.last_local_map
        relative_motion = last_local_map.local_trajectory[-1]
        
        # 计算相对运动的逆变换
        inverse_relative_motion = np.linalg.inv(relative_motion)
        
        # 将点云变换到新的局部坐标系
        transformed_local_map = transform_points(points, inverse_relative_motion)

        # 清空里程计的局部地图并添加变换后的点云
        self.odometry.local_map.clear()
        self.odometry.local_map.add_points(transformed_local_map)
        
        # 重置里程计姿态为单位矩阵（局部坐标系原点）
        self.odometry.last_pose = np.eye(4)

        # === 局部地图图更新部分 ===
        # 记录查询ID（即将完成的局部地图）
        query_id = last_local_map.id
        
        # 获取体素地图中的所有点云作为查询数据
        query_points = self.voxel_grid.point_cloud()
        
        # 完成当前局部地图的构建
        self.local_map_graph.finalize_local_map(self.voxel_grid)
        
        # 清空体素地图并添加变换后的点云，为新局部地图做准备
        self.voxel_grid.clear()
        self.voxel_grid.add_points(transformed_local_map)
        
        # === 姿态图优化器更新部分 ===
        # 向优化器添加新的局部地图节点
        self.optimizer.add_variable(self.local_map_graph.last_id, self.local_map_graph.last_keypose)
        
        # 添加相邻局部地图之间的里程计约束
        self.optimizer.add_factor(
            self.local_map_graph.last_id, query_id, relative_motion, np.eye(6)
        )
        
        # 执行闭环检测
        self.compute_closures(query_id, query_points)

    @property
    def poses(self):
        """
        获取完整的机器人轨迹姿态序列
        
        Returns:
            list: 所有时刻的姿态变换矩阵
        """
        # 初始化姿态列表，起始姿态为单位矩阵
        poses = [np.eye(4)]
        
        # 遍历所有局部地图节点
        for node in self.local_map_graph.local_maps():
            # 跳过第一个轨迹点（已经包含在poses中）
            # 将每个局部轨迹点变换到全局坐标系
            for rel_pose in node.local_trajectory[1:]:
                # 全局姿态 = 节点关键姿态 × 局部相对姿态
                poses.append(node.keypose @ rel_pose)
        
        return poses

    def fine_grained_optimization(self):
        """
        精细化优化：对所有轨迹点进行优化，而不仅仅是关键姿态
        
        Returns:
            tuple: (优化后的所有姿态, 优化器对象)
        """
        # 创建新的姿态图优化器用于精细化优化
        pgo = PoseGraphOptimizer(self.config.pose_graph_optimizer)
        
        # 初始化姿态ID计数器
        id_ = 0
        
        # 添加第一个姿态并固定它作为参考
        pgo.add_variable(id_, self.local_map_graph[id_].keypose)
        pgo.fix_variable(id_)
        
        # 遍历所有局部地图节点
        for node in self.local_map_graph.local_maps():
            # 计算局部轨迹中相邻姿态之间的里程计因子
            odometry_factors = [
                np.linalg.inv(T0) @ T1  # 计算相对变换
                for T0, T1 in zip(node.local_trajectory[:-1], node.local_trajectory[1:])
            ]
            
            # 为每个里程计因子添加变量和约束
            for i, factor in enumerate(odometry_factors):
                # 计算全局姿态：节点关键姿态 × 局部轨迹姿态
                global_pose = node.keypose @ node.local_trajectory[i + 1]
                
                # 添加新的姿态变量
                pgo.add_variable(id_ + 1, global_pose)
                
                # 添加里程计约束（相邻姿态之间的约束）
                pgo.add_factor(id_ + 1, id_, factor, np.eye(6))
                
                # 更新ID计数器
                id_ += 1
            
            # 固定每个局部地图的最后一个姿态
            pgo.fix_variable(id_ - 1)

        # 执行优化
        pgo.optimize()
        
        # 获取所有优化后的姿态
        poses = [x for x in pgo.estimates().values()]
        
        return poses, pgo
