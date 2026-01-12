from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import sys
import os
import random

sys.path.append(os.path.dirname(__file__)) 

import math
import ast

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image



import settings
import constants
from geometry import Heightmap, CollisionDetector
from blender_ops import (
    clear_scene,
    setup_render,
    create_mesh,
    setup_camera,
    setup_light,
    no_physics_render,
    physics_render,
    simple_render,
)

from predict_tf import (
    TrainConfig,
    TowerCollapseDataset,
    TemporalTransformerPredictor,
    eval_one_epoch,
)

# MCTS配置参数
MCTS_CONFIG = {
    "iterations": 50,  # 每次搭建步骤的MCTS迭代次数
    "exploration_constant": 1.414,  # UCB探索系数（平衡探索与利用）
    "height_reward_weight": 0.02,  # 高度奖励权重（避免只追求稳定忽略高度）
    "collapse_penalty": 10.0,  # 高倒塌概率惩罚项
    "max_rollout_layer": 10,  # 模拟阶段最大层数
}

# 积木的结构化描述（与Blender输出对齐）
# @dataclass
# class Block:
#     index: str  # 唯一标识
#     size: Tuple[float, float, float]  # 长宽高
#     position: Tuple[float, float, float]  # 世界坐标
#     rotation: Tuple[float, float, float]  # 欧拉角
#     material: str  # 材质（wood/metal/stone）
#     used: bool = False  # 是否已被使用

class CollapsePredictor:
    def __init__(self, cfg = TrainConfig()):
        self.cfg = cfg
        self.model = TemporalTransformerPredictor(
            image_size=cfg.image_size,
            feat_channels=256,
            nhead=8,
            num_layers=6,
            dim_feedforward=1024,
            dropout=0.1,
            T=22,
        ).to(cfg.device)

        path = Path(__file__).resolve().parent.parent / "model_a.pt"

        state = torch.load(path, map_location=cfg.device)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def predict(self, x):
        x0 =torch.from_numpy(x.transpose(2, 1, 0)).to(dtype=torch.float32).unsqueeze(dim=0)/ 255.0
        x0 = x0.to(self.cfg.device)
        _, logit = self.model(x0)
        prob = torch.sigmoid(logit)
        pred = (prob >= 0.5).float()
        return logit.item()

def get_block_positions(
    existing_blocks,
    heightmap,
    collisiondetector,
    new_size,
    new_rot,
    flag=0,
    num=3,
):
    """
    Generate a block's position.
    Args:
        existing_blocks: dic
        heightmap
        collisiondetector
        size: size of the current block
        new_rot: rotation of the current block
        flag: if it's pedestal then flag equals to 1
    """

    positions = []
    valid_positions = heightmap.get_valid_positions(new_size, new_rot, flag)
    if not valid_positions:
        return positions   

    while valid_positions and len(positions) < num: 
        position = random.choice(valid_positions)
        valid_positions.remove(position)
        if not collisiondetector.check_block_collision(
            existing_blocks, position, new_size, new_rot
        ):
            positions.append(position)

    return positions

# MCTS节点（对应塔状态）
class MCTSTowerNode:
    def __init__(self, blocks: List, current_height: float, current_layer: int):
        # 塔的结构化状态（关键：用于Blender复现与动作生成）
        self.blocks = blocks  # 当前塔的所有积木列表
        self.current_height = current_height  # 当前塔总高度
        self.current_layer = current_layer  # 当前塔层数
        self.is_terminal = False  # 是否为终止状态（达最大层数/高倒塌风险）

        self.heightmap = Heightmap()

        for block in blocks:
            if block["used"]:
                self.heightmap.update_heightmap(block["position"], block["size"], block["rotation"])
        
        # MCTS统计信息
        self.N = 0  # 节点总访问次数
        self.Q = 0.0  # 节点平均价值（稳定性+高度奖励）
        self.children: Dict[Tuple, MCTSTowerNode] = {}  # 子节点：动作→子节点映射
        self.untried_actions: List[Tuple] = None  # 未尝试的合法动作（初始化后生成）
        self.flag = True
        
        # 缓存：避免重复渲染与预测
        self.collapse_prob = None  # 该塔状态的倒塌概率（由已有模型预测）
        self.render_img = None  # 该塔状态的渲染图像

        # 核心补充：父节点引用（初始为None，根节点无父节点）
        self.parent = None

    def get_untried_actions(self, action_generator) -> List[Tuple]:
        """获取当前节点的未尝试合法动作（懒加载）"""
        if self.untried_actions is None and self.flag:
            self.untried_actions = action_generator.generate_legal_actions(self)
            # print(len(self.untried_actions))
            self.flag = False
        return self.untried_actions

    def is_fully_expanded(self, action_generator) -> bool:
        """判断节点是否已完全扩展（所有合法动作均已尝试）"""
        return len(self.get_untried_actions(action_generator)) == 0
    

class LegalActionGenerator:
    def __init__(self, all_blocks: List, max_layer: int, support_threshold: float = 0.01):
        self.all_blocks = all_blocks  # 所有可用积木（初始未使用）
        self.max_layer = max_layer  # 最大层数约束
        self.support_threshold = support_threshold  # 最小支撑面积阈值（与Blender对齐）
        self.collisiondetector = CollisionDetector()

    def generate_legal_actions(self, node: MCTSTowerNode) -> List[Tuple]:
        """
        生成当前塔状态的合法动作列表
        动作格式：(index, target_position, target_rotation)
        """
        legal_actions = []
        
        # 1. 终止条件判断（达最大层数，无合法动作）
        if node.current_layer >= self.max_layer:
            node.is_terminal = True
            return legal_actions
        
        # 2. 筛选可用积木（未被使用的积木）
        available_blocks = [b for b in self.all_blocks if not b["used"]]
        if not available_blocks:
            return legal_actions
        
        existing_blocks = [b for b in self.all_blocks if b["used"]]
        
        # # 3. 生成顶部支撑区域（当前塔顶部的所有可放置位置）
        top_support_regions = self._get_top_support_regions(node)
        
        # 4. 为每个可用积木生成候选位置与角度
        for block in available_blocks:
            # 离散化旋转角度（简化动作空间，也可连续采样）
            candidate_rotations = self._get_candidate_rotations()
            
            # # 离散化顶部放置位置（基于支撑区域采样）
            # candidate_positions = self._sample_positions_on_support(top_support_regions, block)
            
            # 5. 验证每个（位置+角度）的合法性（不重叠、支撑面积达标）
            # for pos in candidate_positions:
            #     
            for rot in candidate_rotations:
                positions = get_block_positions(existing_blocks, 
                                                node.heightmap, 
                                                self.collisiondetector,
                                                block["size"],
                                                rot)
                
                for pos in positions:
                    
                    legal_actions.append((block["index"], pos, rot))
        return legal_actions

        #  # 4. 为每个可用积木生成候选位置与角度
        # for block in available_blocks:
        #     # 离散化旋转角度（简化动作空间，也可连续采样）
        #     candidate_rotations = self._get_candidate_rotations()
            
        #     # 离散化顶部放置位置（基于支撑区域采样）
        #     candidate_positions = self._sample_positions_on_support(top_support_regions, block)
            
        #     # 5. 验证每个（位置+角度）的合法性（不重叠、支撑面积达标）
        #     for pos in candidate_positions:
        #         for rot in candidate_rotations:
        #             if self._is_action_legal(node, block, pos, rot, existing_blocks):
        #                 legal_actions.append((block["index"], pos, rot))
        
        # return legal_actions

    def _get_top_support_regions(self, node: MCTSTowerNode) -> List[Tuple]:
        """获取当前塔顶部的支撑区域（返回可放置的平面区域坐标）"""
        # 简化实现：提取当前塔所有顶部积木的上表面区域（与你的Heightmap类对齐）
        top_regions = []
        if not node.blocks:  # 空白平面（初始状态）
            top_regions.append(((0.0, 0.0), (1.0, 1.0)))  # 中心区域
        else:
            # 找到当前塔最高的积木（顶部层）
            max_z = max(b["position"][2] + b["size"][2]/2 for b in node.blocks)
            top_blocks = [b for b in node.blocks if (b["position"][2] + b["size"][2]/2) == max_z]
            for b in top_blocks:
                # 提取积木上表面的四个角（支撑区域）
                x, y, _ = b["position"]
                l, w, _ = b["size"]
                top_regions.append(((x-l/2, y-w/2), (x+l/2, y+w/2)))
        return top_regions

    def _get_candidate_rotations(self) -> List[Tuple[float, float, float]]:
        """离散化候选旋转角度（仅绕Z轴旋转，与你的Blender代码对齐）"""
        if settings.ROT_DISCRETE:
            return [(0, 0, 0), (0, 0, np.pi/2), (0, 0, np.pi), (0, 0, 3*np.pi/2)]
        else:
            # 连续空间：采样4个角度（平衡效率与探索性）
            return [(0, 0, np.random.uniform(0, 2*np.pi)) for _ in range(10)]

    def _sample_positions_on_support(self, support_regions, block) -> List[Tuple[float, float, float]]:
        """在支撑区域上采样候选放置位置（保证Z轴对齐顶部）"""
        candidate_positions = []
        l, w, h = block["size"]
        for (min_xy, max_xy) in support_regions:
            min_x, min_y = min_xy
            max_x, max_y = max_xy
            # 采样5个候选位置（避免动作空间过大）
            for _ in range(5):
                x = np.random.uniform(min_x + l/4, max_x - l/4)
                y = np.random.uniform(min_y + w/4, max_y - w/4)
                z = (max_xy[0] + min_xy[0])/2  # 对齐支撑区域顶部Z轴（简化，与Blender Heightmap对齐）
                candidate_positions.append((x, y, z + h/2))  # 积木中心Z坐标
        return candidate_positions

    def _is_action_legal(self, node: MCTSTowerNode, block, pos: Tuple, rot: Tuple, exsiting_blocks) -> bool:
        """验证动作合法性：1. 不重叠 2. 支撑面积达标（调用Blender CollisionDetector）"""
        # 1. 调用CollisionDetector检查是否与已有积木重叠
        collision_detector = CollisionDetector()
        new_vertices = collision_detector.get_block_vertices(pos, block["size"], rot)
        for existing_block in exsiting_blocks:
            existing_vertices = collision_detector.get_block_vertices(
                existing_block["position"], existing_block["size"], existing_block["rotation"]
            )
            if collision_detector.separating_axis_theorem(new_vertices, existing_vertices):
                return False
        
        # 2. 检查支撑面积是否达标（与你的Heightmap.intersection_threshold对齐）
        # 此处简化实现：可调用Heightmap计算多边形交集面积
        return True
    

class TowerMCTS:
    def __init__(self, collapse_predictor, action_generator, config: dict = MCTS_CONFIG):
        self.collapse_predictor = collapse_predictor  # 已有倒塌概率预测模型
        self.action_generator = action_generator  # 合法动作生成器
        self.config = config

    def _ucb_score(self, parent_node: MCTSTowerNode, child_node: MCTSTowerNode) -> float:
        """
        计算子节点的UCB评分（适配搭建任务）
        UCB = 节点平均价值 + 探索项（鼓励探索未充分访问的节点）
        """
        if child_node.N == 0:
            # 未访问过的节点：给予高评分鼓励探索
            return float('inf')
        
        # 探索项（传统UCB公式）
        exploration_term = self.config["exploration_constant"] * np.sqrt(
            np.log(parent_node.N) / child_node.N
        )
        
        # 利用项（节点平均价值：稳定性+高度奖励）
        exploitation_term = child_node.Q
        
        return exploitation_term + exploration_term

    def _select(self, node: MCTSTowerNode) -> MCTSTowerNode:
        """
        选择阶段：从当前节点出发，沿UCB最高的子节点向下遍历，直到未完全扩展/终止节点
        """
        # print("select")
        # print(len(node.children))
        # print(node.is_terminal, node.is_fully_expanded(action_generator=self.action_generator))
        while not node.is_terminal and node.is_fully_expanded(action_generator=self.action_generator):
            # 计算所有子节点的UCB评分，选择最高的
            child_scores = [
                (child, self._ucb_score(node, child))
                for child in node.children.values()
            ]
            # print(f"childlen{len(child_scores)}, node{len(node.blocks)}, {node.flag}")
            best_child = max(child_scores, key=lambda x: x[1])[0]
            node = best_child
        
        return node

    def _expand(self, node: MCTSTowerNode) -> Optional[MCTSTowerNode]:
        """
        扩展阶段：从节点的未尝试动作中选择一个，生成子节点（新塔状态）
        """
        if node.is_terminal:
            return None
        
        
        # 选择一个未尝试的合法动作
        untried_actions = node.get_untried_actions(self.action_generator)
        if not untried_actions:
            return None
        
        
        # 随机选择一个未尝试动作（也可优先选择高度收益高的动作）
        action = random.choice(untried_actions)
        index, target_pos, target_rot = action
        
        # 执行动作：生成新塔状态（更新积木列表、高度、层数）
        new_block = next(b for b in self.action_generator.all_blocks if b["index"] == index)
        # new_block["used"] = True  # 标记为已使用
        # new_block["position"] = target_pos
        # new_block["rotation"] = target_rot

        block_data = {
            "index": new_block["index"],
            "color": new_block["color"],
            "material": new_block["material"],
            "size": new_block["size"],
            "position": target_pos,
            "rotation": target_rot,
            "used": True,
        }
        new_blocks = node.blocks.copy() + [block_data]
        
        # 计算新塔的高度与层数
        new_height = max(b["position"][2] + b["size"][2]/2 for b in new_blocks)
        new_layer = node.current_layer + 1
        
        # 创建子节点
        child_node = MCTSTowerNode(
            blocks=new_blocks,
            current_height=new_height,
            current_layer=new_layer
        )
        
        # 缓存子节点的倒塌概率与渲染图像（调用已有模型与Blender）
        child_node.render_img = self._render_tower(child_node)  # 调用Blender渲染
        child_node.collapse_prob = self.collapse_predictor.predict(child_node.render_img)
        # child_node.collapse_prob = 0.5
        
        # 关联父子节点，移除已尝试动作
        node.children[action] = child_node
        node.untried_actions.remove(action)

        # 核心补充：建立父子节点关联（子节点的parent指向当前节点）
        child_node.parent = node

        return child_node

    def _simulate(self, node: MCTSTowerNode) -> float:
        """
        模拟阶段（Rollout）：从当前节点出发，快速模拟搭建直到终止，返回累计价值
        模拟策略：贪心策略（优先选择低倒塌概率+高高度的动作）
        """
        current_node = node
        total_reward = 0.0
        current_layer = current_node.current_layer
        
        while not current_node.is_terminal and current_layer <= self.config["max_rollout_layer"]:
            # 1. 计算当前节点的即时价值
            instant_reward = self._calculate_node_value(current_node)
            total_reward += instant_reward
            
            # 2. 生成合法动作，选择贪心最优动作（低倒塌概率+高高度）
            legal_actions = self.action_generator.generate_legal_actions(current_node)
            if not legal_actions:
                break
            
            # 3. 对每个候选动作快速评估，选择最优
            best_reward = -float('inf')
            best_next_state = None
            for action in legal_actions:
                # 快速生成下一个状态（不创建完整MCTS节点，提升效率）
                next_state = self._fast_generate_next_state(current_node, action)
                next_reward = self._calculate_node_value(next_state)
                if next_reward > best_reward:
                    best_reward = next_reward
                    best_next_state = next_state
            
            # 4. 迭代到下一个状态
            current_node = best_next_state
            current_layer += 1
        
        # 终止状态惩罚（如果倒塌概率过高）
        if current_node.collapse_prob > 0.8:
            total_reward -= self.config["collapse_penalty"]
        
        return total_reward

    def _backpropagate(self, node: MCTSTowerNode, reward: float):
        """
        回溯阶段：将模拟得到的奖励反向传播，更新路径上所有节点的N与Q
        """
        while node is not None:
            # 更新访问次数
            node.N += 1
            
            # 更新平均价值（增量更新，避免存储所有奖励）
            node.Q = node.Q + (reward - node.Q) / node.N
            
            # 向上遍历父节点（此处需补充父节点关联，简化实现可通过路径记录）
            node = node.parent  # 可在扩展阶段记录父节点，此处简化

    def _calculate_node_value(self, node: MCTSTowerNode) -> float:
        """
        计算节点价值（核心：融合稳定性与高度奖励）
        价值 = 稳定性得分（1 - 倒塌概率） + 高度奖励（当前高度 * 权重）
        """
        if node.collapse_prob is None:
            # 未预测倒塌概率，先调用模型预测
            node.render_img = self._render_tower(node)
            node.collapse_prob = self.collapse_predictor.predict(node.render_img)
            # node.collapse_prob = 0.5
        
        # 稳定性得分（0~1，越高越稳定）
        stability_score = 1.0 - node.collapse_prob
        
        # 高度奖励（鼓励搭建更高的塔）
        height_reward = node.current_height * self.config["height_reward_weight"]
        
        return stability_score + height_reward

    def _render_tower(self, node: MCTSTowerNode) -> np.ndarray:
        """调用Blender脚本渲染塔状态，返回图像（与你的已有代码集成）"""
        # 调用你已有的Blender渲染接口，传入node.blocks生成图像
        # 此处为简化实现，返回模拟图像
        clear_scene()

        setup_render(resolution_x=128, resolution_y=128, samples=4)

        create_mesh("PLANE")

        for block in node.blocks:
            create_mesh("BLOCK", block)

        setup_camera()
        setup_light()

        # no_physics_render(i, config_num_colors)
        p = simple_render()        
        clear_scene()
        
        return p

    def _fast_generate_next_state(self, node: MCTSTowerNode, action: Tuple) -> MCTSTowerNode:
        """快速生成下一个塔状态（模拟阶段专用，提升效率）"""
        # 与扩展阶段的动作执行逻辑一致，简化实现
        index, target_pos, target_rot = action
        new_block = next(b for b in self.action_generator.all_blocks if b["index"] == index)
        new_blocks = node.blocks.copy() + [new_block]
        new_height = max(b["position"][2] + b["size"][2]/2 for b in new_blocks)
        new_layer = node.current_layer + 1
        new_node = MCTSTowerNode(new_blocks, new_height, new_layer)
        
        # 快速预测倒塌概率（避免重复渲染）
        new_node.collapse_prob = np.random.uniform(0, 1)  # 替换为已有模型预测
        return new_node

    def search_best_action(self, root_node: MCTSTowerNode) -> Tuple:
        """
        执行MCTS搜索，返回最优动作
        输入：根节点（当前塔状态）
        输出：最优动作（index, target_position, target_rotation）
        """
        # 迭代执行MCTS四步

        for _ in range(self.config["iterations"]):
            # 1. 选择
            selected_node = self._select(root_node)

            # 2. 扩展
            expanded_node = self._expand(selected_node)
            
            # 3. 模拟（扩展成功则模拟子节点，否则模拟选中节点）
            simulate_node = expanded_node if expanded_node is not None else selected_node
            reward = self._simulate(simulate_node)
            
            # 4. 回溯
            self._backpropagate(simulate_node, reward)
        
        # 搜索结束：选择访问次数最多/价值最高的子节点对应的动作
        if not root_node.children:
            raise ValueError("无合法动作可选择")
        
        # 策略1：选择价值最高的子节点（优先稳定+高）
        best_child = max(root_node.children.values(), key=lambda x: x.Q)
        
        # 找到对应动作并返回
        for action, child in root_node.children.items():
            if child == best_child:
                return action
            

def tower_building_planner(
    all_blocks: List,
    collapse_predictor,
):
    """
    积木塔搭建规划主函数
    输出：搭建动作序列（可直接在Blender中执行验证）
    """
    # 1. 初始化组件
    action_generator = LegalActionGenerator(all_blocks, len(all_blocks))
    tower_mcts = TowerMCTS(collapse_predictor, action_generator)
    
    # 2. 初始化根节点（空白平面，无积木）
    existing_block = [block for block in all_blocks if block["used"]]
    # print(f"exsiting{len(existing_block)}")
    root_node = MCTSTowerNode(
        blocks=existing_block,
        current_height=0.0,
        current_layer=len(existing_block)
    )
    
    # 3. 迭代搭建（每一步执行MCTS搜索最优动作）
    build_actions = []
    while not root_node.is_terminal and len(root_node.blocks)<len(all_blocks):
        
        # 执行MCTS搜索，获取最优动作
        best_action = tower_mcts.search_best_action(root_node)
        build_actions.append(best_action)
        
        # 执行最优动作，更新根节点为新塔状态
        index, target_pos, target_rot = best_action
        new_block = next(b for b in all_blocks if b["index"] == index)
        new_block["used"] = True
        new_blocks = root_node.blocks.copy() + [new_block]
        new_block["position"] = target_pos
        new_block["rotation"] = target_rot
        
        
        new_height = max(b["position"][2] + b["size"][2]/2 for b in new_blocks)
        new_layer = root_node.current_layer + 1
        
        # 更新根节点
        root_node = MCTSTowerNode(
            blocks=new_blocks,
            current_height=new_height,
            current_layer=new_layer
        )
        
        print(f"完成第 {new_layer} 层搭建，当前高度：{new_height}，倒塌概率：{root_node.collapse_prob}")
    
    # 4. 输出搭建结果（可在Blender中执行验证）
    print(f"搭建完成，共 {len(build_actions)} 步，最终高度：{root_node.current_height}")
    return build_actions, root_node.blocks

def set_blocks_data(blocks_data, ped_num):
    if blocks_data:
        xs = [b["position"][0] for b in blocks_data]
        ys = [b["position"][1] for b in blocks_data]
        com_x = sum(xs) / len(xs)
        com_y = sum(ys) / len(ys)

        dx = -com_x
        dy = -com_y

        for b in blocks_data:
            x, y, z = b["position"]
            b["position"] = (x + dx, y + dy, z)

        # 额外一步：让底座方块“贴在地面”上，避免出现整体悬空的情况。
        # 这里假设水平地面位于 z=0（即 DEGREE=0 的常见设置）。
        # - 对所有底座方块（index < ped_num），计算它们的底面高度 bottom_z
        # - 找到最小的 bottom_z，并整体下移，使该底面刚好落在 z=0
        if ped_num > 0 and settings.DEGREE == 0:
            pedestal_blocks = [b for b in blocks_data if b["index"] < ped_num]
            if pedestal_blocks:
                min_bottom_z = min(
                    b["position"][2] - b["size"][2] / 2.0 for b in pedestal_blocks
                )
                dz = -min_bottom_z

                for b in blocks_data:
                    x, y, z = b["position"]
                    b["position"] = (x, y, z + dz)


def generate_blocks_data(config):
    """
    Generate blocks data.
    Args:
        config: dictionary from yaml
        heightmap
        collisiondetector
    """    
    heightmap = Heightmap()
    collisiondetector = CollisionDetector()

    blocks_data = []
    num_blocks = config["Scene"]["num_blocks"]  # 29
    ori_color_dic = config["Scene"][
        "num_colors"
    ]  # {"yellow": 13, "blue": 14, "white": 2}#7,9,1
    color_dic = {}
    for key, value in ori_color_dic.items():
        color_dic[key] = value
    ori_size_dic = config["Scene"][
        "sizes"
    ]  # 例如 {(0.5, 0.5, 1.5): 16, (1.5, 0.5, 0.5): 13}
    size_dic = {}
    for key, value in ori_size_dic.items():
        key_t = ast.literal_eval(key)
        size_dic[key_t] = value

    # 所有可用尺寸列表（不考虑数量），后面会根据数量动态过滤
    all_sizes = list(size_dic.keys())
    if not all_sizes:
        raise ValueError("No block sizes configured in yaml 'Scene.sizes'.")

    # 作为“底座”的尺寸：优先选择 z 方向最高的那个
    pedestal_size = max(all_sizes, key=lambda s: s[2])
    if settings.ROT_DISCRETE == False:
        rot_range = config["Scene"]["rot_range"]  # [0, 360]
        assert len(rot_range) == 2
        rot_range = [math.radians(rot_range[0]), math.radians(rot_range[1])]
    else:
        rot_range = config["Scene"]["rot_range"]  # [0, 90, 180, 270]
        rot_range = [math.radians(rot_range[i]) for i in range(len(rot_range))]

    # -------- 材质分配策略（仅支持新版写法）--------
    # Scene.num_materials = {"wood": 10, "metal": 5, ...}
    # 每个方块在生成时从仍有余量的材质中随机选择，从而实现“每块独立控制材质（按数量）”。
    scene_cfg = config["Scene"]
    if "num_materials" not in scene_cfg:
        raise ValueError("Scene.num_materials must be provided in config.")

    material_counts = {}
    for name, cnt in scene_cfg["num_materials"].items():
        if name not in constants.MATERIALS:
            raise ValueError(f"Unknown material '{name}' in Scene.num_materials")
        material_counts[name] = int(cnt)

    if sum(material_counts.values()) != num_blocks:
        raise ValueError("Sum of Scene.num_materials must be equal to Scene.num_blocks")
    
    blocks = []
    
    for i in range(num_blocks):
        # 为当前方块选择具体材质（从仍有余量的材质中随机选择）
        available_mats = [m for m, c in material_counts.items() if c > 0]
        if not available_mats:
            # 理论上不应该出现；兜底避免 KeyError
            available_mats = list(material_counts.keys())
        mat_name = random.choice(available_mats)
        
        # 按数量筛选还可用的尺寸
        candidate_sizes = [s for s, c in size_dic.items() if c > 0]
        if not candidate_sizes:
            # 理论上不应该出现；兜底避免 KeyError
            candidate_sizes = all_sizes
        new_size = random.choice(candidate_sizes)

        block_data = {
            "index": i,
            "color": random.choice(
                [key for key in color_dic.keys() if color_dic[key] > 0]
            ),
            "material": mat_name,
            "size": new_size,
            "position": (0., 0., 0.),
            "rotation": (0., 0., 0.),
            "used": False,
        }

        color_dic[block_data["color"]] -= 1
        size_dic[block_data["size"]] -= 1
        material_counts[mat_name] -= 1
        blocks.append(block_data)

    if num_blocks <= 1:
        ped_num = 0
    else:
        ped_num = min(random.randint(2, 5), num_blocks - 1)


    for i in range(ped_num):
        # ---------- 先生成底座 ----------
        if settings.ROT_DISCRETE is False:
            new_rotation = (0, 0, random.uniform(rot_range[0], rot_range[1]))
        else:
            new_rotation = (0, 0, random.choice(rot_range))
            
        new_position = random.choice(get_block_positions(
            [block for block in blocks if block["used"]],
            heightmap,
            collisiondetector,
            blocks[i]["size"],
            new_rotation,
            1,
        ))

        heightmap.update_heightmap( new_position, blocks[i]["size"], new_rotation)

    #     else:
    #         if settings.ROT_DISCRETE is False:
    #             new_rotation = (0, 0, random.uniform(rot_range[0], rot_range[1]))
    #         else:
    #             new_rotation = (0, 0, random.choice(rot_range))

    #         new_position = get_block_position(
    #             blocks_data,
    #             heightmap,
    #             collisiondetector,
    #             blocks[i]["size"],
    #             new_rotation,
    #         )

        blocks[i]["position"] = new_position
        blocks[i]["rotation"] = new_rotation
        blocks[i]["used"] = True
        # blocks_data.append(blocks[i])

    # 所有方块生成完毕后，将整体在水平面内平移，使总质心尽量靠近原点 (0, 0)

    _, blocks_data = tower_building_planner(
        blocks,
        CollapsePredictor(),
    )

    set_blocks_data(blocks_data, ped_num)

    return blocks_data, ped_num

def main():
    # 确保当前目录在 sys.path 中，以便 Blender 能找到模块
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if dir_path not in sys.path:
        sys.path.append(dir_path)

    # Parse command line arguments
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    if len(argv) < 1:
        raise ValueError("No config file provided.")

    config_path = argv[0]
    config = settings.load_scene_config(config_path)

    config_num_colors = {}
    for key, value in config["Scene"]["num_colors"].items():
        config_num_colors[key] = value

    for i in range(5):

        clear_scene()

        blocks_data, ped_num = generate_blocks_data(config)

        setup_render(resolution_x=128, resolution_y=128, samples=16)

        create_mesh("PLANE")

        print(blocks_data)
        for block_data in blocks_data:
            create_mesh("BLOCK", block_data)

        setup_camera()
        setup_light()

        # no_physics_render(i, config_num_colors)
        physics_render(i, ped_num, config)

        p = simple_render()
        
        predictor = CollapsePredictor()

        print("predict ",predictor.predict(p))
        clear_scene()


if __name__ == "__main__":
    main()

