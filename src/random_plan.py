from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import sys
import os
import random

import math
import ast


sys.path.append(os.path.dirname(__file__)) 


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
        # print(x.dtype)
        x0 =torch.from_numpy(x.transpose(2, 0, 1)).to(dtype=torch.float32).unsqueeze(dim=0) / 255.0
        x0 = x0.to(self.cfg.device)
        _, logit = self.model(x0)
        prob = torch.sigmoid(logit)
        pred = (prob >= 0.5).float()
        return logit.item()
    

def render_tower(blocks) -> np.ndarray:
    """调用Blender脚本渲染塔状态，返回图像（与你的已有代码集成）"""
    # 调用你已有的Blender渲染接口，传入node.blocks生成图像
    # 此处为简化实现，返回模拟图像
    clear_scene()

    setup_render(resolution_x=128, resolution_y=128, samples=8)

    create_mesh("PLANE")

    for block in blocks:
        create_mesh("BLOCK", block)

    setup_camera()
    setup_light()

    # no_physics_render(i, config_num_colors)
    p = simple_render()        
    clear_scene()
    
    return p



# def get_block_position(
#     existing_blocks,
#     heightmap,
#     collisiondetector,
#     new_size,
#     new_rot,
#     flag=0,
# ):
#     """
#     Generate a block's position.
#     Args:
#         existing_blocks: dic
#         heightmap
#         collisiondetector
#         size: size of the current block
#         new_rot: rotation of the current block
#         flag: if it's pedestal then flag equals to 1
#     """
#     valid_positions = heightmap.get_valid_positions(new_size, new_rot, flag)
#     if not valid_positions:
#         raise ValueError("No valid positions available for the block.")

#     while valid_positions:
#         if np.random.uniform(0.0, 1.0) < settings.FATNESS:
#             position = valid_positions[0]
#         else:
#             position = random.choice(valid_positions)
#         valid_positions.remove(position)
#         if not collisiondetector.check_block_collision(
#             existing_blocks, position, new_size, new_rot
#         ):
#             heightmap.update_heightmap(position, new_size, new_rot)
#             return position

#     # 如果所有候选位置都会与已有方块发生碰撞，则认为当前几何配置不可行，
#     # 直接报错让上层决定是否跳过该场景 / 重新采样，而不是强行允许方块重叠。
#     raise ValueError(
#         "All candidate positions collide with existing blocks; "
#         "no non-overlapping placement found for this block."
#     )


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


def generate_blocks_data(config, heightmap, collisiondetector):
    """
    Generate blocks data.
    Args:
        config: dictionary from yaml
        heightmap
        collisiondetector
    """
    collapsepredictor = CollapsePredictor()

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
        ped_num = min(random.randint(2, 4), num_blocks - 1)

    for i in range(num_blocks):
        # ---------- 先生成底座 ----------
        print(i)
        if i < ped_num:
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

        else:
            optim = blocks[i].copy()
            maxi = 10
            for j in range(50):
                if settings.ROT_DISCRETE is False:
                    new_rotation = (0, 0, random.uniform(rot_range[0], rot_range[1]))
                else:
                    new_rotation = (0, 0, random.choice(rot_range))
                    
                new_position = random.choice(get_block_positions(
                    blocks_data,
                    heightmap,
                    collisiondetector,
                    blocks[i]["size"],
                    new_rotation,
                ))

                blocks[i]["position"] = new_position
                blocks[i]["rotation"] = new_rotation

                temp_blocks = blocks_data.copy() + [blocks[i]]

                p = render_tower(temp_blocks) 

                pred = collapsepredictor.predict(p)

                if  pred < maxi:
                    maxi = pred
                    optim["position"] = new_position
                    optim["rotation"] = new_rotation
            
            new_position = optim["position"]
            new_rotation = optim["rotation"]

        heightmap.update_heightmap(new_position, blocks[i]["size"], new_rotation)
        blocks[i]["position"] = new_position
        blocks[i]["rotation"] = new_rotation
        
        blocks[i]["used"] = True
        
        blocks_data.append(blocks[i])

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

        heightmap = Heightmap()
        collisiondetector = CollisionDetector()

        blocks_data, ped_num = generate_blocks_data(
            config, heightmap, collisiondetector
        )

        setup_render(resolution_x=128, resolution_y=128, samples=16)

        create_mesh("PLANE")

        for block_data in blocks_data:
            create_mesh("BLOCK", block_data)

        setup_camera()
        setup_light()

        # no_physics_render(i, config_num_colors)
        physics_render(i, ped_num, config)

        p = simple_render()
        print(p.shape)

        predictor = CollapsePredictor()

        clear_scene()

        print("predict ",predictor.predict(p))

    

if __name__ == "__main__":
    main()
