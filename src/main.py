import sys
import os
import random
import math
import ast
import numpy as np

sys.path.append(os.path.dirname(__file__))

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
)


# 将 get_block_position 和 generate_blocks_data 放在这里，或者单独再开一个 logic.py
def get_block_position(
    existing_blocks,
    heightmap,
    collisiondetector,
    new_size,
    new_rot,
    red_or_green,
    flag=0,
):
    """
    Generate a block's position.
    Args:
        existing_blocks: dic
        heightmap
        collisiondetector
        size: size of the current block
        new_rot: rotation of the current block
        red_or_green: 'red' -> negative
        flag: if it's pedestal then flag equals to 1
    """
    valid_positions = heightmap.get_valid_positions(
        new_size, new_rot, flag, red_or_green
    )
    if not valid_positions:
        raise ValueError("No valid positions available for the block.")
    while valid_positions:
        if np.random.uniform(0.0, 1.0) < settings.FATNESS:
            position = valid_positions[0]
        else:
            position = random.choice(valid_positions)
        valid_positions.remove(position)
        if not collisiondetector.check_block_collision(
            existing_blocks, position, new_size, new_rot
        ):
            heightmap.update_heightmap(position, new_size, new_rot)
            return position
    raise ValueError(
        "No valid position found for the block after checking all options."
    )


def generate_blocks_data(config, heightmap, collisiondetector, red_or_green):
    """
    Generate blocks data.
    Args:
        config: dictionary from yaml
        heightmap
        collisiondetector
        red_or_green: string 'red' or 'green'. If red, more x are negative.
    """
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
    mat = config["Scene"]["material"]  #'wood'

    # pedestal 数量限制：保证至少有 1 个非底座方块
    # 原本范围是 [2, 5]，这里再与 num_blocks-1 取最小值
    if num_blocks <= 1:
        ped_num = 0
    else:
        ped_num = min(random.randint(2, 5), num_blocks - 1)

    for i in range(num_blocks):
        # ---------- 先生成底座 ----------
        if i < ped_num:
            if settings.ROT_DISCRETE is False:
                new_rotation = (0, 0, random.uniform(rot_range[0], rot_range[1]))
            else:
                new_rotation = (0, 0, random.choice(rot_range))

            new_position = get_block_position(
                blocks_data,
                heightmap,
                collisiondetector,
                pedestal_size,
                new_rotation,
                red_or_green,
                1,
            )
            block_size = pedestal_size

        # ---------- 其余方块：部分随机放置，部分优先“堆叠”在已有方块上 ----------
        else:
            # 按数量筛选还可用的尺寸
            candidate_sizes = [s for s, c in size_dic.items() if c > 0]
            if not candidate_sizes:
                # 理论上不应该出现；兜底避免 KeyError
                candidate_sizes = all_sizes
            new_size = random.choice(candidate_sizes)

            if settings.ROT_DISCRETE is False:
                new_rotation = (0, 0, random.uniform(rot_range[0], rot_range[1]))
            else:
                new_rotation = (0, 0, random.choice(rot_range))

            # 决定是否尝试“堆叠”在已有方块上
            use_stack = (
                len(blocks_data) > 0
                and random.random() < settings.STACK_ON_EXISTING_PROB
            )

            if use_stack:
                stacked_position = None
                # 多尝试几次，找一个合适的支撑方块
                for _ in range(10):
                    support_block = random.choice(blocks_data)
                    support_top_z = (
                        support_block["position"][2] + support_block["size"][2] / 2.0
                    )
                    # 支撑平面 z= support_top_z 必须已经在 heightmap 里
                    if support_top_z not in heightmap.height:
                        continue

                    base_x, base_y = (
                        support_block["position"][0],
                        support_block["position"][1],
                    )
                    # 轻微水平扰动，产生 T 形 / 偏心结构，但保证仍有较大重叠面积
                    dx = np.random.uniform(-new_size[0] * 0.25, new_size[0] * 0.25)
                    dy = np.random.uniform(-new_size[1] * 0.25, new_size[1] * 0.25)
                    candidate_pos = (
                        base_x + dx,
                        base_y + dy,
                        support_top_z + new_size[2] / 2.0,
                    )

                    if not collisiondetector.check_block_collision(
                        blocks_data, candidate_pos, new_size, new_rotation
                    ):
                        # 手动更新 heightmap（与 get_block_position 内部逻辑保持一致）
                        heightmap.update_heightmap(
                            candidate_pos, new_size, new_rotation
                        )
                        stacked_position = candidate_pos
                        break

                if stacked_position is not None:
                    new_position = stacked_position
                else:
                    # 如果多次尝试堆叠失败，回退到原来的随机放置逻辑
                    new_position = get_block_position(
                        blocks_data,
                        heightmap,
                        collisiondetector,
                        new_size,
                        new_rotation,
                        red_or_green,
                    )
            else:
                # 保持原有“在斜面上随机找位置”的逻辑
                new_position = get_block_position(
                    blocks_data,
                    heightmap,
                    collisiondetector,
                    new_size,
                    new_rotation,
                    red_or_green,
                )

            block_size = new_size

        block_data = {
            "index": i,
            "color": random.choice(
                [key for key in color_dic.keys() if color_dic[key] > 0]
            ),
            "material": mat,
            "size": block_size,
            "position": new_position,
            "rotation": new_rotation,
        }

        color_dic[block_data["color"]] -= 1
        size_dic[block_data["size"]] -= 1
        blocks_data.append(block_data)

    # 所有方块生成完毕后，将整体在水平面内平移，使总质心尽量靠近原点 (0, 0)
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

    return blocks_data, ped_num


def get_final_tilt_color(block_positions, red_or_green, ped_num):
    """
    Args:
        block_positions: list, positions for all blocks after physics simulation
    """
    count = 0
    for p in block_positions:
        if red_or_green == "green":
            if p[0] >= 0:
                count += 1
        else:
            if p[0] <= 0:
                count += 1
    if count >= (len(block_positions) - ped_num) // 2:
        return red_or_green
    else:
        print("reverse")
        if red_or_green == "green":
            return "red"
        else:
            return "green"


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

    total_scenes = settings.NUM_SCENES
    print(f"Total scenes to generate: {total_scenes}")

    for i in range(total_scenes):
        clear_scene()

        heightmap = Heightmap()
        collisiondetector = CollisionDetector()

        # 注意这里 settings.RED_OR_GREEN
        blocks_data, ped_num = generate_blocks_data(
            config, heightmap, collisiondetector, settings.RED_OR_GREEN
        )

        setup_render()

        create_mesh("PLANE")

        for block_data in blocks_data:
            create_mesh("BLOCK", block_data)

        setup_camera()
        setup_light()

        # no_physics_render(i, config_num_colors)
        physics_render(i, ped_num, config)
        print(f"Finish creating scene {i + 1}/{total_scenes}.")


if __name__ == "__main__":
    main()
