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
    ]  # {(0.5, 0.5, 1.5): 16, (1.5, 0.5, 0.5): 13}#8,9
    size_dic = {}
    for key, value in ori_size_dic.items():
        key_t = ast.literal_eval(key)
        size_dic[key_t] = value
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
        if i < ped_num:
            if settings.ROT_DISCRETE == False:
                new_rotation = (0, 0, random.uniform(rot_range[0], rot_range[1]))
            else:
                new_rotation = (0, 0, random.choice(rot_range))
            new_position = get_block_position(
                blocks_data,
                heightmap,
                collisiondetector,
                (0.5, 0.5, 1.5),
                new_rotation,
                red_or_green,
                1,
            )
            block_data = {
                "index": i,
                "color": random.choice(
                    [key for key in color_dic.keys() if color_dic[key] > 0]
                ),
                "material": mat,
                "size": (0.5, 0.5, 0.5),
                "position": new_position,
                "rotation": new_rotation,
            }
        else:
            if i % 2 == 1 and size_dic[(0.5, 0.5, 0.5)] > 0:
                new_size = (0.5, 0.5, 0.5)
            else:
                new_size = (0.5, 0.5, 0.5)
            if settings.ROT_DISCRETE == False:
                new_rotation = (0, 0, random.uniform(rot_range[0], rot_range[1]))
            else:
                new_rotation = (0, 0, random.choice(rot_range))
            new_position = get_block_position(
                blocks_data,
                heightmap,
                collisiondetector,
                new_size,
                new_rotation,
                red_or_green,
            )
            block_data = {
                "index": i,
                "color": random.choice(
                    [key for key in color_dic.keys() if color_dic[key] > 0]
                ),
                "material": mat,
                "size": new_size,
                "position": new_position,
                "rotation": new_rotation,
            }
        color_dic[block_data["color"]] -= 1
        size_dic[block_data["size"]] -= 1
        blocks_data.append(block_data)
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
