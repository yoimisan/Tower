import yaml
import random
import numpy as np

# 默认全局变量
SEED = 42
INTERSECTION_THRESHOLD = 0.1
FATNESS = 0.5
NUM_SCENES = 1
VIDEO_LEN = 6
FPS = 30
DEGREE = 10
POINT = None
PROJECTION_X = [-1.5, 2.5]
PROJECTION_Y = [-1.5, 1.5]
ROT_DISCRETE = False
OUTPUT_PATH = ""
RENDER_VIDEO = True
SAVE_LAST_FRAME_IMAGE = False
# 是否额外渲染物理开始前的第一帧静态图
SAVE_FIRST_FRAME_IMAGE = False
# 是否导出整个模拟过程中的“所有帧”PNG 序列（共 VIDEO_LEN * FPS 张）
SAVE_ALL_FRAMES_IMAGES = False

# 新增：控制有多少方块优先“堆叠”在已有方块上，从而形成多层结构 / T 形结构
# 取值范围 [0, 1]，0 表示完全不用堆叠逻辑，1 表示只用堆叠逻辑
STACK_ON_EXISTING_PROB = 0.5


def load_scene_config(yml_path="configs/config.yml"):
    global SEED, INTERSECTION_THRESHOLD, FATNESS, NUM_SCENES, VIDEO_LEN, FPS
    global DEGREE, POINT, PROJECTION_X, PROJECTION_Y, ROT_DISCRETE, OUTPUT_PATH, RENDER_VIDEO, SAVE_LAST_FRAME_IMAGE
    global SAVE_FIRST_FRAME_IMAGE, SAVE_ALL_FRAMES_IMAGES
    global STACK_ON_EXISTING_PROB

    with open(yml_path, "r") as f:
        config = yaml.safe_load(f)

    SEED = config["General"].get("SEED", 42)
    INTERSECTION_THRESHOLD = config["General"].get("INTERSECTION_THRESHOLD", 0.01)
    FATNESS = config["General"].get("FATNESS", 0.5)

    NUM_SCENES = config["General"].get("NUM_SCENES", 1)

    VIDEO_LEN = config["General"].get("VIDEO_LEN", 6)
    FPS = config["General"].get("FPS", 30)

    DEGREE = config["General"].get("DEGREE", 10)
    POINT = config["General"].get("POINT", None)

    PROJECTION_X = config["General"].get("PROJECTION_X", [-1.5, 2.5])
    PROJECTION_Y = config["General"].get("PROJECTION_Y", [-1.5, 1.5])

    ROT_DISCRETE = config["General"].get("ROT_DISCRETE", False)

    OUTPUT_PATH = config["General"].get("OUTPUT_PATH")
    RENDER_VIDEO = config["General"].get("RENDER_VIDEO", True)
    SAVE_LAST_FRAME_IMAGE = config["General"].get("SAVE_LAST_FRAME_IMAGE", False)
    SAVE_FIRST_FRAME_IMAGE = config["General"].get("SAVE_FIRST_FRAME_IMAGE", False)
    SAVE_ALL_FRAMES_IMAGES = config["General"].get("SAVE_ALL_FRAMES_IMAGES", False)

    # 场景多样性相关：从配置里读取堆叠概率（可选）
    STACK_ON_EXISTING_PROB = config["General"].get("STACK_ON_EXISTING_PROB", 0.5)

    random.seed(SEED)
    np.random.seed(SEED)
    return config
