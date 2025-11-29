import yaml
import random
import numpy as np

# 默认全局变量
SEED = 42
INTERSECTION_THRESHOLD = 0.01
FATNESS = 0.5
NUM_SCENES = 1
RED_OR_GREEN = None
VIDEO_LEN = 6
FPS = 30
DEGREE = 10
POINT = None
PROJECTION_X = [-1.5, 2.5]
PROJECTION_Y = [-1.5, 1.5]
ROT_DISCRETE = False
OUTPUT_PATH = ""

def load_scene_config(yml_path='configs/config.yml'):
    global SEED, INTERSECTION_THRESHOLD, FATNESS, NUM_SCENES, RED_OR_GREEN, VIDEO_LEN, FPS
    global DEGREE, POINT, PROJECTION_X, PROJECTION_Y, ROT_DISCRETE, OUTPUT_PATH
    
    with open(yml_path, 'r') as f:
        config = yaml.safe_load(f)

    SEED = config['General'].get("SEED", 42) 
    INTERSECTION_THRESHOLD = config['General'].get("INTERSECTION_THRESHOLD", 0.01)
    FATNESS = config['General'].get("FATNESS", 0.5)
    
    NUM_SCENES = config['General'].get("NUM_SCENES", 1)
    RED_OR_GREEN = config['General'].get("RED_OR_GREEN")
    
    VIDEO_LEN = config['General'].get("VIDEO_LEN", 6)
    FPS = config['General'].get("FPS", 30)
    
    DEGREE = config['General'].get("DEGREE", 10)
    POINT = config['General'].get("POINT", None)
    
    PROJECTION_X = config['General'].get("PROJECTION_X", [-1.5, 2.5])
    PROJECTION_Y = config['General'].get("PROJECTION_Y", [-1.5, 1.5])

    ROT_DISCRETE = config['General'].get("ROT_DISCRETE", False)
    
    OUTPUT_PATH = config['General'].get("OUTPUT_PATH")
    
    random.seed(SEED)
    np.random.seed(SEED)
    return config