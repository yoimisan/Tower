COLORS = {
    "red": [1, 0, 0, 1],
    "green": [0, 1, 0, 1],
    "blue": [0, 0.25, 1, 1],
    "yellow": [1, 1, 0, 1],
    "purple": [1, 0, 1, 1],
    "cyan": [0, 1, 1, 1],
    "orange": [1, 0.5, 0, 1],
    "white": [1, 1, 1, 1],
    "gray": [0.5, 0.5, 0.5, 1],
    "black": [0, 0, 0, 1]
    }

MATERIALS = {
    'wood': {
        'Roughness': 0.9,
        'Metallic': 0.05,
        'IOR': 2.0,
        'Specular IOR Level': 0.5  # 原 Specular
    },
    'metal': {
        'Roughness': 0.3,      # 改为大写
        'Metallic': 0.8,       # 改为大写
        'Specular IOR Level': 0.5
    },
    'plastic': {
        'Roughness': 0.5,
        'Metallic': 0.1,
        'Specular IOR Level': 0.6
    },
    'glass': {
        'Roughness': 0.02,
        'Metallic': 0.0,
        'Specular IOR Level': 0.9,
        'Transmission Weight': 0.95, # 原 Transmission 改为 Transmission Weight
        'IOR': 1.52           # 原 ior 改为大写 IOR
    },
    'rubber': {
        'Roughness': 0.8,
        'Metallic': 0.0,
        'Specular IOR Level': 0.1
    },
    'ceramic': {
        'Roughness': 0.1,
        'Metallic': 0.0,
        'Specular IOR Level': 0.8
    }
}