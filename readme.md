# 数据集构建

使用 Blender 物理引擎生成方块塔倒塌数据集，输出：固定视角图像帧序列、塔是否倒塌标签、倒塌过程序列、以及每个方块在整个模拟过程中的三维轨迹（位置 + 姿态）。

---

## 快速运行

1. 安装依赖（示例）：`pip install numpy pyyaml shapely`
2. 设置 Blender 路径（PowerShell 示例，可按需修改版本号）：

```powershell
$env:BLENDER_PATH = "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"
```

3. 在项目根目录执行生成脚本：

- **正方体小塔（方块数从 4 往上扫）**

```powershell
python src\generate_dataset.py
```

- **多种高度 / 尺寸 / 材质的复杂塔**

```powershell
python src\generate_complex_dataset.py
```

（如需单独调试某个 yml，可用 `blender -b main.blend -P src\main.py -- <config.yml>`。）

---

## 输出结构

以某个配置的 `General.OUTPUT_PATH = "output/cubes_4"` 为例：

- `output/cubes_4/0/`, `output/cubes_4/1/`, ...：每个场景一个文件夹
  - `meta.json`：包含
    - `collapse_state`：`"collapsed"` / `"stable"`
    - `per_frame_hit_counts`：每帧有多少非底座方块砸到地面
    - `state_sequence`：每帧每个方块的 `(x, y, z)` 与 `(rx, ry, rz)`
  - `frame_0001.png` ... `frame_XXXX.png`（若 `SAVE_ALL_FRAMES_IMAGES=True`）：完整帧序列
  - `f_init.png`（可选）：初始塔形
  - `p_collapsed.png` / `p_stable.png`（可选）：最后一帧静态图
  - `p_collapsed.mp4` / `p_stable.mp4`（可选）：整段视频
---

# 基线模型

## 1. Static CNN Baseline (`cnn.py`)

### 输出
* 二分类概率：`P(collapse)`

### 快速运行
```bash
python src/cnn.py
```
---

## 2. CNN + Transformer Temporal Reasoning Model (`transformer.py`)


### 输出

* 二分类概率：`P(collapse)`
* 预测视频帧序列

### 训练

```bash
python src/predict_tf.py
```

### 评估

```bash
python src/eval_tf.py --model-path ./model_a.pt --data-root ./all_data --output-dir ./eval_outputs
```

---
