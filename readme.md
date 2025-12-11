## 使用方式

- **命令行运行**
  - 在项目根目录下执行：  
    `blender -b -P src/main.py -- ./tmp/test_config.yml`
  - 需要将 `blender.exe` 所在目录加入系统 `PATH`。

- **配置文件路径**
  - 命令行最后一个参数是场景配置文件（默认示例：`tmp/test_config.yml`）。
  - 所有可调参数都写在该 yml 文件中，`src/settings.py` 会在启动时读取。

## General 段可调参数

- **SEED**：随机种子，固定后每次生成结果可复现。
- **INTERSECTION_THRESHOLD**：
  - 高度图上方块与支撑区域的最小重叠面积阈值。
  - 越大越“保守”，方块越难被放下（塔更稳定但位置更受限）。
- **FATNESS**：
  - 在候选位置列表中，选取“第一个位置”（即最低、最稳位置）的概率 \([0,1]\)。
  - 越大塔越集中、越“胖”；越小位置更分散。
- **NUM_SCENES**：一次性要生成的场景数量（循环调用生成与渲染）。
- **VIDEO_LEN**：视频总时长（秒）。
- **FPS**：视频帧率。
- **DEGREE**：
  - 斜坡的倾斜角度（度数）。
  - 同时影响：高度图的平面倾角、以及 Blender 中地面网格的旋转角。
- **POINT**：高度图中用于定义倾斜平面的一点（一般留空使用默认即可）。
- **PROJECTION_X / PROJECTION_Y**：
  - 在倾斜平面上采样候选位置时的 x / y 范围。
  - 可以通过缩小范围让塔更集中，放大范围让塔更分散。
- **ROT_DISCRETE**：
  - `False`：按连续区间采样旋转角度（见 `Scene.rot_range`）。
  - `True`：按一组离散角度采样（`Scene.rot_range` 里给角度列表）。
- **OUTPUT_PATH**：输出文件目录（视频和图片都会写到这里）。
- **RENDER_VIDEO**：是否渲染整段物理过程视频（mp4）。
- **SAVE_LAST_FRAME_IMAGE**：是否额外保存最后一帧的单张图片（塔倒完后的状态）。
- **SAVE_FIRST_FRAME_IMAGE**：是否额外保存物理模拟开始前的第一帧图片（初始塔形）。
- **STACK_ON_EXISTING_PROB**：
  - 非底座方块优先“堆叠在已有方块上”的概率 \([0,1]\)。 
  - 值越大，多层结构、T 形 / L 形结构出现得越频繁。

## Scene 段可调参数

- **num_blocks**：
  - 场景中方块总数量（包括底座 + 上层方块）。
  - 与 `sizes` 中所有数量之和保持一致。
- **num_colors**：
  - 每种颜色的数量（键为颜色名，值为个数）。
  - 总和应等于 `num_blocks`，颜色名需在 `src/constants.py` 的 `COLORS` 里存在。
- **sizes**：
  - 不同方块尺寸及其数量，键是 `(x, y, z)` 形状的元组字符串，值是该尺寸的个数。
  - 例如：  
    `(0.5, 0.5, 0.5): 4` 小方块；  
    `(0.5, 0.5, 1.5): 3` 竖直长条；  
    `(1.5, 0.5, 0.5): 3` 横向长条。
  - 代码会自动：
    - 选 z 最大的尺寸作为“底座”尺寸；
    - 其余方块在所有仍有余量的尺寸中随机抽取。
- **rot_range**：
  - 当 `ROT_DISCRETE=False` 时：`[min_deg, max_deg]`，在这个区间内连续随机角度（单位：度）。
  - 当 `ROT_DISCRETE=True` 时：`[deg1, deg2, ...]`，在这几个角度中离散采样。
- **num_materials**：
  - 用于“每个方块可以单独控制材料（按数量）”。
  - 形式与 `num_colors` 类似，为一个字典：键是材质名，值是该材质在场景中的方块数量。
  - 例如：`{"wood": 10, "metal": 5, "glass": 4}`。
  - 所有数量之和必须等于 `num_blocks`，同时所有材质名需存在于 `MATERIALS` 中。

## 输出文件命名约定

- **视频文件**（如果 `RENDER_VIDEO=True`）：  
  - `OUTPUT_PATH/{index}_p_{collapse_state}.mp4`  
  - `index` 为场景编号（从 0 开始），`collapse_state` 为塔的状态：`collapsed` / `stable`。
- **最后一帧图片**（如果 `SAVE_LAST_FRAME_IMAGE=True`）：  
  - `OUTPUT_PATH/{index}_p_{collapse_state}.png`
- **第一帧图片（初始塔形）**（如果 `SAVE_FIRST_FRAME_IMAGE=True`）：  
  - `OUTPUT_PATH/{index}_f_init.png`
