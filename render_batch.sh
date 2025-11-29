#!/bin/bash
#SBATCH --partition=IAI_SLURM_3090
#SBATCH --job-name=render
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=singlegpu
#SBATCH --cpus-per-task=10
#SBATCH --time=2-00:00:00
#SBATCH --output=render_%j.log
#SBATCH --error=render_%j.err
#SBATCH --mail-user=2300017477@stu.pku.edu.cn
#SBATCH --mail-type=FAIL,END

# 创建输出目录
mkdir -p TowerTaskVideos

# 配置文件列表
config_files=(
  "configs/config_red_17.yml"
  "configs/config_red_18.yml"
  "configs/config_red_19.yml"
  "configs/config_red_20.yml"
  "configs/config_red_21.yml"
  "configs/config_red_22.yml"
  "configs/config_red_23.yml"
  "configs/config_red_24.yml"
  "configs/config_red_25.yml"
  "configs/config_red_26.yml"
  "configs/config_red_27.yml"
  "configs/config_red_28.yml"
  "configs/config_red_29.yml"
  "configs/config_red_30.yml"
  "configs/config_green_17.yml"
  "configs/config_green_18.yml"
  "configs/config_green_19.yml"
  "configs/config_green_20.yml"
  "configs/config_green_21.yml"
  "configs/config_green_22.yml"
  "configs/config_green_23.yml"
  "configs/config_green_24.yml"
  "configs/config_green_25.yml"
  "configs/config_green_26.yml"
  "configs/config_green_27.yml"
  "configs/config_green_28.yml"
  "configs/config_green_29.yml"
  "configs/config_green_30.yml"
)

# 处理所有配置文件
for config_file in "${config_files[@]}"; do
  echo "开始处理配置文件: $config_file"
  
  # 检查配置文件是否存在
  if [ ! -f "$config_file" ]; then
    echo "错误: 配置文件 $config_file 不存在"
    continue
  fi
  
  # 从配置文件名中提取信息（可选，用于日志记录）
  # 假设文件名格式为 configs/config_{color}_{num_blocks}.yml
  filename=$(basename "$config_file")
  color=$(echo "$filename" | cut -d'_' -f2)
  num_blocks=$(echo "$filename" | cut -d'_' -f3 | cut -d'.' -f1)
  
  echo "处理 $color 颜色的 $num_blocks 个积木块场景"
  
  # 创建输出目录（如果需要）
  output_dir="TowerTaskVideos/${color}/${num_blocks}"
  mkdir -p "$output_dir"
  
  # 将配置文件复制到代码期望的固定位置
  # 请将下面的路径替换为您代码中使用的实际路径
  fixed_config_path="/data/qingjingfan/Documents/TowerTask/configs/config.yml"  # 请替换为实际路径
  cp "$config_file" "$fixed_config_path"
  
  # 运行Blender渲染任务
  blender -b -P obvious_script.py  # 请替换为您的脚本名称
  
  # 检查命令执行状态
  if [ $? -ne 0 ]; then
    echo "错误: 处理配置文件 $config_file 时渲染失败"
    exit 1
  fi
  
  echo "完成配置文件 $config_file 的渲染"
done

echo "所有渲染任务已完成"