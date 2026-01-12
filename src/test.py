# import torch

# print(torch.cuda.is_available())

# 导入内置模块（无需pip安装）
import sys
import platform

# 方式1：sys.version - 输出详细版本信息（包含版本号、编译器、系统）
print("=== 详细版本信息 ===")
print(sys.version)

# 方式2：sys.version_info - 输出结构化版本信息（便于程序判断版本）
print("\n=== 结构化版本信息 ===")
print(f"Python主版本：{sys.version_info.major}")
print(f"Python次版本：{sys.version_info.minor}")
print(f"Python微版本：{sys.version_info.micro}")

# 方式3：platform.python_version() - 输出简洁版本号（字符串格式，便于存储）
print("\n=== 简洁版本号 ===")
print(platform.python_version())