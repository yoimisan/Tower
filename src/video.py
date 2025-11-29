import cv2
import os

def images_to_video(image_folder, output_video, fps=30, size=None):
    """
    将图片序列合成为MP4视频
    
    参数:
        image_folder: 图片所在文件夹路径
        output_video: 输出视频路径（如 'output.mp4'）
        fps: 视频帧率，默认30帧/秒
        size: 视频尺寸 (宽度, 高度)，默认使用第一张图片的尺寸
    """
    # 获取文件夹中所有图片的路径
    images = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    # 按文件名中的数字排序（假设文件名是纯数字或数字开头，如 '1.png', '2.jpg', '003.png'）
    images.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    
    if not images:
        print("错误：未找到图片文件")
        return
    
    # 读取第一张图片获取尺寸
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"错误：无法读取图片 {first_image_path}")
        return
    
    # 设置视频尺寸
    if size is None:
        height, width = frame.shape[:2]
        size = (width, height)
    else:
        width, height = size
    
    # 定义视频编码器，使用MP4格式
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 生成mp4格式
    out = cv2.VideoWriter(output_video, fourcc, fps, size)
    
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"警告：跳过无法读取的图片 {image_path}")
            continue
        
        # 调整图片尺寸以匹配视频尺寸
        if (frame.shape[1], frame.shape[0]) != size:
            frame = cv2.resize(frame, size)
        
        out.write(frame)
        print(f"处理中：{image}", end='\r')
    
    # 释放资源
    out.release()
    cv2.destroyAllWindows()
    print(f"\n视频合成完成：{output_video}")

# 使用示例
if __name__ == "__main__":
    # 图片所在文件夹路径
    image_folder = r"C:\tmp"  # 替换为你的图片文件夹路径
    # 输出视频路径
    output_video = "output.mp4"
    # 帧率设置（根据需要调整，如15、24、30）
    fps = 30
    
    # 调用函数合成视频
    images_to_video(image_folder, output_video, fps)
