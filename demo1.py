import cv2


def resize_image_to_fixed_size(image_path, output_path, target_width, target_height):
    """
    读取图像并强制缩放到固定分辨率（190x217）

    参数：
        image_path (str): 原始图像路径
        output_path (str): 输出图像保存路径
        target_width (int): 目标宽度（190）
        target_height (int): 目标高度（217）
    """
    # 1. 读取原始图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片，请检查路径：{image_path}")

    # 2. 强制缩放图像到固定尺寸
    resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

    # 3. 保存缩放后的图像
    cv2.imwrite(output_path, resized_image)
    print(f"图像已缩放到 {target_width}x{target_height} 分辨率，并保存到：{output_path}")

    # 可视化结果
    cv2.imshow("Original Image", image)
    cv2.imshow("Resized Image", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 示例使用
image_path = "cyc.jpg"  # 替换为原始图片路径
output_path = "resized_hand_imagecyc.jpg"  # 保存缩放图像的路径
target_width = 190  # 目标宽度
target_height = 217  # 目标高度

resize_image_to_fixed_size(image_path, output_path, target_width, target_height)
