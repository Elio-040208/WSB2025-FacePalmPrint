import torch
from model import MobileNetV1  # 替换为你的模型文件路径
import torchvision.transforms as transforms
from PIL import Image


def test_single_image(image_path, model_path, num_classes=291):
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型
    model = MobileNetV1(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 替换为你的模型输入尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 替换为你的数据归一化参数
    ])

    # 加载图像
    try:
        image = Image.open(image_path).convert('RGB')  # 确保图像为 RGB 模式
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # 预处理图像
    input_image = transform(image).unsqueeze(0).to(device)  # 增加 batch 维度

    # 预测
    with torch.no_grad():
        output = model(input_image)
        _, predicted_label = torch.max(output, dim=1)  # 获取预测类别
        predicted_label = predicted_label.item() + 1  # 加1以匹配文件夹编号

    print(f"Predicted Label: {predicted_label}")
    return predicted_label


# 示例调用
if __name__ == "__main__":
    image_path = "resized_hand_image.jpg"  # 替换为你要测试的图像路径
    model_path = "model_best.pth"  # 替换为你的模型文件路径
    predicted_label = test_single_image(image_path, model_path, num_classes=291)
