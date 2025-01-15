import torch.nn as nn

class MobileNetV1FeatureExtractor(nn.Module):
    def __init__(self):
        super(MobileNetV1FeatureExtractor, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 1),
            conv_dw(1024, 1024, 2),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        return self.features(x)


class MobileNetV1(nn.Module):
    def __init__(self, num_classes=291):
        super(MobileNetV1, self).__init__()
        # 单模态，使用一个特征提取器
        self.feature_extractor = MobileNetV1FeatureExtractor()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        # 通过单一特征提取器
        x = self.feature_extractor(x)

        # 全局平均池化
        branch = self.avgpool(x)
        branch = branch.view(branch.size(0), -1)

        # 分类器
        output = self.classifier(branch)
        return output
