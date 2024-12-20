import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import t

def torch_mean_confidence_interval(data, confidence=0.95):
    a = torch.tensor(data, dtype=torch.float32)
    n = a.numel()
    m = torch.mean(a)
    se = torch.std(a, unbiased=True) / torch.sqrt(torch.tensor(n - 1))

    # 根据样本量和置信水平计算 t 分数
    if n > 1:
        df = n - 1
        t_score = t.ppf((1 + confidence) / 2, df=df)  # 双尾 t 分布
    else:
        raise ValueError("Sample size must be greater than 1")
    h = t_score * se
    return m.item(), h.item()

class CNNEncoder(nn.Module):
    """CNN编码器，用于提取特征"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # 展平输出，以便与RelationNetwork的第一个全连接层匹配
        out = out.view(out.size(0), -1)
        return out  # 返回展平后的特征

class RelationNetwork(nn.Module):
    """关系网络，用于计算关系得分"""
    def __init__(self, input_size, hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)  # 展平输出
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))  # 使用sigmoid作为最后一层的激活函数
        return out
