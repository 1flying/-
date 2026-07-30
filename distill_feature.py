from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================== 蒸馏配置（只改这里） =====================
STUDENT = 'yolov8s.pt'  # 轻量学生模型
TEACHER = 'runs/detect/result/aircraft/yolov8-loss/SIOU-LeakyReLU/weights/best.pt'  # 训练好的教师模型
DATA = 'data.yaml'  # 你的数据集配置
EPOCHS = 100
BATCH = 16
DEVICE = 0
ALPHA = 0.5  # 蒸馏损失权重


# ==============================================================

# 特征对齐模块（解决学生/教师通道数不一致问题）
class FeatureAlign(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)


def run_feature_distill():
    # 1. 加载模型
    student = YOLO(STUDENT)
    teacher = YOLO(TEACHER).model.to(DEVICE)
    teacher.eval()

    # 冻结教师模型
    for param in teacher.parameters():
        param.requires_grad = False

    # 2. 获取 YOLOv8 多尺度特征层（3个核心检测层）
    student_model = student.model
    feat_indices = [15, 18, 21]  # YOLOv8 特征输出层索引
    student_feats = []
    teacher_feats = []

    # 3. 注册钩子提取中间特征
    def student_hook(module, input, output):
        student_feats.append(output)

    def teacher_hook(module, input, output):
        teacher_feats.append(output)

    # 给学生/教师对应特征层注册钩子
    for idx in feat_indices:
        student_model.model[idx].register_forward_hook(student_hook)
        teacher.model[idx].register_forward_hook(teacher_hook)

    # 4. 特征对齐层（适配通道数）
    align_layers = nn.ModuleList([
        FeatureAlign(128, 128).to(DEVICE),  # 小尺度
        FeatureAlign(256, 256).to(DEVICE),  # 中尺度
        FeatureAlign(512, 512).to(DEVICE)  # 大尺度
    ])

    # 5. 重写前向传播，加入特征蒸馏损失
    original_forward = student_model.forward

    def forward_with_distill(x):
        student_feats.clear()
        # 学生前向
        s_out = original_forward(x)

        # 教师前向（无梯度）
        with torch.no_grad():
            teacher_feats.clear()
            teacher(x)

        # 计算多尺度特征蒸馏损失（MSE）
        distill_loss = 0.0
        for i, (s_feat, t_feat) in enumerate(zip(student_feats, teacher_feats)):
            # 特征对齐 + 损失计算
            s_feat_aligned = align_layers[i](s_feat)
            distill_loss += F.mse_loss(s_feat_aligned, t_feat.detach())

        # 蒸馏损失融入总损失
        s_out[0] += ALPHA * distill_loss
        return s_out

    student_model.forward = forward_with_distill

    # 6. 启动训练
    student.train(
        data=DATA,
        epochs=EPOCHS,
        batch=BATCH,
        device=DEVICE,
        project="runs_feature_distill",
        name="yolov8_feature_distill"
    )


if __name__ == '__main__':
    run_feature_distill()