from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.nn.functional as nnf


def run_distill():
    # ===================== 【只改这里】 =====================
    STUDENT = 'yolov8s.pt'  # 学生模型（自动下载，不用训练）
    TEACHER = 'runs/detect/result/aircraft/yolov8-loss/SIOU-LeakyReLU/weights/best.pt'  # 你的模型（教师，必须有效）
    DATA = 'data.yaml'  # 你的数据集
    EPOCHS = 50
    BATCH = 16
    DEVICE = 0
    ALPHA = 0.5
    # ======================================================

    # 加载模型
    student = YOLO(STUDENT)
    teacher = YOLO(TEACHER).model.to(DEVICE)
    teacher.eval()

    # 冻结教师
    for p in teacher.parameters():
        p.requires_grad = False

    # 保存原始前向函数
    original_forward = student.model.forward

    def new_forward(x):
        # 学生预测
        s_out = original_forward(x)

        # 教师预测
        with torch.no_grad():
            t_out = teacher(x)

        # 蒸馏损失（输出层对齐）
        distill_loss = 0
        for s, t in zip(s_out, t_out):
            if s.shape == t.shape:
                distill_loss += nnf.mse_loss(s, t)

        # 存到张量里，让训练器可以拿到
        s_out[0] = s_out[0] + ALPHA * distill_loss
        return s_out

    # 替换前向函数
    student.model.forward = new_forward

    # 开始训练（官方原生，最稳定）
    student.train(
        data=DATA,
        epochs=EPOCHS,
        batch=BATCH,
        device=DEVICE,
        project="runs_distill",
        name="final_student"
    )


if __name__ == '__main__':
    run_distill()