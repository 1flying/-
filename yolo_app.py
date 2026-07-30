import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import time
import tempfile
import os
from pathlib import Path

# ===================== 页面配置 =====================
st.set_page_config(
    page_title="YOLOv8 飞机跑道识别系统",
    layout="wide",
    page_icon="✈️"
)

# 自定义美化 CSS
st.markdown("""
<style>
    .main { background-color: #0b1020; color: white; }
    .stButton>button { 
        background: linear-gradient(90deg, #4361ee, #3a0ca3); 
        color: white; border-radius: 10px; height: 3em; font-size: 16px; 
        border: none;
    }
    .stop-button>button { background: linear-gradient(90deg, #e0218a, #f72585); }
    .sidebar .stRadio>label { font-size: 18px; font-weight: bold; }
    div[data-testid="stMetricValue"] { color: #4cc9f0; }
    .log-box { background: #121a33; padding: 15px; border-radius: 10px; height: 250px; overflow-y: auto; }
</style>
""", unsafe_allow_html=True)

# ===================== 加载模型 =====================
@st.cache_resource
def load_model():
    return YOLO("runs/detect/result/aircraft/yolov8-loss/SIOU-LeakyReLU/weights/best.pt")  # 你训练好的权重

model = load_model()

# ===================== 侧边栏：控制面板 =====================
with st.sidebar:
    st.title("✈️ 识别控制面板")
    st.markdown("---")

    # 模型配置
    st.subheader("📦 模型配置")
    conf = st.slider("置信度阈值", 0.1, 1.0, 0.80, 0.01)

    st.markdown("---")

    # 输入模式
    st.subheader("📷 输入源")
    mode = st.radio("选择模式", ["上传图片", "上传视频"], horizontal=True)

    st.markdown("---")

    # 控制按钮
    col1, col2 = st.columns(2)
    with col1:
        run = st.button("▶️ 开始识别")
    with col2:
        stop = st.button("⏹️ 停止识别", type="primary", help="停止识别")

    st.markdown("---")

    # 速度控制
    st.subheader("⚙️ 识别参数")
    speed = st.slider("播放速度", 0.25, 2.0, 1.0, 0.25)

# ===================== 主界面 =====================
st.title("🚀 YOLOv8 飞机跑道智能识别系统")
st.markdown("<h4 style='color:#a0aec0'>实时目标检测 | 跑道/标识/飞机识别</h4>", unsafe_allow_html=True)
st.markdown("---")

# 上传区域
uploaded_file = None
if mode == "上传图片":
    uploaded_file = st.file_uploader("选择图片", type=["jpg", "png", "jpeg"])
else:
    uploaded_file = st.file_uploader("选择视频", type=["mp4", "avi", "mov"])

# 画面展示区域
col_ori, col_det = st.columns(2)
with col_ori:
    st.subheader("📸 原始画面")
    ori_placeholder = st.empty()
with col_det:
    st.subheader("✅ 识别结果")
    det_placeholder = st.empty()

# 日志与状态区域
st.markdown("---")
st.subheader("📊 实时识别日志")
log_container = st.container()
log_area = log_container.empty()
log_text = ""

# 状态指标
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
with metric_col1:
    st.metric("当前帧 / 总帧数", "0 / 0")
with metric_col2:
    st.metric("单帧耗时", "0.000s")
with metric_col3:
    st.metric("置信度阈值", f"{conf:.2f}")
with metric_col4:
    st.metric("识别目标数", "0")

# ===================== 识别逻辑 =====================
def append_log(msg):
    global log_text
    log_text = f"[{time.strftime('%H:%M:%S')}] {msg}\n" + log_text
    log_area.markdown(f"<div class='log-box'>{log_text}</div>", unsafe_allow_html=True)

# 图片识别
if mode == "上传图片" and uploaded_file is not None and run:
    img = Image.open(uploaded_file)
    img_np = np.array(img)
    ori_placeholder.image(img, caption="原始图片", use_column_width=True)

    # 推理
    start = time.time()
    res = model(img_np, conf=conf)
    cost = time.time() - start

    # 绘制结果
    res_img = res[0].plot()
    res_img_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
    det_placeholder.image(res_img_rgb, caption="识别结果", use_column_width=True)

    # 日志
    append_log(f"图片识别完成 | 耗时：{cost:.3f}s | 目标数：{len(res[0].boxes)}")

# 视频识别
if mode == "上传视频" and uploaded_file is not None and run:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0

    append_log(f"视频加载成功 | 总帧数：{total_frames}")

    try:
        while cap.isOpened() and not stop:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            ori_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ori_placeholder.image(ori_rgb, caption=f"原始视频 | 帧：{frame_idx}", use_column_width=True)

            # 推理
            s = time.time()
            res = model(frame, conf=conf)
            cost = time.time() - s

            # 结果图
            det_frame = res[0].plot()
            det_rgb = cv2.cvtColor(det_frame, cv2.COLOR_BGR2RGB)
            det_placeholder.image(det_rgb, caption=f"识别结果 | 帧：{frame_idx}", use_column_width=True)

            # 更新指标
            metric_col1.metric("当前帧 / 总帧数", f"{frame_idx} / {total_frames}")
            metric_col2.metric("单帧耗时", f"{cost:.3f}s")
            metric_col4.metric("识别目标数", f"{len(res[0].boxes)}")

            # 日志
            append_log(f"帧 {frame_idx}/{total_frames} | 耗时：{cost:.3f}s | 目标：{len(res[0].boxes)}")

            # 播放速度
            time.sleep(0.03 / speed)

        append_log("✅ 视频识别已结束")
    except:
        append_log("❌ 识别已停止")

    cap.release()

