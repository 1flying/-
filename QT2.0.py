import sys
import os
import time
import numpy as np
from pathlib import Path
import cv2
from ultralytics import YOLO
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QSlider, QComboBox,
    QTextEdit, QSplitter, QDoubleSpinBox, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QPalette, QBrush, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer


# 视频播放+识别一体化线程（简化逻辑，确保稳定性）
class VideoDetectionThread(QThread):
    # 信号定义
    original_frame_signal = pyqtSignal(QImage)  # 原图帧
    detected_frame_signal = pyqtSignal(QImage)  # 识别后帧
    progress_signal = pyqtSignal(int)  # 进度更新
    info_signal = pyqtSignal(str, float)  # 识别信息、耗时
    error_signal = pyqtSignal(str)  # 错误信息
    finished_signal = pyqtSignal()  # 完成信号

    def __init__(self, video_path, model_path, conf_threshold=0.5):
        super().__init__()
        self.video_path = video_path
        self.model_path = model_path
        self.conf_threshold = conf_threshold

        # 播放控制
        self.is_running = True
        self.is_paused = False
        self.speed = 1.0
        self.target_frame = 0

        # 视频参数
        self.cap = None
        self.total_frames = 0
        self.fps = 0
        self.model = None

    def run(self):
        try:
            # 初始化视频
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                self.error_signal.emit("无法打开视频文件")
                return

            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.model = YOLO(self.model_path)

            # 主播放循环
            while self.is_running:
                if self.is_paused:
                    self.msleep(100)
                    continue

                # 核心修复：强制跳转到目标帧
                current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                if self.target_frame != current_frame:
                    # 强制设置帧位置（关键修复）
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.target_frame)
                    # 验证是否设置成功
                    current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    if abs(current_frame - self.target_frame) > 1:
                        self.error_signal.emit(f"帧跳转失败: 目标{self.target_frame}, 实际{current_frame}")

                # 读取帧
                ret, frame = self.cap.read()
                if not ret:
                    break

                # ========== 1. 处理原图（颜色正确） ==========
                # BGR转RGB（修复颜色问题）
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                # 创建QImage（确保格式正确）
                original_qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.original_frame_signal.emit(original_qimg)

                # ========== 2. 识别并处理结果帧（颜色正确） ==========
                start_time = time.time()
                results = self.model(frame, conf=self.conf_threshold)
                detect_time = round(time.time() - start_time, 3)

                # 生成识别后帧（BGR转RGB修复颜色）
                detected_frame = results[0].plot()
                detected_rgb = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
                h2, w2, ch2 = detected_rgb.shape
                bytes_per_line2 = ch2 * w2
                detected_qimg = QImage(detected_rgb.data, w2, h2, bytes_per_line2, QImage.Format_RGB888)
                self.detected_frame_signal.emit(detected_qimg)

                # ========== 3. 更新进度和信息 ==========
                self.progress_signal.emit(current_frame)

                # 构建识别信息
                info_text = f"当前帧: {current_frame}/{self.total_frames}\n"
                info_text += f"识别耗时: {detect_time}s\n"
                info_text += f"置信度阈值: {self.conf_threshold}\n"
                info_text += f"识别目标数: {len(results[0].boxes)}\n"
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    conf = round(float(box.conf[0]), 3)
                    info_text += f"类别: {self.model.names[cls]}, 置信度: {conf}\n"
                self.info_signal.emit(info_text, detect_time)

                # ========== 4. 控制播放逻辑 ==========
                # 正常播放时递增帧
                if self.target_frame == current_frame:
                    self.target_frame += 1

                # 控制播放速度
                delay = int(1000 / (self.fps * self.speed))
                self.msleep(delay)

        except Exception as e:
            self.error_signal.emit(f"运行错误: {str(e)}")
        finally:
            if self.cap:
                self.cap.release()
            self.finished_signal.emit()

    # 播放/暂停
    def pause_play(self):
        self.is_paused = not self.is_paused

    # 设置播放速度
    def set_speed(self, speed):
        self.speed = speed

    # 设置目标帧（进度条跳转核心）
    def set_target_frame(self, frame_num):
        # 限制帧范围
        self.target_frame = max(0, min(int(frame_num), self.total_frames - 1))

    # 更新置信度
    def update_conf(self, conf):
        self.conf_threshold = conf

    # 停止线程
    def stop(self):
        self.is_running = False
        self.wait()


# 主窗口类
class YOLOv8AircraftUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8 飞机主题识别系统")
        self.setGeometry(100, 100, 1400, 800)

        # 初始化变量
        self.video_thread = None
        self.current_video_path = ""
        self.total_video_frames = 0

        # 设置飞机主题样式
        self.set_aircraft_theme()

        # 创建主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # 左右分割器（1:9）
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)
        main_layout.addWidget(splitter)

        # ---------------------- 左侧控制面板 ----------------------
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)
        left_widget.setMinimumWidth(140)
        left_widget.setMaximumWidth(180)

        # 标题
        title_label = QLabel("✈️ YOLOv8 控制区")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        title_label.setStyleSheet("color: #0066cc; margin-bottom: 10px;")
        left_layout.addWidget(title_label)

        # 模型选择
        model_label = QLabel("📌 选择模型:")
        model_label.setFont(QFont("Arial", 10, QFont.Bold))
        left_layout.addWidget(model_label)

        self.model_combo = QComboBox()
        self.model_combo.setPlaceholderText("加载模型中...")
        left_layout.addWidget(self.model_combo)
        self.load_models_from_path(r"D:\Python\paper\model")

        # 置信度阈值
        conf_label = QLabel("🎯 置信度阈值:")
        conf_label.setFont(QFont("Arial", 10, QFont.Bold))
        left_layout.addWidget(conf_label)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.5)
        self.conf_spin.valueChanged.connect(self.on_conf_changed)
        left_layout.addWidget(self.conf_spin)

        # 文件选择按钮
        file_label = QLabel("📁 选择文件:")
        file_label.setFont(QFont("Arial", 10, QFont.Bold))
        left_layout.addWidget(file_label)

        self.img_btn = QPushButton("🖼️ 上传图片")
        self.img_btn.setMinimumHeight(40)
        self.img_btn.clicked.connect(self.select_image)
        left_layout.addWidget(self.img_btn)

        self.video_btn = QPushButton("🎬 上传视频")
        self.video_btn.setMinimumHeight(40)
        self.video_btn.clicked.connect(self.select_video)
        left_layout.addWidget(self.video_btn)

        self.stop_btn = QPushButton("⏹️ 停止识别")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setStyleSheet("background-color: #ff4444;")
        self.stop_btn.clicked.connect(self.stop_all)
        self.stop_btn.setEnabled(False)
        left_layout.addWidget(self.stop_btn)

        # 视频速度调节
        speed_label = QLabel("🎚️ 视频速度:")
        speed_label.setFont(QFont("Arial", 10, QFont.Bold))
        left_layout.addWidget(speed_label)

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 20)  # 0.1-2.0倍速
        self.speed_slider.setValue(10)
        self.speed_slider.valueChanged.connect(self.on_speed_changed)
        left_layout.addWidget(self.speed_slider)

        self.speed_display = QLabel("当前速度: 1.0x")
        self.speed_display.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.speed_display)

        # 识别时间显示
        time_label = QLabel("⏱️ 识别耗时:")
        time_label.setFont(QFont("Arial", 10, QFont.Bold))
        left_layout.addWidget(time_label)

        self.time_display = QLabel("0.000s")
        self.time_display.setAlignment(Qt.AlignCenter)
        self.time_display.setStyleSheet("font-size: 14px; color: #0066cc; font-weight: bold;")
        left_layout.addWidget(self.time_display)

        # 识别结果显示
        result_label = QLabel("📝 识别结果:")
        result_label.setFont(QFont("Arial", 10, QFont.Bold))
        left_layout.addWidget(result_label)

        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        self.result_display.setMinimumHeight(150)
        left_layout.addWidget(self.result_display)

        left_layout.addStretch()

        # ---------------------- 右侧展示面板 ----------------------
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)

        # 标题
        right_title = QLabel("📊 识别结果展示区")
        right_title.setAlignment(Qt.AlignCenter)
        right_title.setFont(QFont("Arial", 12, QFont.Bold))
        right_title.setStyleSheet("color: #0066cc; margin-bottom: 10px;")
        right_layout.addWidget(right_title)

        # 双图显示区域
        display_layout = QHBoxLayout()
        display_layout.setSpacing(10)

        # 原图显示
        self.original_label = QLabel("📸 未识别内容")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("border: 2px solid #4a90e2; background-color: #f8f8f8; padding: 10px;")
        self.original_label.setMinimumSize(500, 400)
        display_layout.addWidget(self.original_label)

        # 识别后显示
        self.detected_label = QLabel("✅ 已识别内容")
        self.detected_label.setAlignment(Qt.AlignCenter)
        self.detected_label.setStyleSheet("border: 2px solid #4a90e2; background-color: #f8f8f8; padding: 10px;")
        self.detected_label.setMinimumSize(500, 400)
        display_layout.addWidget(self.detected_label)

        right_layout.addLayout(display_layout, stretch=1)

        # 视频控制区
        self.video_control_widget = QWidget()
        video_control_layout = QVBoxLayout(self.video_control_widget)
        video_control_layout.setContentsMargins(5, 5, 5, 5)
        video_control_layout.setSpacing(5)
        self.video_control_widget.setVisible(False)
        self.video_control_widget.setMinimumHeight(80)

        progress_label = QLabel("📺 视频进度控制")
        progress_label.setAlignment(Qt.AlignCenter)
        progress_label.setFont(QFont("Arial", 10, QFont.Bold))
        video_control_layout.addWidget(progress_label)

        self.video_progress = QSlider(Qt.Horizontal)
        self.video_progress.setRange(0, 100)
        self.video_progress.setValue(0)
        # 核心修改：拖动时实时跳转，释放时确认
        self.video_progress.sliderMoved.connect(self.on_progress_moved)
        self.video_progress.sliderReleased.connect(self.on_progress_released)
        video_control_layout.addWidget(self.video_progress)

        self.frame_display = QLabel("当前帧: 0 / 0")
        self.frame_display.setAlignment(Qt.AlignCenter)
        video_control_layout.addWidget(self.frame_display)

        self.play_pause_btn = QPushButton("⏸️ 暂停播放")
        self.play_pause_btn.setMinimumHeight(35)
        self.play_pause_btn.clicked.connect(self.play_pause_video)
        video_control_layout.addWidget(self.play_pause_btn)

        right_layout.addWidget(self.video_control_widget)

        # 添加到分割器
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([150, 1250])

    # 设置飞机主题样式
    def set_aircraft_theme(self):
        palette = QPalette()
        palette.setBrush(QPalette.Window, QBrush(QColor(230, 240, 250)))
        self.setPalette(palette)

        self.setStyleSheet("""
            QMainWindow {background-color: #e6f0fa;}
            QWidget {font-family: Arial; font-size: 9pt;}
            QPushButton {
                background-color: #4a90e2; color: white; border: none;
                border-radius: 8px; font-weight: bold; font-size: 10pt;
            }
            QPushButton:hover {background-color: #357abd;}
            QPushButton:disabled {background-color: #cccccc; color: #666666;}
            QComboBox {padding: 8px; border-radius: 8px; border: 1px solid #4a90e2;}
            QSlider::groove:horizontal {height: 8px; background: #e0e0e0; border-radius: 4px;}
            QSlider::handle:horizontal {
                background: #4a90e2; width: 16px; margin: -4px 0; border-radius: 8px;
            }
            QDoubleSpinBox {padding: 8px; border-radius: 8px; border: 1px solid #4a90e2;}
            QTextEdit {border: 1px solid #4a90e2; border-radius: 8px; background: white; padding: 5px;}
            QLabel {color: #2c3e50;}
        """)

    # 加载模型
    def load_models_from_path(self, model_path):
        try:
            model_dir = Path(model_path)
            if not model_dir.exists():
                self.model_combo.addItem("模型路径不存在")
                return

            model_files = list(model_dir.glob("*.pt"))
            if not model_files:
                self.model_combo.addItem("未找到.pt模型")
                return

            for model_file in model_files:
                self.model_combo.addItem(model_file.name, str(model_file))

        except Exception as e:
            self.result_display.setText(f"加载模型出错: {str(e)}")

    # 选择图片
    def select_image(self):
        self.stop_all()
        self.video_control_widget.setVisible(False)

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if not file_path:
            return

        # 显示原图（颜色正确）
        img_bgr = cv2.imread(file_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.original_label.setPixmap(
            pixmap.scaled(self.original_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # 获取模型路径
        model_idx = self.model_combo.currentIndex()
        model_path = self.model_combo.itemData(model_idx)
        if not model_path or not os.path.exists(model_path):
            self.result_display.setText("❌ 请选择有效的YOLO模型！")
            return

        # 识别图片
        self.result_display.setText("🔍 正在识别图片...")
        self.stop_btn.setEnabled(True)

        model = YOLO(model_path)
        start_time = time.time()
        results = model(img_bgr, conf=self.conf_spin.value())
        detect_time = round(time.time() - start_time, 3)

        # 显示识别结果（颜色正确）
        detected_bgr = results[0].plot()
        detected_rgb = cv2.cvtColor(detected_bgr, cv2.COLOR_BGR2RGB)
        h2, w2, ch2 = detected_rgb.shape
        bytes_per_line2 = ch2 * w2
        detected_qimg = QImage(detected_rgb.data, w2, h2, bytes_per_line2, QImage.Format_RGB888)
        detected_pixmap = QPixmap.fromImage(detected_qimg)
        self.detected_label.setPixmap(
            detected_pixmap.scaled(self.detected_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # 更新信息
        self.time_display.setText(f"{detect_time}s")
        result_text = f"识别耗时: {detect_time}s\n"
        result_text += f"置信度阈值: {self.conf_spin.value()}\n"
        result_text += f"识别目标数: {len(results[0].boxes)}\n"
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = round(float(box.conf[0]), 3)
            result_text += f"类别: {model.names[cls]}, 置信度: {conf}\n"
        self.result_display.setText(result_text)

    # 选择视频
    def select_video(self):
        self.stop_all()

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if not file_path:
            return

        # 获取模型路径
        model_idx = self.model_combo.currentIndex()
        model_path = self.model_combo.itemData(model_idx)
        if not model_path or not os.path.exists(model_path):
            self.result_display.setText("❌ 请选择有效的YOLO模型！")
            return

        # 初始化视频线程
        self.video_thread = VideoDetectionThread(
            video_path=file_path,
            model_path=model_path,
            conf_threshold=self.conf_spin.value()
        )

        # 绑定信号
        self.video_thread.original_frame_signal.connect(self.update_original_frame)
        self.video_thread.detected_frame_signal.connect(self.update_detected_frame)
        self.video_thread.progress_signal.connect(self.update_progress)
        self.video_thread.info_signal.connect(self.update_info)
        self.video_thread.error_signal.connect(self.show_error)
        self.video_thread.finished_signal.connect(self.on_video_finished)

        # 获取视频总帧数（初始化进度条）
        cap = cv2.VideoCapture(file_path)
        self.total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self.video_progress.setRange(0, self.total_video_frames - 1)
        self.frame_display.setText(f"当前帧: 0 / {self.total_video_frames - 1}")

        # 启动线程
        self.video_thread.set_speed(self.speed_slider.value() / 10)
        self.video_thread.start()

        # 更新UI
        self.video_control_widget.setVisible(True)
        self.stop_btn.setEnabled(True)
        self.result_display.setText("🔍 正在播放并识别视频...")
        self.play_pause_btn.setText("⏸️ 暂停播放")

    # 更新原图帧
    @pyqtSlot(QImage)
    def update_original_frame(self, qimg):
        pixmap = QPixmap.fromImage(qimg)
        self.original_label.setPixmap(
            pixmap.scaled(self.original_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # 更新识别帧（修复颜色）
    @pyqtSlot(QImage)
    def update_detected_frame(self, qimg):
        pixmap = QPixmap.fromImage(qimg)
        self.detected_label.setPixmap(
            pixmap.scaled(self.detected_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # 更新进度条
    @pyqtSlot(int)
    def update_progress(self, frame_num):
        self.video_progress.setValue(frame_num)
        self.frame_display.setText(f"当前帧: {frame_num} / {self.total_video_frames - 1}")

    # 更新识别信息
    @pyqtSlot(str, float)
    def update_info(self, info_text, detect_time):
        self.time_display.setText(f"{detect_time}s")
        self.result_display.setText(info_text)

    # 进度条拖动（核心修复：实时跳转）
    def on_progress_moved(self, frame_num):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.set_target_frame(frame_num)
            self.frame_display.setText(f"当前帧: {frame_num} / {self.total_video_frames - 1}")

    # 进度条释放（确认跳转）
    def on_progress_released(self):
        if self.video_thread:
            frame_num = self.video_progress.value()
            self.result_display.append(f"\n🔀 跳转到帧: {frame_num}")

    # 播放/暂停
    def play_pause_video(self):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.pause_play()
            if self.video_thread.is_paused:
                self.play_pause_btn.setText("▶️ 继续播放")
                self.result_display.append("\n⏸️ 视频已暂停")
            else:
                self.play_pause_btn.setText("⏸️ 暂停播放")
                self.result_display.append("\n▶️ 视频已恢复播放")

    # 调节速度
    def on_speed_changed(self):
        speed = self.speed_slider.value() / 10
        self.speed_display.setText(f"当前速度: {speed}x")
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.set_speed(speed)

    # 更新置信度
    def on_conf_changed(self):
        conf = self.conf_spin.value()
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.update_conf(conf)
            self.result_display.append(f"\n🎯 置信度阈值更新为: {conf}")

    # 显示错误
    def show_error(self, error_text):
        self.result_display.setText(f"❌ {error_text}")

    # 视频播放完成
    def on_video_finished(self):
        self.result_display.append("\n✅ 视频播放完成")
        self.play_pause_btn.setText("▶️ 重新播放")

    # 停止所有操作
    def stop_all(self):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
        self.stop_btn.setEnabled(False)
        self.video_control_widget.setVisible(False)
        self.result_display.append("\n🛑 已停止所有操作")

    # 窗口关闭
    def closeEvent(self, event):
        self.stop_all()
        event.accept()


# 主函数
if __name__ == "__main__":
    # 检查依赖
    try:
        import ultralytics
        import cv2
        import numpy as np
    except ImportError:
        print("请先安装依赖：")
        print("pip install ultralytics opencv-python pyqt5 numpy")
        sys.exit(1)

    app = QApplication(sys.argv)
    window = YOLOv8AircraftUI()
    window.show()
    sys.exit(app.exec_())