import sys
import time
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, 
                             QPushButton, QComboBox, QLabel, QTextEdit, QLineEdit, QHBoxLayout)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QPixmap, QImage
import pygetwindow as gw
from simple_poker_analyzer import SimplePokerAnalyzer
from PIL import Image

class WindowCaptureApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("窗口截图与识别")
        self.setGeometry(100, 100, 800, 600)
        
        # 初始化变量
        self.windows = []
        self.selected_window = None
        self.screenshot_interval = 2  # 默认2秒
        self.timer = QTimer()
        self.timer.timeout.connect(self.capture_and_analyze)
        
        # 初始化扑克牌分析器
        self.poker_analyzer = SimplePokerAnalyzer()
        
        # 创建UI
        self.init_ui()
        
        # 刷新窗口列表
        self.refresh_windows()
    
    def init_ui(self):
        # 主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # 窗口选择部分
        window_layout = QHBoxLayout()
        self.window_combo = QComboBox()
        self.window_combo.currentIndexChanged.connect(self.on_window_selected)
        self.refresh_button = QPushButton("刷新窗口")
        self.refresh_button.clicked.connect(self.refresh_windows)
        self.bind_button = QPushButton("绑定窗口")
        self.bind_button.clicked.connect(self.bind_window)
        window_layout.addWidget(QLabel("选择窗口:"))
        window_layout.addWidget(self.window_combo)
        window_layout.addWidget(self.refresh_button)
        window_layout.addWidget(self.bind_button)
        layout.addLayout(window_layout)
        
        # 显示区域
        self.image_label = QLabel()
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setText("等待截图...")
        layout.addWidget(self.image_label)
        
        # 截图间隔设置和按钮
        control_layout = QHBoxLayout()
        
        # 截图间隔设置
        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("截图间隔(秒):"))
        self.interval_input = QLineEdit(str(self.screenshot_interval))
        self.interval_input.setFixedWidth(50)
        interval_layout.addWidget(self.interval_input)
        
        # 将所有按钮放在同一行
        # 识别窗口内容按钮
        self.recognize_button = QPushButton("识别窗口内容")
        self.recognize_button.clicked.connect(self.recognize_window_content)
        
        # 开始截图按钮
        self.start_button = QPushButton("开始截图")
        self.start_button.clicked.connect(self.start_capture)
        
        # 调整窗口大小的输入框和按钮
        self.width_input = QLineEdit("860")  # 设置默认宽度
        self.width_input.setFixedWidth(50)
        self.height_input = QLineEdit("665")  # 设置默认高度
        self.height_input.setFixedWidth(50)
        self.resize_button = QPushButton("调整窗口大小")
        self.resize_button.clicked.connect(self.resize_window)
        
        # 添加控件到布局
        control_layout.addLayout(interval_layout)
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.recognize_button)
        control_layout.addWidget(QLabel("宽度:"))
        control_layout.addWidget(self.width_input)
        control_layout.addWidget(QLabel("高度:"))
        control_layout.addWidget(self.height_input)
        control_layout.addWidget(self.resize_button)
        
        layout.addLayout(control_layout)
        
        # 输出文本框
        self.output_text = QTextEdit()
        self.output_text.setMaximumHeight(150)
        layout.addWidget(self.output_text)
        
        # 状态栏
        self.statusBar().showMessage("就绪")
    
    def refresh_windows(self):
        """刷新窗口列表"""
        # 过滤掉标题为空或只有空格的窗口
        all_windows = gw.getAllWindows()
        self.windows = [w for w in all_windows if w.title and w.title.strip()]
        self.window_combo.clear()
        for window in self.windows:
            self.window_combo.addItem(f"{window.title} ({window.left}, {window.top}, {window.width}, {window.height})")
        self.statusBar().showMessage(f"找到 {len(self.windows)} 个有效窗口 (总计 {len(all_windows)} 个窗口)")
    
    def on_window_selected(self, index):
        """当选择窗口时"""
        self.output_text.append(f"[调试] 窗口选择事件触发，索引: {index}, 窗口总数: {len(self.windows)}")
        if 0 <= index < len(self.windows):
            # 仅更新状态栏显示，不改变已绑定的窗口
            window = self.windows[index]
            # 获取窗口大小
            width = window.width
            height = window.height
            self.statusBar().showMessage(f"已选择窗口: {window.title} (大小: {width}x{height})")
            self.output_text.append(f"[调试] 选择窗口索引: {index}, 标题: {window.title}")
        else:
            self.output_text.append(f"[调试] 窗口选择索引无效: {index}")
    
    def bind_window(self):
        """绑定选定的窗口"""
        index = self.window_combo.currentIndex()
        self.output_text.append(f"[调试] 尝试绑定窗口，当前索引: {index}，窗口总数: {len(self.windows)}")
        if 0 <= index < len(self.windows):
            self.selected_window = self.windows[index]
            # 检查窗口是否有效
            if not self.selected_window.title or not self.selected_window.title.strip():
                self.statusBar().showMessage("无法绑定标题为空的窗口")
                self.output_text.append("[调试] 绑定失败，窗口标题为空")
                return
            # 获取窗口大小
            width = self.selected_window.width
            height = self.selected_window.height
            self.statusBar().showMessage(f"已绑定窗口: {self.selected_window.title} (大小: {width}x{height})")
            self.output_text.append(f"已绑定窗口: {self.selected_window.title} (大小: {width}x{height})")
            self.output_text.append(f"[调试] 绑定成功，窗口标题: {self.selected_window.title}")
        else:
            self.statusBar().showMessage("请选择一个窗口进行绑定")
            self.output_text.append("[调试] 绑定失败，索引无效")
    
    def start_capture(self):
        """开始截图"""
        try:
            self.screenshot_interval = int(self.interval_input.text())
        except ValueError:
            self.screenshot_interval = 5
            self.interval_input.setText("5")
        
        # 如果没有绑定窗口，则使用当前选择的窗口进行绑定
        if not self.selected_window:
            self.output_text.append(f"[调试] 开始截图时未绑定窗口，当前组合框索引: {self.window_combo.currentIndex()}")
            self.bind_window()
        
        if self.selected_window:
            self.timer.start(self.screenshot_interval * 1000)  # 毫秒
            self.start_button.setText("停止截图")
            self.start_button.clicked.disconnect()
            self.start_button.clicked.connect(self.stop_capture)
            self.statusBar().showMessage("开始截图...")
        else:
            self.statusBar().showMessage("请先选择并绑定一个窗口")
    
    def stop_capture(self):
        """停止截图"""
        self.timer.stop()
        self.start_button.setText("开始截图")
        self.start_button.clicked.disconnect()
        self.start_button.clicked.connect(self.start_capture)
        self.statusBar().showMessage("已停止截图")
    
    def capture_and_analyze(self):
        """截图并分析"""
        if not self.selected_window:
            return
        
        try:
            # 获取窗口位置和大小
            left, top, width, height = self.selected_window.left, self.selected_window.top, \
                                       self.selected_window.width, self.selected_window.height
            
            # 截图
            # 使用PIL的ImageGrab来截取屏幕区域
            from PIL import ImageGrab
            screenshot = ImageGrab.grab(bbox=(left, top, left + width, top + height))
            
            if screenshot:
                # 转换为OpenCV格式
                open_cv_image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                
                # 显示截图
                self.display_image(open_cv_image)
                
                # 转换为PIL图像以供OCR使用
                pil_image = Image.fromarray(cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB))
                
                # 识别发牌和加注动作
                action_type = self.poker_analyzer.identify_action_position(pil_image)
                
                # 输出结果
                self.output_text.append(f"窗口大小: {width}x{height}")
                if action_type == 'DEAL':
                    self.output_text.append("识别结果: 发牌动作 (忽略)")
                elif action_type == 'RAISE':
                    self.output_text.append("识别结果: 加注动作 (需要进一步识别)")
                    # 这里应该调用完整的图片识别功能
                else:
                    self.output_text.append("识别结果: 未检测到发牌或加注动作")
                self.output_text.append("---")
                
        except Exception as e:
            self.output_text.append(f"截图错误: {str(e)}")
    
    def recognize_window_content(self):
        """识别窗口内容（使用英文模式）"""
        if not self.selected_window:
            self.statusBar().showMessage("请先选择并绑定一个窗口")
            return
        
        try:
            # 获取窗口位置和大小
            left, top, width, height = self.selected_window.left, self.selected_window.top, \
                                       self.selected_window.width, self.selected_window.height
            
            # 截图
            # 使用PIL的ImageGrab来截取屏幕区域
            from PIL import ImageGrab
            screenshot = ImageGrab.grab(bbox=(left, top, left + width, top + height))
            
            if screenshot:
                # 转换为OpenCV格式
                open_cv_image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                
                # 显示截图
                self.display_image(open_cv_image)
                
                # 转换为PIL图像以供OCR使用
                pil_image = Image.fromarray(cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB))
                
                # 使用PokerRecognizer进行英文识别
                from poker_recognition import PokerRecognizer
                recognizer = PokerRecognizer()
                recognizer.language = 'eng'  # 设置为英文模式
                
                # 识别整个图片
                result = recognizer.recognize_full_image(pil_image)
                
                # 输出结果
                if result and result.get('success'):
                    self.output_text.append(f"Window size: {width}x{height}")
                    self.output_text.append(f"Recognition result: {result['text']}")
                    self.output_text.append(f"Recognition time: {result['time']:.3f} seconds")
                    self.statusBar().showMessage("窗口内容识别完成")
                else:
                    self.output_text.append("Failed to recognize window content")
                    self.statusBar().showMessage("窗口内容识别失败")
                self.output_text.append("---")
            
        except Exception as e:
            self.output_text.append(f"Recognition error: {str(e)}")
            self.statusBar().showMessage("识别过程中发生错误")
    
    def resize_window(self):
        """调整绑定窗口的大小"""
        if not self.selected_window:
            self.statusBar().showMessage("请先选择并绑定一个窗口")
            return
        
        try:
            # 获取用户输入的宽度和高度
            width = int(self.width_input.text())
            height = int(self.height_input.text())
            
            # 使用pygetwindow调整窗口大小
            self.selected_window.resizeTo(width, height)
            
            # 更新状态栏和输出文本框
            self.statusBar().showMessage(f"窗口大小已调整为: {width}x{height}")
            self.output_text.append(f"窗口大小已调整为: {width}x{height}")
            
            # 更新窗口信息显示
            self.on_window_selected(self.window_combo.currentIndex())
        except ValueError:
            self.statusBar().showMessage("请输入有效的宽度和高度值")
        except Exception as e:
            self.statusBar().showMessage(f"调整窗口大小时出错: {str(e)}")
            self.output_text.append(f"调整窗口大小时出错: {str(e)}")
    
    def display_image(self, image):
        """显示图像"""
        # 调整图像大小以适应显示区域
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)
        
        # 缩放图像以适应标签
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, 
                                      Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WindowCaptureApp()
    window.show()
    sys.exit(app.exec())