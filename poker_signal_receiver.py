#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扑克信号接收端
接收网络信号，根据信号类型控制鼠标移动到相应位置并点击
"""

import json
import socket
import threading
import time
import pyautogui
import sys
import win32gui
import win32con
from typing import Dict, List, Optional
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QTextEdit, 
                             QGroupBox, QCheckBox, QSpinBox, QMessageBox,
                             QComboBox)
from PyQt6.QtCore import QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont

class SignalReceiver(QMainWindow):
    """扑克信号接收端主窗口"""
    
    def __init__(self):
        super().__init__()
        self.server_socket = None
        self.client_socket = None
        self.is_running = False
        self.config = {}
        self.bound_window_hwnd = None
        self.bound_window_info = None
        self.init_ui()
        self.load_config()
        
        # 初始化窗口列表
        self.refresh_windows()
        
        # 设置pyautogui安全设置
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("扑克信号接收端")
        self.setGeometry(100, 100, 700, 600)
        
        # 设置字体
        font = QFont("Microsoft YaHei", 10)
        self.setFont(font)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 服务器控制组
        server_group = QGroupBox("服务器控制")
        server_layout = QHBoxLayout(server_group)
        
        self.start_btn = QPushButton("启动服务器")
        self.start_btn.clicked.connect(self.toggle_server)
        server_layout.addWidget(self.start_btn)
        
        self.reload_config_btn = QPushButton("重新载入配置")
        self.reload_config_btn.clicked.connect(self.reload_config)
        server_layout.addWidget(self.reload_config_btn)
        
        self.port_label = QLabel("端口: 8888")
        server_layout.addWidget(self.port_label)
        
        server_layout.addStretch()
        
        # 连接状态
        self.status_label = QLabel("状态: 未启动")
        server_layout.addWidget(self.status_label)
        
        main_layout.addWidget(server_group)
        
        # 窗口绑定组
        window_group = QGroupBox("窗口绑定")
        window_layout = QHBoxLayout(window_group)
        
        self.window_combo = QComboBox()
        self.window_combo.setMinimumWidth(200)
        window_layout.addWidget(self.window_combo)
        
        self.refresh_btn = QPushButton("刷新窗口")
        self.refresh_btn.clicked.connect(self.refresh_windows)
        window_layout.addWidget(self.refresh_btn)
        
        self.bind_btn = QPushButton("绑定窗口")
        self.bind_btn.clicked.connect(self.bind_selected_window)
        window_layout.addWidget(self.bind_btn)
        
        self.window_info_label = QLabel("未绑定窗口")
        window_layout.addWidget(self.window_info_label)
        
        main_layout.addWidget(window_group)
        
        # 鼠标控制组
        mouse_group = QGroupBox("鼠标控制设置")
        mouse_layout = QHBoxLayout(mouse_group)
        
        # 延迟设置
        mouse_layout.addWidget(QLabel("点击延迟(秒):"))
        self.delay_spin = QSpinBox()
        self.delay_spin.setRange(0, 50)  # 0-5秒，以0.1秒为单位
        self.delay_spin.setValue(10)     # 默认1秒
        self.delay_spin.setSingleStep(1) # 步长为1（对应0.1秒）
        mouse_layout.addWidget(self.delay_spin)
        
        # 移动速度设置
        mouse_layout.addWidget(QLabel("移动速度:"))
        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(1, 10)
        self.speed_spin.setValue(3)
        mouse_layout.addWidget(self.speed_spin)
        
        # 安全模式
        self.safe_mode_check = QCheckBox("安全模式")
        self.safe_mode_check.setChecked(True)
        mouse_layout.addWidget(self.safe_mode_check)
        
        mouse_layout.addStretch()
        
        main_layout.addWidget(mouse_group)
        
        # 坐标显示组
        coords_group = QGroupBox("坐标信息")
        coords_layout = QVBoxLayout(coords_group)
        
        self.coords_text = QTextEdit()
        self.coords_text.setMaximumHeight(150)
        coords_layout.addWidget(self.coords_text)
        
        main_layout.addWidget(coords_group)
        
        # 日志显示组
        log_group = QGroupBox("接收日志")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        log_layout.addWidget(self.log_text)
        
        # 清空日志按钮
        clear_btn = QPushButton("清空日志")
        clear_btn.clicked.connect(self.log_text.clear)
        log_layout.addWidget(clear_btn)
        
        main_layout.addWidget(log_group)
        
        # 设置按钮样式
        self.set_button_styles()
        
    def set_button_styles(self):
        """设置按钮样式"""
        button_style = """
        QPushButton {
            background-color: #2196F3;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #1976D2;
        }
        QPushButton:pressed {
            background-color: #0D47A1;
        }
        QPushButton:disabled {
            background-color: #cccccc;
            color: #666666;
        }
        """
        
        self.start_btn.setStyleSheet(button_style)
        self.refresh_btn.setStyleSheet(button_style)
        self.bind_btn.setStyleSheet(button_style)
        self.reload_config_btn.setStyleSheet(button_style)
        
    def refresh_windows(self):
        """刷新窗口列表"""
        self.window_combo.clear()
        
        def enum_windows_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                if window_text:  # 只添加有标题的窗口
                    windows.append({
                        'hwnd': hwnd,
                        'title': window_text,
                        'class': win32gui.GetClassName(hwnd)
                    })
        
        windows = []
        win32gui.EnumWindows(enum_windows_callback, windows)
        
        # 添加窗口到下拉列表
        for window in windows:
            display_text = f"{window['title']} ({window['class']})"
            self.window_combo.addItem(display_text, window)
        
        self.log_message(f"刷新窗口列表，找到 {len(windows)} 个可见窗口")
    
    def bind_selected_window(self):
        """绑定选中的窗口"""
        current_index = self.window_combo.currentIndex()
        if current_index >= 0:
            window_data = self.window_combo.itemData(current_index)
            if window_data:
                self.bound_window_hwnd = window_data['hwnd']
                self.bound_window_info = window_data
                
                # 获取窗口详细信息
                try:
                    rect = win32gui.GetWindowRect(self.bound_window_hwnd)
                    window_width = rect[2] - rect[0]
                    window_height = rect[3] - rect[1]
                    window_info_text = f"已绑定: {window_data['title']} | 位置: {rect[0]},{rect[1]} | 大小: {window_width}x{window_height}"
                    self.window_info_label.setText(window_info_text)
                    self.log_message(f"成功绑定窗口: {window_data['title']}")
                    
                    # 更新并保存reference_size到配置文件
                    self.update_reference_size(window_width, window_height)
                except Exception as e:
                    self.log_message(f"绑定窗口时出错: {e}")
        else:
            self.log_message("请先选择一个窗口")
    
    def get_bound_window_size(self) -> Optional[tuple]:
        """获取绑定窗口的大小"""
        if self.bound_window_hwnd:
            try:
                rect = win32gui.GetWindowRect(self.bound_window_hwnd)
                return (rect[2] - rect[0], rect[3] - rect[1])
            except:
                pass
        return None
    
    def update_reference_size(self, width: int, height: int):
        """更新并保存reference_size到配置文件"""
        try:
            # 更新内存中的配置
            if 'reference_size' in self.config:
                old_width, old_height = self.config['reference_size']
                self.config['reference_size'] = [width, height]
                self.log_message(f"已更新reference_size: {old_width}x{old_height} -> {width}x{height}")
                
                # 保存到文件
                with open('ocr_test_config.json', 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, ensure_ascii=False, indent=2)
                self.log_message("配置文件保存成功")
                
                # 更新坐标显示
                self.update_coords_display()
            else:
                self.log_message("配置文件中未找到reference_size字段")
        except Exception as e:
            self.log_message(f"更新reference_size失败: {e}")
    
    def load_config(self):
        """加载配置文件"""
        try:
            with open('ocr_test_config.json', 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            self.log_message("配置文件加载成功")
            self.update_coords_display()
        except Exception as e:
            self.log_message(f"配置文件加载失败: {e}")
            self.config = {}
            
    def reload_config(self):
        """重新载入配置文件"""
        self.log_message("正在重新载入配置文件...")
        self.load_config()
        self.log_message("配置文件重新载入完成")
            
    def update_coords_display(self):
        """更新坐标显示"""
        coords_info = "可用坐标区域:\n"
        
        # 显示test_regions中的固定坐标
        if 'test_regions' in self.config:
            for name, coords in self.config['test_regions'].items():
                coords_info += f"{name}: {coords}\n"
        
        # 显示adaptive_regions中的自适应区域
        if 'adaptive_regions' in self.config:
            coords_info += "\n自适应区域 (基于比例):\n"
            for name, ratios in self.config['adaptive_regions'].items():
                coords_info += f"{name}: {ratios}\n"
                
        self.coords_text.setText(coords_info)
            
    def toggle_server(self):
        """切换服务器状态"""
        if not self.is_running:
            self.start_server()
        else:
            self.stop_server()
            
    def start_server(self):
        """启动服务器"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)


# 修改后，接受来自任何网络接口的连接
            self.server_socket.bind(('0.0.0.0', 8888))
            self.server_socket.listen(1)
            
            self.is_running = True
            self.start_btn.setText("停止服务器")
            self.status_label.setText("状态: 等待连接")
            self.log_message("服务器启动成功，等待连接...")
            
            # 启动监听线程
            self.listen_thread = threading.Thread(target=self.listen_for_connections)
            self.listen_thread.daemon = True
            self.listen_thread.start()
            
        except Exception as e:
            QMessageBox.warning(self, "启动失败", f"无法启动服务器: {e}")
            self.log_message(f"服务器启动失败: {e}")
            
    def stop_server(self):
        """停止服务器"""
        self.is_running = False
        
        if self.client_socket:
            self.client_socket.close()
            self.client_socket = None
            
        if self.server_socket:
            self.server_socket.close()
            self.server_socket = None
            
        self.start_btn.setText("启动服务器")
        self.status_label.setText("状态: 已停止")
        self.log_message("服务器已停止")
        
    def listen_for_connections(self):
        """监听客户端连接"""
        while self.is_running:
            try:
                self.log_message("等待客户端连接...")
                self.client_socket, address = self.server_socket.accept()
                self.log_message(f"客户端已连接: {address}")
                
                # 更新状态
                self.status_label.setText(f"状态: 已连接 ({address[0]})")
                
                # 启动接收线程
                self.receive_thread = threading.Thread(target=self.receive_signals)
                self.receive_thread.daemon = True
                self.receive_thread.start()
                
            except Exception as e:
                if self.is_running:
                    self.log_message(f"连接错误: {e}")
                break
                
    def receive_signals(self):
        """接收信号"""
        while self.is_running and self.client_socket:
            try:
                data = self.client_socket.recv(1024)
                if not data:
                    break
                    
                message = data.decode('utf-8')
                self.log_message(f"收到信号: {message}")
                
                # 解析信号
                signal_data = json.loads(message)
                self.process_signal(signal_data)
                
            except Exception as e:
                if self.is_running:
                    self.log_message(f"接收错误: {e}")
                break
                
        # 连接断开
        if self.client_socket:
            self.client_socket.close()
            self.client_socket = None
            self.status_label.setText("状态: 等待连接")
            self.log_message("客户端连接断开")
            
    def process_signal(self, signal_data):
        """处理接收到的信号"""
        signal_type = signal_data.get('type', '')
        amount = signal_data.get('amount', 0)
        
        self.log_message(f"处理信号: {signal_type}")
        if amount:
            self.log_message(f"  加注额度: {amount}")
            
        # 根据信号类型执行相应操作
        if signal_type == "发牌":
            self.click_position("发牌")
        elif signal_type == "弃牌":
            self.click_position("弃牌")
        elif signal_type == "让牌":
            self.click_position("跟注")  # 让牌通常和跟注在同一位置
        elif signal_type == "跟注":
            self.click_position("跟注")
        elif signal_type == "加注":
            if amount in [33, 50, 75, 100]:
                # 对于固定额度，直接点击对应的按钮
                self.click_position(f"加注{amount}")
            else:
                # 对于自定义额度，先点击加注按钮，然后输入金额
                self.click_position("加注")
                if amount > 0:
                    self.log_message(f"  需要输入自定义加注额度: {amount}")
                     # 输入加注额度的逻辑
                    self.input_raise_amount(amount)
        elif signal_type == "加注33":
            self.click_position("加注33")
        elif signal_type == "加注50":
            self.click_position("加注50")
        elif signal_type == "加注75":
            self.click_position("加注75")
        elif signal_type == "加注100":
            self.click_position("加注100")
        elif signal_type == "自定义加注":
            # 对于自定义加注，我们先点击加注按钮，然后可能需要输入金额
            self.click_position("加注")
            if amount > 0:
                self.log_message(f"  需要输入自定义加注额度: {amount}")
        else:
            self.log_message(f"未知信号类型: {signal_type}")
            
    def calculate_adaptive_regions(self, window_width: int, window_height: int) -> Dict[str, List[int]]:
        """
        根据窗口尺寸计算自适应区域坐标
        
        Args:
            window_width: 窗口宽度
            window_height: 窗口高度
            
        Returns:
            绝对坐标区域 {区域名: [x1, y1, x2, y2]}
        """
        if 'adaptive_regions' not in self.config or 'reference_size' not in self.config:
            return {}
            
        adaptive_regions = self.config['adaptive_regions']
        ref_width, ref_height = self.config['reference_size']
        absolute_regions = {}
        
        for region_name, ratios in adaptive_regions.items():
            x1_ratio, y1_ratio, x2_ratio, y2_ratio = ratios
            
            # 根据比例计算绝对坐标
            x1 = int(x1_ratio * window_width)
            y1 = int(y1_ratio * window_height)
            x2 = int(x2_ratio * window_width)
            y2 = int(y2_ratio * window_height)
            
            absolute_regions[region_name] = [x1, y1, x2, y2]
            
        return absolute_regions
        
    def click_position(self, position_name):
        """点击指定位置"""
        coords = None
        
        # 首先尝试从test_regions获取固定坐标
        if 'test_regions' in self.config and position_name in self.config['test_regions']:
            coords = self.config['test_regions'][position_name]
        
        # 如果没有固定坐标或固定坐标为空，则尝试计算自适应区域
        if coords is None or len(coords) == 0:
            # 获取绑定窗口的大小
            window_size = self.get_bound_window_size()
            if window_size:
                window_width, window_height = window_size
                adaptive_regions = self.calculate_adaptive_regions(window_width, window_height)
                if position_name in adaptive_regions:
                    coords = adaptive_regions[position_name]
        
        # 如果仍然没有找到坐标
        if coords is None:
            self.log_message(f"错误: 未找到位置 '{position_name}' 的坐标")
            return
            
        try:
            # 获取坐标
            x = (coords[0] + coords[2]) // 2  # 计算中心点X坐标
            y = (coords[1] + coords[3]) // 2  # 计算中心点Y坐标
            
            # 如果绑定了窗口，则需要转换坐标
            if self.bound_window_hwnd:
                try:
                    rect = win32gui.GetWindowRect(self.bound_window_hwnd)
                    left, top = rect[0], rect[1]
                    x += left
                    y += top
                    self.log_message(f"使用窗口绑定坐标: ({x}, {y}) (原坐标: ({x-left}, {y-top}))")
                except Exception as e:
                    self.log_message(f"窗口坐标转换失败: {e}")
            
            # 设置移动速度
            duration = 1.0 / self.speed_spin.value()
            
            # 移动鼠标
            self.log_message(f"移动鼠标到 {position_name}: ({x}, {y})")
            pyautogui.moveTo(x, y, duration=duration)
            
            # 等待延迟
            delay = self.delay_spin.value() / 10.0  # 转换为秒
            if delay > 0:
                time.sleep(delay)
                
            # 点击
            pyautogui.click(x, y)
            self.log_message(f"已点击 {position_name}")
            
        except Exception as e:
            self.log_message(f"点击失败: {e}")
            
    def input_raise_amount(self, amount):
        """输入加注额度"""
        try:
            # 1. 点击加注输入框
            self.click_position("加注框")
            
            # 2. 输入数字
            pyautogui.typewrite(str(amount))
            
            # 3. 点击确认按钮（这里假设确认按钮就是加注按钮）
            self.click_position("加注")
            
            self.log_message(f"已输入加注额度: {amount}")
        except Exception as e:
            self.log_message(f"输入加注额度失败: {e}")
            
    def log_message(self, message):
        """记录日志消息"""
        import time
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        # 在主线程中更新UI
        QTimer.singleShot(0, lambda: self.log_text.append(log_entry))
        
        # 自动滚动到底部
        QTimer.singleShot(0, lambda: self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()))
        
    def closeEvent(self, event):
        """窗口关闭事件"""
        if self.is_running:
            self.stop_server()
        event.accept()

def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用程序信息
    app.setApplicationName("扑克信号接收端")
    app.setApplicationVersion("1.0")
    
    # 创建并显示主窗口
    window = SignalReceiver()
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
