#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扑克信号发送端
使用PyQt6设计界面，发送各种扑克游戏操作信号
"""

import sys
import json
import socket
import threading
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLineEdit, QLabel, 
                             QTextEdit, QGroupBox, QGridLayout, QMessageBox)
from PyQt6.QtCore import QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QIcon

class SignalSender(QMainWindow):
    """扑克信号发送端主窗口"""
    
    def __init__(self):
        super().__init__()
        self.socket = None
        self.server_address = ('localhost', 8888)
        self.init_ui()
        self.load_config()
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("扑克信号发送端")
        self.setGeometry(100, 100, 600, 500)
        
        # 设置字体
        font = QFont("Microsoft YaHei", 10)
        self.setFont(font)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 连接设置组
        connection_group = QGroupBox("连接设置")
        connection_layout = QGridLayout(connection_group)
        
        # IP地址输入
        connection_layout.addWidget(QLabel("服务器IP:"), 0, 0)
        self.ip_input = QLineEdit("localhost")
        connection_layout.addWidget(self.ip_input, 0, 1)
        
        # 端口输入
        connection_layout.addWidget(QLabel("端口:"), 0, 2)
        self.port_input = QLineEdit("8888")
        connection_layout.addWidget(self.port_input, 0, 3)
        
        # 连接按钮
        self.connect_btn = QPushButton("连接")
        self.connect_btn.clicked.connect(self.toggle_connection)
        connection_layout.addWidget(self.connect_btn, 0, 4)
        
        main_layout.addWidget(connection_group)
        
        # 游戏操作组
        game_group = QGroupBox("游戏操作")
        game_layout = QGridLayout(game_group)
        
        # 第一行按钮 - 发牌按钮单独一行
        self.deal_btn = QPushButton("发牌")
        self.deal_btn.clicked.connect(lambda: self.send_signal("发牌"))
        game_layout.addWidget(self.deal_btn, 0, 0)
        
        # 第二行按钮 - 弃牌、让牌、跟注
        self.fold_btn = QPushButton("弃牌")
        self.fold_btn.clicked.connect(lambda: self.send_signal("弃牌"))
        game_layout.addWidget(self.fold_btn, 1, 0)
        
        self.check_btn = QPushButton("让牌")
        self.check_btn.clicked.connect(lambda: self.send_signal("让牌"))
        game_layout.addWidget(self.check_btn, 1, 1)
        
        self.call_btn = QPushButton("跟注")
        self.call_btn.clicked.connect(lambda: self.send_signal("跟注"))
        game_layout.addWidget(self.call_btn, 1, 2)
        
        # 第三行按钮 - 固定额度加注按钮
        self.raise_33_btn = QPushButton("加注33")
        self.raise_33_btn.clicked.connect(lambda: self.send_signal("加注33"))
        game_layout.addWidget(self.raise_33_btn, 2, 0)
        
        self.raise_50_btn = QPushButton("加注50")
        self.raise_50_btn.clicked.connect(lambda: self.send_signal("加注50"))
        game_layout.addWidget(self.raise_50_btn, 2, 1)
        
        self.raise_75_btn = QPushButton("加注75")
        self.raise_75_btn.clicked.connect(lambda: self.send_signal("加注75"))
        game_layout.addWidget(self.raise_75_btn, 2, 2)
        
        self.raise_100_btn = QPushButton("加注100")
        self.raise_100_btn.clicked.connect(lambda: self.send_signal("加注100"))
        game_layout.addWidget(self.raise_100_btn, 2, 3)
        
        # 第四行 - 自定义加注输入框
        self.custom_raise_input = QLineEdit()
        self.custom_raise_input.setFixedHeight(30)  # 设置输入框标准高度
        game_layout.addWidget(self.custom_raise_input, 3, 0, 1, 4)  # 占据整行
        
        # 第五行 - 加注按钮
        self.raise_btn = QPushButton("加注")
        self.raise_btn.setFixedHeight(30)  # 设置按钮标准高度
        self.raise_btn.clicked.connect(self.send_custom_raise)
        game_layout.addWidget(self.raise_btn, 4, 0)
        
        main_layout.addWidget(game_group)
        
        # 日志显示组
        log_group = QGroupBox("发送日志")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)
        
        # 清空日志按钮
        clear_btn = QPushButton("清空日志")
        clear_btn.clicked.connect(self.log_text.clear)
        log_layout.addWidget(clear_btn)
        
        main_layout.addWidget(log_group)
        
        # 状态栏
        self.statusBar().showMessage("未连接")
        
        # 设置按钮样式
        self.set_button_styles()
        
    def set_button_styles(self):
        """设置按钮样式"""
        button_style = """
        QPushButton {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px;
            border-radius: 5px;
            font-size: 12px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #45a049;
        }
        QPushButton:pressed {
            background-color: #3d8b40;
        }
        QPushButton:disabled {
            background-color: #cccccc;
            color: #666666;
        }
        """
        
        # 应用样式到所有按钮
        for btn in [self.deal_btn, self.fold_btn, self.check_btn, 
                   self.call_btn, self.raise_btn, self.connect_btn]:
            btn.setStyleSheet(button_style)
            
        # 特殊样式
        self.deal_btn.setStyleSheet(button_style.replace("#4CAF50", "#2196F3"))
        self.fold_btn.setStyleSheet(button_style.replace("#4CAF50", "#f44336"))
        self.raise_btn.setStyleSheet(button_style.replace("#4CAF50", "#FF9800"))
        
    def load_config(self):
        """加载配置文件"""
        try:
            with open('ocr_test_config.json', 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            self.log_message("配置文件加载成功")
        except Exception as e:
            self.log_message(f"配置文件加载失败: {e}")
            self.config = {}
            
    def toggle_connection(self):
        """切换连接状态"""
        if self.socket is None:
            self.connect_to_server()
        else:
            self.disconnect_from_server()
            
    def connect_to_server(self):
        """连接到服务器"""
        try:
            ip = self.ip_input.text()
            port = int(self.port_input.text())
            self.server_address = (ip, port)
            
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect(self.server_address)
            
            self.connect_btn.setText("断开")
            self.statusBar().showMessage(f"已连接到 {ip}:{port}")
            self.log_message(f"成功连接到服务器 {ip}:{port}")
            
            # 启用游戏按钮
            self.enable_game_buttons(True)
            
        except Exception as e:
            QMessageBox.warning(self, "连接失败", f"无法连接到服务器: {e}")
            self.log_message(f"连接失败: {e}")
            
    def disconnect_from_server(self):
        """断开服务器连接"""
        if self.socket:
            self.socket.close()
            self.socket = None
            
        self.connect_btn.setText("连接")
        self.statusBar().showMessage("未连接")
        self.log_message("已断开连接")
        
        # 禁用游戏按钮
        self.enable_game_buttons(False)
        
    def enable_game_buttons(self, enabled):
        """启用或禁用游戏按钮"""
        for btn in [self.deal_btn, self.fold_btn, self.check_btn, 
                   self.call_btn, self.raise_btn, self.raise_33_btn,
                   self.raise_50_btn, self.raise_75_btn, self.raise_100_btn]:
            btn.setEnabled(enabled)
            
    def send_signal(self, signal_type):
        """发送信号"""
        if not self.socket:
            QMessageBox.warning(self, "未连接", "请先连接到服务器")
            return
            
        try:
            # 构建信号数据
            signal_data = {
                "type": signal_type,
                "timestamp": self.get_timestamp()
            }
            
            # 如果是加注，添加额度信息
            if signal_type == "加注":
                amount = self.custom_raise_input.text()
                if amount:
                    signal_data["amount"] = int(amount)
                    
            # 发送数据
            message = json.dumps(signal_data, ensure_ascii=False)
            self.socket.send(message.encode('utf-8'))
            
            self.log_message(f"发送信号: {signal_type}")
            if signal_type == "加注" and "amount" in signal_data:
                self.log_message(f"  加注额度: {signal_data['amount']}")
                
        except Exception as e:
            self.log_message(f"发送失败: {e}")
            QMessageBox.warning(self, "发送失败", f"无法发送信号: {e}")
            
    def send_custom_raise(self):
        """发送自定义加注信号"""
        amount = self.custom_raise_input.text()
        if amount.isdigit():
            signal_data = {
                "type": "加注",
                "amount": int(amount),
                "timestamp": self.get_timestamp()
            }
            
            try:
                message = json.dumps(signal_data, ensure_ascii=False)
                self.socket.send(message.encode('utf-8'))
                self.log_message(f"发送信号: 加注")
                self.log_message(f"  加注额度: {amount}")
            except Exception as e:
                self.log_message(f"发送失败: {e}")
                QMessageBox.warning(self, "发送失败", f"无法发送信号: {e}")
        else:
            QMessageBox.warning(self, "输入错误", "请输入有效的数字")
            
    def get_timestamp(self):
        """获取当前时间戳"""
        import time
        return time.strftime("%Y-%m-%d %H:%M:%S")
        
    def log_message(self, message):
        """记录日志消息"""
        timestamp = self.get_timestamp()
        log_entry = f"[{timestamp}] {message}"
        self.log_text.append(log_entry)
        
        # 自动滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def closeEvent(self, event):
        """窗口关闭事件"""
        if self.socket:
            self.disconnect_from_server()
        event.accept()

def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用程序信息
    app.setApplicationName("扑克信号发送端")
    app.setApplicationVersion("1.0")
    
    # 创建并显示主窗口
    window = SignalSender()
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
