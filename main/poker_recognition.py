#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整合的扑克牌识别程序
包含OCR数字识别、花色识别和按钮位置识别功能
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import re
import warnings
import pytesseract
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import platform

class PokerRecognizer:
    """整合的扑克牌识别器"""
    
    def __init__(self, template_dir: str = "templates", config_path: str = "main/poker_recognition_config.json"):
        """
        初始化识别器
        
        Args:
            template_dir: 模板文件目录
            config_path: 配置文件路径
        """
        print(f"[DEBUG] PokerRecognizer初始化开始, config_path: {config_path}")
        self.template_dir = template_dir
        self.config_path = config_path
        
        try:
            # 加载配置
            print("[DEBUG] 开始加载配置")
            self.config = self.load_config(config_path)
            print(f"[DEBUG] config加载完成: {self.config}")
        except Exception as e:
            print(f"[ERROR] 加载配置时发生异常: {e}")
            raise
        
        # 从配置中获取参考尺寸
        if self.config and 'reference_size' in self.config:
            self.reference_size = self.config['reference_size']
        else:
            self.reference_size = [860, 665]
            print("[WARNING] 使用默认参考尺寸: [860, 665]")
        
        # 打印花色配置信息
        print(f"[DEBUG] hasattr(self, 'suit_adaptive_regions'): {hasattr(self, 'suit_adaptive_regions')}")
        if hasattr(self, 'suit_adaptive_regions'):
            print(f"[DEBUG] self.suit_adaptive_regions: {self.suit_adaptive_regions}")
        print(f"[DEBUG] hasattr(self, 'suit_reference_size'): {hasattr(self, 'suit_reference_size')}")
        if hasattr(self, 'suit_reference_size'):
            print(f"[DEBUG] self.suit_reference_size: {self.suit_reference_size}")
        
        # 设置Tesseract OCR
        self.setup_tesseract()
        
        # 加载模板
        self.load_suit_templates()
        self.load_hand_templates()
        self.load_btn_template()
        self.load_back_template()
        
        # 花色名称映射
        self.suit_names = {
            'spade': '黑桃',
            'heart': '红心',
            'club': '梅花',
            'diamond': '方块'
        }
        self.load_suit_templates()
        self.load_hand_templates()
        self.load_btn_template()
        self.load_back_template()
        
        # 设置输出目录
        self.setup_output_dir()
        print("[DEBUG] PokerRecognizer初始化完成")
    
    def setup_output_dir(self):
        """设置输出目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join("results", timestamp)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[INFO] 输出目录: {self.output_dir}")
    
    def setup_tesseract(self):
        """设置Tesseract OCR"""
        # 检查系统类型
        system = platform.system()
        
        # 尝试不同的Tesseract路径
        tesseract_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"D:\Tesseract-OCR\tesseract.exe",
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract"
        ]
        
        # 尝试设置Tesseract路径
        tesseract_found = False
        for path in tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                tesseract_found = True
                print(f"[INFO] 找到Tesseract: {path}")
                break
        
        if not tesseract_found:
            print("[WARNING] 未找到Tesseract OCR，请确保已安装并添加到PATH")
        
        # 测试Tesseract是否可用
        try:
            # 尝试使用中文语言包
            test_image = Image.new('RGB', (100, 30), color=(255, 255, 255))
            test_image_draw = ImageDraw.Draw(test_image)
            test_image_draw.text((10, 10), "测试", fill=(0, 0, 0))
            
            # 优先使用英文语言包
            languages_to_try = ['eng', 'chi_sim', 'chi_sim+eng']
            language_found = False
            
            for lang in languages_to_try:
                try:
                    pytesseract.image_to_string(test_image, lang=lang)
                    self.language = lang
                    language_found = True
                    print(f"[INFO] 成功加载语言包: {lang}")
                    break
                except Exception as e:
                    print(f"[INFO] 语言包 {lang} 不可用: {str(e)}")
                    continue
            
            if not language_found:
                # 如果没有找到任何语言包，使用默认英文
                self.language = 'eng'
                print("[WARNING] 未找到任何语言包，使用默认英文识别")
            
            # 设置默认配置
            self.default_config = '--oem 3 --psm 6 --dpi 300'
            self.char_whitelist = ''
            print(f"[INFO] Tesseract OCR初始化成功")
            print(f"[INFO] 使用语言包: {self.language}")
            print(f"[INFO] 默认配置: {self.default_config}")
            
        except Exception as e:
            print(f"[ERROR] Tesseract OCR初始化失败: {str(e)}")
            self.default_config = None
    
    def load_config(self, config_file):
        """加载配置文件"""
        print(f"[DEBUG] 开始加载配置文件: {config_file}")
        try:
            if os.path.exists(config_file):
                print(f"[DEBUG] 配置文件存在: {config_file}")
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"[DEBUG] 加载的配置: {config}")
                
                # OCR配置
                ocr_config = config.get('ocr', {})
                print(f"[DEBUG] OCR配置: {ocr_config}")
                if ocr_config:  # 只有当ocr_config不为空时才更新相关配置
                    self.ocr_config = ocr_config
                    self.test_regions = self.ocr_config.get('test_regions', {})
                    print(f"[DEBUG] test_regions: {self.test_regions}")
                    print(f"[DEBUG] test_regions类型: {type(self.test_regions)}")
                    print(f"[DEBUG] test_regions长度: {len(self.test_regions)}")
                    print(f"[DEBUG] test_regions键: {list(self.test_regions.keys())}")
                    self.adaptive_regions = self.ocr_config.get('adaptive_regions', {})
                    print(f"[DEBUG] loaded adaptive_regions: {self.adaptive_regions}")
                    self.reference_size = self.ocr_config.get('reference_size', [860, 665])
                    self.psm_modes = self.ocr_config.get('psm_modes', [6, 8, 7, 13])
                    self.char_whitelist = self.ocr_config.get('char_whitelist', '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz一二三四五六七八九十百千万亿跟注加注弃牌发牌底池公共牌自己的筹码BB大小王红桃黑桃方块梅花AKQJ')
                    self.recognition_strategies = self.ocr_config.get('recognition_strategies', {})
                
                # 花色识别配置
                suit_config = config.get('suit', {})
                if suit_config:  # 只有当suit_config不为空时才更新相关配置
                    self.suit_test_regions = suit_config.get('test_regions', {})
                    print(f"[DEBUG] suit_test_regions: {self.suit_test_regions}")
                    self.suit_adaptive_regions = suit_config.get('adaptive_regions', {})
                    self.suit_reference_size = suit_config.get('reference_size', [860, 665])
                
                # 按钮识别配置
                btn_config = config.get('btn', {})
                if btn_config:  # 只有当btn_config不为空时才更新相关配置
                    self.btn_test_regions = btn_config.get('test_regions', {})
                    print(f"[DEBUG] btn_test_regions: {self.btn_test_regions}")
                    self.btn_adaptive_regions = btn_config.get('adaptive_regions', {})
                    self.btn_reference_size = btn_config.get('reference_size', [860, 665])
                
                print("✅ 已加载配置文件")
                print(f"   OCR识别策略: {len(self.recognition_strategies)} 种")
                print(f"   OCR自适应区域: {len(self.adaptive_regions)} 个")
                print(f"   花色识别区域: {len(self.suit_test_regions)} 个")
                print(f"   按钮识别区域: {len(self.btn_test_regions)} 个")
                
                return config
            else:
                print(f"[WARNING] 配置文件不存在: {config_file}，使用默认配置")
                return {}
        except Exception as e:
            print(f"[ERROR] 配置文件加载失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
    
    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """加载图片"""
        try:
            if not os.path.exists(image_path):
                print(f"[ERROR] 图片文件不存在: {image_path}")
                return None
            
            # 使用PIL加载图片
            image = Image.open(image_path)
            print(f"[INFO] 图片加载成功: {image_path} ({image.width}x{image.height})")
            return image
        except Exception as e:
            print(f"[ERROR] 图片加载失败: {str(e)}")
            return None
    
    def calculate_adaptive_regions(self, image_width: int, image_height: int) -> Dict:
        """计算自适应区域坐标"""
        if not self.adaptive_regions:
            print("[WARNING] 未配置自适应区域")
            return {}
        
        # 计算缩放比例
        ref_width, ref_height = self.reference_size
        scale_x = image_width / ref_width
        scale_y = image_height / ref_height
        
        # 转换自适应区域坐标
        converted_regions = {}
        for region_name, coords in self.adaptive_regions.items():
            if len(coords) == 4:
                x1_ratio, y1_ratio, x2_ratio, y2_ratio = coords
                x1 = int(x1_ratio * image_width)
                y1 = int(y1_ratio * image_height)
                x2 = int(x2_ratio * image_width)
                y2 = int(y2_ratio * image_height)
                
                # 确保坐标在有效范围内
                x1 = max(0, min(x1, image_width))
                y1 = max(0, min(y1, image_height))
                x2 = max(0, min(x2, image_width))
                y2 = max(0, min(y2, image_height))
                
                # 确保左上角坐标小于右下角坐标
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                
                converted_regions[region_name] = [x1, y1, x2, y2]
        
        return converted_regions
    
    def calculate_suit_adaptive_regions(self, image_width: int, image_height: int) -> Dict:
        """计算花色自适应区域坐标"""
        # 使用配置文件中定义的自适应区域坐标
        print(f"[DEBUG] calculate_suit_adaptive_regions called with image_width: {image_width}, image_height: {image_height}")
        print(f"[DEBUG] hasattr(self, 'suit_adaptive_regions'): {hasattr(self, 'suit_adaptive_regions')}")
        print(f"[DEBUG] hasattr(self, 'suit_reference_size'): {hasattr(self, 'suit_reference_size')}")
        
        # 详细检查suit_adaptive_regions
        if hasattr(self, 'suit_adaptive_regions'):
            print(f"[DEBUG] self.suit_adaptive_regions: {self.suit_adaptive_regions}")
            print(f"[DEBUG] type(self.suit_adaptive_regions): {type(self.suit_adaptive_regions)}")
            print(f"[DEBUG] self.suit_adaptive_regions is not None: {self.suit_adaptive_regions is not None}")
            if self.suit_adaptive_regions is not None:
                print(f"[DEBUG] len(self.suit_adaptive_regions): {len(self.suit_adaptive_regions)}")
        else:
            print("[ERROR] 实例没有suit_adaptive_regions属性")
            
        # 详细检查suit_reference_size
        if hasattr(self, 'suit_reference_size'):
            print(f"[DEBUG] self.suit_reference_size: {self.suit_reference_size}")
            print(f"[DEBUG] type(self.suit_reference_size): {type(self.suit_reference_size)}")
            print(f"[DEBUG] self.suit_reference_size is not None: {self.suit_reference_size is not None}")
            if self.suit_reference_size is not None:
                print(f"[DEBUG] len(self.suit_reference_size): {len(self.suit_reference_size)}")
        else:
            print("[ERROR] 实例没有suit_reference_size属性")
        
        # 逐一检查每个条件
        if not hasattr(self, 'suit_adaptive_regions'):
            print("[ERROR] 缺少suit_adaptive_regions属性")
            return {}
        
        if not hasattr(self, 'suit_reference_size'):
            print("[ERROR] 缺少suit_reference_size属性")
            return {}
        
        if self.suit_adaptive_regions is None:
            print("[ERROR] suit_adaptive_regions为None")
            return {}
        
        if self.suit_reference_size is None:
            print("[ERROR] suit_reference_size为None")
            return {}
        
        if not self.suit_adaptive_regions:
            print("[ERROR] suit_adaptive_regions为空")
            print(f"[ERROR] suit_adaptive_regions值: {self.suit_adaptive_regions}")
            return {}
        
        if not self.suit_reference_size:
            print("[ERROR] suit_reference_size为空")
            print(f"[ERROR] suit_reference_size值: {self.suit_reference_size}")
            return {}
        
        print("[DEBUG] 所有检查通过")
        reference_width, reference_height = self.suit_reference_size
        adaptive_regions = self.suit_adaptive_regions
        
        # 转换自适应区域坐标
        converted_regions = {}
        for region_name, ratios in adaptive_regions.items():
            if len(ratios) == 4:
                x1_ratio, y1_ratio, x2_ratio, y2_ratio = ratios
                x1 = int(x1_ratio * image_width)
                y1 = int(y1_ratio * image_height)
                x2 = int(x2_ratio * image_width)
                y2 = int(y2_ratio * image_height)
                
                # 确保坐标在有效范围内
                x1 = max(0, min(x1, image_width))
                y1 = max(0, min(y1, image_height))
                x2 = max(0, min(x2, image_width))
                y2 = max(0, min(y2, image_height))
                
                # 确保左上角坐标小于右下角坐标
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                
                converted_regions[region_name] = [x1, y1, x2, y2]
        
        return converted_regions

    def preprocess_image_for_chinese(self, image: Image.Image, enable_grayscale=True) -> Image.Image:
        """为中文识别预处理图片"""
        # 转换为灰度图
        if enable_grayscale:
            image = image.convert('L')
        
        # 放大图片2倍以提高识别精度
        width, height = image.size
        image = image.resize((width * 2, height * 2), Image.LANCZOS)
        
        # 增强对比度
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)  # 增加对比度
        
        return image
    
    def get_recognition_strategy(self, region_name: str) -> str:
        """获取区域识别策略"""
        # 查找匹配的识别策略
        for strategy_name, strategy in self.recognition_strategies.items():
            regions = strategy.get('regions', [])
            if region_name in regions:
                config = strategy.get('config', self.default_config)
                # 添加字符白名单到配置
                if self.char_whitelist:
                    config += f' -c tessedit_char_whitelist={self.char_whitelist}'
                return config
        
        # 如果没有找到匹配的策略，使用默认配置
        config = self.default_config
        # 添加字符白名单到配置
        if self.char_whitelist:
            config += f' -c tessedit_char_whitelist={self.char_whitelist}'
        return config
    
    def recognize_regions(self, image, regions=None):
        """识别指定区域"""
        print(f"[DEBUG] recognize_regions called with regions: {regions}")
        print(f"[DEBUG] self.test_regions: {self.test_regions}")
        print(f"[DEBUG] self.test_regions type: {type(self.test_regions)}")
        print(f"[DEBUG] self.test_regions length: {len(self.test_regions)}")
        if not self.default_config:
            print("[ERROR] Tesseract OCR未初始化")
            return None
        
        if regions is None:
            regions = self.test_regions
            print(f"[DEBUG] Using self.test_regions: {self.test_regions}")
        
        if not regions:
            print("[ERROR] 未配置识别区域")
            return None
        print(f"[DEBUG] Final regions to process: {regions}")
        
        # 如果regions中的坐标是相对坐标（0-1之间），需要转换为绝对坐标
        converted_regions = {}
        image_width, image_height = image.size
        for region_name, coords in regions.items():
            # 检查坐标是否为相对坐标（0-1之间）
            if all(0 <= coord <= 1 for coord in coords):
                # 转换为绝对坐标
                x1_ratio, y1_ratio, x2_ratio, y2_ratio = coords
                x1 = int(x1_ratio * image_width)
                y1 = int(y1_ratio * image_height)
                x2 = int(x2_ratio * image_width)
                y2 = int(y2_ratio * image_height)
                
                # 确保坐标在有效范围内
                x1 = max(0, min(x1, image_width))
                y1 = max(0, min(y1, image_height))
                x2 = max(0, min(x2, image_width))
                y2 = max(0, min(y2, image_height))
                
                # 确保左上角坐标小于右下角坐标
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                
                converted_regions[region_name] = [x1, y1, x2, y2]
                # print(f"[INFO] 区域 {region_name} 坐标已转换: {coords} -> [{x1}, {y1}, {x2}, {y2}]")
            else:
                # 假设已经是绝对坐标
                converted_regions[region_name] = coords
                # print(f"[INFO] 区域 {region_name} 使用绝对坐标: {coords}")
        
        # print(f"\n[DIAG] 开始识别指定区域...")
        # print(f"   共 {len(converted_regions)} 个区域")
        
        results = {}
        
        # 为每个区域识别文字
        for i, (region_name, coords) in enumerate(converted_regions.items()):
            try:
                # print(f"\n📍 区域 {i+1}/{len(converted_regions)}: {region_name} {coords}")
                
                # 裁剪区域图片
                x1, y1, x2, y2 = coords
                cropped_image = image.crop((x1, y1, x2, y2))
                
                # 保存裁剪图片（可选）
                # cropped_filename = os.path.join(self.output_dir, f"cropped_{region_name}.png")
                # cropped_image.save(cropped_filename)
                # print(f"   📁 裁剪图片已保存: {cropped_filename}")
                
                # 获取该区域的识别策略
                # print(f"   📌 区域名称: '{region_name}'")
                config = self.get_recognition_strategy(region_name)
                # print(f"   📥 获取到的配置: {config}")
                
                # 检查是否为数字识别配置
                is_digit_only = 'digit' in self.char_whitelist and not any(c in self.char_whitelist for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz一二三四五六七八九十百千万亿跟注加注弃牌发牌底池公共牌自己的筹码BB大小王红桃黑桃方块梅花AKQJ')
                
                # 预处理图片
                processed_image = self.preprocess_image_for_chinese(cropped_image, enable_grayscale=not is_digit_only)
                
                # 识别文字
                # print(f"   🧪 识别配置: {config}")
                # print(f"   🌐 语言设置: {self.language}")
                # print(f"   🧪 调用前配置: {config}")
                text = pytesseract.image_to_string(processed_image, config=config, lang=self.language)
                # print(f"   🧪 调用后配置: {config}")
                
                # 清理文本
                cleaned_text = re.sub(r'\s+', ' ', text.strip())
                
                # 保存结果
                region_results = {
                    'text': cleaned_text,
                    'coords': coords,
                    'success': bool(cleaned_text)
                }
                
                results[region_name] = region_results
                
                if cleaned_text:
                    # print(f"   [SUCCESS] 识别结果: '{cleaned_text}'")
                    pass
                else:
                    # print(f"   [WARNING] 未识别到文字")
                    pass
                    
            except Exception as e:
                # print(f"   [ERROR] 区域识别失败: {str(e)}")
                results[region_name] = {
                    'text': '',
                    'coords': coords,
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    def save_recognition_result(self, results, result_type="unknown"):
        """保存识别结果到文件"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = os.path.join(self.output_dir, f"{result_type}_result_{timestamp}.json")
            
            # 保存结果
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"   [INFO] 结果已保存: {result_file}")
            
        except Exception as e:
            print(f"   [ERROR] 保存结果失败: {str(e)}")
    
    def create_marked_image(self, image, regions=None, results=None):
        """创建标记了识别区域和结果的图片"""
        try:
            # 创建标记图片
            marked_image = image.copy()
            draw = ImageDraw.Draw(marked_image)
            
            # 定义颜色
            colors = [
                (255, 0, 0),    # 红色
                (0, 255, 0),    # 绿色
                (0, 0, 255),    # 蓝色
                (255, 255, 0),  # 黄色
                (255, 0, 255),  # 紫色
            ]
            
            # 尝试加载字体
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                try:
                    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 12)
                except:
                    font = ImageFont.load_default()
            
            if regions is None:
                regions = self.test_regions
            
            # 绘制区域标记
            for i, (region_name, coords) in enumerate(regions.items()):
                x1, y1, x2, y2 = coords
                
                # 确保坐标顺序正确
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                
                color = colors[i % len(colors)]
                
                # 绘制矩形框
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                
                # 绘制区域名称
                text = f"{region_name} ({x1},{y1},{x2},{y2})"
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # 计算文本位置
                text_x = max(0, min(x1, image.width - text_width))
                text_y = max(0, min(y1 - text_height - 5, image.height - text_height))
                
                # 绘制文本背景
                draw.rectangle([text_x-2, text_y-2, text_x+text_width+2, text_y+text_height+2], 
                             fill=(0, 0, 0, 128), outline=color)
                
                # 绘制文本
                draw.text((text_x, text_y), text, fill=color, font=font)
                
                # 如果识别结果，显示识别结果
                if results and region_name in results:
                    region_results = results[region_name]
                    if isinstance(region_results, dict) and region_results.get('success') and region_results.get('text'):
                        best_result = region_results.get('text')
                        
                        # 在区域下方显示识别结果
                        result_text = f"识别: {best_result[:20]}..."
                        result_bbox = draw.textbbox((0, 0), result_text, font=font)
                        result_width = result_bbox[2] - result_bbox[0]
                        result_height = result_bbox[3] - result_bbox[1]
                        
                        result_x = max(0, min(x1, image.width - result_width))
                        result_y = y2 + 5
                        
                        # 绘制结果背景
                        draw.rectangle([result_x-2, result_y-2, result_x+result_width+2, result_y+result_height+2], 
                                     fill=(0, 255, 0, 128), outline=(0, 255, 0))
                        
                        # 绘制结果文本
                        draw.text((result_x, result_y), result_text, fill=(0, 255, 0), font=font)
            
            # 保存标记图片
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            marked_filename = os.path.join(self.output_dir, f"marked_image_{timestamp}.png")
            marked_image.save(marked_filename)
            
            print(f"[INFO] 标记图片已保存: {marked_filename}")
            return marked_filename
            
        except Exception as e:
            print(f"[ERROR] 创建标记图片失败: {str(e)}")
            return None

    # 花色识别相关方法
    def load_suit_templates(self):
        """加载花色模板"""
        self.suit_templates = {}
        
        # 加载公共牌花色模板
        public_template_dir = "1/new_templates/public"
        if os.path.exists(public_template_dir):
            for filename in os.listdir(public_template_dir):
                if filename.endswith(".png") or filename.endswith(".jpg"):
                    suit_name = os.path.splitext(filename)[0]
                    template_path = os.path.join(public_template_dir, filename)
                    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                    if template is not None:
                        self.suit_templates[f"public_{suit_name}"] = template
                        print(f"[INFO] 加载公共牌花色模板: {suit_name}")
                    else:
                        print(f"[ERROR] 无法加载公共牌花色模板: {template_path}")
        else:
            print(f"[WARNING] 公共牌花色模板目录不存在: {public_template_dir}")
        
        # 加载手牌1花色模板
        hand1_template_dir = "1/new_templates/hand1"
        if os.path.exists(hand1_template_dir):
            for filename in os.listdir(hand1_template_dir):
                if filename.endswith(".png") or filename.endswith(".jpg"):
                    suit_name = os.path.splitext(filename)[0]
                    template_path = os.path.join(hand1_template_dir, filename)
                    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                    if template is not None:
                        self.suit_templates[f"hand1_{suit_name}"] = template
                        print(f"[INFO] 加载手牌1花色模板: {suit_name}")
                    else:
                        print(f"[ERROR] 无法加载手牌1花色模板: {template_path}")
        else:
            print(f"[WARNING] 手牌1花色模板目录不存在: {hand1_template_dir}")
        
        # 加载手牌2花色模板
        hand2_template_dir = "1/new_templates/hand2"
        if os.path.exists(hand2_template_dir):
            for filename in os.listdir(hand2_template_dir):
                if filename.endswith(".png") or filename.endswith(".jpg"):
                    suit_name = os.path.splitext(filename)[0]
                    template_path = os.path.join(hand2_template_dir, filename)
                    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                    if template is not None:
                        self.suit_templates[f"hand2_{suit_name}"] = template
                        print(f"[INFO] 加载手牌2花色模板: {suit_name}")
                    else:
                        print(f"[ERROR] 无法加载手牌2花色模板: {template_path}")
        else:
            print(f"[WARNING] 手牌2花色模板目录不存在: {hand2_template_dir}")
    
    def load_hand_templates(self):
        """加载手牌模板"""
        self.hand_templates = {}
        template_dir = "1/new_templates"
        
        if not os.path.exists(template_dir):
            print(f"[WARNING] 手牌模板目录不存在: {template_dir}")
            return
        
        for filename in os.listdir(template_dir):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                hand_name = os.path.splitext(filename)[0]
                template_path = os.path.join(template_dir, filename)
                template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    self.hand_templates[hand_name] = template
                    print(f"[INFO] 加载手牌模板: {hand_name}")
                else:
                    print(f"[ERROR] 无法加载手牌模板: {template_path}")
    
    def load_btn_template(self):
        """加载按钮模板"""
        template_path = "1/new_templates/btn.png"
        if os.path.exists(template_path):
            self.btn_template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            print(f"[INFO] 加载按钮模板: {template_path}")
        else:
            print(f"[ERROR] 按钮模板不存在: {template_path}")
            self.btn_template = None
    
    def load_back_template(self):
        """加载扑克背面模板"""
        template_path = "1/new_templates/back.png"
        if os.path.exists(template_path):
            self.back_template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            print(f"[INFO] 加载扑克背面模板: {template_path}")
        else:
            print(f"[ERROR] 扑克背面模板不存在: {template_path}")
            self.back_template = None

    def match_suit_template(self, image: np.ndarray, template: np.ndarray, 
                      threshold: float = 0.6) -> Tuple[float, Tuple[int, int]]:
        """
        花色模板匹配
        
        Args:
            image: 待匹配图像
            template: 模板图像
            threshold: 匹配阈值
            
        Returns:
            (最大匹配度, 最佳匹配位置)
        """
        # 使用单一匹配方法
        method = cv2.TM_CCOEFF_NORMED
        
        try:
            result = cv2.matchTemplate(image, template, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # 对于不同的匹配方法，处理方式不同
            if method == cv2.TM_SQDIFF_NORMED:
                score = 1 - min_val
                loc = min_loc
            else:
                score = max_val
                loc = max_loc
        except cv2.error as e:
            # 如果匹配失败，返回默认值
            print(f"模板匹配错误: {e}")
            score = -1.0
            loc = (0, 0)
        
        return score, loc

    def recognize_suit(self, image: np.ndarray, threshold: float = 0.6) -> Dict:
        """
        识别图像中的花色
        
        Args:
            image: 输入图像
            threshold: 匹配阈值
            
        Returns:
            识别结果字典
        """
        # 转换为灰度图但不进行其他预处理
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image.copy()
        
        results = {}
        best_match = None
        best_score = 0
        
        # 对每个模板进行多尺度匹配
        for suit_name, template in self.suit_templates.items():
            # 转换模板为灰度图
            if len(template.shape) == 3:
                gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            else:
                gray_template = template.copy()
            
            # 尝试不同的缩放比例
            scales = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
            max_scale_score = -1
            max_scale_position = (0, 0)
            
            for scale in scales:
                # 缩放模板
                h, w = gray_template.shape
                new_h, new_w = int(h * scale), int(w * scale)
                resized_template = cv2.resize(gray_template, (new_w, new_h))
                
                # 确保缩放后的模板不大于图像
                if resized_template.shape[0] > gray_image.shape[0] or resized_template.shape[1] > gray_image.shape[1]:
                    continue
                
                # 进行模板匹配
                score, position = self.match_suit_template(gray_image, resized_template, threshold)
                
                # 更新最佳匹配
                if score > max_scale_score:
                    max_scale_score = score
                    max_scale_position = position
            
            # 保存该模板的匹配结果
            results[suit_name] = {
                'score': max_scale_score,
                'position': max_scale_position,
                'scale': scale  # 使用实际的缩放比例
            }
            
            # 更新全局最佳匹配
            if max_scale_score > best_score:
                best_score = max_scale_score
                # 从suit_name中提取花色名称
                if suit_name.startswith('public_'):
                    display_name = suit_name[7:]  # 去掉'public_'前缀
                elif suit_name.startswith('hand1_'):
                    display_name = suit_name[6:]  # 去掉'hand1_'前缀
                elif suit_name.startswith('hand2_'):
                    display_name = suit_name[6:]  # 去掉'hand2_'前缀
                else:
                    display_name = suit_name
                
                # 进一步提取基本花色名称
                if '_' in display_name:
                    base_suit_name = display_name.split('_')[-1]  # 取最后一个下划线后的内容
                else:
                    base_suit_name = display_name
                
                best_match = {
                    'suit_name': suit_name,
                    'chinese_name': self.suit_names.get(base_suit_name, display_name),
                    'score': max_scale_score,
                    'position': max_scale_position
                }
        
        # 返回结果
        return {
            'results': results,
            'best_match': best_match,
            'threshold': threshold
        }



    def recognize_suits_in_regions(self, image: np.ndarray, regions: Dict, threshold: float = 0.6) -> Dict:
        """在指定区域中识别花色"""
        results = {}
        
        for region_name, coords in regions.items():
            if len(coords) >= 4:
                x1, y1, x2, y2 = coords[:4]
                # 确保坐标是整数
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # 裁剪区域
                cropped_image = image[y1:y2, x1:x2]
                
                # 识别花色
                result = self.recognize_suit(cropped_image, threshold)
                results[region_name] = result
        
        return results

    def visualize_suit_regions(self, image: np.ndarray, regions: Dict, results: Dict, output_path: str) -> str:
        """可视化花色识别区域和结果"""
        # 创建标记图片
        marked_image = image.copy()
        
        # 定义颜色
        colors = [
            (255, 0, 0),    # 红色
            (0, 255, 0),    # 绿色
            (0, 0, 255),    # 蓝色
            (255, 255, 0),  # 黄色
            (255, 0, 255),  # 紫色
        ]
        
        # 绘制区域标记
        for i, (region_name, coords) in enumerate(regions.items()):
            if len(coords) >= 4:
                x1, y1, x2, y2 = coords[:4]
                # 确保坐标是整数
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                color = colors[i % len(colors)]
                
                # 绘制矩形框
                cv2.rectangle(marked_image, (x1, y1), (x2, y2), color, 2)
                
                # 绘制区域名称
                cv2.putText(marked_image, region_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # 如果有识别结果，显示最佳匹配
                if region_name in results:
                    result = results[region_name]
                    if 'best_match' in result and result['best_match']:
                        best = result['best_match']
                        score = best['score']
                        chinese_name = best['chinese_name']
                        
                        # 在区域下方显示识别结果
                        result_text = f"{chinese_name} ({score:.2f})"
                        cv2.putText(marked_image, result_text, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 保存标记图片
        cv2.imwrite(output_path, marked_image)
        print(f"[SAVE] 花色标记图片已保存: {output_path}")
        return output_path

    # 按钮识别相关方法
    def recognize_btn_multiscale(self, image: np.ndarray, threshold: float = 0.6) -> Dict:
        """
        多尺度按钮识别
        
        Args:
            image: 输入图像
            threshold: 匹配阈值
            
        Returns:
            识别结果字典
        """
        if self.btn_template is None:
            return {'error': '按钮模板未加载'}
        
        # 转换为灰度图
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image.copy()
        
        # 尝试不同的缩放比例
        scales = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
        best_score = -1
        best_position = (0, 0)
        best_scale = 1.0
        
        for scale in scales:
            # 缩放模板
            h, w = self.btn_template.shape
            new_h, new_w = int(h * scale), int(w * scale)
            resized_template = cv2.resize(self.btn_template, (new_w, new_h))
            
            # 确保缩放后的模板不大于图像
            if resized_template.shape[0] > gray_image.shape[0] or resized_template.shape[1] > gray_image.shape[1]:
                continue
            
            # 进行模板匹配
            method = cv2.TM_CCOEFF_NORMED
            try:
                result = cv2.matchTemplate(gray_image, resized_template, method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                # 对于不同的匹配方法，处理方式不同
                if method == cv2.TM_SQDIFF_NORMED:
                    score = 1 - min_val
                    loc = min_loc
                else:
                    score = max_val
                    loc = max_loc
                
                # 更新最佳匹配
                if score > best_score and score >= threshold:
                    best_score = score
                    best_position = loc
                    best_scale = scale
            except cv2.error as e:
                # 如果匹配失败，继续尝试其他缩放比例
                print(f"按钮匹配错误 (scale={scale}): {e}")
                continue
        
        # 返回结果
        if best_score >= threshold:
            return {
                'matched': True,
                'position': best_position,
                'score': best_score,
                'scale': best_scale
            }
        else:
            return {
                'matched': False,
                'score': best_score
            }

    def recognize_btn(self, image: np.ndarray, threshold: float = 0.6, multiscale: bool = True) -> Dict:
        """
        识别图像中的按钮
        
        Args:
            image: 输入图像
            threshold: 匹配阈值
            multiscale: 是否使用多尺度匹配
            
        Returns:
            识别结果字典
        """
        if self.btn_template is None:
            return {'error': '按钮模板未加载'}
        
        if multiscale:
            return self.recognize_btn_multiscale(image, threshold)
        
        # 转换为灰度图
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image.copy()
        
        # 进行模板匹配
        method = cv2.TM_CCOEFF_NORMED
        try:
            result = cv2.matchTemplate(gray_image, self.btn_template, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # 对于不同的匹配方法，处理方式不同
            if method == cv2.TM_SQDIFF_NORMED:
                score = 1 - min_val
                loc = min_loc
            else:
                score = max_val
                loc = max_loc
            
            # 返回结果
            if score >= threshold:
                return {
                    'matched': True,
                    'position': loc,
                    'score': score
                }
            else:
                return {
                    'matched': False,
                    'score': score
                }
        except cv2.error as e:
            return {'error': f'按钮匹配错误: {e}'}

    def calculate_btn_adaptive_regions(self, image_width: int, image_height: int) -> Dict:
        """计算按钮自适应区域坐标"""
        # 使用配置文件中定义的按钮自适应区域坐标
        if not hasattr(self, 'config') or not self.config or 'btn' not in self.config or 'adaptive_regions' not in self.config['btn'] or 'reference_size' not in self.config['btn']:
            print("[ERROR] 配置文件缺少必要的按钮配置项")
            return {}
        
        reference_width, reference_height = self.config['btn']['reference_size']
        adaptive_regions = self.config['btn']['adaptive_regions']
        
        # 转换自适应区域坐标
        converted_regions = {}
        for region_name, ratios in adaptive_regions.items():
            if len(ratios) == 4:
                x1_ratio, y1_ratio, x2_ratio, y2_ratio = ratios
                x1 = int(x1_ratio * image_width)
                y1 = int(y1_ratio * image_height)
                x2 = int(x2_ratio * image_width)
                y2 = int(y2_ratio * image_height)
                
                # 确保坐标在有效范围内
                x1 = max(0, min(x1, image_width))
                y1 = max(0, min(y1, image_height))
                x2 = max(0, min(x2, image_width))
                y2 = max(0, min(y2, image_height))
                
                # 确保左上角坐标小于右下角坐标
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                
                converted_regions[region_name] = [x1, y1, x2, y2]
        
        return converted_regions

    def determine_table_position(self, x: int, y: int, image_width: int, image_height: int) -> Tuple[str, float]:
        """
        根据按钮检测位置确定牌桌位置
        
        Args:
            x: 按钮中心x坐标
            y: 按钮中心y坐标
            image_width: 图像宽度
            image_height: 图像高度
            
        Returns:
            (位置名称, 置信度)
        """
        # 定义牌桌区域
        regions = {
            "庄家": (0.45, 0.10, 0.55, 0.20),
            "小盲注": (0.10, 0.30, 0.20, 0.40),
            "大盲注": (0.20, 0.60, 0.30, 0.70),
            "玩家1": (0.40, 0.70, 0.50, 0.80),
            "玩家2": (0.60, 0.70, 0.70, 0.80),
            "玩家3": (0.80, 0.60, 0.90, 0.70),
            "玩家4": (0.80, 0.30, 0.90, 0.40),
            "玩家5": (0.60, 0.10, 0.70, 0.20),
            "玩家6": (0.20, 0.10, 0.30, 0.20)
        }
        
        # 将坐标转换为相对坐标
        x_ratio = x / image_width
        y_ratio = y / image_height
        
        # 查找最匹配的区域
        best_region = "未知"
        best_distance = float('inf')
        
        for region_name, (x1_ratio, y1_ratio, x2_ratio, y2_ratio) in regions.items():
            # 计算区域中心
            center_x = (x1_ratio + x2_ratio) / 2
            center_y = (y1_ratio + y2_ratio) / 2
            
            # 计算距离
            distance = ((x_ratio - center_x) ** 2 + (y_ratio - center_y) ** 2) ** 0.5
            
            # 更新最佳匹配
            if distance < best_distance:
                best_distance = distance
                best_region = region_name
        
        # 计算置信度（基于距离）
        # 假设最大距离为0.5（对角线的一半）
        max_distance = 0.5
        confidence = max(0, 1 - best_distance / max_distance)
        
        return best_region, confidence

    def visualize_btn_result(self, image: np.ndarray, result: Dict, output_path: str) -> str:
        """可视化按钮识别结果"""
        # 创建标记图片
        marked_image = image.copy()
        
        # 如果匹配成功，绘制按钮位置
        if 'error' not in result and result.get('matched', False):
            x, y = result['position']
            score = result['score']
            
            # 如果有缩放信息，计算按钮大小
            if 'scale' in result:
                h, w = int(self.btn_template.shape[0] * result['scale']), int(self.btn_template.shape[1] * result['scale'])
            else:
                h, w = self.btn_template.shape
            
            # 绘制矩形框
            cv2.rectangle(marked_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 绘制匹配分数
            cv2.putText(marked_image, f"BTN ({score:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 保存标记图片
        cv2.imwrite(output_path, marked_image)
        print(f"[SAVE] 按钮标记图片已保存: {output_path}")
        return output_path

    # 扑克背面识别（弃牌检测）相关方法
    def check_player_fold_by_variance(self, image: np.ndarray, variance_threshold: float = 10.0) -> Dict:
        """
        通过颜色方差检测玩家是否弃牌
        
        Args:
            image: 输入图像
            variance_threshold: 方差阈值
            
        Returns:
            检测结果字典
        """
        # 定义玩家区域（相对于参考尺寸）
        player_regions = {
            "玩家1": [0.40, 0.70, 0.50, 0.80],
            "玩家2": [0.60, 0.70, 0.70, 0.80],
            "玩家3": [0.80, 0.60, 0.90, 0.70],
            "玩家4": [0.80, 0.30, 0.90, 0.40],
            "玩家5": [0.60, 0.10, 0.70, 0.20],
            "玩家6": [0.20, 0.10, 0.30, 0.20]
        }
        
        # 计算图像尺寸
        image_height, image_width = image.shape[:2]
        
        # 计算缩放比例
        ref_width, ref_height = self.reference_size
        scale_x = image_width / ref_width
        scale_y = image_height / ref_height
        
        results = {}
        
        # 检查每个玩家区域
        for player_name, coords in player_regions.items():
            if len(coords) == 4:
                x1_ratio, y1_ratio, x2_ratio, y2_ratio = coords
                x1 = int(x1_ratio * image_width)
                y1 = int(y1_ratio * image_height)
                x2 = int(x2_ratio * image_width)
                y2 = int(y2_ratio * image_height)
                
                # 确保坐标在有效范围内
                x1 = max(0, min(x1, image_width))
                y1 = max(0, min(y1, image_height))
                x2 = max(0, min(x2, image_width))
                y2 = max(0, min(y2, image_height))
                
                # 确保区域有效
                if x2 > x1 and y2 > y1:
                    # 裁剪玩家区域
                    player_region = image[y1:y2, x1:x2]
                    
                    # 转换为HSV颜色空间
                    hsv = cv2.cvtColor(player_region, cv2.COLOR_BGR2HSV)
                    
                    # 计算饱和度通道的方差
                    saturation = hsv[:,:,1]
                    variance = np.var(saturation)
                    
                    # 根据方差判断是否弃牌
                    # 如果方差小于阈值，认为是弃牌（灰色扑克背面）
                    folded = variance < variance_threshold
                    
                    # 计算置信度
                    # 假设方差在0-100之间，0表示完全灰色，100表示彩色丰富
                    confidence = max(0, min(1, 1 - variance / 100))
                    
                    results[player_name] = {
                        'folded': folded,
                        'variance': variance,
                        'confidence': confidence
                    }
                else:
                    results[player_name] = {
                        'error': '区域坐标无效'
                    }
            else:
                results[player_name] = {
                    'error': '区域坐标格式错误'
                }
        
        return results

    def visualize_fold_result(self, image: np.ndarray, results: Dict, output_path: str) -> str:
        """可视化弃牌检测结果"""
        # 创建标记图片
        marked_image = image.copy()
        
        # 定义玩家区域（相对于参考尺寸）
        player_regions = {
            "玩家1": [0.40, 0.70, 0.50, 0.80],
            "玩家2": [0.60, 0.70, 0.70, 0.80],
            "玩家3": [0.80, 0.60, 0.90, 0.70],
            "玩家4": [0.80, 0.30, 0.90, 0.40],
            "玩家5": [0.60, 0.10, 0.70, 0.20],
            "玩家6": [0.20, 0.10, 0.30, 0.20]
        }
        
        # 计算图像尺寸
        image_height, image_width = image.shape[:2]
        
        # 计算缩放比例
        ref_width, ref_height = self.reference_size
        scale_x = image_width / ref_width
        scale_y = image_height / ref_height
        
        # 绘制每个玩家区域的检测结果
        for player_name, coords in player_regions.items():
            if len(coords) == 4:
                x1_ratio, y1_ratio, x2_ratio, y2_ratio = coords
                x1 = int(x1_ratio * image_width)
                y1 = int(y1_ratio * image_height)
                x2 = int(x2_ratio * image_width)
                y2 = int(y2_ratio * image_height)
                
                # 确保坐标在有效范围内
                x1 = max(0, min(x1, image_width))
                y1 = max(0, min(y1, image_height))
                x2 = max(0, min(x2, image_width))
                y2 = max(0, min(y2, image_height))
                
                # 绘制矩形框
                if player_name in results and 'folded' in results[player_name]:
                    folded = results[player_name]['folded']
                    variance = results[player_name]['variance']
                    confidence = results[player_name]['confidence']
                    
                    # 根据弃牌状态选择颜色
                    color = (0, 0, 255) if folded else (0, 255, 0)  # 红色表示弃牌，绿色表示未弃牌
                    
                    # 绘制矩形框
                    cv2.rectangle(marked_image, (x1, y1), (x2, y2), color, 2)
                    
                    # 绘制文本
                    status_text = "弃牌" if folded else "未弃牌"
                    text = f"{player_name}: 状态 {status_text} ({variance:.1f})"
                    cv2.putText(marked_image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                elif player_name in results and 'error' in results[player_name]:
                    # 绘制错误状态
                    color = (255, 255, 0)  # 黄色表示错误
                    cv2.rectangle(marked_image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(marked_image, f"{player_name}: 错误", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 保存标记图片
        cv2.imwrite(output_path, marked_image)
        print(f"[SAVE] 弃牌检测标记图片已保存: {output_path}")
        return output_path

    def visualize_back_result(self, image: np.ndarray, results: Dict, output_path: str) -> str:
        """可视化扑克背面识别结果"""
        # 创建标记图片
        marked_image = image.copy()
        
        # 定义玩家区域（相对于参考尺寸）
        player_regions = {
            "玩家1": [95, 425, 120, 450],
            "玩家2": [120, 180, 146, 205],
            "玩家3": [425, 120, 452, 145],
            "玩家4": [732, 180, 760, 205],
            "玩家5": [760, 425, 785, 450]
        }
        
        # 计算图像尺寸
        image_height, image_width = image.shape[:2]
        
        # 计算缩放比例
        ref_width, ref_height = self.reference_size
        scale_x = image_width / ref_width
        scale_y = image_height / ref_height
        
        # 绘制每个玩家区域的检测结果
        for player_name, coords in player_regions.items():
            if len(coords) == 4:
                x1_ref, y1_ref, x2_ref, y2_ref = coords
                x1 = int(x1_ref * scale_x)
                y1 = int(y1_ref * scale_y)
                x2 = int(x2_ref * scale_x)
                y2 = int(y2_ref * scale_y)
                
                # 确保坐标在有效范围内
                x1 = max(0, min(x1, image_width))
                y1 = max(0, min(y1, image_height))
                x2 = max(0, min(x2, image_width))
                y2 = max(0, min(y2, image_height))
                
                # 绘制矩形框
                if player_name in results and 'folded' in results[player_name]:
                    folded = results[player_name]['folded']
                    confidence = results[player_name]['confidence']
                    
                    # 根据弃牌状态选择颜色
                    color = (0, 0, 255) if folded else (0, 255, 0)  # 红色表示弃牌，绿色表示未弃牌
                    
                    # 绘制矩形框
                    cv2.rectangle(marked_image, (x1, y1), (x2, y2), color, 2)
                    
                    # 绘制文本
                    status_text = "弃牌" if folded else "未弃牌"
                    text = f"{player_name}: 状态 {status_text} ({confidence:.3f})"
                    cv2.putText(marked_image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                elif player_name in results and 'error' in results[player_name]:
                    # 绘制错误状态
                    color = (255, 255, 0)  # 黄色表示错误
                    cv2.rectangle(marked_image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(marked_image, f"{player_name}: 错误", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 保存标记图片
        cv2.imwrite(output_path, marked_image)
        print(f"[SAVE] 扑克背面识别标记图片已保存: {output_path}")
        return output_path

    # 主要功能接口
    def recognize_all(self, image_path: str):
        """
        对图像进行所有类型的识别
        
        Args:
            image_path: 图像路径
        """
        print(f"\n{'='*60}")
        print(f"[START] 开始对图像进行所有类型识别: {image_path}")
        print(f"{'='*60}")
        
        # 加载图像
        image_pil = self.load_image(image_path)
        if image_pil is None:
            return
        
        # 转换为OpenCV格式
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        # 1. OCR识别
        print(f"\n{'='*40}")
        print("1. OCR识别")
        print(f"{'='*40}")
        
        # 识别指定区域
        region_results = self.recognize_regions(image_pil)
        
        # 创建标记图片
        if region_results is not None:
            print(f"\n[CREATE] 创建OCR标记图片...")
            self.create_marked_image(image_pil, self.test_regions, region_results)
            
            # 使用自适应区域创建标记图片
            adaptive_regions = self.calculate_adaptive_regions(image_pil.width, image_pil.height)
            self.create_marked_image(image_pil, adaptive_regions, region_results)
        else:
            print("[ERROR] OCR识别失败，跳过创建标记图片")
        
        # # 2. 花色识别
        # print(f"\n{'='*40}")
        # print("2. 花色识别")
        # print(f"{'='*40}")
        
        # 计算自适应区域
        image_height, image_width = image_cv.shape[:2]
        adaptive_suit_regions = self.calculate_suit_adaptive_regions(image_width, image_height)
        
        # 识别所有花色区域
        suit_results = self.recognize_suits_in_regions(image_cv, adaptive_suit_regions, threshold=0.6)
        
        # 调试输出suit_results
        print(f"[DEBUG] suit_results: {suit_results}")
        
        # 可视化结果
        vis_suit_image = self.visualize_suit_regions(image_cv, adaptive_suit_regions, suit_results, 
                                                    f"{self.output_dir}/suit_result.png")
        
        # 3. 按钮识别
        # print(f"\n{'='*40}")
        # print("3. 按钮识别")
        # print(f"{'='*40}")
        
        # 进行按钮识别
        btn_result = self.recognize_btn(image_cv, threshold=0.6, multiscale=True)
        
        # 显示结果
        if 'error' in btn_result:
            # print(f"  错误: {btn_result['error']}")
            pass
        else:
            if btn_result['matched']:
                x, y = btn_result['position']
                if 'scale' in btn_result:
                    h, w = int(self.btn_template.shape[0] * btn_result['scale']), int(self.btn_template.shape[1] * btn_result['scale'])
                else:
                    h, w = self.btn_template.shape
                # 直接输出按钮位置
                table_position, confidence = self.determine_table_position(x, y, image_width, image_height)
                # print(f"按钮位置: {table_position}")
            else:
                # print(f"  [ERROR] 未匹配到按钮")
                pass
        
        # 可视化按钮识别结果
        vis_btn_image = self.visualize_btn_result(image_cv, btn_result, 
                                                f"{self.output_dir}/btn_result.png")
        
        # 4. 扑克背面识别（弃牌检测）
        # print(f"\n{'='*40}")
        # print("[INFO] 4. 扑克背面识别（弃牌检测）")
        # print(f"{'='*40}")
        
        # 检查玩家是否弃牌（使用颜色方差方法）
        fold_status_variance = self.check_player_fold_by_variance(image_cv, variance_threshold=10.0)
        if 'error' in fold_status_variance:
            # print(f"  错误: {fold_status_variance['error']}")
            pass
        else:
            for player, status in fold_status_variance.items():
                if 'error' in status:
                    # print(f"    {player}: 错误 - {status['error']}")
                    pass
                else:
                    if status['folded']:
                        # print(f"    {player}: 未弃牌 (方差: {status['variance']:.2f}, 置信度: {status['confidence']:.3f})")
                        pass
                    else:
                        # print(f"    {player}: 已弃牌 (方差: {status['variance']:.2f}, 置信度: {status['confidence']:.3f})")
                        pass
        
        # 可视化弃牌检测结果
        vis_fold_image = self.visualize_fold_result(image_cv, fold_status_variance, 
                                                 f"{self.output_dir}/fold_result_variance.png")
        
        # 可视化扑克背面识别结果
        vis_back_image = self.visualize_back_result(image_cv, fold_status_variance, 
                                                 f"{self.output_dir}/back_result.png")
        
        # 修改输出格式，只输出指定区域的识别结果
        # print(f"\n{'='*40}")
        # print("最终识别结果")
        # print(f"{'='*40}")
        
        # 输出OCR识别结果
        if region_results:
            # 特别处理raise、deal和min区域
            special_regions = ["raise", "deal", "min"]
            for region_name, result in region_results.items():
                if isinstance(result, dict) and result.get('success') and result.get('text'):
                    # print(f"区域: {region_name} 识别结果: {result['text']}")
                    # 检查是否是特殊区域
                    for special_region in special_regions:
                        if special_region in region_name.lower():
                            print(f"[SPECIAL] {region_name}: {result['text']}")
                    pass
        else:
            # print("[ERROR] OCR识别失败，无结果输出")
            pass
        
        # 合并手牌和花色输出
        suit_output = {}
        if suit_results:
            for region_name, result in suit_results.items():
                if isinstance(result, dict) and 'best_match' in result:
                    best = result['best_match']
                    if isinstance(best, dict):
                        suit_symbol = {'红桃': 'h', '黑桃': 's', '方块': 'd', '梅花': 'c'}.get(best.get('chinese_name', ''), '')
                        suit_output[region_name] = suit_symbol
                        # 输出花色识别结果
                        print(f"{region_name}: {best['chinese_name']} (置信度: {best['confidence']}, 分数: {best['score']:.3f})")
        
        # 输出合并后的手牌和花色结果
        hand_cards = {}
        community_cards = {}
        
        # 处理手牌
        if region_results:
            for i in [1, 2]:
                hand_region = f"自己的手牌{i}"
                suit_region = f"自己的手牌{i}花色"
                if hand_region in region_results and suit_region in suit_output:
                    result = region_results[hand_region]
                    if isinstance(result, dict):
                        hand_text = result.get('text', '')
                        suit_symbol = suit_output[suit_region]
                        if hand_text and suit_symbol:
                            hand_cards[i] = f"{hand_text}{suit_symbol}"
                            print(f"自己的手牌{i}: {hand_cards[i]}")
            
            # 处理公共牌
            for i in range(1, 6):
                community_region = f"公共牌{i}"
                suit_region = f"公共牌{i}花色"
                if community_region in region_results and suit_region in suit_output:
                    result = region_results[community_region]
                    if isinstance(result, dict):
                        community_text = result.get('text', '')
                        suit_symbol = suit_output[suit_region]
                        if community_text and suit_symbol:
                            community_cards[i] = f"{community_text}{suit_symbol}"
                            print(f"公共牌{i}: {community_cards[i]}")
        else:
            print("[ERROR] OCR识别失败，无法处理手牌和公共牌结果")
        
        # 输出按钮位置
        if 'error' not in btn_result and btn_result['matched']:
            table_position, confidence = self.determine_table_position(btn_result['position'][0], btn_result['position'][1], image_width, image_height)
            print(f"按钮位置: {table_position}")
        
        print(f"\n[DONE] 所有识别完成！")
        print(f"[RESULT] 所有结果已保存到目录: {self.output_dir}")
        print(f"[RESULT] 请查看该目录中的文件")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='整合扑克牌识别程序')
    parser.add_argument('image_path', nargs='?', help='图片文件路径')
    parser.add_argument('--english', '-e', action='store_true', help='使用英文识别模式')
    
    args = parser.parse_args()
    
    # print("=" * 60)
    # print("整合扑克牌识别程序")
    # print("支持OCR数字识别、花色识别和按钮位置识别")
    # print("=" * 60)
    
    # 创建识别器
    recognizer = PokerRecognizer()
    
    # 如果指定了英文模式，修改语言设置
    if args.english:
        recognizer.language = 'eng'
        # print("[INFO] 使用英文识别模式")
    
    # 默认使用英文识别模式
    recognizer.language = 'eng'
    
    try:
        # 获取图片路径
        # 获取图片路径
        if args.image_path:
            image_path = args.image_path
        else:
            # 设置默认图片路径
            default_image_path = r"D:\\PycharmProjects\\Poker_Script_Test\\1\\eng\\Snipaste_2025-08-27_23-32-59.png"
            if os.path.exists(default_image_path):
                image_path = default_image_path
                # print(f"[INFO] 使用默认图片路径: {image_path}")
            else:
                # print("\n[INPUT] 请输入图片文件路径:")
                # print("   支持格式: PNG, JPG, JPEG, BMP, TIFF等")
                # print("   示例: test.png 或 C:/path/to/image.jpg")
                
                while True:
                    image_path = input("\n图片路径: ").strip().strip('"')
                    
                    if not image_path:
                        # print("[ERROR] 请输入有效的图片路径")
                        continue
                    
                    # 检查文件是否存在
                    if not os.path.exists(image_path):
                        # print(f"[ERROR] 图片文件不存在: {image_path}")
                        continue
                    
                    break

        # 进行所有识别
        recognizer.recognize_all(image_path)
        
    except Exception as e:
        # print(f"[ERROR] 程序运行出错：{str(e)}")
        pass
    finally:
        # print(f"\n[EXIT] 程序已退出")
        pass


if __name__ == "__main__":
    main()