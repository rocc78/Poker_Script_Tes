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
    
    def __init__(self, config_file='poker_recognition_config.json'):
        """
        初始化扑克牌识别器
        
        Args:
            config_file: 配置文件路径
        """
        print("=== 整合扑克牌识别程序 ===")
        self.setup_output_dir()
        self.setup_tesseract()
        self.load_config(config_file)
        self.suit_names = {
            'club': '梅花',
            'diamond': '方块', 
            'heart': '红桃',
            'spade': '黑桃'
        }
        self.load_suit_templates()
        self.load_btn_template()
        self.load_back_template()
    
    def setup_output_dir(self):
        """设置输出目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"result/poker_recognition_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📁 输出目录: {self.output_dir}")
        
    def setup_tesseract(self):
        """初始化Tesseract OCR"""
        try:
            print("🔄 正在初始化Tesseract OCR...")
            version = pytesseract.get_tesseract_version()
            print(f"✅ Tesseract OCR版本：{version}")
            
            # 默认配置 - 优化中文识别
            self.default_config = '--oem 3 --psm 6 --dpi 300'
            
            # 尝试多种中文语言包配置 - 根据测试结果优化顺序
            language_options = [
                'chi_sim',          # 仅简体中文 - 最佳中文识别效果
                'chi_sim+eng',      # 简体中文+英文 - 备用
                'chi_sim+chi_tra+eng',  # 简体中文+繁体中文+英文
                'eng'               # 仅英文（备用）
            ]
            
            # 测试可用的语言包
            self.language = 'eng'  # 默认英文
            for lang in language_options:
                try:
                    # 测试语言包是否可用
                    test_config = f'--oem 3 --psm 6 -l {lang}'
                    pytesseract.get_tesseract_version()
                    print(f"✅ 语言包 {lang} 可用")
                    self.language = lang
                    break
                except Exception as e:
                    print(f"⚠️  语言包 {lang} 不可用: {str(e)}")
                    continue
            
            print(f"✅ Tesseract OCR初始化成功")
            print(f"   默认配置: {self.default_config}")
            print(f"   使用语言: {self.language}")
            
            # 扩展中文字符白名单
            self.chinese_chars = '一二三四五六七八九十百千万亿跟注加注弃牌发牌底池公共牌自己的筹码BB大小王红桃黑桃方块梅花AKQJ'
            
        except Exception as e:
            print(f"❌ Tesseract OCR初始化失败：{str(e)}")
            print("💡 请确保已安装Tesseract OCR并添加到系统PATH")
            print("💡 对于中文识别，请安装中文语言包：")
            print("   Windows: 下载并安装中文语言包")
            print("   Linux: sudo apt-get install tesseract-ocr-chi-sim")
            print("   macOS: brew install tesseract-lang")
            self.default_config = None
            self.language = 'eng'
            self.chinese_chars = ''
    
    def load_config(self, config_file='poker_recognition_config.json'):
        """加载配置文件"""
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # OCR配置
                self.ocr_config = config.get('ocr', {})
                self.test_regions = self.ocr_config.get('test_regions', {})
                self.adaptive_regions = self.ocr_config.get('adaptive_regions', {})
                self.reference_size = self.ocr_config.get('reference_size', [677, 491])
                self.psm_modes = self.ocr_config.get('psm_modes', [6, 8, 7, 13])
                self.char_whitelist = self.ocr_config.get('char_whitelist', '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz一二三四五六七八九十百千万亿跟注加注弃牌发牌底池公共牌自己的筹码BB大小王红桃黑桃方块梅花AKQJ')
                self.recognition_strategies = self.ocr_config.get('recognition_strategies', {})
                
                # 花色识别配置
                self.suit_config = config.get('suit', {})
                self.suit_test_regions = self.suit_config.get('test_regions', {})
                self.suit_adaptive_regions = self.suit_config.get('adaptive_regions', {})
                self.suit_reference_size = self.suit_config.get('reference_size', [546, 392])
                
                # 按钮识别配置
                self.btn_config = config.get('btn', {})
                self.btn_test_regions = self.btn_config.get('test_regions', {})
                self.btn_adaptive_regions = self.btn_config.get('adaptive_regions', {})
                self.btn_reference_size = self.btn_config.get('reference_size', [546, 392])
                
                print("✅ 已加载配置文件")
                print(f"   OCR识别策略: {len(self.recognition_strategies)} 种")
                print(f"   OCR自适应区域: {len(self.adaptive_regions)} 个")
                print(f"   花色识别区域: {len(self.suit_test_regions)} 个")
                print(f"   按钮识别区域: {len(self.btn_test_regions)} 个")
            else:
                print(f"❌ 配置文件不存在：{config_file}")
                self.ocr_config = {}
                self.test_regions = {}
                self.adaptive_regions = {}
                self.reference_size = [677, 491]
                self.psm_modes = [6, 8, 7, 13]
                self.char_whitelist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz一二三四五六七八九十百千万亿跟注加注弃牌发牌底池公共牌自己的筹码BB大小王红桃黑桃方块梅花AKQJ'
                self.recognition_strategies = {}
                
                self.suit_config = {}
                self.suit_test_regions = {}
                self.suit_adaptive_regions = {}
                self.suit_reference_size = [546, 392]
                
                self.btn_config = {}
                self.btn_test_regions = {}
                self.btn_adaptive_regions = {}
                self.btn_reference_size = [546, 392]
                
        except Exception as e:
            print(f"❌ 加载配置文件失败：{str(e)}")
            # 使用默认配置
            self.ocr_config = {}
            self.test_regions = {}
            self.adaptive_regions = {}
            self.reference_size = [677, 491]
            self.psm_modes = [6, 8, 7, 13]
            self.char_whitelist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz一二三四五六七八九十百千万亿跟注加注弃牌发牌底池公共牌自己的筹码BB大小王红桃黑桃方块梅花AKQJ'
            self.recognition_strategies = {}
            
            self.suit_config = {}
            self.suit_test_regions = {}
            self.suit_adaptive_regions = {}
            self.suit_reference_size = [546, 392]
            
            self.btn_config = {}
            self.btn_test_regions = {}
            self.btn_adaptive_regions = {}
            self.btn_reference_size = [546, 392]
    
    def load_suit_templates(self, template_dir: str = "1/gray_templates"):
        """加载花色模板图片"""
        print("正在加载花色模板...")
        self.suit_templates = {}
        
        # 尝试加载新调整大小的模板
        new_template_dir = Path("1/new_templates/public")
        if new_template_dir.exists():
            template_files = {
                'club': 'club_gray.png',
                'diamond': 'diamond_gray.png', 
                'heart': 'heart_gray.png',
                'spade': 'spade_gray.png'
            }
            
            for suit_name, filename in template_files.items():
                template_path = new_template_dir / filename
                if template_path.exists():
                    # 读取模板图片
                    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
                    if template is not None:
                        self.suit_templates[suit_name] = template
                        print(f"✅ 加载新模板: {filename} -> {self.suit_names[suit_name]}")
                    else:
                        print(f"❌ 无法读取新模板: {filename}")
                else:
                    print(f"❌ 新模板文件不存在: {filename}")
        else:
            print("新模板文件夹不存在，使用原始模板")
            
        # 如果没有加载到新模板，则使用原始模板
        if not self.suit_templates:
            template_files = {
                'club': 'club.png',
                'diamond': 'diamond.png', 
                'heart': 'heart.png',
                'spade': 'spade.png'
            }
            
            for suit_name, filename in template_files.items():
                template_path = Path(template_dir) / filename
                if template_path.exists():
                    # 读取模板图片
                    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
                    if template is not None:
                        self.suit_templates[suit_name] = template
                        print(f"✅ 加载模板: {filename} -> {self.suit_names[suit_name]}")
                    else:
                        print(f"❌ 无法读取模板: {filename}")
                else:
                    print(f"❌ 模板文件不存在: {filename}")
                
        print(f"共加载 {len(self.suit_templates)} 个模板")
    
    def load_btn_template(self, template_path: str = "1/new_templates/btn.png"):
        """加载按钮模板图片"""
        print(f"正在加载按钮模板: {template_path}")
        self.btn_template = None
        
        template_path = Path(template_path)
        if template_path.exists():
            # 读取模板图片
            template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
            if template is not None:
                self.btn_template = template
                print(f"✅ 加载模板成功: {template_path.name}")
                print(f"  模板尺寸: {template.shape[1]}x{template.shape[0]}")
            else:
                print(f"❌ 无法读取模板: {template_path}")
        else:
            print(f"❌ 模板文件不存在: {template_path}")
    
    def load_back_template(self, template_path: str = "1/new_templates/back.png"):
        """加载扑克背面模板图片"""
        print(f"正在加载扑克背面模板: {template_path}")
        self.back_template = None
        
        template_path = Path(template_path)
        if template_path.exists():
            # 读取模板图片
            template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
            if template is not None:
                self.back_template = template
                print(f"✅ 加载模板成功: {template_path.name}")
                print(f"  模板尺寸: {template.shape[1]}x{template.shape[0]}")
            else:
                print(f"❌ 无法读取模板: {template_path}")
        else:
            print(f"❌ 模板文件不存在: {template_path}")
    
    def load_image(self, image_path):
        """加载图片"""
        try:
            if not os.path.exists(image_path):
                print(f"❌ 图片文件不存在：{image_path}")
                return None
            
            # 使用PIL加载图片
            image = Image.open(image_path)
            print(f"✅ 成功加载图片：{image_path}")
            print(f"   图片大小：{image.size}")
            print(f"   图片模式：{image.mode}")
            
            return image
            
        except Exception as e:
            print(f"❌ 加载图片失败：{str(e)}")
            return None
    
    # OCR相关方法
    def preprocess_image_for_chinese(self, image, enable_grayscale=True):
        """为中文识别预处理图片"""
        try:
            # 转换为灰度图（可选）
            if enable_grayscale:
                if image.mode != 'L':
                    gray_image = image.convert('L')
                else:
                    gray_image = image
            else:
                gray_image = image
            
            # 放大图片以提高识别效果
            scale_factor = 2
            enlarged_image = gray_image.resize(
                (gray_image.width * scale_factor, gray_image.height * scale_factor),
                Image.Resampling.LANCZOS
            )
            
            # 增强对比度
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(enlarged_image)
            enhanced_image = enhancer.enhance(1.5)
            
            #print(f"✅ 图片预处理完成：放大{scale_factor}倍，增强对比度")
            return enhanced_image
            
        except Exception as e:
            print(f"⚠️  图片预处理失败：{str(e)}，使用原图")
            return image
    
    def get_recognition_strategy(self, region_name):
        """获取区域的识别策略"""
        if not self.recognition_strategies:
            # 如果没有配置策略，使用默认配置
            print(f"   ⚠️  未配置识别策略，使用默认配置")
            return "--oem 3 --psm 6 --dpi 300 -l chi_sim+eng"
        
        #print(f"   🔍 查找区域 {region_name} 的识别策略")
        #print(f"   📚 可用策略: {list(self.recognition_strategies.keys())}")
        
        # 查找区域对应的策略，优先匹配区域数量较少的策略（更具体）
        sorted_strategies = sorted(self.recognition_strategies.items(), key=lambda x: len(x[1].get('regions', [])))
        
        # 查找区域对应的策略
        for strategy_name, strategy in sorted_strategies:
            regions = strategy.get('regions', [])
            #print(f"   📋 策略 {strategy_name} 包含区域: {regions}")
            if region_name in regions:
                config = strategy.get('config', '')
                description = strategy.get('description', '')
                #print(f"   ✅ 找到匹配策略: {description} ({strategy_name})")
                #print(f"   🛠️  使用配置: {config}")
                return config
        
        # 如果没有找到对应策略，使用默认配置
        #print(f"   ⚠️  未找到区域 {region_name} 的识别策略，使用默认配置")
        return "--oem 3 --psm 6 --dpi 300 -l chi_sim+eng"
    
    def recognize_full_image(self, image, save_result=True):
        """识别整个图片"""
        if not self.default_config:
            print("❌ Tesseract OCR未初始化")
            return None
        
        print(f"\n🔍 开始识别整个图片...")
        
        # 检查是否为数字识别配置
        is_digit_only = 'digit' in self.char_whitelist and not any(c in self.char_whitelist for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz一二三四五六七八九十百千万亿跟注加注弃牌发牌底池公共牌自己的筹码BB大小王红桃黑桃方块梅花AKQJ')
        
        try:
            # 预处理图片以提高中文识别效果（根据配置决定是否启用灰度处理）
            processed_image = self.preprocess_image_for_chinese(image, enable_grayscale=not is_digit_only)
            
            # 构建配置字符串 - 使用最佳中文识别配置
            config = f"--oem 3 --psm 6 --dpi 300 -l chi_sim"
            
            # 记录开始时间
            start_time = time.time()
            
            # 识别文字
            text = pytesseract.image_to_string(processed_image, config=config, lang=self.language)
            
            # 记录结束时间
            end_time = time.time()
            recognition_time = end_time - start_time
            
            # 清理文本
            cleaned_text = re.sub(r'\s+', ' ', text.strip())
            
            results = {
                'text': cleaned_text,
                'time': recognition_time,
                'success': bool(cleaned_text)
            }
            
            # #print(f"✅ PSM 6 识别完成 ({recognition_time:.3f}秒)")
            # if cleaned_text:
            #     #print(f"   识别结果: '{cleaned_text}'")
            #     continue
            # else:
            #     print(f"   未识别到文字")
                
        except Exception as e:
            print(f"❌ PSM 6 识别失败: {str(e)}")
            results = {
                'text': '',
                'time': 0,
                'success': False,
                'error': str(e)
            }
        
        # 保存结果
        if save_result:
            self.save_recognition_result(results, "full_image")
        
        return results
    
    def recognize_chinese_text(self, image, save_result=True):
        """专门识别中文文本，使用优化的配置"""
        if not self.default_config:
            print("❌ Tesseract OCR未初始化")
            return None
        
        print(f"\n🔍 开始中文文本识别...")
        
        # 检查是否为数字识别配置
        is_digit_only = 'digit' in self.char_whitelist and not any(c in self.char_whitelist for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz一二三四五六七八九十百千万亿跟注加注弃牌发牌底池公共牌自己的筹码BB大小王红桃黑桃方块梅花AKQJ')
        
        try:
            # 预处理图片以提高中文识别效果（根据配置决定是否启用灰度处理）
            processed_image = self.preprocess_image_for_chinese(image, enable_grayscale=not is_digit_only)
            
            # 中文识别专用配置 - 根据测试结果优化
            chinese_configs = [
                f"--oem 3 --psm 6 --dpi 300 -l chi_sim",  # 仅中文，PSM 6 - 最佳效果
                f"--oem 3 --psm 6 --dpi 300 -l chi_sim+eng",  # 中文+英文，PSM 6
                f"--oem 3 --psm 7 --dpi 300 -l chi_sim",  # 仅中文，PSM 7
                f"--oem 3 --psm 8 --dpi 300 -l chi_sim",  # 仅中文，PSM 8
            ]
            
            best_result = {'text': '', 'confidence': 0, 'config': '', 'time': 0}
            
            for i, config in enumerate(chinese_configs):
                try:
                    print(f"   尝试配置 {i+1}: PSM {config.split('--psm ')[1].split()[0]}")
                    
                    # 记录开始时间
                    start_time = time.time()
                    
                    # 识别文字
                    text = pytesseract.image_to_string(processed_image, config=config, lang=self.language)
                    
                    # 记录结束时间
                    end_time = time.time()
                    recognition_time = end_time - start_time
                    
                    # 清理文本
                    cleaned_text = re.sub(r'\s+', ' ', text.strip())
                    
                    # 计算中文字符比例
                    chinese_chars_count = len(re.findall(r'[\u4e00-\u9fff]', cleaned_text))
                    total_chars = len(cleaned_text)
                    chinese_ratio = chinese_chars_count / total_chars if total_chars > 0 else 0
                    
                    #print(f"   ✅ 识别结果: '{cleaned_text}' (中文比例: {chinese_ratio:.2f}, 时间: {recognition_time:.3f}秒)")
                    
                    # 选择最佳结果（优先选择中文比例高的）
                    if chinese_ratio > best_result['confidence'] or (chinese_ratio == best_result['confidence'] and len(cleaned_text) > len(best_result['text'])):
                        best_result = {
                            'text': cleaned_text,
                            'confidence': chinese_ratio,
                            'config': config,
                            'time': recognition_time,
                            'success': bool(cleaned_text)
                        }
                        
                except Exception as e:
                    print(f"   ❌ 配置 {i+1} 识别失败: {str(e)}")
                    continue
            
            # 显示最佳结果
            if best_result['success']:
                print(f"\n🏆 最佳识别结果 (配置: PSM {best_result['config'].split('--psm ')[1].split()[0]}):")
                print(f"   识别结果: '{best_result['text']}'")
                print(f"   中文比例: {best_result['confidence']:.2f}")
                print(f"   识别时间: {best_result['time']:.3f}秒")
            else:
                print(f"\n❌ 未能识别到有效文字")
                
        except Exception as e:
            print(f"❌ 中文文本识别失败: {str(e)}")
            best_result = {
                'text': '',
                'confidence': 0,
                'config': '',
                'time': 0,
                'success': False,
                'error': str(e)
            }
        
        # 保存结果
        if save_result:
            self.save_recognition_result(best_result, "chinese_text")
        
        return best_result
    
    def recognize_regions(self, image, regions=None):
        """识别指定区域"""
        if not self.default_config:
            print("❌ Tesseract OCR未初始化")
            return None
        
        if regions is None:
            regions = self.test_regions
        
        if not regions:
            print("❌ 未配置识别区域")
            return None
        
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
                #print(f"📍 区域 {region_name} 坐标已转换: {coords} -> [{x1}, {y1}, {x2}, {y2}]")
            else:
                # 假设已经是绝对坐标
                converted_regions[region_name] = coords
                #print(f"📍 区域 {region_name} 使用绝对坐标: {coords}")
        
        #print(f"\n🔍 开始识别指定区域...")
        #print(f"   共 {len(converted_regions)} 个区域")
        
        results = {}
        
        # 为每个区域识别文字
        for i, (region_name, coords) in enumerate(converted_regions.items()):
            try:
                #print(f"\n📍 区域 {i+1}/{len(converted_regions)}: {region_name} {coords}")
                
                # 裁剪区域图片
                x1, y1, x2, y2 = coords
                cropped_image = image.crop((x1, y1, x2, y2))
                
                # 保存裁剪图片（可选）
                # cropped_filename = os.path.join(self.output_dir, f"cropped_{region_name}.png")
                # cropped_image.save(cropped_filename)
                # print(f"   📁 裁剪图片已保存: {cropped_filename}")
                
                # 获取该区域的识别策略
                #print(f"   📌 区域名称: '{region_name}'")
                config = self.get_recognition_strategy(region_name)
                #print(f"   📥 获取到的配置: {config}")
                
                # 检查是否为数字识别配置
                is_digit_only = 'digit' in self.char_whitelist and not any(c in self.char_whitelist for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz一二三四五六七八九十百千万亿跟注加注弃牌发牌底池公共牌自己的筹码BB大小王红桃黑桃方块梅花AKQJ')
                
                # 预处理图片
                processed_image = self.preprocess_image_for_chinese(cropped_image, enable_grayscale=not is_digit_only)
                
                # 识别文字
                #print(f"   🧪 识别配置: {config}")
                #print(f"   🌐 语言设置: {self.language}")
                text = pytesseract.image_to_string(processed_image, config=config, lang=self.language)
                
                # 清理文本
                cleaned_text = re.sub(r'\s+', ' ', text.strip())
                
                # 保存结果
                region_results = {
                    'text': cleaned_text,
                    'coords': coords,
                    'success': bool(cleaned_text)
                }
                
                results[region_name] = region_results
                
                # if cleaned_text:
                #     print(f"   ✅ 识别结果: '{cleaned_text}'")
                # else:
                #     print(f"   ⚠️  未识别到文字")
                    
            except Exception as e:
                print(f"   ❌ 区域识别失败: {str(e)}")
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
            
            print(f"   📁 结果已保存: {result_file}")
            
        except Exception as e:
            print(f"   ❌ 保存结果失败: {str(e)}")
    
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
                
                # 如果有识别结果，显示识别结果
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
            
            #print(f"📁 标记图片已保存: {marked_filename}")
            return marked_filename
            
        except Exception as e:
            print(f"❌ 创建标记图片失败: {str(e)}")
            return None
    
    # 花色识别相关方法
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
                resized_template = cv2.resize(gray_template, (new_w, new_w))
                
                # 确保缩放后的模板不大于图像
                if new_h <= gray_image.shape[0] and new_w <= gray_image.shape[1]:
                    score, position = self.match_suit_template(gray_image, resized_template, threshold)
                    
                    if score > max_scale_score:
                        max_scale_score = score
                        max_scale_position = position
            
            results[suit_name] = {
                'score': max_scale_score,
                'position': max_scale_position,
                'chinese_name': self.suit_names[suit_name],
                'threshold': threshold,
                'matched': max_scale_score >= threshold
            }
            
            # 记录最佳匹配
            if max_scale_score > best_score:
                best_score = max_scale_score
                best_match = suit_name
                
        # 添加最佳匹配信息
        if best_match:
            # 根据实际分数调整置信度判断
            confidence = 'high' if best_score >= 0.7 else 'medium' if best_score >= 0.5 else 'low'
            results['best_match'] = {
                'suit': best_match,
                'chinese_name': self.suit_names[best_match],
                'score': best_score,
                'confidence': confidence
            }
            
        return results
    
    def calculate_suit_adaptive_regions(self, image_width: int, image_height: int) -> Dict[str, List[int]]:
        """
        根据图像尺寸计算花色自适应区域坐标
        
        Args:
            image_width: 当前图像宽度
            image_height: 当前图像高度
            
        Returns:
            绝对坐标区域 {区域名: [x1, y1, x2, y2]}
        """
        ref_width, ref_height = self.suit_reference_size
        absolute_regions = {}
        
        for region_name, ratios in self.suit_adaptive_regions.items():
            x1_ratio, y1_ratio, x2_ratio, y2_ratio = ratios
            
            # 根据比例计算绝对坐标
            x1 = int(x1_ratio * image_width)
            y1 = int(y1_ratio * image_height)
            x2 = int(x2_ratio * image_width)
            y2 = int(y2_ratio * image_height)
            
            absolute_regions[region_name] = [x1, y1, x2, y2]
            
        return absolute_regions
    
    def recognize_suits_in_regions(self, image: np.ndarray, regions: Dict[str, List[int]], 
                                  threshold: float = 0.3) -> Dict[str, Dict]:
        """
        识别图像中多个区域的花色
        
        Args:
            image: 输入图像
            regions: 区域坐标字典 {区域名: [x1, y1, x2, y2]}
            threshold: 匹配阈值
            
        Returns:
            各区域的识别结果
        """
        results = {}
        
        for region_name, coords in regions.items():
            # 提取区域图像
            x1, y1, x2, y2 = coords
            region_image = image[y1:y2, x1:x2]
            
            if region_image.size > 0:
                # 识别该区域的花色
                region_result = self.recognize_suit(region_image, threshold)
                
                # 添加调试信息
                #print(f"  {region_name} 区域大小: {region_image.shape}")
                if 'best_match' in region_result:
                    best = region_result['best_match']
                    #print(f"    最佳匹配: {best['chinese_name']} (分数: {best['score']:.3f})")
                else:
                    print(f"    未找到匹配的花色")
                    
                # 显示所有花色的匹配分数
                # for suit_name, result in region_result.items():
                #     if suit_name != 'best_match':
                #         print(f"    {result['chinese_name']}: {result['score']:.3f}")
                
                results[region_name] = region_result
            else:
                results[region_name] = {'error': '区域图像为空'}
                
        return results
    
    def visualize_suit_results(self, image: np.ndarray, results: Dict, 
                         save_path: Optional[str] = None) -> np.ndarray:
        """
        可视化花色识别结果
        
        Args:
            image: 原始图像
            results: 识别结果
            save_path: 保存路径
            
        Returns:
            标注后的图像
        """
        # 复制图像用于绘制
        vis_image = image.copy()
        
        # 绘制每个匹配结果
        for suit_name, result in results.items():
            if suit_name == 'best_match':
                continue
                
            if result.get('matched', False):
                score = result['score']
                position = result['position']
                chinese_name = result['chinese_name']
                
                # 获取模板尺寸
                template = self.suit_templates[suit_name]
                h, w = template.shape
                
                # 绘制矩形框
                x, y = position
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # 绘制标签
                label = f"{chinese_name}: {score:.3f}"
                cv2.putText(vis_image, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
        # 保存结果
        if save_path:
            cv2.imwrite(save_path, vis_image)
            #print(f"结果图像已保存: {save_path}")
            
        return vis_image
    
    def visualize_suit_regions(self, image: np.ndarray, regions: Dict[str, List[int]], 
                              results: Dict[str, Dict], save_path: Optional[str] = None) -> np.ndarray:
        """
        可视化花色区域识别结果
        
        Args:
            image: 原始图像
            regions: 区域坐标字典
            results: 识别结果
            save_path: 保存路径
            
        Returns:
            标注后的图像
        """
        # 复制图像用于绘制
        vis_image = image.copy()
        
        # 定义颜色映射
        colors = {
            'high': (0, 255, 0),    # 绿色 - 高置信度
            'medium': (0, 255, 255), # 黄色 - 中等置信度
            'low': (0, 0, 255)      # 红色 - 低置信度
        }
        
        # 绘制每个区域和识别结果
        for region_name, coords in regions.items():
            x1, y1, x2, y2 = coords
            
            # 绘制区域框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
            
            # 获取识别结果
            if region_name in results and 'best_match' in results[region_name]:
                result = results[region_name]['best_match']
                suit_name = result['chinese_name']
                confidence = result['confidence']
                score = result['score']
                
                # 选择颜色
                color = colors.get(confidence, (255, 255, 255))
                
                # 绘制标签
                label = f"{region_name}: {suit_name} ({score:.3f})"
                cv2.putText(vis_image, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 在区域中心显示花色名称
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.putText(vis_image, suit_name, (center_x - 20, center_y + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                # 没有识别结果
                label = f"{region_name}: 未识别"
                cv2.putText(vis_image, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
                
        # 保存结果
        if save_path:
            cv2.imwrite(save_path, vis_image)
            #print(f"结果图像已保存: {save_path}")
            
        return vis_image
    
    # 按钮识别相关方法
    def match_btn_template(self, image: np.ndarray, template: np.ndarray, 
                      threshold: float = 0.6) -> Dict:
        """
        按钮模板匹配
        
        Args:
            image: 待匹配图像
            template: 模板图像
            threshold: 匹配阈值
            
        Returns:
            匹配结果字典
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image.copy()
            
        # 转换模板为灰度图
        if len(template.shape) == 3:
            gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            gray_template = template.copy()
        
        # 使用单一匹配方法
        method = cv2.TM_CCOEFF_NORMED
        
        try:
            result = cv2.matchTemplate(gray_image, gray_template, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # 对于不同的匹配方法，处理方式不同
            if method == cv2.TM_SQDIFF_NORMED:
                score = 1 - min_val
                loc = min_loc
            else:
                score = max_val
                loc = max_loc
                
            # 判断是否匹配成功
            matched = score >= threshold
            
            return {
                'score': score,
                'position': loc,
                'matched': matched,
                'threshold': threshold
            }
        except cv2.error as e:
            # 如果匹配失败，返回默认值
            print(f"模板匹配错误: {e}")
            return {
                'score': -1.0,
                'position': (0, 0),
                'matched': False,
                'threshold': threshold
            }
    
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
            return {'error': '模板未加载'}
            
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
            if new_h <= gray_image.shape[0] and new_w <= gray_image.shape[1]:
                result = self.match_btn_template(gray_image, resized_template, threshold)
                
                if result['score'] > best_score:
                    best_score = result['score']
                    best_position = result['position']
                    best_scale = scale
        
        # 判断是否匹配成功
        matched = best_score >= threshold
        
        return {
            'score': best_score,
            'position': best_position,
            'matched': matched,
            'threshold': threshold,
            'scale': best_scale
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
            return {'error': '模板未加载'}
            
        # 进行模板匹配
        if multiscale:
            result = self.recognize_btn_multiscale(image, threshold)
        else:
            result = self.match_btn_template(image, self.btn_template, threshold)
        
        return result
    
    def calculate_btn_adaptive_regions(self, image_width, image_height):
        """
        根据图像尺寸计算按钮自适应区域
        
        Args:
            image_width: 图像宽度
            image_height: 图像高度
            
        Returns:
            自适应区域字典
        """
        reference_width, reference_height = self.btn_reference_size
        
        # 计算比例
        width_ratio = image_width / reference_width
        height_ratio = image_height / reference_height
        
        # 计算自适应区域
        absolute_regions = {}
        for name, ratios in self.btn_adaptive_regions.items():
            x1_ratio, y1_ratio, x2_ratio, y2_ratio = ratios
            x1 = int(x1_ratio * image_width)
            y1 = int(y1_ratio * image_height)
            x2 = int(x2_ratio * image_width)
            y2 = int(y2_ratio * image_height)
            absolute_regions[name] = [x1, y1, x2, y2]
        
        return absolute_regions
    
    def determine_table_position(self, detected_x, detected_y, image_width, image_height, confidence_threshold=0.8):
        """
        根据识别位置确定牌桌位置
        
        Args:
            detected_x: 检测到的按钮位置x坐标
            detected_y: 检测到的按钮位置y坐标
            image_width: 图像宽度
            image_height: 图像高度
            confidence_threshold: 置信度阈值
            
        Returns:
            位置名称和置信度信息
        """
        # 计算自适应区域
        adaptive_regions = self.calculate_btn_adaptive_regions(image_width, image_height)
        if adaptive_regions is None:
            return "未知位置", 0.0
        
        # 检查检测位置在哪个区域内
        for position_name, region_coords in adaptive_regions.items():
            x1, y1, x2, y2 = region_coords
            if x1 <= detected_x <= x2 and y1 <= detected_y <= y2:
                # 计算置信度（基于位置在区域中心的接近程度）
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                max_distance = ((x2-x1)//2)**2 + ((y2-y1)//2)**2
                actual_distance = (detected_x-center_x)**2 + (detected_y-center_y)**2
                confidence = max(0, 1 - (actual_distance / max_distance))
                
                if confidence >= confidence_threshold:
                    return position_name, confidence
        
        return "未知位置", 0.0
    
    def visualize_btn_result(self, image: np.ndarray, result: Dict, save_path: str = None) -> np.ndarray:
        """
        可视化按钮识别结果
        
        Args:
            image: 原始图像
            result: 识别结果
            save_path: 保存路径
            
        Returns:
            标注后的图像
        """
        # 复制图像用于绘制
        vis_image = image.copy()
        
        if 'error' in result:
            # 在图像上显示错误信息
            cv2.putText(vis_image, f"Error: {result['error']}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif result['matched']:
            # 获取位置信息
            x, y = result['position']
            h, w = self.btn_template.shape
            score = result['score']
            
            # 绘制矩形框
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 绘制标签
            label = f"btn: {score:.3f}"
            cv2.putText(vis_image, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 显示位置信息
            pos_label = f"Pos: ({x}, {y})"
            cv2.putText(vis_image, pos_label, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 显示牌桌位置信息
            table_position, confidence = self.determine_table_position(x, y, image.shape[1], image.shape[0])
            table_label = f"Table Pos: {table_position} ({confidence:.2f})"
            cv2.putText(vis_image, table_label, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # 显示未匹配到的信息
            max_score = result.get('max_score', result.get('score', 0))
            cv2.putText(vis_image, f"No btn detected (max score: {max_score:.3f})", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 保存结果
        if save_path:
            cv2.imwrite(save_path, vis_image)
            #print(f"结果图像已保存: {save_path}")
            
        return vis_image
    
    # 扑克背面识别相关方法
    def check_player_fold_by_variance(self, image: np.ndarray, variance_threshold: float = 10.0) -> Dict:
        """
        基于颜色方差检查玩家是否弃牌（替代方法）
        
        Args:
            image: 输入图像
            variance_threshold: 颜色方差阈值，低于此值认为是均色区域（未弃牌）
            
        Returns:
            玩家弃牌状态字典
        """
        # 计算自适应区域
        image_height, image_width = image.shape[:2]
        adaptive_regions = self.calculate_btn_adaptive_regions(image_width, image_height)
        
        if adaptive_regions is None:
            return {'error': '无法计算自适应区域'}
        
        # 检查每个玩家区域
        player_status = {}
        for i in range(1, 6):
            player_key = f"玩家{i}扑克背面"
            if player_key in adaptive_regions:
                x1, y1, x2, y2 = adaptive_regions[player_key]
                # 提取区域图像
                region_image = image[y1:y2, x1:x2]
                
                if region_image.size > 0:
                    # 转换为灰度图
                    if len(region_image.shape) == 3:
                        gray = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = region_image.copy()
                    
                    # 计算颜色方差
                    variance = np.var(gray)
                    
                    # 如果方差小于阈值，判断为未弃牌
                    is_folded = variance < variance_threshold
                    
                    player_status[f"玩家{i}"] = {
                        'folded': is_folded,
                        'variance': variance,
                        'confidence': variance / 100.0 if variance / 100.0 < 1.0 else 1.0
                    }
                else:
                    player_status[f"玩家{i}"] = {
                        'folded': True,
                        'variance': 0.0,
                        'confidence': 0.0,
                        'error': '区域图像为空'
                    }
            else:
                player_status[f"玩家{i}"] = {
                    'folded': True,
                    'variance': 0.0,
                    'confidence': 0.0,
                    'error': '未找到区域定义'
                }
        
        return player_status
    
    def visualize_fold_result(self, image: np.ndarray, fold_status: Dict, save_path: str = None) -> np.ndarray:
        """
        可视化玩家弃牌状态结果
        
        Args:
            image: 原始图像
            fold_status: 玩家弃牌状态字典
            save_path: 保存路径
            
        Returns:
            标注后的图像
        """
        # 复制图像用于绘制
        vis_image = image.copy()
        
        # 使用PIL处理中文文本显示
        vis_pil = Image.fromarray(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(vis_pil)
        
        # 设置字体
        try:
            # 尝试使用系统中文字体
            font = ImageFont.truetype("simhei.ttf", 20, encoding="utf-8")
        except:
            # 如果找不到中文字体，使用默认字体
            font = ImageFont.load_default()
        
        # 显示每个玩家的状态
        y_offset = 40
        if 'error' in fold_status:
            draw.text((10, y_offset), f"Error: {fold_status['error']}", (255, 0, 0), font=font)
        else:
            for player, status in fold_status.items():
                # 跳过错误信息
                if 'error' in status:
                    continue
                
                # 根据弃牌状态设置颜色
                color = (0, 255, 0) if status['folded'] else (255, 0, 0)  # 绿色表示已弃牌，红色表示未弃牌
                
                # 显示玩家状态
                if 'variance' in status:
                    # 颜色方差方法
                    label = f"{player}: {'未弃牌' if status['folded'] else '已弃牌'} (方差: {status['variance']:.2f})"
                else:
                    # 模板匹配方法
                    label = f"{player}: {'未弃牌' if status['folded'] else '已弃牌'} (置信度: {status['confidence']:.3f})"
                
                draw.text((10, y_offset), label, color, font=font)
                y_offset += 25
        
        # 转换回OpenCV格式
        vis_image = cv2.cvtColor(np.array(vis_pil), cv2.COLOR_RGB2BGR)
        
        # 保存结果
        if save_path:
            cv2.imwrite(save_path, vis_image)
            print(f"弃牌检测结果图像已保存: {save_path}")
            
        return vis_image
    
    # 主要功能接口
    def recognize_all(self, image_path: str):
        """
        对图像进行所有类型的识别
        
        Args:
            image_path: 图像路径
        """
        print(f"\n{'='*60}")
        print(f"开始对图像进行所有类型识别: {image_path}")
        print(f"{'='*60}")
        
        # 加载图像
        image_pil = self.load_image(image_path)
        if image_pil is None:
            return
        
        # 转换为OpenCV格式
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        # 1. OCR识别
        #print(f"\n{'='*40}")
        #print("1. OCR识别")
        #print(f"{'='*40}")
        
        # 识别指定区域
        region_results = self.recognize_regions(image_pil)
        
        # 创建标记图片
        #print(f"\n🎨 创建OCR标记图片...")
        self.create_marked_image(image_pil, self.test_regions, region_results)
        
        # # 2. 花色识别
        # print(f"\n{'='*40}")
        # print("2. 花色识别")
        # print(f"{'='*40}")
        
        # 计算自适应区域
        image_height, image_width = image_cv.shape[:2]
        adaptive_suit_regions = self.calculate_suit_adaptive_regions(image_width, image_height)
        
        # 识别所有花色区域
        suit_results = self.recognize_suits_in_regions(image_cv, adaptive_suit_regions, threshold=0.6)
        
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
            print(f"  错误: {btn_result['error']}")
        else:
            if btn_result['matched']:
                x, y = btn_result['position']
                if 'scale' in btn_result:
                    h, w = int(self.btn_template.shape[0] * btn_result['scale']), int(self.btn_template.shape[1] * btn_result['scale'])
                else:
                    h, w = self.btn_template.shape
                # 直接输出按钮位置
                table_position, confidence = self.determine_table_position(x, y, image_width, image_height)
                #print(f"按钮位置: {table_position}")
            else:
                print(f"  ❌ 未匹配到按钮")
        
        # 可视化按钮识别结果
        vis_btn_image = self.visualize_btn_result(image_cv, btn_result, 
                                                f"{self.output_dir}/btn_result.png")
        
        # 4. 扑克背面识别（弃牌检测）
        print(f"\n{'='*40}")
        print("4. 扑克背面识别（弃牌检测）")
        print(f"{'='*40}")
        
        # 检查玩家是否弃牌（使用颜色方差方法）
        fold_status_variance = self.check_player_fold_by_variance(image_cv, variance_threshold=10.0)
        if 'error' in fold_status_variance:
            print(f"  错误: {fold_status_variance['error']}")
        else:
            for player, status in fold_status_variance.items():
                if 'error' in status:
                    print(f"    {player}: 错误 - {status['error']}")
                else:
                    if status['folded']:
                        print(f"    {player}: 未弃牌 (方差: {status['variance']:.2f}, 置信度: {status['confidence']:.3f})")
                    else:
                        print(f"    {player}: 已弃牌 (方差: {status['variance']:.2f}, 置信度: {status['confidence']:.3f})")
        
        # 可视化弃牌检测结果
        vis_fold_image = self.visualize_fold_result(image_cv, fold_status_variance, 
                                                 f"{self.output_dir}/fold_result_variance.png")
        
        # 修改输出格式，只输出指定区域的识别结果
        print(f"\n{'='*40}")
        print("最终识别结果")
        print(f"{'='*40}")
        
        # 输出OCR识别结果
        if region_results:
            for region_name, result in region_results.items():
                if result.get('success') and result.get('text'):
                    print(f"区域: {region_name} 识别结果: {result['text']}")
        
        # 合并手牌与花色、公共牌与花色输出
        suit_output = {}
        if suit_results:
            for region_name, result in suit_results.items():
                if 'best_match' in result:
                    best = result['best_match']
                    suit_symbol = {'红桃': 'h', '黑桃': 's', '方块': 'd', '梅花': 'c'}.get(best['chinese_name'], '')
                    suit_output[region_name] = suit_symbol
        
        # 输出合并后的手牌和花色结果
        hand_cards = {}
        community_cards = {}
        
        # 处理手牌
        for i in [1, 2]:
            hand_region = f"自己的手牌{i}"
            suit_region = f"自己的手牌{i}花色"
            if hand_region in region_results and suit_region in suit_output:
                hand_text = region_results[hand_region].get('text', '')
                suit_symbol = suit_output[suit_region]
                if hand_text and suit_symbol:
                    hand_cards[i] = f"{hand_text}{suit_symbol}"
                    print(f"自己的手牌{i}: {hand_cards[i]}")
        
        # 处理公共牌
        for i in range(1, 6):
            community_region = f"公共牌{i}"
            suit_region = f"公共牌{i}花色"
            if community_region in region_results and suit_region in suit_output:
                community_text = region_results[community_region].get('text', '')
                suit_symbol = suit_output[suit_region]
                if community_text and suit_symbol:
                    community_cards[i] = f"{community_text}{suit_symbol}"
                    print(f"公共牌{i}: {community_cards[i]}")
        
        # 输出按钮位置
        if 'error' not in btn_result and btn_result['matched']:
            table_position, confidence = self.determine_table_position(btn_result['position'][0], btn_result['position'][1], image_width, image_height)
            print(f"按钮位置: {table_position}")
        
        print(f"\n✅ 所有识别完成！")
        print(f"📁 所有结果已保存到目录: {self.output_dir}")
        print(f"📁 请查看该目录中的文件")


def main():
    """主函数"""
    print("=" * 60)
    print("整合扑克牌识别程序")
    print("支持OCR数字识别、花色识别和按钮位置识别")
    print("=" * 60)
    
    # 创建识别器
    recognizer = PokerRecognizer()
    
    try:
        # 获取图片路径
        print("\n📁 请输入图片文件路径:")
        print("   支持格式: PNG, JPG, JPEG, BMP, TIFF等")
        print("   示例: test.png 或 C:/path/to/image.jpg")
        
        while True:
            image_path = input("\n图片路径: ").strip().strip('"')
            
            if not image_path:
                print("❌ 请输入有效的图片路径")
                continue
            
            # 检查文件是否存在
            if not os.path.exists(image_path):
                print(f"❌ 图片文件不存在: {image_path}")
                continue
            
            break
        
        # 进行所有识别
        recognizer.recognize_all(image_path)
        
    except Exception as e:
        print(f"❌ 程序运行出错：{str(e)}")
    finally:
        print(f"\n👋 程序已退出")


if __name__ == "__main__":
    main()