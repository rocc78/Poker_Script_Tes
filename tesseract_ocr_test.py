#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tesseract OCR识别测试脚本
用于识别整个图片和指定区域
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

# 禁用警告
warnings.filterwarnings("ignore")

class TesseractOCRTest:
    def __init__(self):
        print("=== Tesseract OCR识别测试 ===")
        self.setup_output_dir()
        self.setup_tesseract()
        self.load_config()
    
    def setup_output_dir(self):
        """设置输出目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"ocr_results_{timestamp}"
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
    
    def load_config(self):
        """加载配置文件"""
        try:
            if os.path.exists('ocr_test_config.json'):
                with open('ocr_test_config.json', 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                self.test_regions = config.get('test_regions', {})
                self.psm_modes = config.get('psm_modes', [6, 8, 7, 13])
                self.char_whitelist = config.get('char_whitelist', '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz一二三四五六七八九十百千万亿跟注加注弃牌发牌底池公共牌自己的筹码BB大小王红桃黑桃方块梅花AKQJ')
                
                print("✅ 已加载配置文件")
            else:
                # 默认配置
                self.test_regions = {
                    '左上角': [0, 0, 200, 100],
                    '右上角': [400, 0, 600, 100],
                    '左下角': [0, 300, 200, 400],
                    '右下角': [400, 300, 600, 400],
                    '中心区域': [200, 150, 400, 250]
                }
                self.psm_modes = [6, 8, 7, 13]
                self.char_whitelist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz一二三四五六七八九十百千万亿跟注加注弃牌发牌底池公共牌自己的筹码BB大小王红桃黑桃方块梅花AKQJ'
                print("⚠️  未找到配置文件，使用默认设置")
                
        except Exception as e:
            print(f"❌ 加载配置文件失败：{str(e)}")
            # 使用默认配置
            self.test_regions = {
                '左上角': [0, 0, 200, 100],
                '右上角': [400, 0, 600, 100],
                '左下角': [0, 300, 200, 400],
                '右下角': [400, 300, 600, 400],
                '中心区域': [200, 150, 400, 250]
            }
            self.psm_modes = [6, 8, 7, 13]
            self.char_whitelist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz一二三四五六七八九十百千万亿跟注加注弃牌发牌底池公共牌自己的筹码BB大小王红桃黑桃方块梅花AKQJ'
    
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
    
    def preprocess_image_for_chinese(self, image):
        """为中文识别预处理图片"""
        try:
            # 转换为灰度图
            if image.mode != 'L':
                gray_image = image.convert('L')
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
            
            print(f"✅ 图片预处理完成：放大{scale_factor}倍，增强对比度")
            return enhanced_image
            
        except Exception as e:
            print(f"⚠️  图片预处理失败：{str(e)}，使用原图")
            return image
    
    def recognize_full_image(self, image, save_result=True):
        """识别整个图片"""
        if not self.default_config:
            print("❌ Tesseract OCR未初始化")
            return None
        
        print(f"\n🔍 开始识别整个图片...")
        
        try:
            # 预处理图片以提高中文识别效果
            processed_image = self.preprocess_image_for_chinese(image)
            
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
            
            print(f"✅ PSM 6 识别完成 ({recognition_time:.3f}秒)")
            if cleaned_text:
                print(f"   识别结果: '{cleaned_text}'")
            else:
                print(f"   未识别到文字")
                
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
        
        try:
            # 预处理图片以提高中文识别效果
            processed_image = self.preprocess_image_for_chinese(image)
            
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
                    
                    print(f"   ✅ 识别结果: '{cleaned_text}' (中文比例: {chinese_ratio:.2f}, 时间: {recognition_time:.3f}秒)")
                    
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
                    print(f"   ❌ 配置 {i+1} 失败: {str(e)}")
                    continue
            
            print(f"✅ 中文识别完成，最佳结果: '{best_result['text']}' (中文比例: {best_result['confidence']:.2f})")
            
            # 保存结果
            if save_result:
                self.save_recognition_result(best_result, "chinese_text")
            
            return best_result
            
        except Exception as e:
            print(f"❌ 中文识别失败: {str(e)}")
            return {
                'text': '',
                'confidence': 0,
                'config': '',
                'time': 0,
                'success': False,
                'error': str(e)
            }
    
    def recognize_regions(self, image, regions=None, save_result=True):
        """识别指定区域"""
        if not self.default_config:
            print("❌ Tesseract OCR未初始化")
            return None
        
        if regions is None:
            regions = self.test_regions
        
        print(f"\n🔍 开始识别指定区域...")
        
        all_results = {}
        
        for region_name, coords in regions.items():
            print(f"\n📍 识别区域: {region_name} {coords}")
            
            try:
                # 裁剪区域
                x1, y1, x2, y2 = coords
                cropped_image = image.crop((x1, y1, x2, y2))
                
                print(f"   裁剪区域大小: {cropped_image.size}")
                
                # 保存裁剪的图片
                cropped_filename = os.path.join(self.output_dir, f"cropped_{region_name}.png")
                cropped_image.save(cropped_filename)
                print(f"   已保存裁剪图片: {cropped_filename}")
                
                # 识别该区域 - 使用优化的中文识别配置
                try:
                    # 预处理裁剪区域
                    processed_cropped = self.preprocess_image_for_chinese(cropped_image)
                    
                    # 构建配置字符串 - 使用最佳中文识别配置
                    config = f"--oem 3 --psm 6 --dpi 300 -l chi_sim"
                    
                    # 记录开始时间
                    start_time = time.time()
                    
                    # 识别文字
                    text = pytesseract.image_to_string(processed_cropped, config=config, lang=self.language)
                    
                    # 记录结束时间
                    end_time = time.time()
                    recognition_time = end_time - start_time
                    
                    # 清理文本
                    cleaned_text = re.sub(r'\s+', ' ', text.strip())
                    
                    region_results = {
                        'text': cleaned_text,
                        'time': recognition_time,
                        'success': bool(cleaned_text)
                    }
                    
                    print(f"   ✅ PSM 6: '{cleaned_text}' ({recognition_time:.3f}秒)")
                    
                except Exception as e:
                    print(f"   ❌ PSM 6 失败: {str(e)}")
                    region_results = {
                        'text': '',
                        'time': 0,
                        'success': False,
                        'error': str(e)
                    }
                
                all_results[region_name] = region_results
                
            except Exception as e:
                print(f"❌ 处理区域 {region_name} 失败: {str(e)}")
                all_results[region_name] = {'error': str(e)}
        
        # 保存结果
        if save_result:
            self.save_recognition_result(all_results, "regions")
        
        return all_results
    
    def save_recognition_result(self, results, result_type):
        """保存识别结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"ocr_result_{result_type}_{timestamp}.json")
            
            # 准备保存的数据
            save_data = {
                'timestamp': datetime.now().isoformat(),
                'result_type': result_type,
                'results': results
            }
            
            # 保存JSON文件
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            print(f"📄 识别结果已保存: {filename}")
            
            # 同时保存文本格式
            txt_filename = os.path.join(self.output_dir, f"ocr_result_{result_type}_{timestamp}.txt")
            with open(txt_filename, 'w', encoding='utf-8') as f:
                f.write(f"Tesseract OCR识别结果\n")
                f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"类型: {result_type}\n")
                f.write("="*50 + "\n\n")
                
                if result_type == "full_image":
                    f.write(f"PSM 6:\n")
                    f.write(f"  文本: {results.get('text', '')}\n")
                    f.write(f"  时间: {results.get('time', 0):.3f}秒\n")
                    f.write(f"  成功: {results.get('success', False)}\n\n")
                else:
                    for region_name, region_results in results.items():
                        f.write(f"区域: {region_name}\n")
                        f.write("-"*30 + "\n")
                        if isinstance(region_results, dict):
                            f.write(f"  PSM 6:\n")
                            f.write(f"    文本: {region_results.get('text', '')}\n")
                            f.write(f"    时间: {region_results.get('time', 0):.3f}秒\n")
                            f.write(f"    成功: {region_results.get('success', False)}\n\n")
            
            print(f"📄 文本结果已保存: {txt_filename}")
            
        except Exception as e:
            print(f"❌ 保存结果失败: {str(e)}")
    
    def create_marked_image(self, image, regions=None, results=None):
        """创建带标记的图片"""
        try:
            # 创建副本
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
            
            print(f"📁 标记图片已保存: {marked_filename}")
            return marked_filename
            
        except Exception as e:
            print(f"❌ 创建标记图片失败: {str(e)}")
            return None

def main():
    """主函数"""
    print("="*60)
    print("🤖 Tesseract OCR识别测试程序")
    print("支持识别整个图片和指定区域")
    print("="*60)
    
    # 创建OCR测试器
    ocr_test = TesseractOCRTest()
    
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
            
            # 加载图片
            image = ocr_test.load_image(image_path)
            if image is not None:
                break
            else:
                print("❌ 无法加载图片，请重新输入")
        
        # 选择识别模式
        print(f"\n🔍 请选择识别模式:")
        print("1. 识别整个图片")
        print("2. 识别指定区域")
        print("3. 两种都识别")
        print("4. 专门中文识别")
        print("5. 所有模式都识别")
        
        while True:
            choice = input("\n请选择 (1/2/3/4/5): ").strip()
            if choice in ['1', '2', '3', '4', '5']:
                break
            else:
                print("❌ 请输入 1、2、3、4 或 5")
        
        # 执行识别
        if choice in ['1', '3', '5']:
            print(f"\n{'='*40}")
            print("识别整个图片")
            print(f"{'='*40}")
            full_results = ocr_test.recognize_full_image(image)
        
        if choice in ['4', '5']:
            print(f"\n{'='*40}")
            print("专门中文识别")
            print(f"{'='*40}")
            chinese_results = ocr_test.recognize_chinese_text(image)
        
        if choice in ['2', '3', '5']:
            print(f"\n{'='*40}")
            print("识别指定区域")
            print(f"{'='*40}")
            
            # 显示当前配置的区域
            print(f"\n📋 当前配置的识别区域:")
            for i, (name, coords) in enumerate(ocr_test.test_regions.items()):
                print(f"  {i+1}. {name}: {coords}")
            
            # 询问是否使用自定义区域
            custom_choice = input(f"\n是否使用自定义区域？(y/n): ").strip().lower()
            if custom_choice in ['y', 'yes', '是']:
                # 让用户输入自定义区域
                custom_regions = {}
                print(f"\n请输入自定义区域 (格式: 区域名 x1,y1,x2,y2)")
                print(f"输入 'done' 完成输入")
                
                while True:
                    region_input = input("区域: ").strip()
                    if region_input.lower() == 'done':
                        break
                    
                    try:
                        parts = region_input.split(' ', 1)
                        if len(parts) == 2:
                            region_name = parts[0]
                            coords_str = parts[1]
                            coords = [int(x.strip()) for x in coords_str.split(',')]
                            if len(coords) == 4:
                                custom_regions[region_name] = coords
                                print(f"✅ 添加区域: {region_name} {coords}")
                            else:
                                print("❌ 坐标格式错误，需要4个数字")
                        else:
                            print("❌ 格式错误，请使用: 区域名 x1,y1,x2,y2")
                    except Exception as e:
                        print(f"❌ 输入错误: {str(e)}")
                
                if custom_regions:
                    ocr_test.test_regions = custom_regions
            
            region_results = ocr_test.recognize_regions(image)
            
            # 创建标记图片
            print(f"\n🎨 创建标记图片...")
            ocr_test.create_marked_image(image, ocr_test.test_regions, region_results)
        
        print(f"\n✅ 识别完成！")
        print(f"📁 所有结果已保存到目录: {ocr_test.output_dir}")
        print(f"📁 请查看该目录中的文件")
        
    except Exception as e:
        print(f"❌ 程序运行出错：{str(e)}")
    finally:
        print(f"\n👋 程序已退出")

if __name__ == "__main__":
    main()
