#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于OpenCV的花色识别程序
使用matchTemplate方法识别扑克牌花色
"""

import cv2
import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import platform
import sys

class SuitRecognizer:
    """花色识别器"""
    
    def __init__(self, template_dir: str = "1/gray_templates"):
        """
        初始化花色识别器
        
        Args:
            template_dir: 模板图片目录
        """
        self.template_dir = Path(template_dir)
        self.templates = {}
        self.suit_names = {
            'club': '梅花',
            'diamond': '方块', 
            'heart': '红桃',
            'spade': '黑桃'
        }
        self.load_templates()
        self.font = self._get_chinese_font()
        
    def _get_chinese_font(self):
        """
        获取支持中文的字体
        
        Returns:
            字体路径
        """
        # 根据不同操作系统选择默认中文字体
        if platform.system() == 'Windows':
            return 'C:/Windows/Fonts/simhei.ttf'
        elif platform.system() == 'Linux':
            return '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
        elif platform.system() == 'Darwin':  # macOS
            return '/System/Library/Fonts/PingFang.ttc'
        else:
            # 如果找不到系统字体，返回默认字体
            print("警告: 无法找到中文字体，可能导致中文显示异常")
            return cv2.FONT_HERSHEY_SIMPLEX
        
    def load_templates(self):
        """加载花色模板图片"""
        print("正在加载花色模板...")
        
        # 尝试加载新调整大小的模板
        new_template_dir = Path("1/new_templates")
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
                    template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
                    if template is not None:
                        # 转换为灰度图
                        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                        self.templates[suit_name] = template_gray
                        print(f"✅ 加载新模板: {filename} -> {self.suit_names[suit_name]}")
                    else:
                        print(f"❌ 无法读取新模板: {filename}")
                else:
                    print(f"❌ 新模板文件不存在: {filename}")
        else:
            print("新模板文件夹不存在，使用原始模板")
            
        # 如果没有加载到新模板，则使用原始模板
        if not self.templates:
            template_files = {
                'club': 'club_gray.png',
                'diamond': 'diamond_gray.png', 
                'heart': 'heart_gray.png',
                'spade': 'spade_gray.png'
            }
            
            for suit_name, filename in template_files.items():
                template_path = self.template_dir / filename
                if template_path.exists():
                    # 读取模板图片
                    template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
                    if template is not None:
                        # 转换为灰度图
                        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                        self.templates[suit_name] = template_gray
                        print(f"✅ 加载模板: {filename} -> {self.suit_names[suit_name]}")
                    else:
                        print(f"❌ 无法读取模板: {filename}")
                else:
                    print(f"❌ 模板文件不存在: {filename}")
                
        print(f"共加载 {len(self.templates)} 个模板")
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理
        
        Args:
            image: 输入图像
            
        Returns:
            预处理后的图像
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 调整对比度和亮度
        # 创建一个CLAHE对象(对比度受限的自适应直方图均衡化)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrasted = clahe.apply(gray)
        
        # 轻微的高斯模糊去除噪声
        blurred = cv2.GaussianBlur(contrasted, (3, 3), 0)
        
        # 二值化处理
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
        
    def match_template(self, image: np.ndarray, template: np.ndarray, 
                      threshold: float = 0.6) -> Tuple[float, Tuple[int, int]]:
        """
        模板匹配
        
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
        for suit_name, template in self.templates.items():
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
                if new_h <= gray_image.shape[0] and new_w <= gray_image.shape[1]:
                    score, position = self.match_template(gray_image, resized_template, threshold)
                    
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
        
    def calculate_adaptive_regions(self, image_width: int, image_height: int, 
                                  adaptive_regions: Dict[str, List[float]], 
                                  reference_size: List[int]) -> Dict[str, List[int]]:
        """
        根据图像尺寸计算自适应区域坐标
        
        Args:
            image_width: 当前图像宽度
            image_height: 当前图像高度
            adaptive_regions: 相对坐标区域 {区域名: [x1_ratio, y1_ratio, x2_ratio, y2_ratio]}
            reference_size: 参考尺寸 [width, height]
            
        Returns:
            绝对坐标区域 {区域名: [x1, y1, x2, y2]}
        """
        ref_width, ref_height = reference_size
        absolute_regions = {}
        
        for region_name, ratios in adaptive_regions.items():
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
                print(f"  {region_name} 区域大小: {region_image.shape}")
                if 'best_match' in region_result:
                    best = region_result['best_match']
                    print(f"    最佳匹配: {best['chinese_name']} (分数: {best['score']:.3f})")
                else:
                    print(f"    未找到匹配的花色")
                    
                # 显示所有花色的匹配分数
                for suit_name, result in region_result.items():
                    if suit_name != 'best_match':
                        print(f"    {result['chinese_name']}: {result['score']:.3f}")
                
                results[region_name] = region_result
            else:
                results[region_name] = {'error': '区域图像为空'}
                
        return results
        
    def visualize_results(self, image: np.ndarray, results: Dict, 
                         save_path: Optional[str] = None) -> np.ndarray:
        """
        可视化识别结果
        
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
                template = self.templates[suit_name]
                h, w = template.shape
                
                # 绘制矩形框
                x, y = position
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # 绘制标签
                label = f"{chinese_name}: {score:.3f}"
                if isinstance(self.font, str):
                    # 使用支持中文的字体
                    from PIL import Image, ImageDraw, ImageFont
                    import numpy as np
                     
                    # 将OpenCV图像转换为PIL图像
                    pil_image = Image.fromarray(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_image)
                    
                    try:
                        # 加载字体
                        font = ImageFont.truetype(self.font, 14)
                        # 绘制文本
                        draw.text((x, y - 10), label, font=font, fill=(0, 255, 0))
                        # 转换回OpenCV图像
                        vis_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    except Exception as e:
                        print(f"加载中文字体失败: {e}")
                        # 回退到默认字体
                        cv2.putText(vis_image, label, (x, y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    # 使用默认字体
                    cv2.putText(vis_image, label, (x, y - 10), 
                               self.font, 0.6, (0, 255, 0), 2)
                
        # 保存结果
        if save_path:
            cv2.imwrite(save_path, vis_image)
            print(f"结果图像已保存: {save_path}")
            
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
                if isinstance(self.font, str):
                    # 使用支持中文的字体
                    from PIL import Image, ImageDraw, ImageFont
                    import numpy as np
                     
                    # 将OpenCV图像转换为PIL图像
                    pil_image = Image.fromarray(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_image)
                    
                    try:
                        # 加载字体
                        font = ImageFont.truetype(self.font, 12)
                        # 绘制文本
                        draw.text((x1, y1 - 10), label, font=font, fill=tuple(color))
                        
                        # 在区域中心显示花色名称
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        font_large = ImageFont.truetype(self.font, 14)
                        draw.text((center_x - 20, center_y + 5), suit_name, font=font_large, fill=tuple(color))
                        
                        # 转换回OpenCV图像
                        vis_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    except Exception as e:
                        print(f"加载中文字体失败: {e}")
                        # 回退到默认字体
                        cv2.putText(vis_image, label, (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        cv2.putText(vis_image, suit_name, (center_x - 20, center_y + 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    # 使用默认字体
                    cv2.putText(vis_image, label, (x1, y1 - 10), 
                               self.font, 0.5, color, 2)
                    cv2.putText(vis_image, suit_name, (center_x - 20, center_y + 5), 
                               self.font, 0.6, color, 2)
            else:
                # 没有识别结果
                label = f"{region_name}: 未识别"
                if isinstance(self.font, str):
                    # 使用支持中文的字体
                    from PIL import Image, ImageDraw, ImageFont
                    import numpy as np
                     
                    # 将OpenCV图像转换为PIL图像
                    pil_image = Image.fromarray(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_image)
                    
                    try:
                        # 加载字体
                        font = ImageFont.truetype(self.font, 12)
                        # 绘制文本
                        draw.text((x1, y1 - 10), label, font=font, fill=(128, 128, 128))
                        # 转换回OpenCV图像
                        vis_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    except Exception as e:
                        print(f"加载中文字体失败: {e}")
                        # 回退到默认字体
                        cv2.putText(vis_image, label, (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
                else:
                    # 使用默认字体
                    cv2.putText(vis_image, label, (x1, y1 - 10), 
                               self.font, 0.5, (128, 128, 128), 2)
                
        # 保存结果
        if save_path:
            cv2.imwrite(save_path, vis_image)
            print(f"结果图像已保存: {save_path}")
            
        return vis_image
        
    def save_results_to_txt(self, results: Dict, save_path: str):
        """
        将识别结果保存到txt文件
        
        Args:
            results: 识别结果
            save_path: 保存路径
        """
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("花色识别结果\n")
                f.write("=" * 30 + "\n")
                
                # 写入每个花色的匹配分数
                for suit_name, result in results.items():
                    if suit_name == 'best_match':
                        continue
                    status = "匹配成功" if result['matched'] else "匹配失败"
                    f.write(f"{result['chinese_name']}: {result['score']:.3f} ({status})\n")
                
                # 写入最佳匹配
                if 'best_match' in results:
                    best = results['best_match']
                    f.write("\n最佳匹配:\n")
                    f.write(f"花色: {best['chinese_name']}\n")
                    f.write(f"分数: {best['score']:.3f}\n")
                    f.write(f"置信度: {best['confidence']}\n")
                
            print(f"识别结果已保存到: {save_path}")
        except Exception as e:
            print(f"保存识别结果到txt文件失败: {e}")
            
    def test_recognition(self, test_image_path: str, threshold: float = 0.7):
        """
        测试花色识别功能
        
        Args:
            test_image_path: 测试图像路径
            threshold: 匹配阈值
        """
        print(f"\n测试花色识别: {test_image_path}")
        
        # 读取测试图像
        image = cv2.imread(test_image_path)
        if image is None:
            print(f"❌ 无法读取测试图像: {test_image_path}")
            return
            
        # 进行识别
        results = self.recognize_suit(image, threshold)
        
        # 显示结果
        print("\n识别结果:")
        for suit_name, result in results.items():
            if suit_name == 'best_match':
                continue
            print(f"{result['chinese_name']}: {result['score']:.3f} {'✅' if result['matched'] else '❌'}")
            
        if 'best_match' in results:
            best = results['best_match']
            print(f"\n最佳匹配: {best['chinese_name']} (置信度: {best['confidence']})")
            
        # 可视化结果
        vis_image = self.visualize_results(image, results, f"result_{Path(test_image_path).stem}.png")
        
        # 保存结果到txt文件
        txt_save_path = f"result_{Path(test_image_path).stem}.txt"
        self.save_results_to_txt(results, txt_save_path)
        
        return results

def main():
    """主函数"""
    print("=" * 50)
    print("花色识别程序")
    print("=" * 50)
    
    # 创建识别器
    recognizer = SuitRecognizer()
    
    # 加载配置文件中的花色区域
    try:
        with open('suit_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 获取参考尺寸
        reference_size = config.get('reference_size', [546, 392])
        
        # 提取花色相关的区域
        suit_regions = {}
        for region_name, coords in config['test_regions'].items():
            if '花色' in region_name:
                suit_regions[region_name] = coords
                
        print(f"从配置文件中加载了 {len(suit_regions)} 个花色区域:")
        for region_name in suit_regions.keys():
            print(f"  - {region_name}")
            
    except Exception as e:
        print(f"❌ 加载配置文件失败: {e}")
        return
    
    # 测试识别功能
    test_images = [
        "1/Snipaste_2025-08-13_13-13-44.png",
        "1/Snipaste_2025-08-13_13-15-54.png"
    ]
    
    for test_image in test_images:
        if os.path.exists(test_image):
            print(f"\n{'='*60}")
            print(f"测试图像: {test_image}")
            print(f"{'='*60}")
            
            # 读取测试图像
            image = cv2.imread(test_image)
            if image is None:
                print(f"❌ 无法读取测试图像: {test_image}")
                continue
                
            # 获取图像尺寸
            image_height, image_width = image.shape[:2]
            print(f"图像尺寸: {image_width} x {image_height}")
            
            # 如果图像尺寸与参考尺寸不同，使用自适应区域
            if image_width != reference_size[0] or image_height != reference_size[1]:
                print(f"检测到图像尺寸与参考尺寸不同，使用自适应区域计算")
                print(f"参考尺寸: {reference_size[0]} x {reference_size[1]}")
                
                # 计算自适应区域
                adaptive_suit_regions = recognizer.calculate_adaptive_regions(
                    image_width, image_height, config['adaptive_regions'], reference_size)
                suit_regions = adaptive_suit_regions
                print(f"已计算 {len(suit_regions)} 个自适应花色区域")
            else:
                print(f"图像尺寸与参考尺寸相同，使用配置文件中的固定区域")
                
            # 识别所有花色区域
            results = recognizer.recognize_suits_in_regions(image, suit_regions, threshold=0.6)
            
            # 显示识别结果
            print("\n花色识别结果:")
            for region_name, result in results.items():
                if 'error' in result:
                    print(f"  {region_name}: {result['error']}")
                elif 'best_match' in result:
                    best = result['best_match']
                    print(f"  {region_name}: {best['chinese_name']} (置信度: {best['confidence']}, 分数: {best['score']:.3f})")
                    
            # 可视化结果
            vis_image = recognizer.visualize_suit_regions(image, suit_regions, results, 
                                                        f"result/result_{Path(test_image).stem}.png")
            
            # 保存区域识别结果到txt文件
            txt_save_path = f"result/result_{Path(test_image).stem}.txt"
            try:
                with open(txt_save_path, 'w', encoding='utf-8') as f:
                    f.write("区域花色识别结果\n")
                    f.write("=" * 40 + "\n")
                    
                    for region_name, region_result in results.items():
                        f.write(f"区域: {region_name}\n")
                        
                        if 'error' in region_result:
                            f.write(f"  错误: {region_result['error']}\n")
                        elif 'best_match' in region_result:
                            best = region_result['best_match']
                            f.write(f"  识别结果: {best['chinese_name']}\n")
                            f.write(f"  匹配分数: {best['score']:.3f}\n")
                            f.write(f"  置信度: {best['confidence']}\n")
                        else:
                            f.write("  识别结果: 未识别\n")
                        
                        f.write("-" * 30 + "\n")
                    
                print(f"区域识别结果已保存到: {txt_save_path}")
            except Exception as e:
                print(f"保存区域识别结果到txt文件失败: {e}")
            
        else:
            print(f"测试图像不存在: {test_image}")
            
    print("\n花色识别程序运行完成！")

if __name__ == "__main__":
    main()
