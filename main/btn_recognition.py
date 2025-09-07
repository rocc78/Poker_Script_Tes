#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于OpenCV的btn按钮识别程序
使用matchTemplate方法识别图像中的btn按钮
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Union
import json
from PIL import Image, ImageDraw, ImageFont


class BtnRecognizer:
    """btn按钮识别器"""
    
    def __init__(self, template_path: str = "1/new_templates/btn.png", deal_template_path: str = "1/new_templates/deal_btn.png"):
        """
        初始化btn按钮识别器
        
        Args:
            template_path: btn模板图片路径
            deal_template_path: 发牌按钮模板图片路径
        """
        self.template_path = Path(template_path)
        self.deal_template_path = Path(deal_template_path)
        self.template = None
        self.deal_template = None
        self.back_template_path = Path("1/back.png")
        self.back_template = None
        self.btn_config = self.load_btn_config()
        self.ocr_config = self.load_ocr_config()
        self.load_template()
        self.load_deal_template()
        self.load_back_template()
        
    def load_template(self):
        """加载btn模板图片"""
        print(f"正在加载btn模板: {self.template_path}")
        
        if self.template_path.exists():
            # 读取模板图片
            template = cv2.imread(str(self.template_path), cv2.IMREAD_GRAYSCALE)
            if template is not None:
                self.template = template
                print(f"[SUCCESS] 加载模板成功: {self.template_path.name}")
                print(f"  模板尺寸: {template.shape[1]}x{template.shape[0]}")
            else:
                print(f"❌ 无法读取模板: {self.template_path}")
        else:
            print(f"[ERROR] 模板文件不存在: {self.template_path}")
    
    def load_deal_template(self):
        """加载发牌按钮模板图片"""
        print(f"正在加载发牌按钮模板: {self.deal_template_path}")
        
        if self.deal_template_path.exists():
            # 读取模板图片
            template = cv2.imread(str(self.deal_template_path), cv2.IMREAD_GRAYSCALE)
            if template is not None:
                self.deal_template = template
                print(f"✅ 加载发牌按钮模板成功: {self.deal_template_path.name}")
                print(f"  模板尺寸: {template.shape[1]}x{template.shape[0]}")
            else:
                print(f"❌ 无法读取发牌按钮模板: {self.deal_template_path}")
        else:
            print(f"❌ 发牌按钮模板文件不存在: {self.deal_template_path}")
    
    def load_back_template(self):
        """加载扑克背面模板图片"""
        print(f"正在加载扑克背面模板: {self.back_template_path}")
        
        if self.back_template_path.exists():
            # 读取模板图片
            template = cv2.imread(str(self.back_template_path), cv2.IMREAD_GRAYSCALE)
            if template is not None:
                self.back_template = template
                print(f"✅ 加载扑克背面模板成功: {self.back_template_path.name}")
                print(f"  模板尺寸: {template.shape[1]}x{template.shape[0]}")
            else:
                print(f"❌ 无法读取扑克背面模板: {self.back_template_path}")
        else:
            print(f"❌ 扑克背面模板文件不存在: {self.back_template_path}")
    
    def load_btn_config(self, config_path="main/poker_recognition_config.json"):
        """
        加载btn配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        config_path = Path(config_path)
        if not config_path.exists():
            print(f"[WARNING] btn配置文件不存在: {config_path}")
            return None
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # 返回btn配置部分
                return config.get('btn', None)
        except Exception as e:
            print(f"[ERROR] 加载btn配置文件失败: {e}")
            return None
    
    def load_ocr_config(self, config_path="main/poker_recognition_config.json"):
        """
        加载ocr配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        config_path = Path(config_path)
        if not config_path.exists():
            print(f"[WARNING] ocr配置文件不存在: {config_path}")
            return None
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # 返回ocr配置部分
                return config.get('ocr', None)
        except Exception as e:
            print(f"[ERROR] 加载ocr配置文件失败: {e}")
            return None
    
    def calculate_ocr_adaptive_regions(self, image_width, image_height):
        """
        根据图像尺寸计算ocr自适应区域
        
        Args:
            image_width: 图像宽度
            image_height: 图像高度
            
        Returns:
            自适应区域字典
        """
        if self.ocr_config is None or 'adaptive_regions' not in self.ocr_config or 'reference_size' not in self.ocr_config:
            return None
        
        reference_width, reference_height = self.ocr_config['reference_size']
        adaptive_regions = self.ocr_config['adaptive_regions']
        
        # 计算比例
        width_ratio = image_width / reference_width
        height_ratio = image_height / reference_height
        
        # 计算自适应区域
        absolute_regions = {}
        for name, ratios in adaptive_regions.items():
            x1_ratio, y1_ratio, x2_ratio, y2_ratio = ratios
            x1 = int(x1_ratio * image_width)
            y1 = int(y1_ratio * image_height)
            x2 = int(x2_ratio * image_width)
            y2 = int(y2_ratio * image_height)
            absolute_regions[name] = [x1, y1, x2, y2]
        
        return absolute_regions
    
    def calculate_adaptive_regions(self, image_width, image_height):
        """
        根据图像尺寸计算自适应区域
        
        Args:
            image_width: 图像宽度
            image_height: 图像高度
            
        Returns:
            自适应区域字典
        """
        if self.btn_config is None or 'adaptive_regions' not in self.btn_config or 'reference_size' not in self.btn_config:
            return None
        
        reference_width, reference_height = self.btn_config['reference_size']
        adaptive_regions = self.btn_config['adaptive_regions']
        
        # 计算比例
        width_ratio = image_width / reference_width
        height_ratio = image_height / reference_height
        
        # 计算自适应区域
        absolute_regions = {}
        for name, ratios in adaptive_regions.items():
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
            detected_x: 检测到的btn位置x坐标
            detected_y: 检测到的btn位置y坐标
            image_width: 图像宽度
            image_height: 图像高度
            confidence_threshold: 置信度阈值
            
        Returns:
            位置名称和置信度信息
        """
        # 计算自适应区域
        adaptive_regions = self.calculate_adaptive_regions(image_width, image_height)
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
    
    def match_template(self, image: np.ndarray, template: np.ndarray, 
                      threshold: float = 0.6) -> Dict:
        """
        模板匹配
        
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
        
        # 尝试多种匹配方法并选择最佳结果
        methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF_NORMED]
        best_score = -1.0
        best_result = None
        
        for method in methods:
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
                
                # 更新最佳结果
                if score > best_score:
                    best_score = score
                    best_result = {
                        'score': score,
                        'position': loc,
                        'matched': score >= threshold,
                        'threshold': threshold,
                        'method': method
                    }
            except cv2.error as e:
                # 如果匹配失败，继续尝试其他方法
                print(f"模板匹配错误 (方法 {method}): {e}")
                continue
        
        # 如果所有方法都失败，返回默认值
        if best_result is None:
            return {
                'score': -1.0,
                'position': (0, 0),
                'matched': False,
                'threshold': threshold
            }
        
        return best_result
    
    def recognize_btn_multiscale_with_template(self, image: np.ndarray, template: np.ndarray, threshold: float = 0.6) -> Dict:
        """
        多尺度btn按钮识别（指定模板）
        
        Args:
            image: 输入图像
            template: 模板图像
            threshold: 匹配阈值
            
        Returns:
            识别结果字典
        """
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
            h, w = template.shape
            new_h, new_w = int(h * scale), int(w * scale)
            resized_template = cv2.resize(template, (new_w, new_h))
            
            # 确保缩放后的模板不大于图像
            if new_h <= gray_image.shape[0] and new_w <= gray_image.shape[1]:
                result = self.match_template(gray_image, resized_template, threshold)
                
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
    
    def recognize_btn_multiscale(self, image: np.ndarray, threshold: float = 0.6) -> Dict:
        """
        多尺度btn按钮识别
        
        Args:
            image: 输入图像
            threshold: 匹配阈值
            
        Returns:
            识别结果字典
        """
        if self.template is None:
            return {'error': '模板未加载'}
        
        return self.recognize_btn_multiscale_with_template(image, self.template, threshold)
    
    def recognize_deal_btn(self, image: np.ndarray, threshold: float = 0.6, multiscale: bool = True) -> Dict:
        """
        识别图像中的发牌按钮
        
        Args:
            image: 输入图像
            threshold: 匹配阈值
            multiscale: 是否使用多尺度匹配
            
        Returns:
            识别结果字典
        """
        # 只使用发牌按钮模板
        if self.deal_template is not None:
            template = self.deal_template
        else:
            return {'error': '发牌按钮模板未加载'}
            
        # 进行模板匹配
        if multiscale:
            result = self.recognize_btn_multiscale_with_template(image, template, threshold)
        else:
            result = self.match_template(image, template, threshold)
        
        return result
    
    def recognize_btn(self, image: np.ndarray, threshold: float = 0.6, multiscale: bool = True, is_deal_action: bool = False) -> Dict:
        """
        识别图像中的btn按钮
        
        Args:
            image: 输入图像
            threshold: 匹配阈值
            multiscale: 是否使用多尺度匹配
            is_deal_action: 是否为发牌动作
            
        Returns:
            识别结果字典
        """
        # 根据是否为发牌动作选择模板
        if is_deal_action and self.deal_template is not None:
            template = self.deal_template
        elif self.template is not None:
            template = self.template
        else:
            return {'error': '模板未加载'}
            
        # 进行模板匹配
        if multiscale:
            result = self.recognize_btn_multiscale_with_template(image, template, threshold)
        else:
            result = self.match_template(image, template, threshold)
        
        return result
    
    def get_btn_position(self, image_path: str, threshold: float = 0.6, multiscale: bool = True, is_deal_action: bool = False) -> Union[Dict, List[Dict]]:
        """
        获取btn在图中的位置信息
        
        Args:
            image_path: 图像路径
            threshold: 匹配阈值
            multiscale: 是否使用多尺度匹配
            is_deal_action: 是否为发牌动作
            
        Returns:
            位置信息字典或列表
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            return {'error': f'无法读取图像: {image_path}'}
        
        # 进行识别
        if multiscale:
            result = self.recognize_btn_multiscale(image, threshold)
        else:
            result = self.recognize_btn(image, threshold, is_deal_action=is_deal_action)
        
        # 处理结果
        if 'error' in result:
            return result
        
        # 根据是否为发牌动作选择模板
        if is_deal_action and self.deal_template is not None:
            template = self.deal_template
        elif self.template is not None:
            template = self.template
        else:
            return {'error': '模板未加载'}
        
        if result['matched']:
            x, y = result['position']
            if multiscale and 'scale' in result:
                h, w = int(template.shape[0] * result['scale']), int(template.shape[1] * result['scale'])
            else:
                h, w = template.shape
            
            return {
                'matched': True,
                'score': result['score'],
                'position': (x, y),
                'region': (x, y, x + w, y + h),
                'scale': result.get('scale', 1.0)
            }
        else:
            return {
                'matched': False,
                'max_score': result['score']
            }
    
    def visualize_result(self, image: np.ndarray, result: Dict, save_path: str = None) -> np.ndarray:
        """
        可视化识别结果
        
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
            h, w = self.template.shape
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
            if self.btn_config is not None:
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
            print(f"结果图像已保存: {save_path}")
            
        return vis_image
    
    def visualize_fold_result(self, image: np.ndarray, fold_status: Dict, method_name: str, save_path: str = None) -> np.ndarray:
        """
        可视化玩家弃牌状态结果
        
        Args:
            image: 原始图像
            fold_status: 玩家弃牌状态字典
            method_name: 检测方法名称
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
        
        # 在图像顶部显示方法名称
        draw.text((10, 10), f"Fold Detection: {method_name}", (0, 255, 255), font=font)
        
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
                color = (255, 0, 0) if status['folded'] else (0, 255, 0)  # 红色表示已弃牌，绿色表示未弃牌
                
                # 显示玩家状态（交换已弃牌和未弃牌的显示）
                if 'variance' in status:
                    # 颜色方差方法
                    label = f"{player}: {'已弃牌' if status['folded'] else '未弃牌'} (方差: {status['variance']:.2f})"
                else:
                    # 模板匹配方法
                    label = f"{player}: {'已弃牌' if status['folded'] else '未弃牌'} (置信度: {status['confidence']:.3f})"
                
                draw.text((10, y_offset), label, color, font=font)
                y_offset += 25
        
        # 转换回OpenCV格式
        vis_image = cv2.cvtColor(np.array(vis_pil), cv2.COLOR_RGB2BGR)
        
        # 保存结果
        if save_path:
            cv2.imwrite(save_path, vis_image)
            print(f"弃牌检测结果图像已保存: {save_path}")
            
        return vis_image
    
    def check_player_fold(self, image: np.ndarray, threshold: float = 0.3) -> Dict:
        """
        检查玩家是否弃牌
        
        Args:
            image: 输入图像
            threshold: 匹配阈值
            
        Returns:
            玩家弃牌状态字典
        """
        if self.back_template is None:
            return {'error': '扑克背面模板未加载'}
        
        # 计算自适应区域
        image_height, image_width = image.shape[:2]
        adaptive_regions = self.calculate_adaptive_regions(image_width, image_height)
        
        # 检查每个玩家区域
        player_status = {}
        for i in range(1, 6):
            player_key = f"玩家{i}扑克背面"
            if player_key in adaptive_regions:
                x1, y1, x2, y2 = adaptive_regions[player_key]
                # 提取区域图像
                region_image = image[y1:y2, x1:x2]
                
                if region_image.size > 0:
                    # 对区域图像进行预处理以提高纯色匹配效果
                    if len(region_image.shape) == 3:
                        gray_region = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY)
                    else:
                        gray_region = region_image.copy()
                    
                    # 使用多种预处理方法来提高匹配效果
                    # 1. 高斯模糊
                    blurred = cv2.GaussianBlur(gray_region, (3, 3), 0)
                    # 2. 中值滤波
                    median = cv2.medianBlur(gray_region, 3)
                    # 3. 双边滤波
                    bilateral = cv2.bilateralFilter(gray_region, 9, 75, 75)
                    
                    # 尝试多种匹配方法并选择最佳结果
                    methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]
                    best_score = 0
                    
                    for method in methods:
                        # 对每种预处理图像进行模板匹配
                        for processed_img in [blurred, median, bilateral]:
                            # 模板匹配
                            res = cv2.matchTemplate(processed_img, self.back_template, method)
                            _, max_val, _, _ = cv2.minMaxLoc(res)
                            
                            # 更新最佳匹配分数
                            if max_val > best_score:
                                best_score = max_val
                    
                    # 调整判断逻辑：如果匹配分数较高，表示匹配到了扑克背面，玩家未弃牌
                    # 如果匹配分数较低，表示没有匹配到扑克背面，玩家可能已弃牌
                    if best_score > threshold:
                        player_status[f"玩家{i}"] = {
                            'folded': False,
                            'confidence': best_score
                        }
                    else:
                        player_status[f"玩家{i}"] = {
                            'folded': True,
                            'confidence': best_score
                        }
                else:
                    player_status[f"玩家{i}"] = {
                        'folded': True,
                        'confidence': 0.0,
                        'error': '区域图像为空'
                    }
            else:
                player_status[f"玩家{i}"] = {
                    'folded': True,
                    'confidence': 0.0,
                    'error': '未找到区域定义'
                }
        
        return player_status
    
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
        adaptive_regions = self.calculate_ocr_adaptive_regions(image_width, image_height)
        
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


def main():
    """主函数 - 测试btn识别功能"""
    print("=" * 50)
    print("btn按钮识别程序")
    print("=" * 50)
    
    # 创建识别器
    recognizer = BtnRecognizer()
    
    # 测试图像
    test_images = [
        "1/eng/Snipaste_2025-09-02_23-17-25.png",
        "1/eng/Snipaste_2025-08-27_23-32-01.png"
    ]
    
    for test_image_path in test_images:
        if Path(test_image_path).exists():
            print(f"\n测试图像: {test_image_path}")
            
            # 读取测试图像
            image = cv2.imread(test_image_path)
            if image is None:
                print(f"[ERROR] 无法读取测试图像: {test_image_path}")
                continue
                
            # 进行btn识别（默认使用多尺度匹配）
            result = recognizer.recognize_btn(image, threshold=0.6, multiscale=True)
            
            # 显示btn识别结果
            if 'error' in result:
                print(f"    [ERROR]: {result['error']}")
            else:
                if result['matched']:
                    x, y = result['position']
                    if 'scale' in result:
                        h, w = int(recognizer.template.shape[0] * result['scale']), int(recognizer.template.shape[1] * result['scale'])
                    else:
                        h, w = recognizer.template.shape
                    print(f"  [DONE] btn匹配成功!")
                    print(f"    置信度: {result['score']:.3f}")
                    if 'scale' in result:
                        print(f"    缩放比例: {result['scale']:.1f}")
                    print(f"    位置: ({x}, {y})")
                    print(f"    区域: ({x}, {y}) - ({x+w}, {y+h})")
                else:
                    print(f"  [ERROR] 未匹配到btn按钮 (最高分数: {result['score']:.3f})")
            
            # 检查玩家是否弃牌（使用颜色方差方法）
            print("  \n检查玩家弃牌状态 (颜色方差方法):")
            fold_status_variance = recognizer.check_player_fold_by_variance(image, variance_threshold=10.0)
            if 'error' in fold_status_variance:
                print(f"    [ERROR]: {fold_status_variance['error']}")
            else:
                for player, status in fold_status_variance.items():
                    if 'error' in status:
                        print(f"    [ERROR] {player}: 错误 - {status['error']}")
                    else:
                        # 交换已弃牌和未弃牌的显示
                        if status['folded']:
                            print(f"    {player}: 未弃牌 (方差: {status['variance']:.2f}, 置信度: {status['confidence']:.3f})")
                        else:
                            print(f"    {player}: 已弃牌 (方差: {status['variance']:.2f}, 置信度: {status['confidence']:.3f})")
            
            # 创建result目录（如果不存在）
            result_dir = Path("result")
            result_dir.mkdir(exist_ok=True)
            
            # 可视化btn识别结果并保存到result目录
            vis_image = recognizer.visualize_result(image, result, 
                                                  f"result/result_btn_{Path(test_image_path).stem}.png")
            
            # 可视化弃牌检测结果（颜色方差方法）并保存到result目录
            if 'error' not in fold_status_variance:
                recognizer.visualize_fold_result(image, fold_status_variance, "颜色方差", 
                                               f"result/result_fold_variance_{Path(test_image_path).stem}.png")
        else:
            print(f"测试图像不存在: {test_image_path}")

if __name__ == "__main__":
    main()