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


class BtnRecognizer:
    """btn按钮识别器"""
    
    def __init__(self, template_path: str = "1/new_templates/btn.png"):
        """
        初始化btn按钮识别器
        
        Args:
            template_path: btn模板图片路径
        """
        self.template_path = Path(template_path)
        self.template = None
        self.load_template()
        
    def load_template(self):
        """加载btn模板图片"""
        print(f"正在加载btn模板: {self.template_path}")
        
        if self.template_path.exists():
            # 读取模板图片
            template = cv2.imread(str(self.template_path), cv2.IMREAD_GRAYSCALE)
            if template is not None:
                self.template = template
                print(f"✅ 加载模板成功: {self.template_path.name}")
                print(f"  模板尺寸: {template.shape[1]}x{template.shape[0]}")
            else:
                print(f"❌ 无法读取模板: {self.template_path}")
        else:
            print(f"❌ 模板文件不存在: {self.template_path}")
    
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
        多尺度btn按钮识别
        
        Args:
            image: 输入图像
            threshold: 匹配阈值
            
        Returns:
            识别结果字典
        """
        if self.template is None:
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
            h, w = self.template.shape
            new_h, new_w = int(h * scale), int(w * scale)
            resized_template = cv2.resize(self.template, (new_w, new_h))
            
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
    
    def recognize_btn(self, image: np.ndarray, threshold: float = 0.6, multiscale: bool = True) -> Dict:
        """
        识别图像中的btn按钮
        
        Args:
            image: 输入图像
            threshold: 匹配阈值
            multiscale: 是否使用多尺度匹配
            
        Returns:
            识别结果字典
        """
        if self.template is None:
            return {'error': '模板未加载'}
            
        # 进行模板匹配
        if multiscale:
            result = self.recognize_btn_multiscale(image, threshold)
        else:
            result = self.match_template(image, self.template, threshold)
        
        return result
    
    def get_btn_position(self, image_path: str, threshold: float = 0.6, multiscale: bool = True) -> Union[Dict, List[Dict]]:
        """
        获取btn在图中的位置信息
        
        Args:
            image_path: 图像路径
            threshold: 匹配阈值
            multiscale: 是否使用多尺度匹配
            
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
            result = self.recognize_btn(image, threshold)
        
        # 处理结果
        if 'error' in result:
            return result
        
        if result['matched']:
            x, y = result['position']
            if multiscale and 'scale' in result:
                h, w = int(self.template.shape[0] * result['scale']), int(self.template.shape[1] * result['scale'])
            else:
                h, w = self.template.shape
            
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
        else:
            # 显示未匹配到的信息
            cv2.putText(vis_image, f"No btn detected (max score: {result['score']:.3f})", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 保存结果
        if save_path:
            cv2.imwrite(save_path, vis_image)
            print(f"结果图像已保存: {save_path}")
            
        return vis_image


def main():
    """主函数 - 测试btn识别功能"""
    print("=" * 50)
    print("btn按钮识别程序")
    print("=" * 50)
    
    # 创建识别器
    recognizer = BtnRecognizer()
    
    # 测试图像
    test_images = [
        "1/Snipaste_2025-08-13_13-13-44.png",
        "1/Snipaste_2025-08-13_13-15-54.png"
    ]
    
    for test_image_path in test_images:
        if Path(test_image_path).exists():
            print(f"\n测试图像: {test_image_path}")
            
            # 读取测试图像
            image = cv2.imread(test_image_path)
            if image is None:
                print(f"❌ 无法读取测试图像: {test_image_path}")
                continue
                
            # 进行识别（默认使用多尺度匹配）
            result = recognizer.recognize_btn(image, threshold=0.6, multiscale=True)
            
            # 显示结果
            if 'error' in result:
                print(f"  错误: {result['error']}")
            else:
                if result['matched']:
                    x, y = result['position']
                    if 'scale' in result:
                        h, w = int(recognizer.template.shape[0] * result['scale']), int(recognizer.template.shape[1] * result['scale'])
                    else:
                        h, w = recognizer.template.shape
                    print(f"  ✅ 匹配成功!")
                    print(f"    置信度: {result['score']:.3f}")
                    if 'scale' in result:
                        print(f"    缩放比例: {result['scale']:.1f}")
                    print(f"    位置: ({x}, {y})")
                    print(f"    区域: ({x}, {y}) - ({x+w}, {y+h})")
                else:
                    print(f"  ❌ 未匹配到btn按钮 (最高分数: {result['score']:.3f})")
            
            # 创建result目录（如果不存在）
            result_dir = Path("result")
            result_dir.mkdir(exist_ok=True)
            
            # 可视化结果并保存到result目录
            vis_image = recognizer.visualize_result(image, result, 
                                                  f"result/result_btn_{Path(test_image_path).stem}.png")
        else:
            print(f"测试图像不存在: {test_image_path}")

if __name__ == "__main__":
    main()