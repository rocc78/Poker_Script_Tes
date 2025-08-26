#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扑克牌局分析器
按顺序处理1目录下以Snipaste开头的图片，识别跟注位置并记录牌局信息
"""

import os
import json
from datetime import datetime
from pathlib import Path

# 尝试导入必要的库，处理可能的兼容性问题
try:
    import cv2
    import numpy as np
    from poker_recognition import PokerRecognizer
    from btn_recognition import BtnRecognizer
except ImportError as e:
    print(f"[ERROR] 导入错误: {e}")
    print("请确保已安装所有必要的依赖库，并且NumPy版本兼容")
    print("可以尝试降级NumPy版本: pip install numpy<2")
    exit(1)


class PokerGameAnalyzer:
    """扑克牌局分析器"""
    
    def __init__(self):
        """初始化分析器"""
        try:
            self.poker_recognizer = PokerRecognizer('poker_recognition_config.json')
            self.btn_recognizer = BtnRecognizer(
                template_path="1/new_templates/btn.png",
                deal_template_path="1/new_templates/deal_btn.png"
            )
        except Exception as e:
            print(f"[ERROR] 初始化识别器失败: {e}")
            raise
        self.game_history = []
        self.current_hand = None
        self.action_history = []
        
    def get_snipaste_images(self, directory='1\eng'):
        """获取1目录下所有以Snipaste开头的图片文件"""
        snipaste_images = []
        for filename in os.listdir(directory):
            if filename.startswith('Snipaste') and filename.endswith(('.png', '.jpg', '.jpeg')):
                snipaste_images.append(os.path.join(directory, filename))
        # 按文件名排序
        snipaste_images.sort()
        return snipaste_images
    
    def identify_action_position(self, image_path, language='chi_sim'):
        """识别发牌、加注动作位置 - 仅使用OCR"""
        print(f"\n[INFO] 识别动作位置: {image_path}")
        
        # 加载图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] 无法加载图片: {image_path}")
            return None
        
        # 转换为PIL图像以供OCR使用
        from PIL import Image
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # 设置识别语言
        original_language = self.poker_recognizer.language
        self.poker_recognizer.language = language
        
        # 只识别加注位置和发牌位置
        target_regions = {}
        all_regions = self.poker_recognizer.test_regions
        
        # 根据语言设置选择相应的区域名称
        if language == 'chi_sim':
            # 中文模式下识别'加注'和'发牌'区域
            if '加注' in all_regions:
                target_regions['加注'] = all_regions['加注']
            if '发牌' in all_regions:
                target_regions['发牌'] = all_regions['发牌']
        else:
            # 英文模式下也识别'加注'和'发牌'区域（因为配置文件中只有中文区域名称）
            if '加注' in all_regions:
                target_regions['加注'] = all_regions['加注']
            if '发牌' in all_regions:
                target_regions['发牌'] = all_regions['发牌']
        
        # 使用OCR识别指定区域
        ocr_results = self.poker_recognizer.recognize_regions(pil_image, target_regions)
        
        # 恢复原始语言设置
        self.poker_recognizer.language = original_language
        
        # 检查所有识别结果中是否包含相应文字
        print(f"[DEBUG] OCR识别结果: {ocr_results}")
        
        # 遍历所有区域
        for region_name, result in ocr_results.items():
            if result.get('success', False):
                text = result.get('text', '').lower()
                print(f"[DEBUG] 检查区域 '{region_name}': '{text}'")
                if language == 'chi_sim':
                    if '发牌' in text:
                        print(f"[DONE] 检测到发牌动作")
                        return 'DEAL'
                    elif '加注' in text:
                        print(f"[DONE] 检测到加注动作")
                        return 'RAISE'
                else:  # 英文模式
                    # 检查其他动作指示
                    print(f"[DEBUG] 检查 'deal' in '{text}': {'deal' in text}")
                    print(f"[DEBUG] 检查 'raise' in '{text}': {'raise' in text}")
                    print(f"[DEBUG] 检查 'bet' in '{text}': {'bet' in text}")
                    if 'deal' in text:
                        print(f"[DONE] 检测到发牌动作")
                        return 'DEAL'
                    elif 'raise' in text or 'bet' in text:
                        print(f"[DONE] 检测到加注动作")
                        return 'RAISE'
        
        print(f"[INFO] 未检测到发牌或加注动作，跳过图片")
        return None
    
    def analyze_full_image(self, image_path):
        """分析完整图片，提取所有相关信息"""
        print(f"\n[INFO] 分析完整图片: {image_path}")
        
        # 加载图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] 无法加载图片: {image_path}")
            return None
        
        # 转换为PIL图像以供OCR使用
        from PIL import Image
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # 识别所有区域
        results = self.poker_recognizer.recognize_regions(pil_image)
        
        # 添加花色识别结果
        image_height, image_width = image.shape[:2]
        adaptive_suit_regions = self.poker_recognizer.calculate_suit_adaptive_regions(image_width, image_height)
        suit_results = self.poker_recognizer.recognize_suits_in_regions(image, adaptive_suit_regions, threshold=0.6)
        
        # 合并花色识别结果到OCR结果中
        if suit_results:
            for region_name, result in suit_results.items():
                if 'best_match' in result:
                    best = result['best_match']
                    suit_symbol = {'红桃': 'h', '黑桃': 's', '方块': 'd', '梅花': 'c'}.get(best['chinese_name'], '')
                    results[region_name] = {'text': suit_symbol, 'success': True}
        
        # 添加扑克背面识别结果（弃牌检测）
        fold_status = self.poker_recognizer.check_player_fold_by_variance(image, variance_threshold=10.0)
        if 'error' not in fold_status:
            for player, status in fold_status.items():
                if 'error' not in status:
                    # 将弃牌状态添加到结果中，用于后续处理
                    results[f"{player}弃牌状态"] = {'text': '已弃牌' if status['folded'] else '未弃牌', 'success': True}
        
        # 提取关键信息
        game_info = self.extract_game_info(results)
        
        return game_info
    
    def extract_game_info(self, ocr_results):
        """从OCR结果中提取游戏信息"""
        game_info = {
            'hand_cards': [],
            'community_cards': [],
            'game_stage': 'pre_flop',  # 翻牌前
            'my_chips': '',
            'fold_results': [],
            'call_amount': '',
            'pot': '',
        }
        
        # 提取手牌和花色
        hand1 = ocr_results.get('自己的手牌1', {}).get('text', '')
        hand2 = ocr_results.get('自己的手牌2', {}).get('text', '')
        
        # 获取花色信息
        suit1 = ocr_results.get('自己的手牌1花色', {}).get('text', '')
        suit2 = ocr_results.get('自己的手牌2花色', {}).get('text', '')
        
        # 合并手牌和花色
        if hand1:
            hand1 = f"{hand1}{suit1}"
        if hand2:
            hand2 = f"{hand2}{suit2}"
            
        if hand1 or hand2:
            game_info['hand_cards'] = [hand1, hand2]
        
        # 提取公共牌和花色并确定游戏阶段
        community_cards = []
        for i in range(1, 6):
            card = ocr_results.get(f'公共牌{i}', {}).get('text', '')
            suit = ocr_results.get(f'公共牌{i}花色', {}).get('text', '')
            if card:
                card = f"{card}{suit}"
                community_cards.append(card)
        
        game_info['community_cards'] = community_cards
        
        # 确定游戏阶段
        if len(community_cards) == 0:
            game_info['game_stage'] = 'pre_flop'  # 翻牌前
        elif len(community_cards) == 3:
            game_info['game_stage'] = 'flop'      # 翻牌后
        elif len(community_cards) == 4:
            game_info['game_stage'] = 'turn'      # 转牌
        elif len(community_cards) == 5:
            game_info['game_stage'] = 'river'     # 河牌
        
        # 提取筹码信息
        game_info['my_chips'] = ocr_results.get('自己的筹码', {}).get('text', '')
        game_info['call_amount'] = ocr_results.get('跟注', {}).get('text', '')
        game_info['pot'] = ocr_results.get('底池', {}).get('text', '')
        
        # 显示扑克背面检测结果 (反转所有弃牌状态)
        fold_results = []
        for i in range(1, 6):
            fold_status = ocr_results.get(f'玩家{i}弃牌状态', {}).get('text', '已弃牌')
            # 反转弃牌状态：未弃牌改为已弃牌，已弃牌改为未弃牌
            if fold_status == '未弃牌':
                fold_status = '已弃牌'
            elif fold_status == '已弃牌':
                fold_status = '未弃牌'
            fold_results.append(f'玩家{i}: {fold_status}')
        game_info['fold_results'] = fold_results
        
        return game_info
    
    def record_action(self, game_info, action_type):
        """记录玩家动作"""
        # 动作编码
        action_codes = {
            'CHECK_CALL': 1,
            'RAISE_HALF_POT': 2,
            'RAISE_FULL_POT': 3,
            'RAISE_CUSTOM': 4,
            'FOLD': 5
        }
        
        # 记录动作
        action_code = action_codes.get(action_type, 0)
        if action_code > 0:
            self.action_history.append(action_code)
            print(f"[INFO] 记录动作: {action_type} (编码: {action_code})")
        
        # 如果是新手牌，初始化手牌记录
        hand_cards = ''.join(game_info['hand_cards'])
        if self.current_hand != hand_cards:
            self.current_hand = hand_cards
            self.action_history = [action_code]  # 重新开始记录
            print(f"[INFO] 新手牌: {hand_cards}")
    
    def process_images(self):
        """处理所有Snipaste图片"""
        print("=== 开始处理图片 ===")
        
        # 获取所有Snipaste图片
        images = self.get_snipaste_images()
        print(f"[INFO] 找到 {len(images)} 张图片")
        
        # 依次处理每张图片
        for image_path in images:
            print(f"\n--- 处理图片: {os.path.basename(image_path)} ---")
            
            # 根据图片路径确定语言
            language = 'eng' if 'eng' in image_path else 'chi_sim'
            
            # 识别动作位置
            action_type = self.identify_action_position(image_path, language)
            
            # 如果识别到发牌动作或未识别到任何动作，直接跳过图片识别
            if action_type == 'DEAL' or action_type is None:
                print(f"[SKIP] 跳过图片 (检测到发牌动作或未检测到动作)")
                continue  # 直接处理下一张图片
            
            # 如果识别到加注动作，分析完整图片
            if action_type == 'RAISE':
                game_info = self.analyze_full_image(image_path)
                
                if game_info:
                    print(f"\n[RESULT] 识别结果:")
                    print(f"   手牌: {game_info['hand_cards']}")
                    print(f"   公共牌: {game_info['community_cards']}")
                    print(f"   游戏阶段: {game_info['game_stage']}")
                    print(f"   我的筹码: {game_info['my_chips']}")
                    print(f"   跟注金额: {game_info['call_amount']}")
                    print(f"   底池: {game_info['pot']}")
                    print(f"   扑克背面检测结果: {game_info['fold_results']}")
                    
                    # 记录动作
                    self.record_action(game_info, action_type)
                    
                    # 保存游戏信息
                    self.game_history.append({
                        'image': os.path.basename(image_path),
                        'info': game_info,
                        'actions': self.action_history.copy()
                    })
            else:
                print(f"[SKIP] 跳过图片 (未检测到加注动作)")
        
        # 保存分析结果
        self.save_analysis_results()
    
    def save_analysis_results(self):
        """保存分析结果"""
        # 创建结果目录
        output_dir = 'result'
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存游戏历史
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_file = os.path.join(output_dir, f'game_history_{timestamp}.json')
        
        # 准备保存的数据
        data = {
            'timestamp': timestamp,
            'game_history': self.game_history,
            'final_action_history': self.action_history
        }
        
        # 保存为JSON文件
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\n[SAVE] 分析结果已保存到: {history_file}")
        print(f"   总共处理 {len(self.game_history)} 个游戏状态")
        print(f"   最终动作历史: {self.action_history}")


def main():
    """主函数"""
    analyzer = PokerGameAnalyzer()
    analyzer.process_images()


if __name__ == '__main__':
    main()