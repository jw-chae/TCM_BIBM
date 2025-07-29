#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TCM Case JSON Processing Module
Features:
1. Parse TCM case data from txt files
2. Generate zhenduan.json and gold_standard.json files
3. Provide JSON data saving and loading functionality

"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple


class TCMJSONProcessor:
    """TCM Case JSON Processor"""
    
    def __init__(self):
        """Initialize processor"""
        pass
    
    def parse_case_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Parse case data from text
        
        Args:
            text: Text content containing multiple cases
            
        Returns:
            List[Dict]: Parsed case list, each case contains various fields
        """
        cases = []
        
        # Split cases using delimiter
        case_blocks = text.split('--------------------------------------------------')
        
        for i, block in enumerate(case_blocks):
            if not block.strip():
                continue
                
            case_data = {}
            
            # Extract various fields
            date_match = re.search(r'日期:\s*([^\n]+)', block)
            if date_match:
                case_data['date'] = date_match.group(1).strip()
            
            # Tongue diagnosis
            shezhen_match = re.search(r'舌诊:\s*([^\n]+)', block)
            if shezhen_match:
                case_data['shezhen'] = shezhen_match.group(1).strip()
            
            # Pulse diagnosis
            maizhen_match = re.search(r'脉诊:\s*([^\n]+)', block)
            if maizhen_match:
                case_data['maizhen'] = maizhen_match.group(1).strip()
            
            # Chief complaint
            zhusu_match = re.search(r'主诉:\s*([^\n]+)', block)
            if zhusu_match:
                case_data['zhusu'] = zhusu_match.group(1).strip()
            
            # Present illness
            xianbingshi_match = re.search(r'现病史:\s*([^\n]+(?:\n(?!诊断:|辩证:|处方:)[^\n]*)*)', block)
            if xianbingshi_match:
                case_data['xianbingshi'] = xianbingshi_match.group(1).strip()
            
            # Diagnosis
            zhenduan_match = re.search(r'诊断:\s*([^\n]+)', block)
            if zhenduan_match:
                case_data['zhenduan'] = zhenduan_match.group(1).strip()
            
            # Syndrome differentiation
            bianzheng_match = re.search(r'辩证:\s*([^\n]+)', block)
            if bianzheng_match:
                case_data['bianzheng'] = bianzheng_match.group(1).strip()
            
            # Prescription
            chufang_match = re.search(r'处方:\s*([^\n]+(?:\n(?!--)[^\n]*)*)', block)
            if chufang_match:
                case_data['chufang'] = chufang_match.group(1).strip()
            
            # Only add cases that contain necessary fields
            if any(key in case_data for key in ['shezhen', 'maizhen', 'zhusu', 'xianbingshi']):
                cases.append(case_data)
        
        return cases
    
    def process_txt_files(self, data_dir: str) -> Tuple[List[Dict], List[Dict]]:
        """Process all txt files and extract case data
        
        Args:
            data_dir: Directory path containing txt files
            
        Returns:
            Tuple[List[Dict], List[Dict]]: (Diagnosis case list, Gold standard case list)
        """
        zhenduan_cases = []
        gold_standard_cases = []
        
        data_path = Path(data_dir)
        txt_files = list(data_path.glob('*.txt'))
        
        print(f"找到 {len(txt_files)} 个txt文件")
        
        for txt_file in txt_files:
            if txt_file.name.startswith('正文-') or txt_file.name.startswith('文前-'):
                continue  # 跳过这些文档文件
                
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                cases = self.parse_case_from_text(content)
                
                if not cases:
                    continue
                    
                # 根据案例数量选择
                if len(cases) <= 5:
                    selected_cases = cases[:1]  # 选择1个案例
                else:
                    selected_cases = cases[:2]  # 选择2个案例
                
                for idx, case in enumerate(selected_cases):
                    case_id = f"{txt_file.stem}_{idx+1}"
                    
                    # 创建诊断数据
                    if all(key in case for key in ['shezhen', 'maizhen', 'zhusu', 'xianbingshi']):
                        zhenduan_case = {
                            "CaseID": case_id,
                            "shezhen": case.get('shezhen', ''),
                            "maizhen": case.get('maizhen', ''),
                            "zhusu": case.get('zhusu', ''),
                            "xianbingshi": case.get('xianbingshi', '')
                        }
                        zhenduan_cases.append(zhenduan_case)
                    
                    # 创建金标准数据
                    if all(key in case for key in ['zhenduan', 'bianzheng', 'chufang']):
                        gold_case = {
                            "CaseID": case_id,
                            "zhenduan": case.get('zhenduan', ''),
                            "bianzheng": case.get('bianzheng', ''),
                            "chufang": case.get('chufang', '')
                        }
                        gold_standard_cases.append(gold_case)
                        
            except Exception as e:
                print(f"处理文件 {txt_file.name} 时出错: {e}")
                continue
        
        return zhenduan_cases, gold_standard_cases
    
    def save_json_file(self, data: List[Dict], filename: str, description: str = "") -> bool:
        """保存数据到JSON文件
        
        Args:
            data: 要保存的数据列表
            filename: 保存的文件名
            description: 文件描述（用于日志输出）
            
        Returns:
            bool: 保存是否成功
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            if description:
                print(f"已保存 {description}：{filename} ({len(data)} 条记录)")
            else:
                print(f"已保存文件：{filename} ({len(data)} 条记录)")
            
            return True
            
        except Exception as e:
            print(f"保存文件 {filename} 时出错: {e}")
            return False
    
    def load_json_file(self, filename: str) -> List[Dict]:
        """从JSON文件加载数据
        
        Args:
            filename: JSON文件路径
            
        Returns:
            List[Dict]: 加载的数据列表，如果失败返回空列表
        """
        try:
            if not os.path.exists(filename):
                print(f"文件 {filename} 不存在")
                return []
                
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"已加载文件：{filename} ({len(data)} 条记录)")
            return data
            
        except Exception as e:
            print(f"加载文件 {filename} 时出错: {e}")
            return []
    
    def generate_case_files(self, data_dir: str, output_dir: str = ".") -> Tuple[bool, Dict[str, int]]:
        """生成诊断和金标准JSON文件
        
        Args:
            data_dir: 包含txt文件的数据目录
            output_dir: 输出文件的目录，默认为当前目录
            
        Returns:
            Tuple[bool, Dict]: (是否成功, 统计信息字典)
        """
        print("开始生成案例JSON文件...")
        
        # 处理txt文件
        zhenduan_data, gold_standard_data = self.process_txt_files(data_dir)
        
        if not zhenduan_data and not gold_standard_data:
            print("未提取到任何有效案例数据")
            return False, {}
        
        # 构造输出文件路径
        zhenduan_file = os.path.join(output_dir, 'zhenduan.json')
        gold_standard_file = os.path.join(output_dir, 'gold_standard.json')
        
        # 保存文件
        zhenduan_success = self.save_json_file(zhenduan_data, zhenduan_file, "诊断案例数据")
        gold_success = self.save_json_file(gold_standard_data, gold_standard_file, "金标准案例数据")
        
        # 统计信息
        stats = {
            'zhenduan_count': len(zhenduan_data),
            'gold_standard_count': len(gold_standard_data),
            'zhenduan_success': zhenduan_success,
            'gold_success': gold_success
        }
        
        success = zhenduan_success and gold_success
        
        if success:
            print(f"\n案例文件生成完成:")
            print(f"- 诊断案例: {stats['zhenduan_count']} 条")
            print(f"- 金标准案例: {stats['gold_standard_count']} 条")
        else:
            print("部分文件生成失败，请检查错误信息")
            
        return success, stats
    
    def validate_case_data(self, cases: List[Dict], required_fields: List[str]) -> Dict[str, int]:
        """验证案例数据的完整性
        
        Args:
            cases: 案例数据列表
            required_fields: 必需字段列表
            
        Returns:
            Dict[str, int]: 验证统计信息
        """
        total_cases = len(cases)
        complete_cases = 0
        field_stats = {field: 0 for field in required_fields}
        
        for case in cases:
            # 统计每个字段的完整性
            case_complete = True
            for field in required_fields:
                if field in case and case[field] and case[field].strip():
                    field_stats[field] += 1
                else:
                    case_complete = False
            
            if case_complete:
                complete_cases += 1
        
        return {
            'total_cases': total_cases,
            'complete_cases': complete_cases,
            'completeness_rate': complete_cases / total_cases * 100 if total_cases > 0 else 0,
            'field_stats': field_stats
        }
    
    def print_validation_report(self, zhenduan_cases: List[Dict], gold_cases: List[Dict]):
        """打印数据验证报告
        
        Args:
            zhenduan_cases: 诊断案例数据
            gold_cases: 金标准案例数据
        """
        print("\n" + "="*50)
        print("数据验证报告")
        print("="*50)
        
        # 验证诊断案例数据
        zhenduan_stats = self.validate_case_data(zhenduan_cases, ['shezhen', 'maizhen', 'zhusu', 'xianbingshi'])
        print(f"\n诊断案例数据:")
        print(f"总案例数: {zhenduan_stats['total_cases']}")
        print(f"完整案例数: {zhenduan_stats['complete_cases']}")
        print(f"完整率: {zhenduan_stats['completeness_rate']:.1f}%")
        print("字段完整性:")
        for field, count in zhenduan_stats['field_stats'].items():
            rate = count / zhenduan_stats['total_cases'] * 100 if zhenduan_stats['total_cases'] > 0 else 0
            print(f"  {field}: {count}/{zhenduan_stats['total_cases']} ({rate:.1f}%)")
        
        # 验证金标准案例数据
        gold_stats = self.validate_case_data(gold_cases, ['zhenduan', 'bianzheng', 'chufang'])
        print(f"\n金标准案例数据:")
        print(f"总案例数: {gold_stats['total_cases']}")
        print(f"完整案例数: {gold_stats['complete_cases']}")
        print(f"完整率: {gold_stats['completeness_rate']:.1f}%")
        print("字段完整性:")
        for field, count in gold_stats['field_stats'].items():
            rate = count / gold_stats['total_cases'] * 100 if gold_stats['total_cases'] > 0 else 0
            print(f"  {field}: {count}/{gold_stats['total_cases']} ({rate:.1f}%)")


def main():
    """主函数 - 演示如何使用TCMJSONProcessor"""
    processor = TCMJSONProcessor()
    
    # 数据目录路径
    data_dir = "./data"
    
    if not os.path.exists(data_dir):
        print(f"数据目录 {data_dir} 不存在，请确认路径是否正确")
        return
    
    # 生成案例JSON文件
    success, stats = processor.generate_case_files(data_dir)
    
    if success:
        # 加载生成的文件进行验证
        zhenduan_data = processor.load_json_file('zhenduan.json')
        gold_data = processor.load_json_file('gold_standard.json')
        
        # 打印验证报告
        if zhenduan_data or gold_data:
            processor.print_validation_report(zhenduan_data, gold_data)
    else:
        print("文件生成失败")


if __name__ == "__main__":
    main()
