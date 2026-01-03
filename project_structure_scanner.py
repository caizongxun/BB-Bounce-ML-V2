#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
專案結構掃描工具
功能: 遞迴掃描本地專案所有檔案，生成完整的檔案架構報告
用法: python project_structure_scanner.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json


class ProjectStructureScanner:
    """專案結構掃描器"""
    
    def __init__(self, root_path=None, output_file='project_structure_report.txt'):
        """
        初始化掃描器
        
        Args:
            root_path: 專案根目錄路徑（預設為當前目錄）
            output_file: 輸出檔案名稱
        """
        self.root_path = Path(root_path or os.getcwd())
        self.output_file = output_file
        
        # 需要忽略的目錄
        self.ignore_dirs = {
            '__pycache__',
            '.git',
            '.idea',
            '.pytest_cache',
            '.venv',
            'venv',
            '.egg-info',
            'dist',
            'build',
            '__MACOSX',
            '.DS_Store',
            'node_modules',
            '.vscode',
        }
        
        # 需要忽略的檔案
        self.ignore_files = {
            '.DS_Store',
            'Thumbs.db',
            '*.pyc',
        }
        
        # 統計信息
        self.stats = {
            'total_files': 0,
            'total_dirs': 0,
            'file_types': {},
            'total_size': 0,
        }
        
        self.output_lines = []
    
    def should_ignore(self, path):
        """
        判斷是否應該忽略此路徑
        """
        name = path.name
        
        # 檢查目錄
        if path.is_dir() and name in self.ignore_dirs:
            return True
        
        # 檢查檔案
        if path.is_file():
            if name in self.ignore_files:
                return True
            # 檢查隱藏檔案（以點開頭）
            if name.startswith('.'):
                return True
        
        return False
    
    def get_file_size(self, file_path):
        """
        獲取檔案大小
        """
        try:
            size = file_path.stat().st_size
            return size
        except:
            return 0
    
    def format_size(self, bytes_size):
        """
        格式化檔案大小為可讀格式
        """
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024
        return f"{bytes_size:.1f} TB"
    
    def get_file_extension(self, file_path):
        """
        獲取檔案副檔名
        """
        return file_path.suffix or '(no extension)'
    
    def scan_directory(self, directory, prefix='', is_last=True):
        """
        遞迴掃描目錄
        """
        try:
            items = sorted(directory.iterdir())
        except PermissionError:
            return
        
        # 過濾忽略項目
        items = [item for item in items if not self.should_ignore(item)]
        
        # 分離目錄和檔案
        dirs = [item for item in items if item.is_dir()]
        files = [item for item in items if item.is_file()]
        
        # 先顯示檔案
        for i, file_path in enumerate(files):
            is_last_file = (i == len(files) - 1) and len(dirs) == 0
            connector = '└── ' if is_last_file else '├── '
            
            file_size = self.get_file_size(file_path)
            self.stats['total_size'] += file_size
            
            # 統計檔案類型
            ext = self.get_file_extension(file_path)
            self.stats['file_types'][ext] = self.stats['file_types'].get(ext, 0) + 1
            self.stats['total_files'] += 1
            
            size_str = self.format_size(file_size)
            line = f"{prefix}{connector}{file_path.name} ({size_str})"
            self.output_lines.append(line)
        
        # 顯示目錄
        for i, dir_path in enumerate(dirs):
            is_last_dir = (i == len(dirs) - 1)
            connector = '└── ' if is_last_dir else '├── '
            
            line = f"{prefix}{connector}{dir_path.name}/"
            self.output_lines.append(line)
            
            self.stats['total_dirs'] += 1
            
            # 遞迴掃描子目錄
            if is_last_dir:
                new_prefix = prefix + '    '
            else:
                new_prefix = prefix + '│   '
            
            self.scan_directory(dir_path, new_prefix, is_last_dir)
    
    def generate_report(self):
        """
        生成完整報告
        """
        # 標題
        self.output_lines.insert(0, '')
        self.output_lines.insert(0, '='*80)
        self.output_lines.insert(0, 'BB Bounce ML V2 - 專案結構報告')
        self.output_lines.insert(0, '='*80)
        
        # 掃描信息
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.output_lines.insert(4, f'掃描時間: {timestamp}')
        self.output_lines.insert(5, f'專案根目錄: {self.root_path}')
        self.output_lines.insert(6, '')
        
        # 樹狀結構
        self.output_lines.insert(7, f'{self.root_path.name}/')
        
        # 掃描
        start_index = len(self.output_lines)
        self.scan_directory(self.root_path)
        
        # 統計信息
        self.output_lines.append('')
        self.output_lines.append('='*80)
        self.output_lines.append('統計信息')
        self.output_lines.append('='*80)
        self.output_lines.append(f'總檔案數: {self.stats["total_files"]}')
        self.output_lines.append(f'總目錄數: {self.stats["total_dirs"]}')
        self.output_lines.append(f'總大小: {self.format_size(self.stats["total_size"])}')
        self.output_lines.append('')
        
        # 檔案類型統計
        self.output_lines.append('檔案類型分布:')
        self.output_lines.append('-'*80)
        for ext in sorted(self.stats['file_types'].keys()):
            count = self.stats['file_types'][ext]
            self.output_lines.append(f'  {ext}: {count} 個檔案')
        
        self.output_lines.append('')
        self.output_lines.append('='*80)
    
    def save_to_file(self):
        """
        保存報告到檔案
        """
        try:
            output_path = self.root_path / self.output_file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.output_lines))
            print(f'報告已保存到: {output_path}')
            return output_path
        except Exception as e:
            print(f'保存檔案時出錯: {e}')
            return None
    
    def print_to_console(self):
        """
        在控制台打印報告
        """
        for line in self.output_lines:
            print(line)
    
    def save_json_report(self):
        """
        保存為JSON格式報告
        """
        try:
            json_file = self.root_path / 'project_structure_report.json'
            
            report_data = {
                'scan_time': datetime.now().isoformat(),
                'project_root': str(self.root_path),
                'statistics': {
                    'total_files': self.stats['total_files'],
                    'total_dirs': self.stats['total_dirs'],
                    'total_size_bytes': self.stats['total_size'],
                    'total_size_formatted': self.format_size(self.stats['total_size']),
                },
                'file_types': self.stats['file_types'],
                'directory_tree': '\n'.join(self.output_lines),
            }
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            print(f'JSON報告已保存到: {json_file}')
            return json_file
        except Exception as e:
            print(f'保存JSON檔案時出錯: {e}')
            return None
    
    def run(self, save_txt=True, save_json=True, print_console=True):
        """
        執行完整的掃描流程
        
        Args:
            save_txt: 是否保存為TXT檔案
            save_json: 是否保存為JSON檔案
            print_console: 是否在控制台打印
        """
        print('開始掃描專案結構...')
        print(f'掃描目錄: {self.root_path}')
        print()
        
        # 生成報告
        self.generate_report()
        
        # 打印到控制台
        if print_console:
            self.print_to_console()
        
        # 保存為TXT
        if save_txt:
            self.save_to_file()
        
        # 保存為JSON
        if save_json:
            self.save_json_report()
        
        print()
        print('掃描完成！')
        print(f'總計: {self.stats["total_files"]} 個檔案, {self.stats["total_dirs"]} 個目錄')


def main():
    """
    主函數
    """
    # 獲取專案根目錄（BB-Bounce-ML-V2所在目錄）
    project_root = Path.cwd()
    
    # 驗證是否在正確的目錄
    if not (project_root / 'realtime_service.py').exists():
        print('警告: 未檢測到BB-Bounce-ML-V2的核心檔案')
        print('請確保在專案根目錄執行此腳本')
        print(f'當前目錄: {project_root}')
        print()
    
    # 創建掃描器
    scanner = ProjectStructureScanner(root_path=project_root)
    
    # 執行掃描（同時生成TXT和JSON報告，並在控制台打印）
    scanner.run(save_txt=True, save_json=True, print_console=True)


if __name__ == '__main__':
    main()
