#!/usr/bin/env python3
"""
测试报告自动生成模块

生成 HTML、PDF、Markdown 格式的测试报告
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import jinja2
from loguru import logger


class ReportGenerator:
    """测试报告生成器"""
    
    def __init__(self, results: Dict, output_dir: str = "tests/benchmarks/reports"):
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置 Jinja2 模板
        self.template_dir = Path(__file__).parent / "templates"
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.template_dir)),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
    def generate_html_report(self, template_name: str = "report_template.html") -> str:
        """生成 HTML 报告"""
        try:
            template = self.jinja_env.get_template(template_name)
        except jinja2.TemplateNotFound:
            # 使用内置模板
            html_content = self._generate_builtin_html()
        else:
            html_content = template.render(**self.results)
            
        output_path = self.output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        logger.info(f"HTML 报告已生成：{output_path}")
        return str(output_path)
        
    def generate_markdown_report(self) -> str:
        """生成 Markdown 报告"""
        md_content = self._generate_markdown()
        
        output_path = self.output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
            
        logger.info(f"Markdown 报告已生成：{output_path}")
        return str(output_path)
        
    def generate_json_report(self) -> str:
        """生成 JSON 报告"""
        output_path = self.output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
            
        logger.info(f"JSON 报告已生成：{output_path}")
        return str(output_path)
        
    def _generate_builtin_html(self) -> str:
        """生成内置 HTML 报告"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 安全获取指标值
        def safe_get_metric(metrics, key, default='N/A'):
            val = metrics.get(key, default)
            if isinstance(val, float):
                return f"{val:.4f}"
            return val
            
        tests = self.results.get('tests', {})
        rec_metrics = tests.get('recognition', {}).get('metrics', {})
        
        auc_val = safe_get_metric(rec_metrics, 'auc')
        eer_val = safe_get_metric(rec_metrics, 'eer')
        fnmr_1e4 = safe_get_metric(rec_metrics, 'FNMR@FMR=1e-4')
        total_time = f"{self.results.get('total_time', 0):.1f}s"
        
        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>人脸识别模型测试报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; }}
        .metric-label {{ font-size: 0.9em; opacity: 0.9; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #4CAF50; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .pass {{ color: #4CAF50; font-weight: bold; }}
        .fail {{ color: #f44336; font-weight: bold; }}
        .warning {{ color: #ff9800; font-weight: bold; }}
        .timestamp {{ color: #999; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 人脸识别模型测试报告</h1>
        <p class="timestamp">生成时间：{timestamp}</p>
        
        <h2>📊 测试结果总览</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{auc_val}</div>
                <div class="metric-label">验证 AUC</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{eer_val}</div>
                <div class="metric-label">等错误率 (EER)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{fnmr_1e4}</div>
                <div class="metric-label">FNMR@FMR=10⁻⁴</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{total_time}</div>
                <div class="metric-label">总测试时间</div>
            </div>
        </div>

        <h2>📋 详细测试结果</h2>
"""
        
        # 添加识别测试结果
        if 'tests' in self.results and 'recognition' in self.results['tests']:
            rec_test = self.results['tests']['recognition']
            html += f"""
        <h3>人脸识别测试 ({rec_test.get('dataset', 'N/A')})</h3>
        <table>
            <tr><th>指标</th><th>值</th></tr>
"""
            metrics = rec_test.get('metrics', {})
            for metric, value in metrics.items():
                if metric not in ['roc', 'det']:  # 跳过复杂数据
                    if isinstance(value, float):
                        html += f"            <tr><td>{metric}</td><td>{value:.4f}</td></tr>\n"
                    else:
                        html += f"            <tr><td>{metric}</td><td>{value}</td></tr>\n"
            html += """        </table>
"""
        
        html += """
        <h2>🔧 环境信息</h2>
        <table>
            <tr><th>项目</th><th>值</th></tr>
"""
        
        if 'config' in self.results:
            env = self.results['config'].get('environment', {})
            for key, value in env.items():
                html += f"            <tr><td>{key}</td><td>{value}</td></tr>\n"
                
        html += f"""            <tr><td>测试时间</td><td>{timestamp}</td></tr>
        </table>
        
        <h2>📈 性能分析</h2>
        <p>详细分析请参考图表附件。</p>
        
        <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; color: #999; text-align: center;">
            <p>人脸识别端到端自动化测试框架 | Generated by Test Benchmark System</p>
        </footer>
    </div>
</body>
</html>
"""
        return html
        
    def _generate_markdown(self) -> str:
        """生成 Markdown 报告"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        md = f"""# 🎯 人脸识别模型测试报告

**生成时间**: {timestamp}

---

## 📊 测试结果总览

| 指标 | 值 |
|-----|-----|
"""
        
        # 添加主要指标
        if 'tests' in self.results and 'recognition' in self.results['tests']:
            metrics = self.results['tests']['recognition'].get('metrics', {})
            md += f"| 验证 AUC | {metrics.get('auc', 'N/A'):.4f if isinstance(metrics.get('auc'), float) else 'N/A'} |\n"
            md += f"| 等错误率 (EER) | {metrics.get('eer', 'N/A'):.4f if isinstance(metrics.get('eer'), float) else 'N/A'} |\n"
            md += f"| FNMR@FMR=10⁻⁴ | {metrics.get('FNMR@FMR=1e-4', 'N/A'):.4f if isinstance(metrics.get('FNMR@FMR=1e-4'), float) else 'N/A'} |\n"
            md += f"| FNMR@FMR=10⁻⁶ | {metrics.get('FNMR@FMR=1e-6', 'N/A'):.4f if isinstance(metrics.get('FNMR@FMR=1e-6'), float) else 'N/A'} |\n"
            
        md += f"| 总测试时间 | {self.results.get('total_time', 0):.1f} 秒 |\n"
        
        md += """
---

## 📋 详细测试结果

### 人脸识别测试

"""
        
        if 'tests' in self.results and 'recognition' in self.results['tests']:
            metrics = self.results['tests']['recognition'].get('metrics', {})
            for metric, value in metrics.items():
                if metric not in ['roc', 'det', 'full_fmr', 'full_fnmr']:
                    if isinstance(value, float):
                        md += f"- **{metric}**: {value:.4f}\n"
                    else:
                        md += f"- **{metric}**: {value}\n"
                        
        md += """
---

## 🔧 环境信息

"""
        
        if 'config' in self.results:
            env = self.results['config'].get('environment', {})
            for key, value in env.items():
                md += f"- **{key}**: {value}\n"
                
        md += f"\n- **测试时间**: {timestamp}\n"
        
        md += """
---

## 📈 性能分析

### 优势
- 待分析...

### 需要改进
- 待分析...

### 优化建议
1. 待补充...

---

*报告由人脸识别端到端自动化测试框架自动生成*
"""
        
        return md


def main():
    """测试报告生成"""
    import argparse
    
    parser = argparse.ArgumentParser(description="生成测试报告")
    parser.add_argument("--results", type=str, required=True, help="测试结果 JSON 文件路径")
    parser.add_argument("--format", type=str, choices=["html", "markdown", "json", "all"], default="all",
                        help="报告格式")
    parser.add_argument("--output", type=str, default="tests/benchmarks/reports",
                        help="输出目录")
    
    args = parser.parse_args()
    
    # 加载结果
    with open(args.results, 'r', encoding='utf-8') as f:
        results = json.load(f)
        
    # 生成报告
    generator = ReportGenerator(results, args.output)
    
    if args.format == "html":
        generator.generate_html_report()
    elif args.format == "markdown":
        generator.generate_markdown_report()
    elif args.format == "json":
        generator.generate_json_report()
    elif args.format == "all":
        generator.generate_html_report()
        generator.generate_markdown_report()
        generator.generate_json_report()
        
    print("报告生成完成!")


if __name__ == "__main__":
    main()
