#!/usr/bin/env python
"""Markdown → PDF(中文 + 图片 + 表格)。
pandoc 转 HTML 片段 → 套 Noto CJK 字体 CSS → weasyprint 渲染。
用法: python scripts/md_to_pdf.py <input.md> [output.pdf]
"""
import os
import sys
import subprocess
from weasyprint import HTML

src = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0] + ".pdf"
base = os.path.dirname(os.path.abspath(src))

# 1) pandoc: GFM markdown → HTML 片段(保留表格)
body = subprocess.run(
    ["pandoc", "-f", "gfm", "-t", "html5", src],
    capture_output=True, text=True, check=True,
).stdout

CSS = """
@page { size: A4; margin: 1.8cm 1.6cm; }
* { font-family: "Noto Sans CJK SC", "DejaVu Sans", sans-serif; }
body { font-size: 11pt; line-height: 1.55; color: #1a1a1a; }
h1 { font-size: 19pt; border-bottom: 2px solid #1a73e8; padding-bottom: 4px; }
h2 { font-size: 15pt; color: #1a73e8; margin-top: 18px; border-bottom: 1px solid #ddd; padding-bottom: 2px; }
h3 { font-size: 12.5pt; color: #333; margin-top: 14px; }
code { background: #f4f4f4; padding: 1px 4px; border-radius: 3px; font-size: 9.5pt; }
blockquote { color: #666; border-left: 3px solid #ccc; margin: 8px 0; padding: 2px 10px; font-size: 10pt; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9.5pt; }
th, td { border: 1px solid #bbb; padding: 4px 8px; text-align: left; }
th { background: #eef2fb; }
img { max-width: 100%; display: block; margin: 6px auto; }
hr { border: none; border-top: 1px solid #ddd; margin: 14px 0; }
strong { color: #c0392b; }
"""

html = f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"

# 2) weasyprint: base_url 指向 md 所在目录,使 figs/ 相对图片可解析
HTML(string=html, base_url=base + "/").write_pdf(out)
print("wrote", out)
