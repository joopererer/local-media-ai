"""确定性 OCR 测试图片生成器（27.3 引擎对比用）。

渲染中英文截图/海报/表格，含难点：小字号、竖排、复杂背景、形近字。
真值以 exact text（逐行）写入同名 .json；图片完全由代码生成，零版权风险。

依赖：Pillow（pip install pillow）。字体：优先系统字体
  - 中文：msyh.ttc（微软雅黑）/ simhei.ttf
  - 英文：arial.ttf / DejaVuSans.ttf
找不到则列出可用字体目录并退出（不静默降级为缺字形）。

用法：python eval/tools/gen_ocr_fixtures.py --out eval/public/ocr [--check-fonts]
"""

import argparse
import io
import json
import os
import random
import sys

sys.stdout.reconfigure(encoding="utf-8")

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("need Pillow: pip install pillow")
    sys.exit(3)

W, H = 1280, 800

ZH_BODY = [
    "语义检索评测用例",
    "电话：138-0000-1234  邮箱：test@example.com",
    "会议纪要：周三下午三点在三楼会议室",
    "合计金额：人民币壹万贰仟叁佰元整",
    "己已巳已辨认清楚 0O0oIl1 形近字测试",
]
EN_BODY = [
    "Meeting notes: Wednesday 3pm, Room 301",
    "Total: USD 12,300.00  invoice INV-2026-1016",
    "The quick brown fox jumps over 13 lazy dogs",
    "Error E_CONN_TIMEOUT after 3 retries",
    "Il1i O0o0 lI|! confusables test",
]
ZH_VERTICAL = "竖排文字测试从右向左排列"
TABLE = [
    ("项目", "数量", "单价", "金额"),
    ("苹果", "10", "5.50", "55.00"),
    ("香蕉", "6", "3.20", "19.20"),
    ("橙子", "8", "4.00", "32.00"),
]
POSTER_ZH = ["社区文化节", "时间：10月18日 上午9点", "地点：中心广场", "免费入场 欢迎参加"]
POSTER_EN = ["Autumn Book Fair", "Oct 18, 9:00 AM", "Central Square", "Free entry"]


def find_font(candidates):
    dirs = [r"C:\Windows\Fonts", "/usr/share/fonts", "/System/Library/Fonts"]
    for d in dirs:
        for c in candidates:
            p = os.path.join(d, c)
            if os.path.exists(p):
                return p
    return None


def cx(font_path, size):
    return ImageFont.truetype(font_path, size)


def noisy_bg(seed, w=W, h=H):
    rnd = random.Random(seed)
    img = Image.new("RGB", (w, h), (242, 240, 235))
    d = ImageDraw.Draw(img)
    for _ in range(60):
        x, y = rnd.randrange(w), rnd.randrange(h)
        r = rnd.randrange(8, 60)
        shade = rnd.randrange(215, 245)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(shade, shade - 5, shade - 12))
    return img


def screenshot_zh(font):
    """中英文混排截图：标题 + 正文 + 底部小字（小字号难点）。"""
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 90], fill=(30, 60, 120))
    d.text((40, 18), "产品发布会 · 产品发布会", font=cx(font, 44), fill="white")
    lines = ZH_BODY + EN_BODY[:2]
    y = 130
    truth = ["产品发布会 · 产品发布会"]
    for ln in lines:
        d.text((40, y), ln, font=cx(font, 34), fill="black")
        truth.append(ln)
        y += 56
    small = "版本 v2.6.1 · 版权所有 © 2026 示例公司 · 保留所有权利"
    d.text((40, H - 60), small, font=cx(font, 20), fill=(90, 90, 90))
    truth.append(small)
    return img, truth


def poster_en(font):
    """英文海报：大字标题 + 三行信息，色块背景。"""
    img = Image.new("RGB", (W, H), (20, 30, 60))
    d = ImageDraw.Draw(img)
    truth = []
    y = 90
    for ln in POSTER_EN[:1]:
        d.text((80, y), ln, font=cx(font, 96), fill=(255, 210, 60))
        truth.append(ln)
        y += 140
    for ln in POSTER_EN[1:]:
        d.text((80, y), ln, font=cx(font, 52), fill="white")
        truth.append(ln)
        y += 90
    return img, truth


def poster_zh(font):
    img = Image.new("RGB", (W, H), (150, 20, 20))
    d = ImageDraw.Draw(img)
    truth = []
    y = 90
    for ln in POSTER_ZH[:1]:
        d.text((80, y), ln, font=cx(font, 96), fill=(255, 240, 200))
        truth.append(ln)
        y += 150
    for ln in POSTER_ZH[1:]:
        d.text((80, y), ln, font=cx(font, 52), fill="white")
        truth.append(ln)
        y += 95
    return img, truth


def table_img(font):
    img = Image.new("RGB", (W, 560), "white")
    d = ImageDraw.Draw(img)
    truth = []
    cw, rh, x0, y0 = 260, 64, 60, 60
    for r, row in enumerate(TABLE):
        for c, cell in enumerate(row):
            x, y = x0 + c * cw, y0 + r * rh
            d.rectangle([x, y, x + cw, y + rh], outline="black", width=2)
            d.text((x + 16, y + 10), cell, font=cx(font, 36), fill="black")
        truth.append(" ".join(row))
    return img, truth


def vertical_img(font):
    """竖排中文：从右向左列排（阅读顺序为列序）。"""
    img = Image.new("RGB", (W, H), (250, 248, 240))
    d = ImageDraw.Draw(img)
    f = cx(font, 56)
    cols = [ZH_VERTICAL[i:i + 8] for i in range(0, len(ZH_VERTICAL), 8)]
    truth = []
    x = W - 140
    for col in cols:
        y = 80
        for ch in col:
            d.text((x, y), ch, font=f, fill="black")
            y += 66
        truth.append(col)
        x -= 90
    return img, truth


def noisy_img(font):
    """复杂背景 + 小字号 + 形近字。"""
    img = noisy_bg(20261016)
    d = ImageDraw.Draw(img)
    truth = ZH_BODY[4:5] + EN_BODY[4:5] + ["服务热线 400-820-0000"]
    y = 120
    for ln in truth:
        d.text((60, y), ln, font=cx(font, 30), fill=(40, 40, 40))
        y += 70
    return img, truth


def mixed_poster(font):
    """中英文混排海报（大标题中英 + 信息行）。"""
    img = Image.new("RGB", (W, H), (25, 60, 40))
    d = ImageDraw.Draw(img)
    truth = ["社区文化节 Community Festival", "时间 Time：10月18日 Oct 18 上午9点 9AM",
             "地点 Place：中心广场 Central Square", "免费入场 Free entry"]
    y = 100
    d.text((70, y), truth[0], font=cx(font, 72), fill=(255, 235, 150))
    y += 130
    for ln in truth[1:]:
        d.text((70, y), ln, font=cx(font, 48), fill="white")
        y += 95
    return img, truth


def body_en(font):
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    truth = EN_BODY + ZH_BODY[0:1]
    y = 100
    for ln in truth:
        d.text((60, y), ln, font=cx(font, 36), fill="black")
        y += 62
    return img, truth


CASES = [
    ("ocr-01-zh-screenshot.png", screenshot_zh, "zh-en", "screenshot"),
    ("ocr-02-en-poster.png", poster_en, "en", "poster"),
    ("ocr-03-zh-poster.png", poster_zh, "zh", "poster"),
    ("ocr-04-table.png", table_img, "zh", "table"),
    ("ocr-05-zh-vertical.png", vertical_img, "zh", "vertical"),
    ("ocr-06-noisy-confusables.png", noisy_img, "zh-en", "noisy"),
    ("ocr-07-mixed-poster.png", mixed_poster, "zh-en", "poster"),
    ("ocr-08-en-body.png", body_en, "en", "document"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/public/ocr")
    ap.add_argument("--check-fonts", action="store_true")
    args = ap.parse_args()

    zh = find_font(["msyh.ttc", "simhei.ttf", "NotoSansCJK-Regular.ttc"])
    en = find_font(["arial.ttf", "DejaVuSans.ttf"]) or zh
    print("zh font:", zh)
    print("en font:", en)
    if args.check_fonts or not zh:
        if not zh:
            print("ERROR: no CJK font found; refusing to render tofu")
            sys.exit(2)
        return
    font = zh

    os.makedirs(args.out, exist_ok=True)
    manifest = []
    for fname, fn, lang, kind in CASES:
        img, truth = fn(font)
        path = os.path.join(args.out, fname)
        img.save(path)
        manifest.append({"file": fname, "lang": lang, "kind": kind,
                         "truth_lines": truth})
        with io.open(path + ".json", "w", encoding="utf-8") as f:
            f.write(json.dumps({"file": fname, "truth_lines": truth},
                               ensure_ascii=False, indent=1) + "\n")
        print(f"  {fname} lines={len(truth)}")
    with io.open(os.path.join(args.out, "manifest.jsonl"), "w",
                 encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"total={len(manifest)} -> {args.out}")


if __name__ == "__main__":
    main()
