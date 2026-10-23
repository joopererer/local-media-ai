//! PaddleOCR（PP-OCRv5 mobile）ONNX 三段管线：det（DB）→ cls（LCNet 方向分类）→ rec（CTC）。
//!
//! 与 RapidOCR 参考实现对齐的关键点（逐项复刻 `rapidocr` 3.10）：
//! - 全局预处理：最长边 >2000 缩小、最短边 <30 放大（均为 32 的倍数）；
//! - 竖排补白：宽高比 >8 或高 <30 时上下各补白到 `max(w/8, 30)*2`；
//! - det：min 边缩到 736（32 对齐），(x/255-0.5)/0.5，输出 p 图阈值 0.3、膨胀 2x2、
//!   连通域→框、框上 p 均值 ≥0.5、按 unclip_ratio=1.6 外扩；
//! - cls：高 48、宽 ≤192 等比缩放，(x/255-0.5)/0.5；label=180 且 score>0.9 旋转；
//! - rec：高 48、宽=int(48*max_wh_ratio)（≥320），CTC 贪心解码，blank=0，末位空格；
//! - 检测不到文字框时**直接返回空结果**，不运行 cls/rec（27.3b 要求）。

use crate::OcrError;
use ort::session::Session;
use ort::value::Tensor;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// 模型三件套 + 字符字典（rec 的 ONNX metadata `character`，'\n' 分隔）。
pub struct PaddleOcr {
    det: Mutex<Session>,
    cls: Mutex<Session>,
    rec: Mutex<Session>,
    charset: Vec<String>,
    threads: usize,
    pub model_tag: String,
}

const CLS_THRESH: f32 = 0.9;
const DET_THRESH: f32 = 0.3;
const BOX_THRESH: f32 = 0.5;
const UNCLIP_RATIO: f32 = 1.6;
const MIN_SIZE: f32 = 3.0;

fn inf<E: std::fmt::Display>(e: E) -> OcrError {
    OcrError::Failed(e.to_string())
}

fn build_session(path: &Path, threads: usize) -> Result<Session, OcrError> {
    Session::builder()
        .map_err(inf)?
        .with_intra_threads(threads.max(1))
        .map_err(inf)?
        .with_inter_threads(1)
        .map_err(inf)?
        .commit_from_file(path)
        .map_err(inf)
}

/// 一个检测框（原图坐标，轴对齐 \[x0,y0,x1,y1\]）。
#[derive(Debug, Clone, Copy)]
pub struct DetBox {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

pub struct PaddleLine {
    pub text: String,
    pub score: f32,
    pub bbox: [f32; 4],
}

/// 公开给诊断工具：detect 之前的全部前处理（边界缩放 + 竖排补白）。
pub fn prep_for_det(img: &RgbImageData) -> RgbImageData {
    let img = resize_within_bounds(img, 30.0, 2000.0);
    vertical_pad(&img, 8.0, 30).0
}

impl PaddleOcr {
    /// `model_dir` 内含 `det.onnx` / `cls.onnx` / `rec.onnx`（清单固定 sha256）。
    pub fn load(
        model_dir: &Path,
        threads: usize,
        model_tag: impl Into<String>,
    ) -> Result<Self, OcrError> {
        let det_path = model_dir.join("det.onnx");
        let cls_path = model_dir.join("cls.onnx");
        let rec_path = model_dir.join("rec.onnx");
        for p in [&det_path, &cls_path, &rec_path] {
            if !p.is_file() {
                return Err(OcrError::Unavailable(format!(
                    "OCR 模型缺失: {}",
                    p.display()
                )));
            }
        }
        let det = build_session(&det_path, threads)?;
        let cls = build_session(&cls_path, threads)?;
        let rec = build_session(&rec_path, threads)?;
        // 字符字典内嵌于 rec.onnx 的 metadata（键 `character`，按行分隔）。
        let raw: Option<String> = rec.metadata().ok().and_then(|m| m.custom("character"));
        let raw = raw
            .ok_or_else(|| OcrError::Unavailable("rec.onnx 缺少 character metadata".to_string()))?;
        // 复刻 CTCLabelDecode：blank 插入索引 0，末尾追加空格。
        let mut charset: Vec<String> = raw.split('\n').map(|s| s.to_string()).collect();
        if charset.last().map(|s| s.is_empty()).unwrap_or(false) {
            charset.pop();
        }
        charset.insert(0, "blank".to_string());
        charset.push(" ".to_string());
        Ok(Self {
            det: Mutex::new(det),
            cls: Mutex::new(cls),
            rec: Mutex::new(rec),
            charset,
            threads,
            model_tag: model_tag.into(),
        })
    }

    /// 识别 RGB 像素（行优先 h×w×3）。无文字时返回空 Vec。
    /// 返回 (lines, det_ms, cls_ms, rec_ms)，便于耗时统计。
    pub fn recognize_rgb(
        &self,
        w: u32,
        h: u32,
        rgb: &[u8],
    ) -> Result<(Vec<PaddleLine>, f64, f64, f64), OcrError> {
        if rgb.len() != w as usize * h as usize * 3 {
            return Err(OcrError::Decode("像素长度与尺寸不符".to_string()));
        }
        let mut img = RgbImageData {
            w,
            h,
            data: rgb.to_vec(),
        };

        // 1) 全局预处理（边界缩放）+ 竖排补白。
        img = resize_within_bounds(&img, 30.0, 2000.0);
        let (padded, _pad_top) = vertical_pad(&img, 8.0, 30);
        img = padded;

        // 2) det。
        let t_det = std::time::Instant::now();
        let boxes = self.detect(&img)?;
        let det_ms = t_det.elapsed().as_secs_f64() * 1000.0;
        if boxes.is_empty() {
            return Ok((Vec::new(), det_ms, 0.0, 0.0));
        }

        // 3) 裁剪 + cls + rec。
        let crops: Vec<RgbImageData> = boxes
            .iter()
            .map(|b| crop_axis_aligned(&img, b))
            .collect::<Result<_, _>>()?;
        let t_cls = std::time::Instant::now();
        let crops = self.classify_rotate(crops)?;
        let cls_ms = t_cls.elapsed().as_secs_f64() * 1000.0;

        let t_rec = std::time::Instant::now();
        let mut lines = Vec::with_capacity(crops.len());
        for (i, c) in crops.iter().enumerate() {
            let (text, score) = self.recognize_line(c)?;
            lines.push(PaddleLine {
                text,
                score,
                bbox: [boxes[i].x0, boxes[i].y0, boxes[i].x1, boxes[i].y1],
            });
        }
        let rec_ms = t_rec.elapsed().as_secs_f64() * 1000.0;
        Ok((lines, det_ms, cls_ms, rec_ms))
    }

    // ── det ────────────────────────────────────────────────────────────────

    /// 一个检测框（原图坐标，轴对齐 \[x0,y0,x1,y1\]）。调试用原始分量输出。
    pub fn detect_debug(&self, img: &RgbImageData) -> Result<Vec<(DetBox, f32)>, OcrError> {
        self.detect_components(img)
    }

    /// 检测：连通域→框（复刻 boxes_from_bitmap），返回原始框与分数。
    fn detect_components(&self, img: &RgbImageData) -> Result<Vec<(DetBox, f32)>, OcrError> {
        // resize：min 边 736，32 对齐（limit_type=min）。
        let (rw, rh) = min_side_resize(img.w, img.h, 736);
        let resized = resize_bilinear(img, rw, rh);
        // 归一化 (x/255-0.5)/0.5 → CHW。
        let mut chw = vec![0f32; 3 * rh as usize * rw as usize];
        for y in 0..rh as usize {
            for x in 0..rw as usize {
                let i = (y * rw as usize + x) * 3;
                for c in 0..3 {
                    let v = resized.data[i + c] as f32 / 255.0;
                    chw[c * (rh as usize) * (rw as usize) + y * rw as usize + x] = (v - 0.5) / 0.5;
                }
            }
        }
        let input =
            Tensor::from_array(([1usize, 3, rh as usize, rw as usize], chw)).map_err(inf)?;
        let (ph, pw, data) = {
            let mut session = self.det.lock().map_err(|_| inf("det 锁已损坏"))?;
            let outputs = session.run(ort::inputs! { "x" => input }).map_err(inf)?;
            let (shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(inf)?;
            let dims: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
            (dims[2], dims[3], data.to_vec())
        };
        {
            let (_n, _c): (usize, usize) = (0, 0);
        }
        if ph == 0 || pw == 0 || data.len() < ph * pw {
            return Err(inf(format!("det 输出形状异常: {:?}x{:?}", ph, pw)));
        }

        // 阈值化 + 2x2 膨胀。
        let mut mask: Vec<u8> = data[..ph * pw]
            .iter()
            .map(|&v| (v > DET_THRESH) as u8)
            .collect();
        let orig = mask.clone();
        for y in 0..ph {
            for x in 0..pw {
                if orig[y * pw + x] == 1 {
                    for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                        let yy = y + dy;
                        let xx = x + dx;
                        if yy < ph && xx < pw {
                            mask[yy * pw + xx] = 1;
                        }
                    }
                }
            }
        }

        // 连通域（8 邻域 BFS）→ 轴对齐框。
        let mut visited = vec![false; ph * pw];
        let mut boxes = Vec::new();
        for start in 0..ph * pw {
            if mask[start] == 0 || visited[start] {
                continue;
            }
            let mut stack = vec![start];
            visited[start] = true;
            let (mut x0, mut y0, mut x1, mut y1) = (pw, ph, 0usize, 0usize);
            // 紧致包围盒（膨胀前掩膜上求）：更贴近 cv2 轮廓的 minAreaRect。
            let (mut t0, mut t1, mut t2, mut t3) = (pw, ph, 0usize, 0usize);
            while let Some(idx) = stack.pop() {
                let y = idx / pw;
                let x = idx % pw;
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
                if orig[idx] == 1 {
                    t0 = t0.min(x);
                    t1 = t1.min(y);
                    t2 = t2.max(x);
                    t3 = t3.max(y);
                }
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let (ny, nx) = (y as i32 + dy, x as i32 + dx);
                        if ny < 0 || nx < 0 || ny >= ph as i32 || nx >= pw as i32 {
                            continue;
                        }
                        let nidx = ny as usize * pw + nx as usize;
                        if mask[nidx] == 1 && !visited[nidx] {
                            visited[nidx] = true;
                            stack.push(nidx);
                        }
                    }
                }
            }
            if t0 > t2 || t1 > t3 {
                continue; // 该连通域仅由膨胀产生（无原始像素）
            }
            let bw = (t2 - t0 + 1) as f32;
            let bh = (t3 - t1 + 1) as f32;
            if bw.min(bh) < MIN_SIZE {
                continue;
            }
            // 框上概率均值（fast score 的轴对齐近似）。
            let mut sum = 0f32;
            let mut cnt = 0usize;
            for y in t1..=t3 {
                for x in t0..=t2 {
                    sum += data[y * pw + x];
                    cnt += 1;
                }
            }
            let score = if cnt > 0 { sum / cnt as f32 } else { 0.0 };
            if score < BOX_THRESH {
                continue;
            }
            // unclip：按 Paddle 公式外扩（面积/周长 * ratio）。
            let area = bw * bh;
            let peri = 2.0 * (bw + bh);
            let dist = area * UNCLIP_RATIO / peri;
            let (ux0, uy0, ux1, uy1) = (
                t0 as f32 - dist,
                t1 as f32 - dist,
                t2 as f32 + dist,
                t3 as f32 + dist,
            );
            if (ux1 - ux0).min(uy1 - uy0) < MIN_SIZE + 2.0 {
                continue;
            }
            // 映射回原图坐标（除以 det 缩放，原图已含补白，故补白 offset 为 0）。
            let sx = img.w as f32 / pw as f32;
            let sy = img.h as f32 / ph as f32;
            boxes.push((
                DetBox {
                    x0: (ux0 * sx).clamp(0.0, img.w as f32),
                    y0: (uy0 * sy).clamp(0.0, img.h as f32),
                    x1: (ux1 * sx).clamp(0.0, img.w as f32),
                    y1: (uy1 * sy).clamp(0.0, img.h as f32),
                },
                score,
            ));
        }
        Ok(boxes)
    }

    /// 检测 + 行内排序（按 y 阈值 10 归并成行，行内按 x）。
    fn detect(&self, img: &RgbImageData) -> Result<Vec<DetBox>, OcrError> {
        let mut boxes: Vec<DetBox> = self
            .detect_components(img)?
            .into_iter()
            .map(|(b, _)| b)
            .collect();
        boxes.sort_by(|a, b| a.y0.total_cmp(&b.y0));
        let mut rows: Vec<Vec<DetBox>> = Vec::new();
        for b in boxes {
            match rows.last_mut() {
                Some(row) if (b.y0 - row[0].y0).abs() < 10.0 => row.push(b),
                _ => rows.push(vec![b]),
            }
        }
        let mut ordered = Vec::new();
        for mut row in rows {
            row.sort_by(|a, b| a.x0.total_cmp(&b.x0));
            ordered.extend(row);
        }
        Ok(ordered)
    }

    // ── cls ────────────────────────────────────────────────────────────────

    fn classify_rotate(&self, crops: Vec<RgbImageData>) -> Result<Vec<RgbImageData>, OcrError> {
        let mut out = Vec::with_capacity(crops.len());
        for mut c in crops {
            let (rw, rh) = (192usize, 48usize);
            let ratio = c.w as f32 / c.h as f32;
            let resized_w = ((rh as f32 * ratio).ceil() as usize).min(rw);
            let small = resize_bilinear(&c, resized_w as u32, rh as u32);
            let mut chw = vec![0f32; 3 * rh * rw];
            for y in 0..rh {
                for x in 0..small.w as usize {
                    let i = (y * small.w as usize + x) * 3;
                    for ch in 0..3 {
                        let v = small.data[i + ch] as f32 / 255.0;
                        chw[ch * rh * rw + y * rw + x] = (v - 0.5) / 0.5;
                    }
                }
            }
            let probs = {
                let input = Tensor::from_array(([1usize, 3, rh, rw], chw)).map_err(inf)?;
                let mut session = self.cls.lock().map_err(|_| inf("cls 锁已损坏"))?;
                let outputs = session.run(ort::inputs! { "x" => input }).map_err(inf)?;
                let (_shape, probs) = outputs[0].try_extract_tensor::<f32>().map_err(inf)?;
                probs.to_vec()
            };
            if probs.len() >= 2 {
                // label_list = ["0", "180"]；index 1 = 180°。
                let (p0, p180) = (probs[0], probs[1]);
                if std::env::var("FO_OCR_DEBUG_CLS").is_ok() {
                    eprintln!("[cls] p0={p0:.3} p180={p180:.3} crop={}x{}", c.w, c.h);
                }
                if p180 > p0 && p180 > CLS_THRESH {
                    c = rotate_180(&c);
                }
            }
            out.push(c);
        }
        Ok(out)
    }

    // ── rec ────────────────────────────────────────────────────────────────

    fn recognize_line(&self, c: &RgbImageData) -> Result<(String, f32), OcrError> {
        let rh = 48usize;
        let max_wh_ratio = (320.0f32 / 48.0).max(c.w as f32 / c.h as f32);
        let rw = (rh as f32 * max_wh_ratio) as usize;
        let ratio = c.w as f32 / c.h as f32;
        let resized_w = ((rh as f32 * ratio).ceil() as usize).min(rw).max(1);
        let small = resize_bilinear(c, resized_w as u32, rh as u32);
        let mut chw = vec![0f32; 3 * rh * rw];
        for y in 0..rh {
            for x in 0..small.w as usize {
                let i = (y * small.w as usize + x) * 3;
                for ch in 0..3 {
                    let v = small.data[i + ch] as f32 / 255.0;
                    chw[ch * rh * rw + y * rw + x] = (v - 0.5) / 0.5;
                }
            }
        }
        let (dims, probs) = {
            let input = Tensor::from_array(([1usize, 3, rh, rw], chw)).map_err(inf)?;
            let mut session = self.rec.lock().map_err(|_| inf("rec 锁已损坏"))?;
            let outputs = session.run(ort::inputs! { "x" => input }).map_err(inf)?;
            let (shape, probs) = outputs[0].try_extract_tensor::<f32>().map_err(inf)?;
            (
                shape.iter().map(|d| *d as usize).collect::<Vec<_>>(),
                probs.to_vec(),
            )
        };
        let (_b, steps, classes) = if dims.len() == 3 {
            (dims[0], dims[1], dims[2])
        } else {
            return Err(inf(format!("rec 输出形状异常: {dims:?}")));
        };
        // CTC 贪心：argmax + 去重（blank=0）+ 概率均值。
        let mut last = usize::MAX;
        let mut out = String::new();
        let mut confs = Vec::new();
        for t in 0..steps {
            let row = &probs[t * classes..(t + 1) * classes];
            let mut best = 0usize;
            let mut best_p = row[0];
            for (i, &p) in row.iter().enumerate() {
                if p > best_p {
                    best_p = p;
                    best = i;
                }
            }
            if best != 0 && best != last && best < self.charset.len() {
                out.push_str(&self.charset[best]);
                confs.push(best_p);
            }
            last = best;
        }
        let score = if confs.is_empty() {
            0.0
        } else {
            confs.iter().sum::<f32>() / confs.len() as f32
        };
        Ok((out, score))
    }

    /// 逐词框/行框坐标使用的线程数（诊断用）。
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// 模型目录约定（供上层构造路径）。
    pub fn model_paths(model_dir: &Path) -> (PathBuf, PathBuf, PathBuf) {
        (
            model_dir.join("det.onnx"),
            model_dir.join("cls.onnx"),
            model_dir.join("rec.onnx"),
        )
    }
}

// ── 图像工具（RgbImageData + 缩放/补白/裁剪/旋转） ─────────────────────────

#[derive(Clone)]
pub struct RgbImageData {
    pub w: u32,
    pub h: u32,
    pub data: Vec<u8>,
}

/// 最长边限 max、最短边保 min（均为 32 对齐），复刻 resize_image_within_bounds。
fn resize_within_bounds(img: &RgbImageData, min_side: f32, max_side: f32) -> RgbImageData {
    let mut out = img.clone();
    let maxv = out.w.max(out.h) as f32;
    if maxv > max_side {
        let ratio = max_side / maxv;
        let (rw, rh) = (align32(out.w as f32 * ratio), align32(out.h as f32 * ratio));
        out = resize_bilinear(&out, rw.max(32), rh.max(32));
    }
    let minv = out.w.min(out.h) as f32;
    if minv < min_side {
        let ratio = min_side / minv;
        let (rw, rh) = (
            align32(out.w as f32 * ratio).max(32),
            align32(out.h as f32 * ratio).max(32),
        );
        out = resize_bilinear(&out, rw, rh);
    }
    out
}

/// 竖排补白：高 ≤ min_height 或 宽/高 > ratio 时，上下各补白。
fn vertical_pad(img: &RgbImageData, ratio: f32, min_height: u32) -> (RgbImageData, u32) {
    let use_limit_ratio = img.w as f32 / img.h as f32 > ratio;
    if img.h <= min_height || use_limit_ratio {
        let new_h = ((img.w as f32 / ratio) as u32).max(min_height) * 2;
        let pad = ((new_h as i64 - img.h as i64).unsigned_abs() / 2) as u32;
        let out_h = img.h + pad * 2;
        let mut data = vec![0u8; out_h as usize * img.w as usize * 3];
        let row = img.w as usize * 3;
        data[pad as usize * row..(pad as usize + img.h as usize) * row].copy_from_slice(&img.data);
        return (
            RgbImageData {
                w: img.w,
                h: out_h,
                data,
            },
            pad,
        );
    }
    (img.clone(), 0)
}

fn align32(v: f32) -> u32 {
    ((v / 32.0).round() as u32).max(1) * 32
}

fn min_side_resize(w: u32, h: u32, limit: u32) -> (u32, u32) {
    let (mut rw, mut rh) = (w, h);
    if w.min(h) < limit {
        let ratio = limit as f32 / w.min(h) as f32;
        rw = (w as f32 * ratio) as u32;
        rh = (h as f32 * ratio) as u32;
    }
    (align32(rw as f32).max(32), align32(rh as f32).max(32))
}

fn crop_axis_aligned(img: &RgbImageData, b: &DetBox) -> Result<RgbImageData, OcrError> {
    let x0 = b.x0.floor().max(0.0) as u32;
    let y0 = b.y0.floor().max(0.0) as u32;
    let x1 = (b.x1.ceil() as u32).min(img.w);
    let y1 = (b.y1.ceil() as u32).min(img.h);
    if x1 <= x0 + 1 || y1 <= y0 + 1 {
        return Err(OcrError::Failed("裁剪框过小".to_string()));
    }
    let (cw, ch) = (x1 - x0, y1 - y0);
    let mut data = Vec::with_capacity(cw as usize * ch as usize * 3);
    for y in y0..y1 {
        let row = (y * img.w + x0) as usize * 3;
        data.extend_from_slice(&img.data[row..row + cw as usize * 3]);
    }
    let mut out = RgbImageData { w: cw, h: ch, data };
    // 复刻 get_rotate_crop_image：高/宽 ≥ 1.5 时逆时针 90°（竖排转横读）。
    if out.h as f32 / out.w as f32 >= 1.5 {
        out = rotate_90_ccw(&out);
    }
    Ok(out)
}

fn rotate_90_ccw(img: &RgbImageData) -> RgbImageData {
    let (w, h) = (img.h, img.w); // 新尺寸
    let mut data = vec![0u8; img.data.len()];
    for y in 0..img.h {
        for x in 0..img.w {
            let src = ((y * img.w + x) * 3) as usize;
            let nx = y;
            let ny = img.w - 1 - x;
            let dst = ((ny * w + nx) * 3) as usize;
            data[dst..dst + 3].copy_from_slice(&img.data[src..src + 3]);
        }
    }
    RgbImageData { w, h, data }
}

fn rotate_180(img: &RgbImageData) -> RgbImageData {
    let mut data = img.data.clone();
    let row = img.w as usize * 3;
    for y in 0..img.h as usize / 2 {
        let a = y * row;
        let b = (img.h as usize - 1 - y) * row;
        for i in 0..row / 3 {
            let ia = a + i * 3;
            let ib = b + (row / 3 - 1 - i) * 3;
            for c in 0..3 {
                data.swap(ia + c, ib + c);
            }
        }
    }
    if img.h % 2 == 1 {
        let m = (img.h as usize / 2) * row;
        for i in 0..row / 6 {
            let ia = m + i * 3;
            let ib = m + (row / 3 - 1 - i) * 3;
            for c in 0..3 {
                data.swap(ia + c, ib + c);
            }
        }
    }
    RgbImageData {
        w: img.w,
        h: img.h,
        data,
    }
}

/// 双线性缩放（cv2.INTER_LINEAR 近似）。
fn resize_bilinear(img: &RgbImageData, nw: u32, nh: u32) -> RgbImageData {
    if nw == img.w && nh == img.h {
        return img.clone();
    }
    let mut data = vec![0u8; nw as usize * nh as usize * 3];
    let x_ratio = img.w as f32 / nw as f32;
    let y_ratio = img.h as f32 / nh as f32;
    for y in 0..nh {
        let fy = (y as f32 + 0.5) * y_ratio - 0.5;
        let y0 = fy.floor().max(0.0) as u32;
        let y1 = (y0 + 1).min(img.h - 1);
        let wy = (fy - y0 as f32).clamp(0.0, 1.0);
        for x in 0..nw {
            let fx = (x as f32 + 0.5) * x_ratio - 0.5;
            let x0 = fx.floor().max(0.0) as u32;
            let x1 = (x0 + 1).min(img.w - 1);
            let wx = (fx - x0 as f32).clamp(0.0, 1.0);
            let i00 = ((y0 * img.w + x0) * 3) as usize;
            let i01 = ((y0 * img.w + x1) * 3) as usize;
            let i10 = ((y1 * img.w + x0) * 3) as usize;
            let i11 = ((y1 * img.w + x1) * 3) as usize;
            let dst = ((y * nw + x) * 3) as usize;
            for c in 0..3 {
                let v00 = img.data[i00 + c] as f32;
                let v01 = img.data[i01 + c] as f32;
                let v10 = img.data[i10 + c] as f32;
                let v11 = img.data[i11 + c] as f32;
                let top = v00 + (v01 - v00) * wx;
                let bottom = v10 + (v11 - v10) * wx;
                data[dst + c] = (top + (bottom - top) * wy).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    RgbImageData { w: nw, h: nh, data }
}
