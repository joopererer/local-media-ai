//! EmbeddingGemma 2 图片预处理（Phase 25.0.4b）：复现 transformers `Gemma4ImageProcessor`
//! （torchvision 后端，sentence-transformers 默认使用的实现）。
//!
//! 依据（transformers 源码，venv 中的版本）：
//! - `models/gemma4/image_processing_gemma4.py`：`get_aspect_ratio_preserving_size`（目标尺寸）、
//!   `aspect_ratio_preserving_resize`（`tvF.resize`，BICUBIC，`antialias=True`）、`convert_image_to_patches`、
//!   位置 id（meshgrid `indexing="xy"`，即 (x, y)）、`pad_along_first_dim`（补丁补 0、位置补 -1）。
//! - torchvision 对 uint8 图像的双三次抗锯齿缩放走 ATen 的 int16 定点路径：先水平后垂直，
//!   中间结果取整为 uint8，权重精度取“最大权重 × 2^(p+1) < 32768”的最大 p（上限 22），
//!   舍入偏置 `1 << (p-1)`。该路径在 Python 原型中对 3 张图片复现出 0 误差（见 docs/spikes/onnx_rust_image_prep.md）。
//! - 输入 id：`<bos> <|image> <|image|>×n <image|> <eos>`，n = 实际补丁数 / 9；
//!   对 75 张图片的 Python 输入逐一核对过该模式。

use crate::EmbedError;
use image::metadata::Orientation;
use image::{DynamicImage, ImageDecoder, ImageReader, RgbImage};
use image::Limits;
use std::path::Path;
use std::io::Read;

pub const PATCH_SIZE: u32 = 16;
pub const POOLING_KERNEL: u32 = 3;
/// 每个补丁的特征数：16 × 16 × 3。
pub const PATCH_DIM: usize = (PATCH_SIZE * PATCH_SIZE * 3) as usize;
pub const SUPPORTED_SOFT_TOKENS: [u32; 5] = [70, 140, 280, 560, 1120];

pub const BOS_ID: i64 = 2;
pub const EOS_ID: i64 = 1;
pub const BOI_ID: i64 = 255_999; // <|image>
pub const IMAGE_TOKEN_ID: i64 = 258_880; // <|image|>
pub const EOI_ID: i64 = 258_882; // <image|>

#[derive(Debug, Clone)]
pub struct ImageInputs {
    /// (max_patches, 768)，行优先；补丁内按 (行, 列, 通道) 排列，值域 [0, 1]。
    pub pixel_values: Vec<f32>,
    /// (max_patches, 2)，每行 (x, y)；补齐部分为 -1。
    pub position_ids: Vec<i64>,
    pub input_ids: Vec<i64>,
    pub soft_tokens: usize,
    pub max_patches: usize,
    /// 缩放后的 (高, 宽)。
    pub resized: (u32, u32),
}

/// 后端边界：按预算缩放后的 RGB 图（27.2b 架构：缩放/切块/位置是共享组件）。
///
/// 27.2a 决定性实验证明：图片向量的运行库间残差全部来自预处理。
/// 因此所有后端（ORT、llama.cpp、未来 GPU）都从这个类型开始计算：
/// 后端只做"本类型 → 视觉特征 → 句子向量"，缩放/切块/position/input_ids
/// 由本模块的纯函数产出，不随后端分叉。
#[derive(Debug, Clone)]
pub struct SharedPixels {
    /// 按预算缩放后的 RGB 像素，行优先，(h, w, 3)，uint8。
    pub rgb: Vec<u8>,
    /// 行优先尺寸 (高, 宽)，均为 48 的倍数（见 [`aspect_ratio_preserving_size`]）。
    pub size_hw: (u32, u32),
    /// soft token 上限（决定预算；实际值见 [`Self::patchify`] 结果）。
    pub max_soft_tokens: u32,
}

impl SharedPixels {
    /// 切块 → (ImageInputs)。唯一的切块实现：所有后端的
    /// "RGB 像素 → 视觉特征"输入都由此产出（含 ONNX 的 pixel_values /
    /// position_ids / input_ids；给 llama.cpp 的路径见 [`SharedPixels::to_png`]）。
    pub fn patchify(&self) -> Result<ImageInputs, EmbedError> {
        patchify_pixels(&self.rgb, self.size_hw.0, self.size_hw.1, self.max_soft_tokens)
    }
    /// 兼容旧名（历史上返回 Result）。
    pub fn patchify_result(&self) -> Result<ImageInputs, EmbedError> {
        self.patchify()
    }

    /// 导出为 PNG（供走"先缩放 PNG 再解码"路径的后端恒等测试，如 27.2a 的 llama.cpp 对照）。
    pub fn to_png(&self) -> Result<Vec<u8>, EmbedError> {
        let (th, tw) = self.size_hw;
        let img = image::RgbImage::from_raw(tw, th, self.rgb.clone())
            .ok_or_else(|| bad("SharedPixels 尺寸与像素长度不一致"))?;
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png)
            .map_err(|e| bad(format!("PNG 编码失败: {e}")))?;
        Ok(buf.into_inner())
    }
}

fn bad(msg: impl Into<String>) -> EmbedError {
    EmbedError::InvalidImage(msg.into())
}

/// 对应 Python `get_aspect_ratio_preserving_size`；返回 (高, 宽)。
pub fn aspect_ratio_preserving_size(
    height: u32,
    width: u32,
    max_patches: u32,
) -> Result<(u32, u32), EmbedError> {
    if height == 0 || width == 0 {
        return Err(bad("图片尺寸为 0"));
    }
    let side_mult = (POOLING_KERNEL * PATCH_SIZE) as f64;
    let total_px = height as f64 * width as f64;
    let target_px = max_patches as f64 * (PATCH_SIZE * PATCH_SIZE) as f64;
    let factor = (target_px / total_px).sqrt();
    let ideal_h = factor * height as f64;
    let ideal_w = factor * width as f64;

    let mut th = (ideal_h / side_mult).floor() as u32 * side_mult as u32;
    let mut tw = (ideal_w / side_mult).floor() as u32 * side_mult as u32;
    if th == 0 && tw == 0 {
        return Err(bad("缩放后尺寸为 0 x 0"));
    }
    let max_side = (max_patches / (POOLING_KERNEL * POOLING_KERNEL)) * side_mult as u32;
    if th == 0 {
        th = side_mult as u32;
        tw = (((width as f64 / height as f64).floor() as u32) * side_mult as u32).min(max_side);
    } else if tw == 0 {
        tw = side_mult as u32;
        th = (((height as f64 / width as f64).floor() as u32) * side_mult as u32).min(max_side);
    }
    if th as f64 * tw as f64 > target_px {
        return Err(bad(format!("缩放后 {th}x{tw} 超出补丁预算")));
    }
    Ok((th, tw))
}

fn cubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

struct AxisWeights {
    precision: u32,
    /// 每个输出位置：(起始输入下标, 定点权重)。
    taps: Vec<(usize, Vec<i32>)>,
}

fn axis_weights(inn: usize, out: usize) -> AxisWeights {
    let scale = inn as f64 / out as f64;
    let (support, inv) = if scale >= 1.0 {
        (2.0 * scale, 1.0 / scale)
    } else {
        (2.0, 1.0)
    };
    let mut float_taps: Vec<(usize, Vec<f64>)> = Vec::with_capacity(out);
    let mut max_w = 0.0f64;
    for i in 0..out {
        let center = scale * (i as f64 + 0.5);
        let xmin = ((center - support + 0.5) as i64).max(0) as usize;
        let xmax = (((center + support + 0.5) as i64).max(0) as usize).min(inn);
        let mut w: Vec<f64> = (xmin..xmax)
            .map(|j| cubic((j as f64 - center + 0.5) * inv))
            .collect();
        let sum: f64 = w.iter().sum();
        for v in &mut w {
            *v /= sum;
            max_w = max_w.max(v.abs());
        }
        float_taps.push((xmin, w));
    }
    let mut precision = 0u32;
    while precision < 22 && max_w * ((1u64 << (precision + 1)) as f64) < 32768.0 {
        precision += 1;
    }
    let one = (1u64 << precision) as f64;
    let taps = float_taps
        .into_iter()
        .map(|(xmin, w)| {
            let q = w
                .iter()
                .map(|&v| (v * one + if v < 0.0 { -0.5 } else { 0.5 }) as i32)
                .collect();
            (xmin, q)
        })
        .collect();
    AxisWeights { precision, taps }
}

/// 双三次抗锯齿缩放（RGB，uint8），先水平后垂直，中间结果取整为 uint8。
fn resize_bicubic_aa(src: &[u8], w: usize, h: usize, tw: usize, th: usize) -> Vec<u8> {
    let wx = axis_weights(w, tw);
    let wy = axis_weights(h, th);

    let mut mid = vec![0u8; h * tw * 3];
    let bias = 1i32 << (wx.precision - 1);
    for y in 0..h {
        let row = &src[y * w * 3..(y + 1) * w * 3];
        for (x, (xmin, ws)) in wx.taps.iter().enumerate() {
            let mut acc = [bias; 3];
            for (k, &wt) in ws.iter().enumerate() {
                let p = (xmin + k) * 3;
                acc[0] += wt * row[p] as i32;
                acc[1] += wt * row[p + 1] as i32;
                acc[2] += wt * row[p + 2] as i32;
            }
            for c in 0..3 {
                mid[(y * tw + x) * 3 + c] = (acc[c] >> wx.precision).clamp(0, 255) as u8;
            }
        }
    }

    let mut out = vec![0u8; th * tw * 3];
    let bias = 1i32 << (wy.precision - 1);
    let mut acc = vec![0i32; tw * 3];
    for (y, (ymin, ws)) in wy.taps.iter().enumerate() {
        acc.iter_mut().for_each(|a| *a = bias);
        for (k, &wt) in ws.iter().enumerate() {
            let row = &mid[(ymin + k) * tw * 3..(ymin + k + 1) * tw * 3];
            for (a, &v) in acc.iter_mut().zip(row) {
                *a += wt * v as i32;
            }
        }
        for (i, a) in acc.iter().enumerate() {
            out[y * tw * 3 + i] = (a >> wy.precision).clamp(0, 255) as u8;
        }
    }
    out
}

/// 完整预处理：缩放 → 缩放到 [0,1] → 切补丁 → 位置 id → 补齐 → input_ids。
///
/// 27.2b：本体已拆为 [`shared_pixels_from_image`]（缩放，后端边界）+
/// [`patchify_pixels`]（切块/位置/token，纯函数）。保留本函数为两者的组合，
/// 语义与拆分前逐字节相同（由 `shared_pixels_patchify_matches_prepare_image` 测试与
/// 27.1A 一致性工具保证）。
pub fn prepare_image(img: &RgbImage, max_soft_tokens: u32) -> Result<ImageInputs, EmbedError> {
    shared_pixels_from_image(img, max_soft_tokens)?.patchify()
}

/// 切块：已缩放的 RGB 像素 → (pixel_values / position_ids / input_ids)。
/// 纯函数（不读盘、不缩放），是所有后端共用的唯一切块实现。
pub fn patchify_pixels(
    pixels: &[u8],
    th: u32,
    tw: u32,
    max_soft_tokens: u32,
) -> Result<ImageInputs, EmbedError> {
    if !SUPPORTED_SOFT_TOKENS.contains(&max_soft_tokens) {
        return Err(bad(format!(
            "max_soft_tokens 必须是 {SUPPORTED_SOFT_TOKENS:?} 之一"
        )));
    }
    let max_patches = (max_soft_tokens * POOLING_KERNEL * POOLING_KERNEL) as usize;
    if th == 0 || tw == 0 || !th.is_multiple_of(PATCH_SIZE) || !tw.is_multiple_of(PATCH_SIZE) {
        return Err(bad(format!("切块尺寸非法: {th}x{tw}")));
    }
    if pixels.len() != (th as usize) * (tw as usize) * 3 {
        return Err(bad(format!(
            "像素长度与尺寸不一致: {} vs {th}x{tw}",
            pixels.len()
        )));
    }

    let ph = (th / PATCH_SIZE) as usize;
    let pw = (tw / PATCH_SIZE) as usize;
    let real = ph * pw;
    let ps = PATCH_SIZE as usize;
    let scale = 1.0f32 / 255.0;

    let mut pixel_values = vec![0f32; max_patches * PATCH_DIM];
    let mut position_ids = vec![-1i64; max_patches * 2];
    for r in 0..ph {
        for c in 0..pw {
            let idx = r * pw + c;
            let dst = &mut pixel_values[idx * PATCH_DIM..(idx + 1) * PATCH_DIM];
            for py in 0..ps {
                let src_row = ((r * ps + py) * tw as usize + c * ps) * 3;
                let d = py * ps * 3;
                for (o, &v) in dst[d..d + ps * 3]
                    .iter_mut()
                    .zip(&pixels[src_row..src_row + ps * 3])
                {
                    *o = v as f32 * scale;
                }
            }
            position_ids[idx * 2] = c as i64;
            position_ids[idx * 2 + 1] = r as i64;
        }
    }

    let soft_tokens = real / (POOLING_KERNEL * POOLING_KERNEL) as usize;
    let mut input_ids = Vec::with_capacity(soft_tokens + 4);
    input_ids.push(BOS_ID);
    input_ids.push(BOI_ID);
    input_ids.extend(std::iter::repeat_n(IMAGE_TOKEN_ID, soft_tokens));
    input_ids.push(EOI_ID);
    input_ids.push(EOS_ID);

    Ok(ImageInputs {
        pixel_values,
        position_ids,
        input_ids,
        soft_tokens,
        max_patches,
        resized: (th, tw),
    })
}

/// 识别支持的图片格式（JPEG, PNG, GIF, BMP, TIFF, WebP）的魔数
fn is_supported_image_magic(head: &[u8]) -> bool {
    if head.starts_with(b"\xFF\xD8\xFF") { return true; } // JPEG
    if head.starts_with(b"\x89PNG\r\n\x1a\n") { return true; } // PNG
    if head.starts_with(b"GIF87a") || head.starts_with(b"GIF89a") { return true; } // GIF
    if head.starts_with(b"BM") { return true; } // BMP
    if head.starts_with(b"II\x2A\x00") || head.starts_with(b"MM\x00\x2A") { return true; } // TIFF
    if head.len() >= 12 && head.starts_with(b"RIFF") && &head[8..12] == b"WEBP" { return true; } // WEBP
    false
}

/// 沿 JPEG 段结构查找主图像的 EOI 标记。解码器（zune-jpeg）对截断文件会静默补灰，
/// 而 Python（PIL）会报错，所以这里单独检查，避免产生静默错误的向量。
fn jpeg_is_complete(b: &[u8]) -> bool {
    if b.len() < 4 || b[0] != 0xFF || b[1] != 0xD8 {
        return false;
    }
    let mut i = 2;
    while i + 1 < b.len() {
        if b[i] != 0xFF {
            return false;
        }
        let m = b[i + 1];
        match m {
            0xFF => i += 1,
            0xD9 => return true,
            0x01 | 0xD0..=0xD8 => i += 2,
            _ => {
                if i + 3 >= b.len() {
                    return false;
                }
                let len = u16::from_be_bytes([b[i + 2], b[i + 3]]) as usize;
                i += 2 + len;
                if m == 0xDA {
                    // 熵编码数据：跳过 FF00 与 RSTn，直到下一个真正的标记
                    while i + 1 < b.len() {
                        if b[i] == 0xFF && b[i + 1] != 0x00 && !(0xD0..=0xD7).contains(&b[i + 1]) {
                            break;
                        }
                        i += 1;
                    }
                }
            }
        }
    }
    false
}

/// 读取并解码图片，规则对齐 transformers `image_utils.load_image(path)`：
/// `PIL.Image.open` → `ImageOps.exif_transpose`（按 EXIF 方向自动旋转/翻转）→ `convert("RGB")`。
/// `convert("RGB")` 对 RGBA/LA 直接丢弃 alpha（不合成底色），对调色板按调色板取色，灰度复制三通道；
/// 与此一致，这里用 `to_rgb8`。GIF 只取第一帧。无法识别或损坏的格式返回 `InvalidImage`。
pub fn load_rgb_image(path: &Path, skip_sniff: bool) -> Result<RgbImage, EmbedError> {
    let name = path.display();
    let mut head = [0u8; 32];
    let n = std::fs::File::open(path)
        .map_err(|e| bad(format!("{name}: 无法打开: {e}")))?
        .read(&mut head)
        .unwrap_or(0);

    if !skip_sniff && !is_supported_image_magic(&head[..n]) {
        return Err(EmbedError::UnsupportedImage(format!(
            "{name}: 文件头未匹配到支持的图片格式 (JPEG/PNG/WebP/GIF/BMP/TIFF)"
        )));
    }

    let mut reader = ImageReader::open(path)
        .map_err(|e| bad(format!("{name}: 无法打开: {e}")))?
        .with_guessed_format()
        .map_err(|e| bad(format!("{name}: 无法读取: {e}")))?;

    if reader.format().is_none() {
        return Err(EmbedError::UnsupportedImage(format!("{name}: 未能识别为支持的图片格式")));
    }

    let mut limits = Limits::default();
    limits.max_image_width = Some(16384);
    limits.max_image_height = Some(16384);
    limits.max_alloc = Some(512 * 1024 * 1024); // 512MB
    reader.limits(limits);
    if reader.format() == Some(image::ImageFormat::Jpeg) {
        let bytes = std::fs::read(path).map_err(|e| bad(format!("{name}: 无法读取: {e}")))?;
        if !jpeg_is_complete(&bytes) {
            return Err(bad(format!(
                "{name}: JPEG 文件不完整（缺少结束标记 EOI，疑似被截断）"
            )));
        }
    }
    let mut decoder = reader
        .into_decoder()
        .map_err(|e| match e {
            image::ImageError::Unsupported(_) => {
                EmbedError::UnsupportedImage(format!("{name}: 不支持的图片格式: {e}"))
            }
            image::ImageError::Limits(_) => {
                EmbedError::ImageTooLarge(format!("{name}: 图片过大或分配超限: {e}"))
            }
            _ => bad(format!("{name}: 文件损坏或无法解码: {e}")),
        })?;
    let orientation = decoder.orientation().unwrap_or(Orientation::NoTransforms);
    let mut img =
        DynamicImage::from_decoder(decoder).map_err(|e| match e {
            image::ImageError::Limits(_) => EmbedError::ImageTooLarge(format!("{name}: 解码时内存分配超限: {e}")),
            _ => bad(format!("{name}: 解码失败: {e}")),
        })?;
    img.apply_orientation(orientation);
    Ok(img.to_rgb8())
}

/// 读取图片文件并预处理。
pub fn prepare_image_file(path: &Path, max_soft_tokens: u32, skip_sniff: bool) -> Result<ImageInputs, EmbedError> {
    shared_pixels_file(path, max_soft_tokens, skip_sniff)?.patchify_result()
}

/// 缩放为共享像素（后端边界）。唯一的缩放实现：所有后端的图片输入都由此产出。
pub fn shared_pixels_from_image(
    img: &RgbImage,
    max_soft_tokens: u32,
) -> Result<SharedPixels, EmbedError> {
    if !SUPPORTED_SOFT_TOKENS.contains(&max_soft_tokens) {
        return Err(bad(format!(
            "max_soft_tokens 必须是 {SUPPORTED_SOFT_TOKENS:?} 之一"
        )));
    }
    let max_patches = (max_soft_tokens * POOLING_KERNEL * POOLING_KERNEL) as usize;
    let (h, w) = (img.height(), img.width());
    let (th, tw) = aspect_ratio_preserving_size(h, w, max_patches as u32)?;
    let rgb: Vec<u8> = if th == h && tw == w {
        img.as_raw().clone()
    } else {
        resize_bicubic_aa(img.as_raw(), w as usize, h as usize, tw as usize, th as usize)
    };
    Ok(SharedPixels {
        rgb,
        size_hw: (th, tw),
        max_soft_tokens,
    })
}

/// 读取图片文件并缩放为共享像素（含 EXIF 方向与格式嗅探，见 [`load_rgb_image`]）。
pub fn shared_pixels_file(
    path: &Path,
    max_soft_tokens: u32,
    skip_sniff: bool,
) -> Result<SharedPixels, EmbedError> {
    shared_pixels_from_image(&load_rgb_image(path, skip_sniff)?, max_soft_tokens)
}
#[cfg(test)]
mod tests {
    use super::*;

    // 期望值取自 Python get_aspect_ratio_preserving_size（max_patches=1260）的实测输出（见 onnx_rust_image_prep.md）。
    #[test]
    fn jpeg_trailing_data_and_embedded_thumbnail() {
        let mut ok = vec![
            0xFF, 0xD8, 0xFF, 0xDA, 0x00, 0x02, 0x12, 0xFF, 0x00, 0x34, 0xFF, 0xD9,
        ];
        // EOI 之后附带任意数据（如动态照片的视频）仍应视为完整
        ok.extend_from_slice(b"\x00\x00\x00\x18ftypmp42 trailing FF D9 FF D9");
        assert!(jpeg_is_complete(&ok));
        // APP1 内嵌缩略图自带 EOI，主图被截断时不能被它骗过
        let trunc = [
            0xFF, 0xD8, 0xFF, 0xE1, 0x00, 0x08, 0xFF, 0xD8, 0xFF, 0xD9, 0x00, 0xFF, 0xDA, 0x00,
            0x02, 0x12, 0x34, 0x56,
        ];
        assert!(!jpeg_is_complete(&trunc));
    }



    #[test]
    fn jpeg_completeness_check() {
        let ok = [
            0xFF, 0xD8, 0xFF, 0xDA, 0x00, 0x02, 0x12, 0xFF, 0x00, 0x34, 0xFF, 0xD9,
        ];
        assert!(jpeg_is_complete(&ok));
        assert!(!jpeg_is_complete(&ok[..ok.len() - 3]));
        assert!(!jpeg_is_complete(b"not a jpeg"));
    }

    #[test]
    fn target_size_matches_python_samples() {
        assert_eq!(
            aspect_ratio_preserving_size(768, 1024, 1260).unwrap(),
            (480, 624)
        );
        assert_eq!(
            aspect_ratio_preserving_size(1152, 768, 1260).unwrap(),
            (672, 432)
        );
        assert_eq!(
            aspect_ratio_preserving_size(2306, 1538, 1260).unwrap(),
            (672, 432)
        );
        assert_eq!(
            aspect_ratio_preserving_size(2000, 3000, 1260).unwrap(),
            (432, 672)
        );
    }

    #[test]
    fn extreme_aspect_ratio_uses_edge_branch() {
        let (h, w) = aspect_ratio_preserving_size(20, 4000, 1260).unwrap();
        assert_eq!(h, 48);
        assert!(w % 48 == 0 && h * w <= 1260 * 256);
        assert!(aspect_ratio_preserving_size(0, 10, 1260).is_err());
    }

    #[test]
    fn identity_size_skips_resize_and_builds_layout() {
        // 624x480 恰为目标尺寸：不缩放，直接切补丁。
        let mut img = RgbImage::new(624, 480);
        for (x, y, p) in img.enumerate_pixels_mut() {
            p.0 = [(x % 251) as u8, (y % 241) as u8, ((x + y) % 239) as u8];
        }
        let out = prepare_image(&img, 140).unwrap();
        assert_eq!(out.resized, (480, 624));
        assert_eq!(out.pixel_values.len(), 1260 * PATCH_DIM);
        assert_eq!(out.soft_tokens, 130);
        assert_eq!(out.input_ids.len(), 134);
        assert_eq!(&out.input_ids[..3], &[BOS_ID, BOI_ID, IMAGE_TOKEN_ID]);
        assert_eq!(&out.input_ids[out.input_ids.len() - 2..], &[EOI_ID, EOS_ID]);
        // 位置：x 为列，先走完一行；补齐为 -1
        assert_eq!(&out.position_ids[..6], &[0, 0, 1, 0, 2, 0]);
        assert_eq!(&out.position_ids[39 * 2..39 * 2 + 2], &[0, 1]);
        assert_eq!(&out.position_ids[1170 * 2..1170 * 2 + 2], &[-1, -1]);
        assert!(out.pixel_values[1170 * PATCH_DIM..]
            .iter()
            .all(|&v| v == 0.0));
        // 补丁内排列为 (行, 列, 通道)：补丁 1（第二列）第 2 行第 3 个像素的 G 通道
        let (px, py) = (16 + 2, 1);
        let want = (py % 241) as f32 * (1.0f32 / 255.0);
        let got = out.pixel_values[PATCH_DIM + py * 16 * 3 + 2 * 3 + 1];
        assert_eq!(got, want, "px={px}");
    }

    #[test]
    fn resize_preserves_constant_color() {
        let mut img = RgbImage::new(1000, 700);
        img.pixels_mut().for_each(|p| p.0 = [10, 128, 250]);
        let out = prepare_image(&img, 140).unwrap();
        let real = out.soft_tokens * 9;
        for i in 0..real {
            let patch = &out.pixel_values[i * PATCH_DIM..(i + 1) * PATCH_DIM];
            assert_eq!(patch[0], 10.0f32 * (1.0 / 255.0));
            assert_eq!(patch[1], 128.0f32 * (1.0 / 255.0));
            assert_eq!(patch[2], 250.0f32 * (1.0 / 255.0));
        }
    }

    #[test]
    fn rejects_unsupported_soft_tokens() {
        let img = RgbImage::new(64, 64);
        assert!(matches!(
            prepare_image(&img, 100),
            Err(EmbedError::InvalidImage(_))
        ));
    }

    /// 27.2b：拆分后 `shared_pixels_from_image + patchify` 必须与旧路径 `prepare_image`
    /// 逐元素相同（后端共用的边界不改变结果）。
    #[test]
    fn shared_pixels_patchify_matches_prepare_image() {
        // 非整除尺寸：必走缩放分支，覆盖 bicubic-AA 路径。
        let mut img = RgbImage::new(1000, 701);
        for (x, y, p) in img.enumerate_pixels_mut() {
            p.0 = [(x % 251) as u8, (y % 241) as u8, ((x * 3 + y) % 233) as u8];
        }
        let old = prepare_image(&img, 140).unwrap();
        let sp = shared_pixels_from_image(&img, 140).unwrap();
        assert_eq!(sp.size_hw, old.resized, "缩放尺寸必须一致");
        assert_eq!(sp.rgb.len(), old.resized.0 as usize * old.resized.1 as usize * 3);
        let new = sp.patchify().unwrap();
        assert_eq!(new.soft_tokens, old.soft_tokens);
        assert_eq!(new.max_patches, old.max_patches);
        assert_eq!(new.resized, old.resized);
        assert_eq!(new.input_ids, old.input_ids);
        assert_eq!(new.position_ids, old.position_ids);
        assert_eq!(new.pixel_values, old.pixel_values);
    }
}
