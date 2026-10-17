//! 27.5：工作进程 stdio 帧编解码（主进程 router 与 `fo-llama-worker` 共用）。
//!
//! 帧（小端）：`[len u32][req_id u64][kind u8][payload]`，`len` = 其后字节数。
//! kind：1=hello、2=text、3=image、4=shutdown、5=cancel。
//! - text：`[nlen u32][utf8]`；image：`[h u32][w u32][soft u32][rgb h*w*3]`。
//! - 响应：`[len u32][req_id u64][status u8][payload]`；status 0=ok（`[dim u32][f32..][ms f64]`），
//!   非零=错误（`[utf8]`）。hello 响应 payload 为 JSON：`{backend_tag, vulkan, device, tokenizer_sha256}`。

use std::io::{self, Read, Write};

pub const KIND_HELLO: u8 = 1;
pub const KIND_TEXT: u8 = 2;
pub const KIND_IMAGE: u8 = 3;
pub const KIND_SHUTDOWN: u8 = 4;
pub const KIND_CANCEL: u8 = 5;

pub const STATUS_OK: u8 = 0;

pub const HELLO_JSON_MAX: usize = 4096;

#[derive(Debug)]
pub enum FrameError {
    Io(io::Error),
    TooLarge(u32),
    BadKind(u8),
    Truncated,
}

impl From<io::Error> for FrameError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl std::fmt::Display for FrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io: {e}"),
            Self::TooLarge(n) => write!(f, "frame too large: {n}"),
            Self::BadKind(k) => write!(f, "bad kind: {k}"),
            Self::Truncated => write!(f, "truncated frame"),
        }
    }
}

/// 请求体（不含 req_id；由收发双方配对）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Request {
    Hello,
    Text(String),
    Image { h: u32, w: u32, soft: u32, rgb: Vec<u8> },
    Shutdown,
    Cancel(u64),
}

impl Request {
    pub fn kind(&self) -> u8 {
        match self {
            Self::Hello => KIND_HELLO,
            Self::Text(_) => KIND_TEXT,
            Self::Image { .. } => KIND_IMAGE,
            Self::Shutdown => KIND_SHUTDOWN,
            Self::Cancel(_) => KIND_CANCEL,
        }
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut p = Vec::new();
        match self {
            Self::Hello | Self::Shutdown => {}
            Self::Text(s) => {
                p.extend_from_slice(&(s.len() as u32).to_le_bytes());
                p.extend_from_slice(s.as_bytes());
            }
            Self::Image { h, w, soft, rgb } => {
                p.extend_from_slice(&h.to_le_bytes());
                p.extend_from_slice(&w.to_le_bytes());
                p.extend_from_slice(&soft.to_le_bytes());
                p.extend_from_slice(rgb);
            }
            Self::Cancel(id) => p.extend_from_slice(&id.to_le_bytes()),
        }
        p
    }

    pub fn decode(kind: u8, p: &[u8]) -> Result<Self, FrameError> {
        match kind {
            KIND_HELLO => Ok(Self::Hello),
            KIND_SHUTDOWN => Ok(Self::Shutdown),
            KIND_TEXT => {
                if p.len() < 4 {
                    return Err(FrameError::Truncated);
                }
                let n = u32::from_le_bytes(p[0..4].try_into().unwrap()) as usize;
                if p.len() < 4 + n {
                    return Err(FrameError::Truncated);
                }
                let s = std::str::from_utf8(&p[4..4 + n]).map_err(|_| FrameError::Truncated)?;
                Ok(Self::Text(s.to_string()))
            }
            KIND_IMAGE => {
                if p.len() < 12 {
                    return Err(FrameError::Truncated);
                }
                let h = u32::from_le_bytes(p[0..4].try_into().unwrap());
                let w = u32::from_le_bytes(p[4..8].try_into().unwrap());
                let soft = u32::from_le_bytes(p[8..12].try_into().unwrap());
                let rgb = p[12..].to_vec();
                if rgb.len() != h as usize * w as usize * 3 {
                    return Err(FrameError::Truncated);
                }
                Ok(Self::Image { h, w, soft, rgb })
            }
            KIND_CANCEL => {
                if p.len() < 8 {
                    return Err(FrameError::Truncated);
                }
                Ok(Self::Cancel(u64::from_le_bytes(
                    p[0..8].try_into().unwrap(),
                )))
            }
            k => Err(FrameError::BadKind(k)),
        }
    }
}

const MAX_FRAME: u32 = 64 * 1024 * 1024;

/// 写一帧请求。`limit` 为 payload 上限（默认 [`MAX_FRAME`]）。
pub fn write_request(
    w: &mut impl Write,
    req_id: u64,
    req: &Request,
    limit: u32,
) -> Result<(), FrameError> {
    let p = req.encode();
    if p.len() as u64 > u64::from(limit) {
        return Err(FrameError::TooLarge(p.len() as u32));
    }
    let body_len = 8 + 1 + p.len() as u32;
    w.write_all(&body_len.to_le_bytes())?;
    w.write_all(&req_id.to_le_bytes())?;
    w.write_all(&[req.kind()])?;
    w.write_all(&p)?;
    w.flush()?;
    Ok(())
}

/// 读一帧请求，返回 `(req_id, Request)`。
pub fn read_request(r: &mut impl Read, limit: u32) -> Result<(u64, Request), FrameError> {
    let mut len_b = [0u8; 4];
    r.read_exact(&mut len_b).map_err(|e| {
        if e.kind() == io::ErrorKind::UnexpectedEof {
            FrameError::Truncated
        } else {
            FrameError::Io(e)
        }
    })?;
    let len = u32::from_le_bytes(len_b);
    if len > limit || len < 9 {
        return Err(FrameError::TooLarge(len));
    }
    let mut body = vec![0u8; len as usize];
    r.read_exact(&mut body).map_err(|_| FrameError::Truncated)?;
    let req_id = u64::from_le_bytes(body[0..8].try_into().unwrap());
    let req = Request::decode(body[8], &body[9..])?;
    Ok((req_id, req))
}

/// 写一帧响应（ok=`Ok((dim, f32le, ms))`，err=`Err(msg)`）。
pub fn write_response(
    w: &mut impl Write,
    req_id: u64,
    res: Result<(&[f32], f64), &str>,
) -> Result<(), FrameError> {
    let mut body = Vec::new();
    body.extend_from_slice(&req_id.to_le_bytes());
    match res {
        Ok((v, ms)) => {
            body.push(STATUS_OK);
            body.extend_from_slice(&(v.len() as u32).to_le_bytes());
            for x in v {
                body.extend_from_slice(&x.to_le_bytes());
            }
            body.extend_from_slice(&ms.to_le_bytes());
        }
        Err(msg) => {
            body.push(1);
            body.extend_from_slice(msg.as_bytes());
        }
    }
    w.write_all(&(body.len() as u32).to_le_bytes())?;
    w.write_all(&body)?;
    w.flush()?;
    Ok(())
}

/// 写 hello 响应（payload 为 JSON 字符串）。
pub fn write_hello_response(
    w: &mut impl Write,
    req_id: u64,
    json: &str,
) -> Result<(), FrameError> {
    let mut body = Vec::new();
    body.extend_from_slice(&req_id.to_le_bytes());
    body.push(STATUS_OK);
    body.extend_from_slice(&(json.len() as u32).to_le_bytes());
    body.extend_from_slice(json.as_bytes());
    w.write_all(&(body.len() as u32).to_le_bytes())?;
    w.write_all(&body)?;
    w.flush()?;
    Ok(())
}

/// 读一帧响应。ok 返回 `Ok((req_id, Ok((vector, ms))))`；
/// hello 响应（dim 字段实为 JSON 长度）返回 `Ok((req_id, Err(json)))` 由调用方解析；
/// 错误返回 `Ok((req_id, Err(msg)))`。
pub type Response = (u64, Result<(Vec<f32>, f64), String>);

pub fn read_response(r: &mut impl Read) -> Result<Response, FrameError> {
    let mut len_b = [0u8; 4];
    r.read_exact(&mut len_b).map_err(|_| FrameError::Truncated)?;
    let len = u32::from_le_bytes(len_b);
    if !(10..=MAX_FRAME).contains(&len) {
        return Err(FrameError::TooLarge(len));
    }
    let mut body = vec![0u8; len as usize];
    r.read_exact(&mut body).map_err(|_| FrameError::Truncated)?;
    let req_id = u64::from_le_bytes(body[0..8].try_into().unwrap());
    let status = body[8];
    if status != STATUS_OK {
        let msg = String::from_utf8_lossy(&body[9..]).into_owned();
        return Ok((req_id, Err(msg)));
    }
    if body.len() < 13 {
        return Err(FrameError::Truncated);
    }
    let dim = u32::from_le_bytes(body[9..13].try_into().unwrap()) as usize;
    let floats = &body[13..];
    // hello 响应：payload 为 JSON（dim = JSON 字节数；可选后跟 8 字节占位）。
    if floats.len() == dim || floats.len() == dim + 8 {
        let json = String::from_utf8_lossy(&floats[..dim]).into_owned();
        return Ok((req_id, Err(json)));
    }
    if floats.len() != dim * 4 + 8 {
        return Err(FrameError::Truncated);
    }
    let mut v = Vec::with_capacity(dim);
    for i in 0..dim {
        let b: [u8; 4] = floats[i * 4..i * 4 + 4].try_into().unwrap();
        v.push(f32::from_le_bytes(b));
    }
    let ms_b: [u8; 8] = floats[dim * 4..dim * 4 + 8].try_into().unwrap();
    Ok((req_id, Ok((v, f64::from_le_bytes(ms_b)))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn roundtrip(req: &Request) {
        let mut buf = Vec::new();
        write_request(&mut buf, 7, req, MAX_FRAME).unwrap();
        let (id, back) = read_request(&mut Cursor::new(&buf), MAX_FRAME).unwrap();
        assert_eq!(id, 7);
        assert_eq!(&back, req);
    }

    #[test]
    fn text_image_hello_shutdown_cancel_roundtrip() {
        roundtrip(&Request::Hello);
        roundtrip(&Request::Text("木椅 blah".into()));
        roundtrip(&Request::Shutdown);
        roundtrip(&Request::Cancel(123));
        roundtrip(&Request::Image {
            h: 2,
            w: 3,
            soft: 140,
            rgb: vec![9u8; 18],
        });
    }

    #[test]
    fn bad_pixel_len_rejected() {
        assert!(matches!(
            Request::decode(KIND_IMAGE, &[2, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 9]),
            Err(FrameError::Truncated)
        ));
        assert!(matches!(
            Request::decode(99, &[]),
            Err(FrameError::BadKind(99))
        ));
    }

    #[test]
    fn vector_response_roundtrip() {
        let mut buf = Vec::new();
        let v = vec![0.5f32, -1.25, 3.0];
        write_response(&mut buf, 42, Ok((&v, 12.5))).unwrap();
        let (id, res) = read_response(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(id, 42);
        let (back, ms) = res.unwrap();
        assert_eq!(back, v);
        assert!((ms - 12.5).abs() < 1e-9);
    }

    #[test]
    fn error_response_roundtrip() {
        let mut buf = Vec::new();
        write_response(&mut buf, 9, Err("boom")).unwrap();
        let (id, res) = read_response(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(id, 9);
        assert_eq!(res.unwrap_err(), "boom");
    }
}
