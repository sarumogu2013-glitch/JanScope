//! JanScope 牌認識: 公開・自作の ONNX モデルを読み込んで、画像の中の牌を見つけて種類を当てる。
//!
//! - **検出モデル**（必須）: Ultralytics YOLO（v8 / 11 / 26 など）を ONNX に書き出したもの。
//!   - クラスが牌の種類（1m…7z）なら、検出だけで種類まで分かる
//!   - クラスが「牌」1つなら、分類モデルで種類を当てる
//!   - 出力の形は `[1, 4+クラス数(+32), N]`（v8 / 11。セグメンテーションでも可）と `[1, N, 6(+32)]`（26 / v10 の NMS なし）を受け付ける
//! - **分類モデル**（任意）: 牌1枚の画像 → 種類（出力 `[1, クラス数]`）
//! - クラス名・入力の大きさ・正規化は、ONNX の中の情報（Ultralytics の `names`・`imgsz`）か、
//!   モデルと同じ名前の `.json`（学習ツールの `export` が書くもの）から読む
//!
//! 処理は学習ツールの `recognize.py` と同じ: 検出 → 枠を少し広げて切り抜く（横長は90°回す）→ 分類。
//! 推論は tract（Rust だけで動く ONNX の実行器。追加の DLL は要らない）。

pub mod labels;

use std::path::{Path, PathBuf};

use serde::Deserialize;
use tract_onnx::pb::type_proto;
use tract_onnx::prelude::*;

pub use labels::{Label, label_of};

type Plan = std::sync::Arc<TypedRunnableModel>;

/// モデルの隣の `.json`（どれも省略できる）
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct Sidecar {
    kind: Option<String>,
    /// 検出の入力の大きさ（正方形なら数、[高さ, 幅] でも可）
    imgsz: Option<serde_json::Value>,
    classes: Option<Vec<String>>,
    /// "yolov8"（[1,4+nc,N]）か "end2end"（[1,N,6]）
    format: Option<String>,
    input: Option<SideInput>,
    score: Option<f32>,
    iou: Option<f32>,
    pad: Option<f32>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct SideInput {
    width: Option<usize>,
    height: Option<usize>,
    scale: Option<f32>,
    mean: Option<[f32; 3]>,
    std: Option<[f32; 3]>,
}

/// 読み込んだ ONNX 1つ
struct Onnx {
    plan: Plan,
    /// 入力の幅・高さ
    w: usize,
    h: usize,
    names: Vec<String>,
    side: Sidecar,
    /// 出力の形（読み込んだときに1回動かして調べる）
    out_shape: Vec<usize>,
    file: PathBuf,
}

fn em(file: &Path, e: impl std::fmt::Display) -> String {
    format!("{}: {e}", file.display())
}

/// Ultralytics の `names`（Python の辞書の文字列 `{0: 'tile', 1: '1m'}`）を読む
fn parse_names(s: &str) -> Vec<String> {
    let mut v: Vec<(usize, String)> = vec![];
    let s = s.trim().trim_start_matches('{').trim_end_matches('}');
    let mut rest = s;
    while let Some(colon) = rest.find(':') {
        let key = rest[..colon].trim().trim_start_matches(',').trim();
        let after = rest[colon + 1..].trim_start();
        let Some(q) = after.chars().next().filter(|c| *c == '\'' || *c == '"') else {
            break;
        };
        let body = &after[1..];
        let Some(end) = body.find(q) else { break };
        if let Ok(k) = key.parse::<usize>() {
            v.push((k, body[..end].to_string()));
        }
        rest = &body[end + 1..];
    }
    v.sort_by_key(|(k, _)| *k);
    v.into_iter().map(|(_, n)| n).collect()
}

/// `[640, 640]` や `640` を（高さ, 幅）に
fn parse_size(v: &serde_json::Value) -> Option<(usize, usize)> {
    match v {
        serde_json::Value::Number(n) => n.as_u64().map(|n| (n as usize, n as usize)),
        serde_json::Value::Array(a) if a.len() == 2 => {
            Some((a[0].as_u64()? as usize, a[1].as_u64()? as usize))
        }
        serde_json::Value::String(s) => serde_json::from_str(s).ok().and_then(|v| parse_size(&v)),
        _ => None,
    }
}

impl Onnx {
    /// `default` は、モデルの入力の大きさが決まっていないときの（幅, 高さ）
    fn load(file: &Path, default: impl Fn(&Sidecar) -> (usize, usize)) -> Result<Onnx, String> {
        let side: Sidecar = std::fs::read_to_string(file.with_extension("json"))
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default();
        let proto = tract_onnx::onnx()
            .proto_model_for_path(file)
            .map_err(|x| em(file, x))?;
        let meta = |k: &str| {
            proto
                .metadata_props
                .iter()
                .find(|p| p.key == k)
                .map(|p| p.value.clone())
        };
        // 入力の大きさ: モデルに書いてあればそれ、なければ .json・Ultralytics の imgsz・既定
        let graph = proto
            .graph
            .as_ref()
            .ok_or("グラフがありません")
            .map_err(|x| em(file, x))?;
        let inits: Vec<&str> = graph.initializer.iter().map(|i| i.name.as_str()).collect();
        let input = graph
            .input
            .iter()
            .find(|i| !inits.contains(&i.name.as_str()))
            .ok_or("入力がありません")
            .map_err(|x| em(file, x))?;
        let dims: Vec<Option<usize>> = match input.r#type.as_ref().and_then(|t| t.value.as_ref()) {
            Some(type_proto::Value::TensorType(t)) => t
                .shape
                .as_ref()
                .map(|s| {
                    s.dim
                        .iter()
                        .map(|d| match &d.value {
                            Some(
                                tract_onnx::pb::tensor_shape_proto::dimension::Value::DimValue(v),
                            ) if *v > 0 => Some(*v as usize),
                            _ => None,
                        })
                        .collect()
                })
                .unwrap_or_default(),
            _ => vec![],
        };
        if dims.len() != 4 || dims[1].is_some_and(|c| c != 3) {
            return Err(em(
                file,
                format!("入力が画像（[1, 3, 高さ, 幅]）ではありません: {dims:?}"),
            ));
        }
        let (mut w, mut h) = default(&side);
        if let Some((hh, ww)) = side
            .imgsz
            .as_ref()
            .and_then(parse_size)
            .or_else(|| meta("imgsz").and_then(|s| parse_size(&serde_json::Value::String(s))))
        {
            (w, h) = (ww, hh);
        }
        if let Some(i) = &side.input {
            w = i.width.unwrap_or(w);
            h = i.height.unwrap_or(h);
        }
        if let (Some(hh), Some(ww)) = (dims[2], dims[3]) {
            (w, h) = (ww, hh);
        }
        let names = side
            .classes
            .clone()
            .or_else(|| meta("names").map(|s| parse_names(&s)))
            .unwrap_or_default();
        let model = tract_onnx::onnx()
            .model_for_proto_model(&proto)
            .map_err(|x| em(file, x))?
            .with_input_fact(0, f32::fact([1, 3, h, w]).into())
            .map_err(|x| em(file, x))?;
        let plan = model
            .into_optimized()
            .map_err(|x| em(file, x))?
            .into_runnable()
            .map_err(|x| em(file, x))?;
        let mut m = Onnx {
            plan,
            w,
            h,
            names,
            side,
            out_shape: vec![],
            file: file.to_path_buf(),
        };
        let out = m.run(vec![0.0; 3 * w * h])?;
        m.out_shape = out.shape().to_vec();
        Ok(m)
    }

    fn run(&self, chw: Vec<f32>) -> Result<tract_ndarray::ArrayD<f32>, String> {
        let t = tract_ndarray::Array4::from_shape_vec((1, 3, self.h, self.w), chw)
            .map_err(|x| em(&self.file, x))?;
        let out = self
            .plan
            .run(tvec!(Tensor::from(t).into()))
            .map_err(|x| em(&self.file, x))?;
        Ok(out[0]
            .to_plain_array_view::<f32>()
            .map_err(|x| em(&self.file, x))?
            .to_owned())
    }
}

/// RGBA の画像（幅・高さ・画素）
#[derive(Clone)]
pub struct Rgba<'a> {
    pub w: usize,
    pub h: usize,
    pub px: &'a [u8],
}

impl Rgba<'_> {
    /// (x, y) の色（範囲外は端の色）を双線形で
    fn sample(&self, x: f32, y: f32) -> [f32; 3] {
        let x = x.clamp(0.0, (self.w - 1) as f32);
        let y = y.clamp(0.0, (self.h - 1) as f32);
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.w - 1), (y0 + 1).min(self.h - 1));
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let p = |xx: usize, yy: usize, c: usize| self.px[(yy * self.w + xx) * 4 + c] as f32;
        std::array::from_fn(|c| {
            let a = p(x0, y0, c) * (1.0 - fx) + p(x1, y0, c) * fx;
            let b = p(x0, y1, c) * (1.0 - fx) + p(x1, y1, c) * fx;
            a * (1.0 - fy) + b * fy
        })
    }
}

/// 見つけた牌1枚
#[derive(Clone, Debug)]
pub struct Detection {
    /// 画像の中の枠（左上 x, y・右下 x, y。画素）
    pub bbox: [f32; 4],
    pub label: Label,
    /// モデルのクラス名
    pub name: String,
    /// 確信度（検出 × 分類）
    pub score: f32,
}

/// 検出（＋分類）
pub struct Recognizer {
    det: Onnx,
    cls: Option<Onnx>,
    pub score_threshold: f32,
    pub iou_threshold: f32,
}

/// フォルダの中から、検出モデルと分類モデルを探す
pub fn find_models(dir: &Path) -> (Vec<PathBuf>, Vec<PathBuf>) {
    let mut det = vec![];
    let mut cls = vec![];
    let Ok(rd) = std::fs::read_dir(dir) else {
        return (det, cls);
    };
    let mut files: Vec<PathBuf> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .is_some_and(|x| x.eq_ignore_ascii_case("onnx"))
        })
        .collect();
    files.sort();
    for f in files {
        let kind: Option<String> = std::fs::read_to_string(f.with_extension("json"))
            .ok()
            .and_then(|t| serde_json::from_str::<Sidecar>(&t).ok())
            .and_then(|s| s.kind);
        let name = f
            .file_name()
            .map(|n| n.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        let is_cls = match kind.as_deref() {
            Some(k) if k.contains("classifier") => true,
            Some(k) if k.contains("detector") => false,
            _ => name.contains("classif") || name.contains("cls") || name.contains("resnet"),
        };
        if is_cls { cls.push(f) } else { det.push(f) }
    }
    (det, cls)
}

impl Recognizer {
    pub fn load(detector: &Path, classifier: Option<&Path>) -> Result<Recognizer, String> {
        let det = Onnx::load(detector, |_| (640, 640))?;
        if det.out_shape.len() != 3 {
            return Err(format!(
                "{}: 検出モデルの出力の形が分かりません: {:?}",
                detector.display(),
                det.out_shape
            ));
        }
        let cls = match classifier {
            Some(p) => {
                let c = Onnx::load(p, |_| (96, 128))?;
                if c.out_shape.len() != 2 {
                    return Err(format!(
                        "{}: 分類モデルの出力の形が分かりません: {:?}",
                        p.display(),
                        c.out_shape
                    ));
                }
                Some(c)
            }
            None => None,
        };
        let score_threshold = det.side.score.unwrap_or(0.35);
        let iou_threshold = det.side.iou.unwrap_or(0.5);
        Ok(Recognizer {
            det,
            cls,
            score_threshold,
            iou_threshold,
        })
    }

    /// 読み込んだモデルの説明（画面に出す）
    pub fn describe(&self) -> String {
        let d = &self.det;
        let mut s = format!(
            "検出: {}（{}×{}・クラス {}）",
            d.file.file_name().unwrap_or_default().to_string_lossy(),
            d.w,
            d.h,
            if d.names.is_empty() {
                "名前なし".to_string()
            } else {
                d.names.len().to_string()
            }
        );
        if let Some(c) = &self.cls {
            s += &format!(
                " ／ 分類: {}（{}×{}・クラス {}）",
                c.file.file_name().unwrap_or_default().to_string_lossy(),
                c.w,
                c.h,
                c.names.len()
            );
        }
        let known = self
            .class_names()
            .iter()
            .filter(|n| label_of(n) != Label::Other)
            .count();
        if known == 0 {
            s += " ／ ⚠ 牌の名前として読めるクラスがありません（.json の classes で名前を付けてください）";
        }
        s
    }

    /// 種類を決めるのに使うクラス名（分類があれば分類の）
    pub fn class_names(&self) -> &[String] {
        match &self.cls {
            Some(c) => &c.names,
            None => &self.det.names,
        }
    }

    /// 画像の中の牌を見つける
    pub fn recognize(&self, img: &Rgba) -> Result<Vec<Detection>, String> {
        if img.w < 2 || img.h < 2 || img.px.len() < img.w * img.h * 4 {
            return Err("画像が空です".into());
        }
        let mut dets = self.detect(img)?;
        if let Some(cls) = &self.cls {
            let pad = self.det.side.pad.or(cls.side.pad).unwrap_or(0.06);
            for d in &mut dets {
                let (k, p) = classify(cls, img, d.bbox, pad)?;
                d.name = cls.names.get(k).cloned().unwrap_or_else(|| format!("#{k}"));
                d.label = label_of(&d.name);
                d.score *= p;
            }
        }
        Ok(dets)
    }

    fn detect(&self, img: &Rgba) -> Result<Vec<Detection>, String> {
        let m = &self.det;
        // 縦横比を保って縮め、余りは灰色（114）で埋める（Ultralytics と同じ）
        let r = (m.w as f32 / img.w as f32).min(m.h as f32 / img.h as f32);
        let (nw, nh) = (img.w as f32 * r, img.h as f32 * r);
        let (ox, oy) = ((m.w as f32 - nw) / 2.0, (m.h as f32 - nh) / 2.0);
        let mut chw = vec![114.0 / 255.0; 3 * m.w * m.h];
        let plane = m.w * m.h;
        for y in 0..m.h {
            let fy = y as f32 + 0.5 - oy;
            if fy < 0.0 || fy >= nh {
                continue;
            }
            for x in 0..m.w {
                let fx = x as f32 + 0.5 - ox;
                if fx < 0.0 || fx >= nw {
                    continue;
                }
                let c = img.sample(fx / r - 0.5, fy / r - 0.5);
                for k in 0..3 {
                    chw[k * plane + y * m.w + x] = c[k] / 255.0;
                }
            }
        }
        let out = m.run(chw)?;
        let s = out.shape();
        let nc = m.names.len();
        // チャンネル（4+クラス数、セグメンテーションは +32）がどちらの次元か
        let is_ch = |d: usize| nc > 0 && (d == 4 + nc || d == 4 + nc + 32);
        let end2end = match m.side.format.as_deref() {
            Some("end2end") => true,
            Some(_) => false,
            // [1, N, 6(+32)]: 1行 = 左上・右下・確信度・クラス
            None => (s[2] == 6 || s[2] == 38) && !is_ch(s[1]) && !(is_ch(s[2]) && s[1] > s[2] * 4),
        };
        let mut boxes: Vec<(f32, [f32; 4], usize)> = vec![];
        let unmap = |b: [f32; 4]| -> [f32; 4] {
            [
                ((b[0] - ox) / r).clamp(0.0, img.w as f32),
                ((b[1] - oy) / r).clamp(0.0, img.h as f32),
                ((b[2] - ox) / r).clamp(0.0, img.w as f32),
                ((b[3] - oy) / r).clamp(0.0, img.h as f32),
            ]
        };
        if end2end {
            for i in 0..s[1] {
                let sc = out[[0, i, 4]];
                if sc >= self.score_threshold {
                    let b = [
                        out[[0, i, 0]],
                        out[[0, i, 1]],
                        out[[0, i, 2]],
                        out[[0, i, 3]],
                    ];
                    boxes.push((sc, unmap(b), out[[0, i, 5]].max(0.0) as usize));
                }
            }
        } else {
            // [1, 4+nc(+32), N]。クラス数が分からなければ、セグメンテーションの 32 を引いた残り
            let (ch, n, t) = if is_ch(s[1]) {
                (s[1], s[2], false)
            } else if is_ch(s[2]) || s[1] > s[2] {
                (s[2], s[1], true)
            } else {
                (s[1], s[2], false)
            };
            let ncls = if nc > 0 && 4 + nc <= ch {
                nc
            } else if ch > 4 + 32 {
                ch - 4 - 32
            } else {
                ch - 4
            };
            let at = |c: usize, i: usize| if t { out[[0, i, c]] } else { out[[0, c, i]] };
            for i in 0..n {
                let (mut best, mut k) = (0.0f32, 0usize);
                for c in 0..ncls {
                    let v = at(4 + c, i);
                    if v > best {
                        best = v;
                        k = c;
                    }
                }
                if best >= self.score_threshold {
                    let (cx, cy, w, h) = (at(0, i), at(1, i), at(2, i), at(3, i));
                    let b = [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0];
                    boxes.push((best, unmap(b), k));
                }
            }
            boxes = nms(boxes, self.iou_threshold);
        }
        Ok(boxes
            .into_iter()
            .filter(|(_, b, _)| b[2] - b[0] >= 2.0 && b[3] - b[1] >= 2.0)
            .map(|(score, bbox, k)| {
                let name = m.names.get(k).cloned().unwrap_or_else(|| format!("#{k}"));
                Detection {
                    bbox,
                    label: label_of(&name),
                    name,
                    score,
                }
            })
            .collect())
    }
}

/// 重なった枠のうち、確信度の低いほうを捨てる（クラスに関係なく。牌は重ならないので）
fn nms(mut v: Vec<(f32, [f32; 4], usize)>, iou: f32) -> Vec<(f32, [f32; 4], usize)> {
    v.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut keep: Vec<(f32, [f32; 4], usize)> = vec![];
    for d in v {
        if keep.iter().all(|k| iou_of(&k.1, &d.1) < iou) {
            keep.push(d);
        }
    }
    keep
}

fn iou_of(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let w = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let h = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let i = w * h;
    let u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i;
    if u <= 0.0 { 0.0 } else { i / u }
}

/// 牌1枚を切り抜いて分類する（枠を少し広げる。横長なら時計回りに90°回して縦長にそろえる）
fn classify(m: &Onnx, img: &Rgba, b: [f32; 4], pad: f32) -> Result<(usize, f32), String> {
    let (bw, bh) = (b[2] - b[0], b[3] - b[1]);
    let x1 = (b[0] - bw * pad).max(0.0);
    let y1 = (b[1] - bh * pad).max(0.0);
    let x2 = (b[2] + bw * pad).min(img.w as f32);
    let y2 = (b[3] + bh * pad).min(img.h as f32);
    let (cw, ch) = (x2 - x1, y2 - y1);
    let rotate = cw > ch * 1.1;
    let inp = m.side.input.as_ref();
    let scale = inp.and_then(|i| i.scale).unwrap_or(1.0 / 255.0);
    let mean = inp.and_then(|i| i.mean).unwrap_or([0.485, 0.456, 0.406]);
    let std = inp.and_then(|i| i.std).unwrap_or([0.229, 0.224, 0.225]);
    let plane = m.w * m.h;
    let mut chw = vec![0.0; 3 * plane];
    for y in 0..m.h {
        for x in 0..m.w {
            let (u, v) = ((x as f32 + 0.5) / m.w as f32, (y as f32 + 0.5) / m.h as f32);
            // 回すときは、出力の (u, v) が元の (v, 1-u) に当たる（時計回り 90°）
            let (sx, sy) = if rotate {
                (x1 + v * cw, y1 + (1.0 - u) * ch)
            } else {
                (x1 + u * cw, y1 + v * ch)
            };
            let c = img.sample(sx - 0.5, sy - 0.5);
            for k in 0..3 {
                chw[k * plane + y * m.w + x] = (c[k] * scale - mean[k]) / std[k];
            }
        }
    }
    let out = m.run(chw)?;
    let logits: Vec<f32> = out.iter().copied().collect();
    // 確率になっていなければ softmax
    let is_prob = logits.iter().all(|v| (0.0..=1.0).contains(v))
        && (logits.iter().sum::<f32>() - 1.0).abs() < 0.01;
    let probs: Vec<f32> = if is_prob {
        logits
    } else {
        let mx = logits.iter().copied().fold(f32::MIN, f32::max);
        let ex: Vec<f32> = logits.iter().map(|v| (v - mx).exp()).collect();
        let sum: f32 = ex.iter().sum();
        ex.iter().map(|v| v / sum).collect()
    };
    let (k, p) = probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(k, p)| (k, *p))
        .unwrap_or((0, 0.0));
    Ok((k, p))
}

/// 左から右に並べた牌（手牌の読み取りの確認用）
pub fn tiles_left_to_right(dets: &[Detection]) -> Vec<Label> {
    let mut v: Vec<&Detection> = dets.iter().collect();
    v.sort_by(|a, b| a.bbox[0].total_cmp(&b.bbox[0]));
    v.into_iter().map(|d| d.label).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_from_ultralytics() {
        assert_eq!(parse_names("{0: 'tile'}"), vec!["tile"]);
        assert_eq!(
            parse_names("{0: '1m', 1: \"2m\", 2: 'back'}"),
            vec!["1m", "2m", "back"]
        );
        assert_eq!(
            parse_size(&serde_json::Value::String("[640, 480]".into())),
            Some((640, 480))
        );
    }

    #[test]
    fn nms_drops_overlaps() {
        let v = nms(
            vec![
                (0.9, [0.0, 0.0, 10.0, 10.0], 0),
                (0.8, [1.0, 1.0, 10.0, 10.0], 1),
                (0.7, [20.0, 0.0, 30.0, 10.0], 2),
            ],
            0.5,
        );
        assert_eq!(v.len(), 2);
    }
}
