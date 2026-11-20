//! 牌認識を試す窓（映像サンドボックスから開く）。
//!
//! 公開・自作の ONNX モデル（検出＋任意で分類）を「モデルのフォルダ」に置いて読み込み、
//! サンドボックスの枠（手牌・河など）の切り出し画像で認識を試す。結果は枠と牌の名前を画像に重ねて出す。

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bevy_egui::egui;
use janscope_recog::{Area, Detection, Found, Label, Recognizer, Rgba, find_models};
use janscope_video::http;
use serde::{Deserialize, Serialize};

use crate::config::ClientConfig;
use crate::filepick::{self, Pick};

/// 牌認識の設定（入力担当の設定ファイルに入る）
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RecogConfig {
    /// モデルを置くフォルダ（空なら実行ファイルの隣の models）
    pub dir: String,
    /// 検出モデルのファイル名
    pub detector: String,
    /// 分類モデルのファイル名（空なら使わない）
    pub classifier: String,
    /// 試す枠（face-A・hand-A など）
    pub region: String,
    /// 領域モデル（手牌・河・副露などの場所を見つける。空なら使わない）
    pub regions: String,
    /// 検出のしきい値（0 ならモデルの既定 0.25）
    pub score: f32,
}

/// 実行ファイルの隣の models
pub fn default_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(|d| d.join("models")))
        .unwrap_or_else(|| PathBuf::from("models"))
}

impl RecogConfig {
    pub fn dir_path(&self) -> PathBuf {
        if self.dir.trim().is_empty() {
            default_dir()
        } else {
            PathBuf::from(self.dir.trim())
        }
    }
}

/// 1回の認識の結果
struct Shot {
    w: usize,
    h: usize,
    rgba: Vec<u8>,
    dets: Vec<Detection>,
    areas: Vec<Area>,
    info: String,
    ms: u128,
    seq: u64,
}

#[derive(Default)]
pub struct RecogUi {
    pub open: bool,
    model: Arc<Mutex<Option<Arc<Recognizer>>>>,
    /// 読み込みの様子（読み込み中・説明・エラー）
    status: Arc<Mutex<String>>,
    loading: Arc<AtomicBool>,
    /// 続けて認識する
    running: Arc<AtomicBool>,
    /// 1回だけ認識する合図
    once: Arc<AtomicBool>,
    /// 認識に使う画像の URL（枠を変えたら変わる）
    url: Arc<Mutex<String>>,
    shot: Arc<Mutex<Option<Shot>>>,
    error: Arc<Mutex<Option<String>>>,
    worker: Option<Arc<AtomicBool>>,
    seq: Arc<AtomicU64>,
    tex: Option<(u64, egui::TextureHandle)>,
    /// 見つけたモデル（検出, 分類）
    found: Option<Found>,
    /// 検出のしきい値（f32 のビット）
    score: Arc<AtomicU32>,
    min_score: f32,
}

impl Drop for RecogUi {
    fn drop(&mut self) {
        if let Some(s) = &self.worker {
            s.store(true, Ordering::SeqCst);
        }
    }
}

fn file_name(p: &Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// フォルダをエクスプローラーなどで開く（なければ作る）
fn open_folder(dir: &Path) {
    let _ = std::fs::create_dir_all(dir);
    #[cfg(windows)]
    let cmd = "explorer";
    #[cfg(target_os = "macos")]
    let cmd = "open";
    #[cfg(all(unix, not(target_os = "macos")))]
    let cmd = "xdg-open";
    let _ = std::process::Command::new(cmd).arg(dir).spawn();
}

impl RecogUi {
    fn load(&mut self, cfg: &RecogConfig) {
        if self.loading.swap(true, Ordering::SeqCst) {
            return;
        }
        let dir = cfg.dir_path();
        let det = dir.join(&cfg.detector);
        let cls = (!cfg.classifier.is_empty()).then(|| dir.join(&cfg.classifier));
        let reg = (!cfg.regions.is_empty()).then(|| dir.join(&cfg.regions));
        let (model, status, loading) = (
            self.model.clone(),
            self.status.clone(),
            self.loading.clone(),
        );
        *status.lock().unwrap() = "読み込んでいます…（大きいモデルは数十秒かかります）".into();
        std::thread::spawn(move || {
            let t = Instant::now();
            match Recognizer::load(&det, cls.as_deref(), reg.as_deref()) {
                Ok(r) => {
                    *status.lock().unwrap() = format!(
                        "読み込みました（{:.1}秒）: {}",
                        t.elapsed().as_secs_f32(),
                        r.describe()
                    );
                    *model.lock().unwrap() = Some(Arc::new(r));
                }
                Err(e) => {
                    *status.lock().unwrap() = format!("読み込めません: {e}");
                    *model.lock().unwrap() = None;
                }
            }
            loading.store(false, Ordering::SeqCst);
        });
    }

    /// 認識する裏の処理（窓を開いている間だけ）
    fn ensure_worker(&mut self) {
        if self.worker.is_some() {
            return;
        }
        let stop = Arc::new(AtomicBool::new(false));
        self.worker = Some(stop.clone());
        let score = self.score.clone();
        let (model, running, once, url, shot, error, seq) = (
            self.model.clone(),
            self.running.clone(),
            self.once.clone(),
            self.url.clone(),
            self.shot.clone(),
            self.error.clone(),
            self.seq.clone(),
        );
        std::thread::spawn(move || {
            while !stop.load(Ordering::SeqCst) {
                let go = once.swap(false, Ordering::SeqCst) || running.load(Ordering::SeqCst);
                let m = model.lock().unwrap().clone();
                let (Some(m), true) = (m, go) else {
                    std::thread::sleep(Duration::from_millis(100));
                    continue;
                };
                let u = url.lock().unwrap().clone();
                let res = http::get(&u, Duration::from_secs(3))
                    .map_err(|e| format!("枠の画像を取れません（{u}）: {e}"))
                    .and_then(|j| {
                        janscope_video::decode_rgba(&j)
                            .map_err(|e| format!("画像を読めません: {e}"))
                    })
                    .and_then(|(w, h, px)| {
                        let t = Instant::now();
                        let th = f32::from_bits(score.load(Ordering::SeqCst));
                        let th = if th > 0.0 { th } else { m.default_score() };
                        let res = m.recognize(
                            &Rgba {
                                w: w as usize,
                                h: h as usize,
                                px: &px,
                            },
                            th,
                        )?;
                        Ok(Shot {
                            w: w as usize,
                            h: h as usize,
                            rgba: px,
                            dets: res.tiles,
                            areas: res.areas,
                            info: res.info,
                            ms: t.elapsed().as_millis(),
                            seq: seq.fetch_add(1, Ordering::SeqCst) + 1,
                        })
                    });
                match res {
                    Ok(s) => {
                        *shot.lock().unwrap() = Some(s);
                        *error.lock().unwrap() = None;
                    }
                    Err(e) => *error.lock().unwrap() = Some(e),
                }
                // 続けるときも、間を空けて CPU を使いすぎない
                std::thread::sleep(Duration::from_millis(150));
            }
        });
    }

    /// 閉じる（認識も止める）
    pub fn close(&mut self) {
        self.open = false;
        self.stop_worker();
    }

    fn stop_worker(&mut self) {
        if let Some(s) = self.worker.take() {
            s.store(true, Ordering::SeqCst);
        }
        self.running.store(false, Ordering::SeqCst);
        self.tex = None;
    }
}

/// 牌の名前（日本語）
fn label_text(l: Label, name: &str) -> String {
    match l {
        Label::Tile(t) => t.name_ja(),
        Label::Back => "裏".into(),
        Label::Other => name.to_string(),
    }
}

/// 窓を出す。`regions` は (枠の id, 名前)、`base` はサンドボックスの HTTP
pub fn window(
    ctx: &egui::Context,
    r: &mut RecogUi,
    cfg: &mut ClientConfig,
    base: &str,
    regions: &[(String, String)],
) {
    if !r.open {
        if r.worker.is_some() {
            r.stop_worker();
        }
        return;
    }
    r.ensure_worker();
    if r.min_score == 0.0 {
        r.min_score = 0.6;
    }
    let mut open = r.open;
    let mut changed = false;
    egui::Window::new("牌認識を試す")
        .open(&mut open)
        .default_size([760.0, 720.0])
        .resizable(true)
        .show(ctx, |ui| {
            ui.label(
                "公開されている・自分で学習した ONNX のモデルを「モデルのフォルダ」に置いて読み込みます。\
                 検出モデル（YOLO）だけでも、検出＋分類モデルでも使えます。",
            );
            // モデルのフォルダ
            let rc = &mut cfg.recog;
            ui.horizontal(|ui| {
                ui.label("モデルのフォルダ");
                let mut d = if rc.dir.is_empty() {
                    default_dir().to_string_lossy().into_owned()
                } else {
                    rc.dir.clone()
                };
                if ui
                    .add(egui::TextEdit::singleline(&mut d).desired_width(380.0))
                    .changed()
                    | filepick::button(ui, "recog_dir", Pick::Folder, &mut d)
                {
                    rc.dir = if PathBuf::from(&d) == default_dir() { String::new() } else { d };
                    r.found = None;
                    changed = true;
                }
                if ui.button("フォルダを開く").clicked() {
                    open_folder(&rc.dir_path());
                }
                if ui.button("探し直す").clicked() {
                    r.found = None;
                }
            });
            let dir = rc.dir_path();
            let found = r.found.get_or_insert_with(|| find_models(&dir)).clone();
            let (dets, clss, regs) = (found.detectors, found.classifiers, found.regions);
            if dets.is_empty() {
                ui.colored_label(
                    egui::Color32::from_rgb(255, 200, 80),
                    format!(
                        "このフォルダに .onnx のモデルがありません（{}）。モデルを置いてから「探し直す」を押してください",
                        dir.display()
                    ),
                );
            }
            // 使うモデル
            if rc.detector.is_empty() || !dets.iter().any(|p| file_name(p) == rc.detector) {
                if let Some(p) = dets.first() {
                    rc.detector = file_name(p);
                    changed = true;
                }
            }
            egui::Grid::new("recog_models").num_columns(2).show(ui, |ui| {
                ui.label("検出モデル");
                egui::ComboBox::from_id_salt("recog_det")
                    .width(300.0)
                    .selected_text(if rc.detector.is_empty() { "（なし）".into() } else { rc.detector.clone() })
                    .show_ui(ui, |ui| {
                        for p in &dets {
                            let n = file_name(p);
                            changed |= ui.selectable_value(&mut rc.detector, n.clone(), n).changed();
                        }
                    });
                ui.end_row();
                ui.label("分類モデル");
                egui::ComboBox::from_id_salt("recog_cls")
                    .width(300.0)
                    .selected_text(if rc.classifier.is_empty() {
                        "使わない（検出モデルのクラスで決める）".into()
                    } else {
                        rc.classifier.clone()
                    })
                    .show_ui(ui, |ui| {
                        changed |= ui
                            .selectable_value(&mut rc.classifier, String::new(), "使わない（検出モデルのクラスで決める）")
                            .changed();
                        for p in &clss {
                            let n = file_name(p);
                            changed |= ui.selectable_value(&mut rc.classifier, n.clone(), n).changed();
                        }
                    });
                ui.end_row();
                ui.label("領域モデル");
                egui::ComboBox::from_id_salt("recog_reg")
                    .width(300.0)
                    .selected_text(if rc.regions.is_empty() {
                        "使わない（枠の画像全体で牌を探す）".into()
                    } else {
                        rc.regions.clone()
                    })
                    .show_ui(ui, |ui| {
                        changed |= ui
                            .selectable_value(&mut rc.regions, String::new(), "使わない（枠の画像全体で牌を探す）")
                            .changed();
                        for p in &regs {
                            let n = file_name(p);
                            changed |= ui.selectable_value(&mut rc.regions, n.clone(), n).changed();
                        }
                    })
                    .response
                    .on_hover_text("手牌・河・副露などの場所を見つけるモデル（regions-yolo など）。卓全体の枠で使うと、場所ごとに牌を探して、どこの牌かも出します");
                ui.end_row();
            });
            ui.horizontal(|ui| {
                let busy = r.loading.load(Ordering::SeqCst);
                if ui
                    .add_enabled(!busy && !rc.detector.is_empty(), egui::Button::new("モデルを読み込む"))
                    .clicked()
                {
                    let c = rc.clone();
                    r.load(&c);
                }
                ui.label(r.status.lock().unwrap().clone());
            });
            ui.separator();
            // 試す枠
            if regions.is_empty() {
                ui.weak("サンドボックスに枠（手牌確認・河/鳴き確認・卓全体）を置くと、その切り出しで試せます");
            } else {
                if !regions.iter().any(|(id, _)| *id == rc.region) {
                    rc.region = regions
                        .iter()
                        .find(|(id, _)| id.starts_with("hand"))
                        .unwrap_or(&regions[0])
                        .0
                        .clone();
                }
                ui.horizontal(|ui| {
                    ui.label("試す枠");
                    let cur = regions
                        .iter()
                        .find(|(id, _)| *id == rc.region)
                        .map(|(_, n)| n.clone())
                        .unwrap_or_default();
                    egui::ComboBox::from_id_salt("recog_region")
                        .width(220.0)
                        .selected_text(cur)
                        .show_ui(ui, |ui| {
                            for (id, n) in regions {
                                changed |= ui.selectable_value(&mut rc.region, id.clone(), n).changed();
                            }
                        });
                    let loaded = r.model.lock().unwrap().is_some();
                    if ui.add_enabled(loaded, egui::Button::new("今の画で認識")).clicked() {
                        r.once.store(true, Ordering::SeqCst);
                    }
                    let mut run = r.running.load(Ordering::SeqCst);
                    if ui.add_enabled(loaded, egui::Checkbox::new(&mut run, "続けて認識")).changed() {
                        r.running.store(run, Ordering::SeqCst);
                    }
                    ui.add(egui::Slider::new(&mut r.min_score, 0.0..=1.0).text("赤にする確信度"));
                });
                ui.horizontal(|ui| {
                    let mut th = if rc.score > 0.0 { rc.score } else { 0.25 };
                    if ui
                        .add(egui::Slider::new(&mut th, 0.01..=0.9).text("検出のしきい値"))
                        .on_hover_text("下げると、自信の低い牌も拾います（見つからない牌があるときは下げる）")
                        .changed()
                    {
                        rc.score = th;
                        changed = true;
                    }
                    r.score.store(th.to_bits(), Ordering::SeqCst);
                });
                *r.url.lock().unwrap() = format!("{}/region/{}.jpg", base.trim_end_matches('/'), rc.region);
            }
            if let Some(e) = r.error.lock().unwrap().clone() {
                ui.colored_label(egui::Color32::from_rgb(255, 120, 120), e);
            }
            // 結果
            let shot = r.shot.lock().unwrap();
            if let Some(s) = shot.as_ref() {
                if r.tex.as_ref().is_none_or(|(q, _)| *q != s.seq) {
                    let img = egui::ColorImage::from_rgba_unmultiplied([s.w, s.h], &s.rgba);
                    match &mut r.tex {
                        Some((q, t)) => {
                            t.set(img, egui::TextureOptions::LINEAR);
                            *q = s.seq;
                        }
                        None => {
                            r.tex = Some((s.seq, ctx.load_texture("recog-shot", img, egui::TextureOptions::LINEAR)));
                        }
                    }
                }
                ui.label(format!("{} 枚（{} ms）", s.dets.len(), s.ms));
                ui.label(egui::RichText::new(&s.info).weak().small())
                    .on_hover_text("モデルの出力の様子。見つからない牌があるときの手がかりになります");
                // 場所（領域）ごとに、左から並べる
                let mut groups: Vec<(Option<String>, Vec<&Detection>)> = vec![];
                for d in &s.dets {
                    match groups.iter_mut().find(|(a, _)| *a == d.area) {
                        Some((_, v)) => v.push(d),
                        None => groups.push((d.area.clone(), vec![d])),
                    }
                }
                for (area, v) in &mut groups {
                    v.sort_by(|a, b| a.bbox[0].total_cmp(&b.bbox[0]));
                ui.horizontal_wrapped(|ui| {
                    ui.label(match area {
                        Some(a) => format!("{a}（左から）:"),
                        None => "左から:".into(),
                    });
                    for d in v.iter() {
                        let t = label_text(d.label, &d.name);
                        let c = if d.score < r.min_score {
                            egui::Color32::from_rgb(255, 110, 110)
                        } else {
                            egui::Color32::from_rgb(120, 230, 140)
                        };
                        ui.colored_label(c, t).on_hover_text(format!("{}（{:.2}）", d.name, d.score));
                    }
                });
                }
                if let Some((_, t)) = &r.tex {
                    let avail = ui.available_size();
                    let sc = (avail.x / s.w as f32).min((avail.y.max(200.0)) / s.h as f32).min(2.0);
                    let size = egui::vec2(s.w as f32 * sc, s.h as f32 * sc);
                    let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
                    let p = ui.painter_at(rect);
                    p.image(
                        t.id(),
                        rect,
                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                        egui::Color32::WHITE,
                    );
                    // 領域（手牌・河など）
                    for a in &s.areas {
                        let b = egui::Rect::from_min_max(
                            rect.min + egui::vec2(a.bbox[0] * sc, a.bbox[1] * sc),
                            rect.min + egui::vec2(a.bbox[2] * sc, a.bbox[3] * sc),
                        );
                        let c = egui::Color32::from_rgb(90, 160, 255);
                        p.rect_stroke(b, 0.0, egui::Stroke::new(2.0, c), egui::StrokeKind::Inside);
                        p.text(
                            b.left_bottom() + egui::vec2(2.0, -2.0),
                            egui::Align2::LEFT_BOTTOM,
                            format!("{} {:.2}", a.name, a.score),
                            egui::FontId::proportional(13.0),
                            c,
                        );
                    }
                    for d in &s.dets {
                        let c = if d.score < r.min_score {
                            egui::Color32::from_rgb(255, 80, 80)
                        } else {
                            egui::Color32::from_rgb(80, 230, 110)
                        };
                        let b = egui::Rect::from_min_max(
                            rect.min + egui::vec2(d.bbox[0] * sc, d.bbox[1] * sc),
                            rect.min + egui::vec2(d.bbox[2] * sc, d.bbox[3] * sc),
                        );
                        p.rect_stroke(b, 0.0, egui::Stroke::new(1.5, c), egui::StrokeKind::Outside);
                        p.text(
                            b.left_top() - egui::vec2(0.0, 1.0),
                            egui::Align2::LEFT_BOTTOM,
                            format!("{} {:.2}", label_text(d.label, &d.name), d.score),
                            egui::FontId::proportional(12.0),
                            c,
                        );
                    }
                }
            } else if r.model.lock().unwrap().is_some() {
                ui.weak("「今の画で認識」を押すと、ここに結果が出ます");
            }
        });
    if changed {
        let _ = cfg.save();
    }
    r.open = open;
}
