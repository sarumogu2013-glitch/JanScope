//! 小さな ONNX（出力が決まっているもの）で、読み込み・出力の読み取り・分類・領域までを確かめる。
//! モデルは `JANSCOPE_RECOG_TESTDIR` のフォルダに置く（なければ飛ばす）。

use janscope_recog::{Label, Recognizer, Rgba, find_models};
use std::path::PathBuf;

fn dir() -> Option<PathBuf> {
    std::env::var_os("JANSCOPE_RECOG_TESTDIR").map(PathBuf::from)
}

fn t(s: &str) -> Label {
    Label::Tile(s.parse().unwrap())
}

#[test]
fn yolov8_style() {
    let Some(d) = dir() else { return };
    let r = Recognizer::load(&d.join("det_v8.onnx"), None, None).unwrap();
    let px = vec![128u8; 128 * 128 * 4];
    let res = r
        .recognize(
            &Rgba {
                w: 128,
                h: 128,
                px: &px,
            },
            0.35,
        )
        .unwrap();
    println!("{} / {}", r.describe(), res.info);
    let dets = res.tiles;
    assert_eq!(dets.len(), 2, "重なった枠は1つになる");
    assert_eq!(dets[0].label, t("1m"));
    assert_eq!(dets[1].label, t("7z"));
    // 64 → 128 なので2倍
    let b = dets[0].bbox;
    assert!(
        (b[0] - 24.0).abs() < 0.01
            && (b[1] - 20.0).abs() < 0.01
            && (b[2] - 40.0).abs() < 0.01
            && (b[3] - 44.0).abs() < 0.01,
        "{b:?}"
    );
    // しきい値を上げると減る
    let res = r
        .recognize(
            &Rgba {
                w: 128,
                h: 128,
                px: &px,
            },
            0.85,
        )
        .unwrap();
    assert_eq!(res.tiles.len(), 1);
}

#[test]
fn end2end_with_classifier() {
    let Some(d) = dir() else { return };
    let f = find_models(&d);
    assert!(
        f.classifiers
            .iter()
            .any(|p| p.ends_with("tile_classifier.onnx"))
    );
    assert!(f.detectors.iter().any(|p| p.ends_with("det_e2e.onnx")));
    assert!(f.regions.iter().any(|p| p.ends_with("regions-test.onnx")));
    let r = Recognizer::load(
        &d.join("det_e2e.onnx"),
        Some(&d.join("tile_classifier.onnx")),
        None,
    )
    .unwrap();
    // 横長の画像（余白あり）
    let px = vec![200u8; 256 * 128 * 4];
    let res = r
        .recognize(
            &Rgba {
                w: 256,
                h: 128,
                px: &px,
            },
            0.35,
        )
        .unwrap();
    println!("{} / {}", r.describe(), res.info);
    let dets = res.tiles;
    assert_eq!(dets.len(), 2, "確信度の低いものは捨てる");
    assert!(dets.iter().all(|d| d.label == t("7z")));
    // 64 に 256×128 を入れると 1/4、上下に 16 の余白 → x1 = 12*4 = 48, y1 = (10-16)*4 → 0
    let b = dets[0].bbox;
    assert!(
        (b[0] - 48.0).abs() < 0.01 && b[1] == 0.0 && (b[3] - 24.0).abs() < 0.01,
        "{b:?}"
    );
}

#[test]
fn end2end_written_as_center() {
    // NMS なし形式なのに中心+幅高さで書いてあるモデル: 左上+右下として読むと左端の1枚しか残らない
    let Some(d) = dir() else { return };
    let r = Recognizer::load(&d.join("det_e2e_cxcywh.onnx"), None, None).unwrap();
    let px = vec![100u8; 64 * 64 * 4];
    let res = r
        .recognize(
            &Rgba {
                w: 64,
                h: 64,
                px: &px,
            },
            0.25,
        )
        .unwrap();
    println!("{}", res.info);
    assert_eq!(res.tiles.len(), 5, "5枚とも出る");
    let b = res.tiles.iter().map(|d| d.bbox[0]).fold(f32::MAX, f32::min);
    assert!((b - 3.0).abs() < 0.01, "{b}");
}

#[test]
fn with_regions() {
    let Some(d) = dir() else { return };
    let r = Recognizer::load(
        &d.join("det_v8.onnx"),
        None,
        Some(&d.join("regions-test.onnx")),
    )
    .unwrap();
    let px = vec![90u8; 64 * 64 * 4];
    let res = r
        .recognize(
            &Rgba {
                w: 64,
                h: 64,
                px: &px,
            },
            0.35,
        )
        .unwrap();
    println!("{} / {}", r.describe(), res.info);
    assert_eq!(res.areas.len(), 2);
    assert_eq!(res.areas[0].name, "area0");
    // 領域ごとに牌を探す（左の領域 x 0〜32、右の領域 x 32〜64）
    assert!(res.tiles.iter().any(|t| t.area.as_deref() == Some("area0")));
    assert!(res.tiles.iter().any(|t| t.area.as_deref() == Some("area3")));
    assert!(res.tiles.iter().all(|t| t.bbox[2] <= 64.0));
}
