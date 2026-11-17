//! 小さな ONNX（出力が決まっているもの）で、読み込み・出力の読み取り・分類までを確かめる。
//! モデルは `JANSCOPE_RECOG_TESTDIR` のフォルダに置く（なければ飛ばす）。

use janscope_recog::{Label, Recognizer, Rgba, find_models};
use std::path::PathBuf;

fn dir() -> Option<PathBuf> {
    std::env::var_os("JANSCOPE_RECOG_TESTDIR").map(PathBuf::from)
}

fn img() -> Vec<u8> {
    vec![128; 128 * 128 * 4]
}

#[test]
fn yolov8_style() {
    let Some(d) = dir() else { return };
    let r = Recognizer::load(&d.join("det_v8.onnx"), None).unwrap();
    let px = img();
    let dets = r
        .recognize(&Rgba {
            w: 128,
            h: 128,
            px: &px,
        })
        .unwrap();
    println!("{} {:?}", r.describe(), dets);
    assert_eq!(dets.len(), 2, "重なった枠は1つになる");
    assert_eq!(dets[0].label, Label::Tile("1m".parse().unwrap()));
    assert_eq!(dets[1].label, Label::Tile("7z".parse().unwrap()));
    // 64 → 128 なので2倍
    let b = dets[0].bbox;
    assert!(
        (b[0] - 24.0).abs() < 0.01
            && (b[1] - 20.0).abs() < 0.01
            && (b[2] - 40.0).abs() < 0.01
            && (b[3] - 44.0).abs() < 0.01,
        "{b:?}"
    );
}

#[test]
fn end2end_with_classifier() {
    let Some(d) = dir() else { return };
    let (det, cls) = find_models(&d);
    assert!(cls.iter().any(|p| p.ends_with("tile_classifier.onnx")));
    assert!(det.iter().any(|p| p.ends_with("det_e2e.onnx")));
    let r = Recognizer::load(
        &d.join("det_e2e.onnx"),
        Some(&d.join("tile_classifier.onnx")),
    )
    .unwrap();
    // 横長の画像（余白あり）
    let px = vec![200u8; 256 * 128 * 4];
    let dets = r
        .recognize(&Rgba {
            w: 256,
            h: 128,
            px: &px,
        })
        .unwrap();
    println!("{} {:?}", r.describe(), dets);
    assert_eq!(dets.len(), 2, "確信度の低いものは捨てる");
    assert!(
        dets.iter()
            .all(|d| d.label == Label::Tile("7z".parse().unwrap()))
    );
    // 64 に 256×128 を入れると 1/4、上下に 16 の余白 → x1 = 12*4 = 48, y1 = (10-16)*4 → 0
    let b = dets[0].bbox;
    assert!(
        (b[0] - 48.0).abs() < 0.01 && b[1] == 0.0 && (b[3] - 24.0).abs() < 0.01,
        "{b:?}"
    );
}
