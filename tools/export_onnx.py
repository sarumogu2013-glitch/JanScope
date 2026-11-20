"""公開されている牌認識モデル（PyTorch の .pt）を、JanScope で読める ONNX に書き出す。

使い方（Python 3.10 以上。CPU だけで動く）:

    pip install ultralytics                      # 検出モデル（YOLO）を書き出すのに使う
    python export_onnx.py detector  tile-yolo-26n.pt           [--out models]
    python export_onnx.py detector  regions-yolo-26n.pt        [--out models]   # 領域（手牌・河・副露など）
    python export_onnx.py classifier classifier-resnet50.pt    [--out models] [--size 224 224] [--classes classes.txt]

- detector:  Ultralytics YOLO（v8 / 11 / 26 など。検出・セグメンテーションどちらでも）→ <名前>.onnx と <名前>.json
- classifier: torchvision の ResNet（18 / 34 / 50 / 101）の重み → <名前>.onnx と <名前>.json
  - クラス名（1m, 2m, …）は .pt の中にあればそれを使う。なければ --classes で1行1つのテキストを渡す
  - 入力の大きさ・正規化は --size（幅 高さ）・--mean・--std で変えられる（既定は 224×224・ImageNet）
- 書き出した .onnx と .json を、JanScope の models フォルダに置く

このスクリプトは JanScope の一部（MIT OR Apache-2.0）。書き出したモデルのライセンスは、元のモデルのライセンスに従う。
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def export_detector(pt: Path, out: Path, imgsz: int | None) -> None:
    from ultralytics import YOLO

    model = YOLO(str(pt))
    kw = {"format": "onnx", "opset": 17, "simplify": False, "dynamic": False}
    if imgsz:
        kw["imgsz"] = imgsz
    path = Path(model.export(**kw))
    out.mkdir(parents=True, exist_ok=True)
    dst = out / path.name
    if path.resolve() != dst.resolve():
        shutil.copyfile(path, dst)
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    meta = {
        "kind": "janscope-region-detector" if "region" in pt.name.lower() else "janscope-tile-detector",
        "framework": "ultralytics-yolo",
        "task": getattr(model, "task", ""),
        "classes": [names[k] for k in sorted(names)],
        "source": pt.name,
    }
    if imgsz:
        meta["imgsz"] = imgsz
    dst.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"書き出しました: {dst}（クラス {len(meta['classes'])}: {', '.join(meta['classes'][:8])}…）")


def _find_state_dict(obj):
    import torch

    if isinstance(obj, torch.nn.Module):
        return obj, None
    if isinstance(obj, dict):
        for k in ("model", "state_dict", "model_state_dict", "net", "weights"):
            v = obj.get(k)
            if isinstance(v, torch.nn.Module):
                return v, obj
            if isinstance(v, dict) and any(hasattr(t, "shape") for t in v.values()):
                return v, obj
        if any(hasattr(t, "shape") for t in obj.values()):
            return obj, None
    raise SystemExit("この .pt の中身が分かりません（モデルか state_dict が入っていません）")


def _class_names(meta: dict | None) -> list[str] | None:
    if not meta:
        return None
    for k in ("classes", "class_names", "names", "labels", "idx_to_class"):
        v = meta.get(k)
        if isinstance(v, dict):
            return [str(v[i]) for i in sorted(v)]
        if isinstance(v, (list, tuple)) and v:
            return [str(x) for x in v]
    c2i = meta.get("class_to_idx")
    if isinstance(c2i, dict):
        return [n for n, _ in sorted(c2i.items(), key=lambda kv: kv[1])]
    return None


def _resnet_from_state(sd: dict):
    import torchvision

    sd = {k.removeprefix("module.").removeprefix("model.").removeprefix("backbone."): v for k, v in sd.items()}
    fc = next((k for k in sd if k.endswith("fc.weight")), None)
    if fc is None:
        raise SystemExit("ResNet の重みではないようです（fc.weight がありません）")
    nc = sd[fc].shape[0]
    bottleneck = any(k.startswith("layer1.0.conv3") for k in sd)
    blocks = [len({k.split(".")[1] for k in sd if k.startswith(f"layer{i}.")}) for i in range(1, 5)]
    arch = {
        (False, (2, 2, 2, 2)): "resnet18",
        (False, (3, 4, 6, 3)): "resnet34",
        (True, (3, 4, 6, 3)): "resnet50",
        (True, (3, 4, 23, 3)): "resnet101",
    }.get((bottleneck, tuple(blocks)))
    if arch is None:
        raise SystemExit(f"ResNet の種類が分かりません（ブロック {blocks}）")
    m = getattr(torchvision.models, arch)(num_classes=nc)
    m.load_state_dict(sd, strict=True)
    return m, arch, nc


def export_classifier(pt: Path, out: Path, size: tuple[int, int], classes_file: Path | None,
                      mean: list[float], std: list[float]) -> None:
    import torch

    obj = torch.load(str(pt), map_location="cpu", weights_only=False)
    model, meta = _find_state_dict(obj)
    arch = type(model).__name__
    if isinstance(model, dict):
        model, arch, nc = _resnet_from_state(model)
    model.eval()
    names = None
    if classes_file:
        names = [l.strip() for l in classes_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    names = names or _class_names(meta if isinstance(meta, dict) else (obj if isinstance(obj, dict) else None))
    w, h = size
    with torch.no_grad():
        nc = model(torch.zeros(1, 3, h, w)).shape[-1]
    if not names:
        print(f"⚠ クラス名が見つかりません。{nc} 個の名前を 1 行 1 つで書いたファイルを --classes で渡してください。"
              " JanScope では名前が牌として読めないと結果が #番号 になります")
        names = [f"#{i}" for i in range(nc)]
    if len(names) != nc:
        raise SystemExit(f"クラス名の数（{len(names)}）がモデルの出力（{nc}）と合いません")
    out.mkdir(parents=True, exist_ok=True)
    dst = out / (pt.stem + ".onnx")
    torch.onnx.export(model, torch.zeros(1, 3, h, w), str(dst), input_names=["input"], output_names=["logits"],
                      opset_version=17, dynamo=False)
    meta_out = {
        "kind": "janscope-tile-classifier",
        "arch": arch,
        "input": {"width": w, "height": h, "layout": "NCHW", "color": "RGB", "scale": 1 / 255, "mean": mean, "std": std},
        "classes": names,
        "source": pt.name,
    }
    dst.with_suffix(".json").write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"書き出しました: {dst}（{arch}・{w}×{h}・クラス {nc}: {', '.join(names[:8])}…）")


def main() -> None:
    p = argparse.ArgumentParser(description="牌認識モデル（.pt）を JanScope 用の ONNX に書き出す")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detector", help="YOLO（Ultralytics）の検出・セグメンテーションモデル")
    d.add_argument("pt", type=Path)
    d.add_argument("--out", type=Path, default=Path("models"))
    d.add_argument("--imgsz", type=int, default=None, help="入力の大きさ（省略するとモデルの学習時の大きさ）")
    c = sub.add_parser("classifier", help="ResNet などの分類モデル")
    c.add_argument("pt", type=Path)
    c.add_argument("--out", type=Path, default=Path("models"))
    c.add_argument("--size", type=int, nargs=2, default=[224, 224], metavar=("幅", "高さ"))
    c.add_argument("--classes", type=Path, default=None, help="クラス名（1行1つ）")
    c.add_argument("--mean", type=float, nargs=3, default=[0.485, 0.456, 0.406])
    c.add_argument("--std", type=float, nargs=3, default=[0.229, 0.224, 0.225])
    a = p.parse_args()
    if a.cmd == "detector":
        export_detector(a.pt, a.out, a.imgsz)
    else:
        export_classifier(a.pt, a.out, tuple(a.size), a.classes, a.mean, a.std)


if __name__ == "__main__":
    main()
