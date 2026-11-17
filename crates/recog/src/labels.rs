//! モデルのクラス名 → 牌。学習ツール（training/janscope_train/tiles.py）の別名と同じものを受け付ける。

use janscope_core::tile::Tile;

/// 認識した1枚の中身
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Label {
    Tile(Tile),
    /// 裏向き
    Back,
    /// 牌でないもの・分からない名前（検出だけのモデルの「牌」など）
    Other,
}

const HONOR_JA: [&str; 7] = ["東", "南", "西", "北", "白", "發", "中"];
const HONOR_EN: [&str; 7] = ["east", "south", "west", "north", "white", "green", "red"];
const HONOR_FLUFFY: [&str; 7] = ["ton", "nan", "shaa", "pei", "haku", "hatsu", "chun"];
const NUM_JA: [&str; 9] = ["一", "二", "三", "四", "五", "六", "七", "八", "九"];

/// クラス名を牌にする（大文字・小文字は区別しない）
pub fn label_of(name: &str) -> Label {
    let k = name.trim().to_lowercase();
    let k = k.as_str();
    if matches!(
        k,
        "back" | "ura" | "裏" | "facedown" | "face_down" | "hidden" | "unknown_back"
    ) {
        return Label::Back;
    }
    // 1m 0p 7z など（正規の表記）
    if let Ok(t) = k.parse::<Tile>() {
        if t != Tile::UNKNOWN {
            return Label::Tile(t);
        }
    }
    // 5mr r5m 5m-red 5m_red
    for (s, c) in [("m", 'm'), ("p", 'p'), ("s", 's')] {
        if [
            format!("5{s}r"),
            format!("r5{s}"),
            format!("5{s}-red"),
            format!("5{s}_red"),
            format!("0{s}"),
        ]
        .iter()
        .any(|a| a == k)
        {
            return Label::Tile(format!("0{c}").parse().unwrap());
        }
    }
    // 字牌の名前
    for i in 0..7 {
        if k == HONOR_JA[i] || k == HONOR_EN[i] || k == HONOR_FLUFFY[i] || (i == 5 && k == "発") {
            return Label::Tile(format!("{}z", i + 1).parse().unwrap());
        }
    }
    // Man1 Pin5-Dora Sou9（FluffyStuff）・m1 p5 s9・1萬 一萬 1筒 1索
    let fluffy = [
        ("man", 'm'),
        ("pin", 'p'),
        ("sou", 's'),
        ("so", 's'),
        ("bamboo", 's'),
        ("dot", 'p'),
        ("dots", 'p'),
        ("character", 'm'),
        ("characters", 'm'),
        ("crak", 'm'),
        ("bam", 's'),
    ];
    let strip_red = |r: &str| -> (String, bool) {
        for suf in ["-dora", "_dora", "-red", "_red", "r", "-aka", "_aka"] {
            if let Some(x) = r.strip_suffix(suf) {
                return (x.to_string(), true);
            }
        }
        (r.to_string(), false)
    };
    for (word, c) in fluffy {
        // 例: man1 / man_1 / man-1 / 1man
        for rest in [k.strip_prefix(word), k.strip_suffix(word)]
            .into_iter()
            .flatten()
        {
            let (rest, red) =
                strip_red(rest.trim_matches(|ch| ch == '_' || ch == '-' || ch == ' '));
            if let Ok(n) = rest.parse::<u8>() {
                if (1..=9).contains(&n) {
                    let n = if red && n == 5 { 0 } else { n };
                    return Label::Tile(format!("{n}{c}").parse().unwrap());
                }
            }
        }
    }
    // m1 p5 s9（記号が先）
    if k.len() == 2 {
        let b = k.as_bytes();
        if matches!(b[0], b'm' | b'p' | b's') && b[1].is_ascii_digit() {
            if let Ok(t) = format!("{}{}", b[1] as char, b[0] as char).parse::<Tile>() {
                return Label::Tile(t);
            }
        }
    }
    // 漢字（1萬・一萬・1万・5筒赤・9索）
    for (suits, c) in [
        (&["萬", "万", "m"][..], 'm'),
        (&["筒", "餅", "p"][..], 'p'),
        (&["索", "条", "s"][..], 's'),
    ] {
        for su in suits {
            if let Some(head) = k
                .strip_suffix(su)
                .or_else(|| k.strip_suffix(&format!("{su}赤")).map(|h| h))
            {
                let red = k.ends_with("赤") || k.starts_with("赤");
                let head = head.trim_start_matches('赤');
                let n = head
                    .parse::<u8>()
                    .ok()
                    .or_else(|| NUM_JA.iter().position(|j| *j == head).map(|i| i as u8 + 1));
                if let Some(n) = n.filter(|n| (1..=9).contains(n)) {
                    let n = if red && n == 5 { 0 } else { n };
                    return Label::Tile(format!("{n}{c}").parse().unwrap());
                }
            }
        }
    }
    Label::Other
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(s: &str) -> Label {
        Label::Tile(s.parse().unwrap())
    }

    #[test]
    fn names() {
        assert_eq!(label_of("1m"), t("1m"));
        assert_eq!(label_of("0p"), t("0p"));
        assert_eq!(label_of("5sr"), t("0s"));
        assert_eq!(label_of("Man1"), t("1m"));
        assert_eq!(label_of("Pin5-Dora"), t("0p"));
        assert_eq!(label_of("Sou9"), t("9s"));
        assert_eq!(label_of("Chun"), t("7z"));
        assert_eq!(label_of("中"), t("7z"));
        assert_eq!(label_of("発"), t("6z"));
        assert_eq!(label_of("north"), t("4z"));
        assert_eq!(label_of("一萬"), t("1m"));
        assert_eq!(label_of("5筒"), t("5p"));
        assert_eq!(label_of("赤5索"), t("0s"));
        assert_eq!(label_of("p3"), t("3p"));
        assert_eq!(label_of("back"), Label::Back);
        assert_eq!(label_of("tile"), Label::Other);
    }
}
