//! 入力担当に映像と様子を渡す HTTP（std だけ。1つの接続に1つの要求、HTTP/1.0）。

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::time::Duration;

use janscope_video::sandbox::{MediaStatus, SandboxConfig, SourceStatus};

use crate::{State, now};

pub fn serve(l: TcpListener, st: Arc<State>) {
    for c in l.incoming().flatten() {
        let st = st.clone();
        std::thread::spawn(move || {
            let _ = handle(c, &st);
        });
    }
}

fn respond(s: &mut TcpStream, code: u16, ctype: &str, body: &[u8]) -> std::io::Result<()> {
    let reason = match code {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Error",
    };
    write!(
        s,
        "HTTP/1.0 {code} {reason}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    s.write_all(body)
}

fn json<T: serde::Serialize>(s: &mut TcpStream, v: &T) -> std::io::Result<()> {
    let b = serde_json::to_vec(v).unwrap_or_default();
    respond(s, 200, "application/json; charset=utf-8", &b)
}

/// 今の様子。入力担当の中のメディアなら、メディアサーバー（RTMP）の様子も合わせて出す
pub fn status(st: &State) -> MediaStatus {
    let rs = st.rtmp.as_ref().map(|r| r.status());
    let cfg = st.config.lock().unwrap().clone();
    let errors = st.errors.lock().unwrap().clone();
    let origin = if st.rtmp.is_some() {
        "メディアサーバー"
    } else {
        "この PC"
    };
    let mut sources: Vec<SourceStatus> = st
        .all_sources()
        .into_iter()
        .map(|s| {
            let bus = st.bus(&format!("preview/{}", s.id));
            let pubi = match (&s.kind, &rs) {
                (janscope_video::sandbox::SourceKind::Rtmp { app, key }, Some(rs)) => rs
                    .publishers
                    .iter()
                    .find(|p| p.app == *app && p.key == *key)
                    .cloned(),
                _ => None,
            };
            SourceStatus {
                configured: cfg.sources.iter().any(|c| c.id == s.id),
                id: s.id.clone(),
                name: s.name.clone(),
                kind: s.kind.label().into(),
                live: s.enabled && bus.live(),
                fps: pubi.as_ref().and_then(|p| p.fps).unwrap_or(bus.fps()),
                width: pubi.as_ref().and_then(|p| p.width),
                height: pubi.as_ref().and_then(|p| p.height),
                error: errors.get(&s.id).cloned(),
                enabled: s.enabled,
                origin: origin.into(),
                detail: s.kind.detail(),
            }
        })
        .collect();
    let mut config = cfg;
    let upstream = st.upstream.lock().unwrap().clone();
    let mut upstream_error = None;
    if st.rtmp.is_none() && upstream.is_some() {
        match st.upstream_status.lock().unwrap().clone() {
            Some(Ok(u)) => {
                sources.extend(u.sources);
                config.sources.extend(u.config.sources);
                config.regions.extend(u.config.regions);
            }
            Some(Err(e)) => upstream_error = Some(e),
            None => {}
        }
    }
    MediaStatus {
        version: env!("CARGO_PKG_VERSION").into(),
        upstream,
        upstream_error,
        rtmp_port: rs.as_ref().map(|r| r.port).unwrap_or(0),
        rtmp_error: rs.and_then(|r| r.bind_error),
        sources,
        config,
    }
}

/// 入力担当の中のメディア: メディアサーバーの様子を1秒ごとに取り直す
pub fn poll_upstream(st: Arc<State>) {
    loop {
        let up = st.upstream.lock().unwrap().clone();
        let r = up.map(|u| {
            janscope_video::http::get(
                &format!("{}/status", u.trim_end_matches('/')),
                Duration::from_millis(1500),
            )
            .and_then(|b| serde_json::from_slice::<MediaStatus>(&b).map_err(|e| e.to_string()))
        });
        *st.upstream_status.lock().unwrap() = r;
        std::thread::sleep(Duration::from_secs(1));
    }
}

/// 設定を保存する。入力担当の中のメディアなら、メディアサーバーのソースの分はメディアサーバーへ送る
fn save_config(st: &State, mut c: SandboxConfig) -> Result<(), String> {
    c.normalize();
    let upstream = st.upstream.lock().unwrap().clone();
    let (Some(up), None) = (upstream, &st.rtmp) else {
        return st.set_config(c);
    };
    let up_ids: Vec<String> = st
        .upstream_ok()
        .map(|u| u.sources.iter().map(|s| s.id.clone()).collect())
        .unwrap_or_default();
    let is_up = |id: &str, kind: Option<&janscope_video::sandbox::SourceKind>| {
        up_ids.iter().any(|x| x == id)
            || matches!(kind, Some(janscope_video::sandbox::SourceKind::Rtmp { .. }))
    };
    let (up_sources, local_sources): (Vec<_>, Vec<_>) = c
        .sources
        .into_iter()
        .partition(|s| is_up(&s.id, Some(&s.kind)));
    let (up_regions, local_regions): (Vec<_>, Vec<_>) =
        c.regions.into_iter().partition(|r| is_up(&r.source, None));
    let mine = SandboxConfig {
        sources: local_sources,
        regions: local_regions,
    };
    st.set_config(mine)?;
    if st.upstream_ok().is_some() {
        let theirs = SandboxConfig {
            sources: up_sources,
            regions: up_regions,
        };
        let body = serde_json::to_vec(&theirs).map_err(|e| e.to_string())?;
        janscope_video::http::put_json(
            &format!("{}/config", up.trim_end_matches('/')),
            &body,
            Duration::from_secs(5),
        )
        .map_err(|e| format!("メディアサーバーに保存できません: {e}"))?;
    }
    Ok(())
}

/// メディアサーバーへ取り次ぐ（入力担当の中のメディアで、ここにない映像）
fn proxy(s: &mut TcpStream, url: &str, stream: bool) -> std::io::Result<()> {
    if !stream {
        return match janscope_video::http::get(url, Duration::from_secs(3)) {
            Ok(b) => respond(s, 200, "image/jpeg", &b),
            Err(e) => respond(s, 404, "text/plain; charset=utf-8", e.as_bytes()),
        };
    }
    let mut r = match janscope_video::http::stream(url, Duration::from_secs(10)) {
        Ok(r) => r,
        Err(e) => return respond(s, 502, "text/plain; charset=utf-8", e.as_bytes()),
    };
    write!(
        s,
        "HTTP/1.0 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=frame\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n"
    )?;
    let mut buf = vec![0u8; 128 * 1024];
    loop {
        let n = r.read(&mut buf)?;
        if n == 0 {
            return Ok(());
        }
        s.write_all(&buf[..n])?;
    }
}

/// Webカメラの一覧（Windows は ffmpeg の DirectShow で調べる）
pub fn devices(st: &State) -> Vec<String> {
    if cfg!(windows) {
        let ffmpeg = st.ffmpeg.lock().unwrap().clone();
        let mut cmd = std::process::Command::new(&ffmpeg);
        cmd.args([
            "-hide_banner",
            "-list_devices",
            "true",
            "-f",
            "dshow",
            "-i",
            "dummy",
        ]);
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x0800_0000);
        }
        let Ok(out) = cmd.output() else {
            return vec![];
        };
        let text = String::from_utf8_lossy(&out.stderr).into_owned();
        parse_dshow(&text)
    } else {
        let mut v: Vec<String> = std::fs::read_dir("/dev")
            .map(|d| {
                d.flatten()
                    .map(|e| e.path().to_string_lossy().into_owned())
                    .filter(|p| p.starts_with("/dev/video"))
                    .collect()
            })
            .unwrap_or_default();
        v.sort();
        v
    }
}

/// `[dshow @ ...] "名前" (video)` の行から名前を取り出す
pub fn parse_dshow(text: &str) -> Vec<String> {
    text.lines()
        .filter(|l| l.contains("(video)"))
        .filter_map(|l| {
            let a = l.find('"')?;
            let b = l[a + 1..].find('"')? + a + 1;
            Some(l[a + 1..b].to_string())
        })
        .collect()
}

fn handle(mut s: TcpStream, st: &State) -> std::io::Result<()> {
    s.set_read_timeout(Some(Duration::from_secs(5)))?;
    let mut r = BufReader::new(s.try_clone()?);
    let mut line = String::new();
    r.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts
        .next()
        .unwrap_or("/")
        .split('?')
        .next()
        .unwrap_or("/")
        .to_string();
    let mut len = 0usize;
    loop {
        let mut h = String::new();
        if r.read_line(&mut h)? == 0 || h == "\r\n" || h == "\n" {
            break;
        }
        if let Some(v) = h.to_ascii_lowercase().strip_prefix("content-length:") {
            len = v.trim().parse().unwrap_or(0);
        }
    }
    let mut body = vec![0u8; len.min(4 * 1024 * 1024)];
    r.read_exact(&mut body)?;
    let seg: Vec<&str> = path.trim_start_matches('/').split('/').collect();
    match (method.as_str(), seg.as_slice()) {
        ("GET", ["status"]) => json(&mut s, &status(st)),
        ("GET", ["config"]) => json(&mut s, &*st.config.lock().unwrap()),
        ("PUT", ["config"]) => match serde_json::from_slice::<SandboxConfig>(&body) {
            Ok(c) => match save_config(st, c) {
                Ok(()) => {
                    println!("{} 設定を変えました（枠・ソース）", now());
                    json(&mut s, &status(st))
                }
                Err(e) => respond(&mut s, 500, "text/plain; charset=utf-8", e.as_bytes()),
            },
            Err(e) => respond(
                &mut s,
                400,
                "text/plain; charset=utf-8",
                format!("設定を読めません: {e}").as_bytes(),
            ),
        },
        ("GET", ["devices"]) => json(&mut s, &devices(st)),
        ("GET", [kind @ ("preview" | "snapshot" | "region"), file]) => {
            let (name, ext) = file.rsplit_once('.').unwrap_or((file, ""));
            // 入力担当の中のメディアで、ここにない映像はメディアサーバーへ取り次ぐ
            if st.rtmp.is_none() {
                let here = match *kind {
                    "region" => st.is_local_region(name),
                    _ => st.is_local_source(name),
                };
                let up = st.upstream.lock().unwrap().clone();
                if let (false, Some(up)) = (here, up) {
                    let url = format!("{}/{kind}/{file}", up.trim_end_matches('/'));
                    return proxy(&mut s, &url, ext == "mjpg");
                }
            }
            let bus_name = match *kind {
                "region" => format!("region/{name}"),
                _ => format!("preview/{name}"),
            };
            let bus = st.bus(&bus_name);
            if ext == "mjpg" {
                stream(&mut s, &bus)
            } else {
                match bus.latest() {
                    Some((_, f)) => respond(&mut s, 200, "image/jpeg", &f),
                    None => respond(
                        &mut s,
                        404,
                        "text/plain; charset=utf-8",
                        "まだ映像がありません".as_bytes(),
                    ),
                }
            }
        }
        ("GET", [""]) => respond(
            &mut s,
            200,
            "text/plain; charset=utf-8",
            "JanScope メディアサーバー（/status /config /preview/<ソース>.mjpg /region/<枠>.mjpg）"
                .as_bytes(),
        ),
        _ => respond(&mut s, 404, "text/plain; charset=utf-8", b"not found"),
    }
}

/// MJPEG の流れ（multipart/x-mixed-replace）。相手が切るまで
fn stream(s: &mut TcpStream, bus: &crate::Bus) -> std::io::Result<()> {
    write!(
        s,
        "HTTP/1.0 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=frame\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n"
    )?;
    s.set_write_timeout(Some(Duration::from_secs(5)))?;
    let mut seq = 0;
    if let Some((q, f)) = bus.latest() {
        seq = q;
        part(s, &f)?;
    }
    loop {
        if let Some((q, f)) = bus.wait_newer(seq, Duration::from_secs(5)) {
            seq = q;
            part(s, &f)?;
        } else {
            // 映像が止まっていても、相手が切ったか確かめる
            s.write_all(b"\r\n")?;
        }
    }
}

fn part(s: &mut TcpStream, f: &[u8]) -> std::io::Result<()> {
    write!(
        s,
        "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
        f.len()
    )?;
    s.write_all(f)?;
    s.write_all(b"\r\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use janscope_video::sandbox::{Region, RegionRole, SourceDef, SourceKind};

    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("janscope-media-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d.join("sandbox.json")
    }

    /// 入力担当の中のメディアで保存すると、メディアサーバーのソースの枠はメディアサーバーへ、この PC の分はこの PC に
    #[test]
    fn local_save_splits_to_upstream() {
        let remote = State::new(0, "ffmpeg-none".into(), tmp("remote"));
        remote
            .set_config(SandboxConfig {
                sources: vec![SourceDef {
                    id: "url-9".into(),
                    name: "遠く".into(),
                    kind: SourceKind::Url {
                        url: "rtmp://example/x".into(),
                    },
                    enabled: false,
                }],
                regions: vec![],
            })
            .unwrap();
        let l = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let addr = l.local_addr().unwrap();
        {
            let r = remote.clone();
            std::thread::spawn(move || serve(l, r));
        }
        let local = State::new_local("ffmpeg-none".into(), tmp("local"));
        *local.upstream.lock().unwrap() = Some(format!("http://{addr}"));
        let up =
            janscope_video::http::get(&format!("http://{addr}/status"), Duration::from_secs(2))
                .unwrap();
        *local.upstream_status.lock().unwrap() = Some(Ok(serde_json::from_slice(&up).unwrap()));
        // 合わせた一覧には両方が出る
        let st = status(&local);
        assert!(
            st.sources
                .iter()
                .any(|s| s.id == "url-9" && s.origin == "メディアサーバー" && !s.enabled)
        );
        let merged = SandboxConfig {
            sources: vec![
                SourceDef {
                    id: "img-1".into(),
                    name: "画像".into(),
                    kind: SourceKind::Image {
                        path: "a.png".into(),
                    },
                    enabled: true,
                },
                st.config
                    .sources
                    .iter()
                    .find(|s| s.id == "url-9")
                    .unwrap()
                    .clone(),
            ],
            regions: vec![
                Region {
                    source: "img-1".into(),
                    role: RegionRole::Face(0),
                    x: 0.0,
                    y: 0.0,
                    w: 0.5,
                    h: 0.5,
                },
                Region {
                    source: "url-9".into(),
                    role: RegionRole::Face(1),
                    x: 0.5,
                    y: 0.0,
                    w: 0.5,
                    h: 0.5,
                },
            ],
        };
        save_config(&local, merged).unwrap();
        let lc = local.config.lock().unwrap().clone();
        assert_eq!(lc.sources.len(), 1);
        assert_eq!(lc.regions.len(), 1);
        assert_eq!(lc.regions[0].role, RegionRole::Face(0));
        let rc = remote.config.lock().unwrap().clone();
        assert_eq!(rc.sources.len(), 1);
        assert_eq!(rc.regions.len(), 1);
        assert_eq!(rc.regions[0].role, RegionRole::Face(1));
        assert!(local.is_local_region("face-A") && !local.is_local_region("face-B"));
    }

    #[test]
    fn dshow_names() {
        let t = "[dshow @ 0x1] \"USB Video\" (video)\n[dshow @ 0x1]   Alternative name \"@device\"\n[dshow @ 0x1] \"Mic\" (audio)\n";
        assert_eq!(super::parse_dshow(t), vec!["USB Video".to_string()]);
    }
}
