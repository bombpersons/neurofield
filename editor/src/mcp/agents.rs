//! Letting agents join over HTTP: the server switched on and off from the Agents
//! window (Tools > Agents) or `--mcp-listen`, and the window that shows how to
//! connect.

use super::{EditorServer, Servers};
use crate::ui::Ui;
use bevy::prelude::*;
use bevy_egui::egui;
use std::net::{Ipv4Addr, SocketAddr};
use std::time::Instant;
use tokio_util::sync::CancellationToken;

/// What the Agents window shows and sets; the server follows `on`, `addr` and `token`.
pub(crate) struct Agents {
    pub window: bool,
    pub on: bool,
    pub addr: SocketAddr,
    pub token: String,
    /// Where the server is listening, while it is.
    pub listening: Option<SocketAddr>,
    /// Why it isn't, when starting it failed.
    pub error: Option<String>,
    /// When an agent last asked something over HTTP.
    pub last_request: Option<Instant>,
}

impl Agents {
    /// From the settings, with a token made (and kept) the first time.
    pub fn new(settings: &mut crate::settings::Settings) -> Agents {
        if settings.agents_token.is_empty() {
            settings.agents_token = super::new_token();
            settings.save();
        }
        Agents {
            window: false,
            on: settings.agents_on,
            addr: (Ipv4Addr::LOCALHOST, settings.agents_port).into(),
            token: settings.agents_token.clone(),
            listening: None,
            error: None,
            last_request: None,
        }
    }

    fn url(&self) -> String {
        format!("http://{}/mcp", self.listening.unwrap_or(self.addr))
    }

    fn claude_command(&self) -> String {
        format!("claude mcp add --transport http neurofield-editor {} --header \"Authorization: Bearer {}\"", self.url(), self.token)
    }

    /// An agent asked something in the last few seconds.
    pub fn active(&self) -> bool {
        self.last_request.is_some_and(|t| t.elapsed().as_secs_f32() < 5.0)
    }
}

pub(super) struct Http {
    /// As asked for (`listening` has the port actually bound).
    addr: SocketAddr,
    token: String,
    stop: CancellationToken,
}

/// Start, stop or restart the HTTP server to match the Agents window.
pub(super) fn switch(mut ui: ResMut<Ui>, mut servers: ResMut<Servers>) {
    let a = &mut ui.agents;
    let want = a.on.then(|| (a.addr, a.token.clone()));
    if want == servers.http.as_ref().map(|h| (h.addr, h.token.clone())) {
        return;
    }
    if let Some(h) = servers.http.take() {
        h.stop.cancel();
        a.listening = None;
        eprintln!("editor MCP: stopped serving http://{}/mcp", h.addr);
    }
    a.error = None;
    let Some((addr, token)) = want else { return };
    match start(&mut servers, addr, token.clone()) {
        Ok((bound, stop)) => {
            a.listening = Some(bound);
            servers.http = Some(Http { addr, token, stop });
            eprintln!("editor MCP: http://{bound}/mcp (header \"Authorization: Bearer {}\")", a.token);
        }
        Err(e) => {
            eprintln!("editor MCP: {e}");
            a.error = Some(e);
            a.on = false;
        }
    }
}

/// Bind here (so a port in use is reported at once) and serve on the runtime.
fn start(servers: &mut Servers, addr: SocketAddr, token: String) -> Result<(SocketAddr, CancellationToken), String> {
    if !addr.ip().is_loopback() {
        return Err(format!("{addr} isn't a local address"));
    }
    let listener = std::net::TcpListener::bind(addr).map_err(|e| format!("can't listen on {addr}: {e}"))?;
    let bound = listener.local_addr().map_err(|e| e.to_string())?;
    listener.set_nonblocking(true).map_err(|e| e.to_string())?;
    let server = EditorServer { sender: servers.sender.clone(), slots: servers.slots.clone(), testing: false, remote: true };
    let runtime = servers.runtime();
    let _entered = runtime.enter();
    let listener = tokio::net::TcpListener::from_std(listener).map_err(|e| e.to_string())?;
    let stop = CancellationToken::new();
    runtime.spawn(serve(listener, bound.port(), token, server, stop.clone()));
    Ok((bound, stop))
}

async fn serve(listener: tokio::net::TcpListener, port: u16, token: String, server: EditorServer, stop: CancellationToken) {
    use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
    use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService};
    // Host names a browser could be tricked into sending (DNS rebinding) are refused.
    let hosts = ["localhost", "127.0.0.1", "[::1]"].map(|h| format!("{h}:{port}"));
    // Stopping ends the sessions' event streams too, so shutting down doesn't wait on them.
    let config = StreamableHttpServerConfig::default().with_allowed_hosts(hosts).with_cancellation_token(stop.clone());
    let service = StreamableHttpService::new(move || Ok(server.clone()), LocalSessionManager::default().into(), config);
    let wanted = format!("Bearer {token}");
    let router = axum::Router::new().nest_service("/mcp", service).layer(axum::middleware::from_fn(
        move |request: axum::extract::Request, next: axum::middleware::Next| {
            let ok = request.headers().get("authorization").and_then(|v| v.to_str().ok()) == Some(wanted.as_str());
            async move {
                if ok {
                    next.run(request).await
                } else {
                    axum::response::IntoResponse::into_response((axum::http::StatusCode::UNAUTHORIZED, "bearer token needed"))
                }
            }
        },
    ));
    if let Err(e) = axum::serve(listener, router).with_graceful_shutdown(stop.cancelled_owned()).await {
        eprintln!("editor MCP over HTTP: {e}");
    }
}

/// Tools > Agents: switch the server on and off, and how to connect to it.
pub(crate) fn window(ctx: &egui::Context, s: &mut Ui) {
    if !s.agents.window {
        return;
    }
    let mut open = true;
    let mut changed = false;
    egui::Window::new("Agents").open(&mut open).collapsible(false).resizable(false).default_width(420.0).show(ctx, |ui| {
        let a = &mut s.agents;
        ui.label("Let AI agents (Claude Code, say) read this level and build on it while you work. Each change they make is one undo step, and they never take over your mouse.");
        ui.add_space(4.0);
        changed |= ui.checkbox(&mut a.on, "Let agents connect").changed();
        let (text, colour) = match (&a.listening, &a.error) {
            (Some(_), _) if a.active() => ("An agent is working".to_string(), egui::Color32::from_rgb(120, 220, 120)),
            (Some(addr), _) => (format!("Listening on {addr}, for this computer only"), ui.visuals().text_color()),
            (None, Some(e)) => (e.clone(), ui.visuals().error_fg_color),
            (None, None) => ("Off".to_string(), ui.visuals().weak_text_color()),
        };
        ui.colored_label(colour, text);
        ui.separator();
        let mut port = a.addr.port();
        egui::Grid::new("agents").num_columns(3).spacing([8.0, 4.0]).show(ui, |ui| {
            ui.label("Port");
            if ui.add(egui::DragValue::new(&mut port).range(1024..=65535)).changed() {
                a.addr.set_port(port);
                changed = true;
            }
            ui.end_row();
            let url = a.url();
            ui.label("Address");
            ui.monospace(&url);
            if ui.small_button("Copy").clicked() {
                ui.ctx().copy_text(url);
            }
            ui.end_row();
            ui.label("Token");
            ui.monospace(&a.token);
            ui.horizontal(|ui| {
                if ui.small_button("Copy").clicked() {
                    ui.ctx().copy_text(a.token.clone());
                }
                if ui.small_button("New").on_hover_text("Agents with the old token are disconnected").clicked() {
                    a.token = super::new_token();
                    changed = true;
                }
            });
            ui.end_row();
        });
        ui.separator();
        ui.label("To add the editor to Claude Code:");
        let command = a.claude_command();
        ui.add(egui::Label::new(egui::RichText::new(&command).monospace().size(10.0)).wrap());
        if ui.button("Copy command").clicked() {
            ui.ctx().copy_text(command);
        }
    });
    s.agents.window = open;
    if changed {
        let a = &s.agents;
        s.settings.agents_on = a.on;
        s.settings.agents_port = a.addr.port();
        s.settings.agents_token = a.token.clone();
        s.settings.save();
    }
}
