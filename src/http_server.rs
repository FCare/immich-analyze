use axum::{Json, Router, extract::State, routing::get, routing::post};
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

/// Shared flag so a second /trigger call while a feedback loop is already
/// running just reports back instead of starting an overlapping run.
#[derive(Default)]
pub struct TriggerState {
    running: AtomicBool,
}

impl TriggerState {
    pub fn try_start(&self) -> bool {
        self.running
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    }

    pub fn finish(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}

#[derive(Clone)]
struct AppState {
    trigger_state: Arc<TriggerState>,
    trigger_tx: tokio::sync::mpsc::Sender<()>,
}

/// Serves POST /trigger (fire-and-forget: asks the feedback loop to run, no-op
/// if one is already in progress) and GET /status (is a run currently active).
/// Internal-only: no auth, meant to be reached from the "ansible" Docker network
/// (e.g. by the family-graph web UI), never exposed publicly.
pub async fn serve(port: u16, trigger_state: Arc<TriggerState>, trigger_tx: tokio::sync::mpsc::Sender<()>) {
    let state = AppState { trigger_state, trigger_tx };
    let app = Router::new()
        .route("/trigger", post(trigger_handler))
        .route("/status", get(status_handler))
        .with_state(state);

    let listener = match tokio::net::TcpListener::bind(("0.0.0.0", port)).await {
        Ok(listener) => listener,
        Err(e) => {
            eprintln!("Failed to bind trigger HTTP server on port {}: {}", port, e);
            return;
        }
    };
    println!("Trigger HTTP server listening on 0.0.0.0:{}", port);
    if let Err(e) = axum::serve(listener, app).await {
        eprintln!("Trigger HTTP server stopped: {}", e);
    }
}

async fn trigger_handler(State(state): State<AppState>) -> Json<Value> {
    if state.trigger_state.is_running() {
        return Json(json!({"status": "already_running"}));
    }
    match state.trigger_tx.try_send(()) {
        Ok(()) => Json(json!({"status": "triggered"})),
        Err(_) => Json(json!({"status": "already_running"})),
    }
}

async fn status_handler(State(state): State<AppState>) -> Json<Value> {
    Json(json!({"running": state.trigger_state.is_running()}))
}
