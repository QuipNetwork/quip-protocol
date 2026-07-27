//! REST dashboard endpoint (agm.4).
//!
//! Static file serving of the per-qblock attempt logs the [`crate::attempt`]
//! writer appends, plus a small index of available qblocks. No query or DB
//! work: `tower_http::ServeDir` serves the JSONL files directly, so the
//! coordinator carries no request-time overhead beyond a file read.

use axum::{extract::State, routing::get, Json, Router};
use std::path::PathBuf;
use tower_http::services::ServeDir;

/// Build the dashboard router rooted at `data_dir`:
/// - `GET /qblocks` → JSON array of available qblock ids (directory names)
/// - `GET /healthz` → `ok`
/// - everything else → static files under `data_dir`, e.g.
///   `GET /<qblock_id>/attempts.jsonl`.
pub fn router(data_dir: PathBuf) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/qblocks", get(list_qblocks))
        .fallback_service(ServeDir::new(&data_dir))
        .with_state(data_dir)
}

async fn healthz() -> &'static str {
    "ok"
}

/// List the qblock directories under the data root (sorted). Returns an empty
/// list if the root doesn't exist yet.
async fn list_qblocks(State(data_dir): State<PathBuf>) -> Json<Vec<String>> {
    let mut ids = Vec::new();
    if let Ok(mut rd) = tokio::fs::read_dir(&data_dir).await {
        while let Ok(Some(entry)) = rd.next_entry().await {
            let is_dir = entry.file_type().await.is_ok_and(|t| t.is_dir());
            if is_dir {
                if let Some(name) = entry.file_name().to_str() {
                    ids.push(name.to_string());
                }
            }
        }
    }
    ids.sort();
    Json(ids)
}

/// Serve the dashboard on `listen` (e.g. `127.0.0.1:9090`) until the task is
/// aborted. A bind failure is logged and the task exits without taking down the
/// coordinator.
pub async fn serve(listen: String, data_dir: PathBuf) {
    let listener = match tokio::net::TcpListener::bind(&listen).await {
        Ok(l) => l,
        Err(e) => {
            tracing::warn!("dashboard: bind {listen} failed: {e}");
            return;
        }
    };
    tracing::info!(
        "dashboard: serving {} at http://{listen}",
        data_dir.display()
    );
    if let Err(e) = axum::serve(listener, router(data_dir)).await {
        tracing::warn!("dashboard: server error: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt; // oneshot

    #[tokio::test]
    async fn serves_attempts_file_and_lists_qblocks() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("42")).unwrap();
        std::fs::write(
            tmp.path().join("42").join("attempts.jsonl"),
            "{\"job_id\":\"ab\"}\n",
        )
        .unwrap();

        let app = router(tmp.path().to_path_buf());

        // File download.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/42/attempts.jsonl")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        assert!(String::from_utf8_lossy(&body).contains("\"job_id\":\"ab\""));

        // Index lists the qblock dir.
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/qblocks")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        let ids: Vec<String> = serde_json::from_slice(&body).unwrap();
        assert_eq!(ids, vec!["42".to_string()]);
    }
}
