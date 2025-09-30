#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::time::{SystemTime, UNIX_EPOCH};
use tauri::{
    menu::{Menu, MenuItem, Submenu},
    AppHandle, Emitter, Runtime, State, Window,
};
use vrp_solver::{
    default_logger_path, fingerprint_problem, solve_vrp_with_callbacks, CancelCallback,
    HybridConfig, LogFormat, LoggerConfig, Problem, ProgressCallback, RunRecorder, Solution,
};

#[derive(Default)]
struct OptimizationState {
    cancel_flag: Mutex<Option<Arc<AtomicBool>>>,
}

#[derive(Deserialize)]
struct OptimizePayload {
    problem: Problem,
    config: Option<HybridConfig>,
    logging: Option<LoggingOptions>,
}

#[derive(Clone, Serialize)]
struct ProgressPayload {
    current: usize,
    total: usize,
}

#[derive(Deserialize)]
struct LoggingOptions {
    enabled: bool,
    path: Option<String>,
    flush_interval: Option<usize>,
    format: Option<String>,
}

#[tauri::command]
async fn optimize(
    window: Window,
    state: State<'_, OptimizationState>,
    payload: OptimizePayload,
) -> Result<Solution, String> {
    let OptimizePayload {
        problem,
        config,
        logging,
    } = payload;
    let config = config.unwrap_or_default();

    let cancel_flag = Arc::new(AtomicBool::new(false));
    {
        let mut guard = state.cancel_flag.lock().expect("cancel flag poisoned");
        *guard = Some(cancel_flag.clone());
    }

    let mut logger: Option<RunRecorder> = None;
    if let Some(logging) = logging {
        if logging.enabled {
            let fingerprint = fingerprint_problem(&problem);
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis();
            let base_path = logging
                .path
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("logs"));
            let output_path = if base_path
                .extension()
                .map(|ext| ext == "jsonl")
                .unwrap_or(false)
            {
                base_path
            } else {
                default_logger_path(&base_path, &format!("{}_{}", fingerprint, timestamp))
            };
            let format = match logging.format.as_deref() {
                Some("json") | Some("jsonl") | None => LogFormat::JsonLines,
                Some(other) => {
                    return Err(format!("Unsupported log format: {}", other));
                }
            };

            let logger_config = LoggerConfig {
                output_path: output_path.clone(),
                format,
                flush_interval: logging.flush_interval.unwrap_or(100),
            };

            logger = Some(
                logger_config
                    .ensure_writer()
                    .map_err(|err| format!("Failed to prepare log writer: {}", err))?,
            );

            let _ = window.emit(
                "optimization-log-path",
                output_path.to_string_lossy().to_string(),
            );
        }
    }

    let progress_cb: ProgressCallback = {
        let window = window.clone();
        Box::new(move |current: usize, total: usize| {
            let _ = window.emit("optimization-progress", ProgressPayload { current, total });
        })
    };

    let cancel_cb: CancelCallback = {
        let flag = cancel_flag.clone();
        Box::new(move || flag.load(Ordering::Relaxed))
    };

    let solution =
        solve_vrp_with_callbacks(problem, config, Some(progress_cb), Some(cancel_cb), logger).await;

    let cancelled = cancel_flag.load(Ordering::Relaxed);
    {
        let mut guard = state.cancel_flag.lock().expect("cancel flag poisoned");
        guard.take();
    }

    let event = if cancelled {
        "optimization-cancelled"
    } else {
        "optimization-done"
    };
    let _ = window.emit(event, ());

    Ok(solution)
}

#[tauri::command]
fn cancel_optimization(state: State<'_, OptimizationState>) {
    if let Some(flag) = state
        .cancel_flag
        .lock()
        .expect("cancel flag poisoned")
        .as_ref()
        .cloned()
    {
        flag.store(true, Ordering::Relaxed);
    }
}

fn create_app_menu<R: Runtime>(app_handle: &AppHandle<R>) -> tauri::Result<Menu<R>> {
    let quit_item = MenuItem::new(app_handle, "Quit", true, Some("CmdOrCtrl+Q"))?;
    let submenu = Submenu::with_items(app_handle, "File", true, &[&quit_item])?;
    let app_menu = Menu::with_items(app_handle, &[&submenu])?;
    Ok(app_menu)
}

fn main() {
    tauri::Builder::default()
        .manage(OptimizationState::default())
        .menu(create_app_menu)
        .on_menu_event(|app, event| {
            if event.id() == "Quit" {
                app.exit(0);
            }
        })
        .invoke_handler(tauri::generate_handler![optimize, cancel_optimization])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
