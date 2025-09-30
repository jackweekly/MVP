use crate::layered::meta::GeneticHyperParams;
use crate::{Problem, ProblemContext, SolutionState, Vehicle};
use blake3::Hasher;
use serde::{Deserialize, Serialize};
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize)]
pub struct LoggerConfig {
    pub output_path: PathBuf,
    pub format: LogFormat,
    pub flush_interval: usize,
}

#[derive(Clone, Debug, Serialize)]
pub enum LogFormat {
    JsonLines,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RunEvent {
    RunStart {
        instance_fingerprint: String,
        vehicles: usize,
        customers: usize,
        config: crate::HybridConfig,
        initial_state: LoggedState,
    },
    Iteration {
        iteration: usize,
        temperature: f64,
        acceptance_probability: f64,
        current_cost: f64,
        best_cost: f64,
    },
    MoveAccepted {
        iteration: usize,
        operator: MoveKind,
        delta_cost: f64,
        current_cost: f64,
        best_cost: f64,
        state: LoggedState,
    },
    LayeredPhase {
        cluster_index: usize,
        heuristic_scale: f64,
        layer_penalty: f64,
        ga_params: GeneticHyperParams,
        best_cost: f64,
    },
    RunComplete {
        iterations_run: usize,
        best_cost: f64,
        runtime_ms: Option<u128>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MoveKind {
    TwoOpt {
        route_idx: usize,
        i: usize,
        j: usize,
    },
    OrOpt {
        from_route: usize,
        to_route: usize,
        start: usize,
        segment_len: usize,
        insert_pos: usize,
    },
    Swap {
        route_a: usize,
        pos_a: usize,
        route_b: usize,
        pos_b: usize,
    },
    Shuffle {
        route_idx: usize,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoggedRoute {
    pub vehicle_id: String,
    pub customers: Vec<String>,
    pub load: u32,
    pub distance: f64,
    pub duration: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoggedState {
    pub total_distance: f64,
    pub total_duration: f64,
    pub unassigned: Vec<String>,
    pub routes: Vec<LoggedRoute>,
}

pub struct RunRecorder {
    config: LoggerConfig,
    writer: BufWriter<File>,
    events_written: usize,
}

impl LoggerConfig {
    pub fn ensure_writer(&self) -> std::io::Result<RunRecorder> {
        if let Some(parent) = self.output_path.parent() {
            if !parent.as_os_str().is_empty() {
                create_dir_all(parent)?;
            }
        }
        let file = File::create(&self.output_path)?;
        let writer = BufWriter::new(file);
        Ok(RunRecorder {
            config: self.clone(),
            writer,
            events_written: 0,
        })
    }
}

impl RunRecorder {
    pub fn record_event(&mut self, event: &RunEvent) {
        match self.config.format {
            LogFormat::JsonLines => {
                if serde_json::to_writer(&mut self.writer, event).is_ok() {
                    let _ = self.writer.write_all(b"\n");
                    self.events_written += 1;
                    if self.events_written % self.config.flush_interval == 0 {
                        let _ = self.writer.flush();
                    }
                }
            }
        }
    }

    pub fn finalize(&mut self) {
        let _ = self.writer.flush();
    }
}

impl Drop for RunRecorder {
    fn drop(&mut self) {
        let _ = self.writer.flush();
    }
}

pub fn fingerprint_problem(problem: &Problem) -> String {
    let mut hasher = Hasher::new();
    if let Ok(serialized) = serde_json::to_vec(problem) {
        hasher.update(&serialized);
    }
    hasher.finalize().to_hex().to_string()
}

pub fn snapshot_state(
    state: &SolutionState,
    ctx: &ProblemContext,
    vehicles: &[Vehicle],
) -> LoggedState {
    let routes = state
        .routes
        .iter()
        .zip(vehicles.iter())
        .map(|(route_state, vehicle)| LoggedRoute {
            vehicle_id: vehicle.id.clone(),
            customers: route_state
                .customers
                .iter()
                .map(|idx| ctx.id_for(*idx).to_string())
                .collect(),
            load: route_state.load,
            distance: route_state.distance,
            duration: route_state.duration,
        })
        .collect();

    let unassigned = state
        .unassigned
        .iter()
        .map(|idx| ctx.id_for(*idx).to_string())
        .collect();

    LoggedState {
        total_distance: state.total_distance,
        total_duration: state.total_duration,
        unassigned,
        routes,
    }
}

pub fn default_logger_path(base: impl AsRef<Path>, fingerprint: &str) -> PathBuf {
    let mut path = base.as_ref().to_path_buf();
    let filename = format!("run_{}.jsonl", fingerprint);
    path.push(filename);
    path
}
