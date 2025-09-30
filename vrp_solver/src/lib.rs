use linfa::prelude::*;
use linfa_clustering::KMeans;
use ndarray::Array;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

mod layered;
mod logging;

pub use logging::{
    default_logger_path, fingerprint_problem, LogFormat, LoggerConfig, MoveKind, RunEvent,
    RunRecorder,
};

use layered::plan::ProgressTracker;
use layered::{LayeredConfig, LayeredCoordinator, LayeredPlan, LayeredTelemetry};
use layered::nn::MoveRegressorPredictor;

const EPSILON: f64 = 1e-6;
const MAPBOX_MATRIX_LIMIT: usize = 25;
const EARTH_RADIUS_METERS: f64 = 6_371_000.0;
const FALLBACK_SPEED_METERS_PER_SECOND: f64 = 13.89; // ~50 km/h

pub type ProgressCallback = Box<dyn FnMut(usize, usize) + Send>;
pub type CancelCallback = Box<dyn Fn() -> bool + Send + Sync>;

fn should_abort(cancel: Option<&CancelCallback>) -> bool {
    cancel.map(|cb| cb()).unwrap_or(false)
}

fn emit_progress(progress: &mut Option<ProgressCallback>, current: usize, total: usize) {
    if let Some(cb) = progress.as_mut() {
        (**cb)(current, total);
    }
}

// --- Data Structures ---
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Location {
    pub id: String,
    pub x: f64,
    pub y: f64,
    pub demand: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Vehicle {
    pub id: String,
    pub capacity: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Problem {
    pub depot: Location,
    pub locations: Vec<Location>,
    pub vehicles: Vec<Vehicle>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Route {
    pub vehicle_id: String,
    pub locations: Vec<String>,
    pub total_distance: f64,
    pub total_duration: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Solution {
    pub routes: Vec<Route>,
    pub unassigned_locations: Vec<String>,
    pub total_cost: f64,
}

#[derive(Deserialize, Debug)]
struct MatrixResponse {
    distances: Vec<Vec<f64>>,
    durations: Vec<Vec<f64>>,
}

pub(crate) struct ProblemContext {
    depot_id: String,
    customer_ids: Vec<String>,
    demands: Vec<u32>,
    vehicle_capacities: Vec<u32>,
    distance: Vec<Vec<f64>>,
    duration: Vec<Vec<f64>>,
}

impl ProblemContext {
    fn new(problem: &Problem, distance: Vec<Vec<f64>>, duration: Vec<Vec<f64>>) -> Self {
        let depot_id = problem.depot.id.clone();
        let customer_ids = problem
            .locations
            .iter()
            .map(|loc| loc.id.clone())
            .collect::<Vec<_>>();

        let mut demands = vec![0];
        demands.extend(problem.locations.iter().map(|loc| loc.demand));

        let vehicle_capacities = problem
            .vehicles
            .iter()
            .map(|vehicle| vehicle.capacity)
            .collect::<Vec<_>>();

        Self {
            depot_id,
            customer_ids,
            demands,
            vehicle_capacities,
            distance,
            duration,
        }
    }

    fn id_for(&self, index: usize) -> &str {
        if index == 0 {
            &self.depot_id
        } else {
            &self.customer_ids[index - 1]
        }
    }

    fn demand(&self, index: usize) -> u32 {
        self.demands[index]
    }

    fn capacity_for_vehicle(&self, vehicle_idx: usize) -> u32 {
        self.vehicle_capacities[vehicle_idx]
    }

    pub(crate) fn duration_matrix(&self) -> &Vec<Vec<f64>> {
        &self.duration
    }

    pub(crate) fn distance_matrix(&self) -> &Vec<Vec<f64>> {
        &self.distance
    }
}

#[derive(Clone)]
struct RouteState {
    vehicle_idx: usize,
    customers: Vec<usize>,
    load: u32,
    distance: f64,
    duration: f64,
}

#[derive(Clone)]
struct SolutionState {
    routes: Vec<RouteState>,
    unassigned: Vec<usize>,
    total_distance: f64,
    total_duration: f64,
}

impl SolutionState {
    fn new(routes: Vec<RouteState>, unassigned: Vec<usize>) -> Self {
        let total_distance = routes.iter().map(|route| route.distance).sum();
        let total_duration = routes.iter().map(|route| route.duration).sum();
        Self {
            routes,
            unassigned,
            total_distance,
            total_duration,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct HybridConfig {
    pub iterations: usize,
    pub initial_temperature: f64,
    pub minimum_temperature: f64,
    pub cooling_factor: f64,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            iterations: 5000,
            initial_temperature: 50.0,
            minimum_temperature: 0.1,
            cooling_factor: 0.95,
        }
    }
}

// --- Mapbox API Call ---
async fn get_matrix_data(
    locations: &[Location],
) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>), Box<dyn std::error::Error>> {
    let access_token = "pk.eyJ1IjoiamFja3dlZWtseSIsImEiOiJjbWc0aHR1cjExbGR0MmxuMGVkNnJ3bzBxIn0.ay9ucOZV_GVfgr7ZKLMS4w";
    let coords = locations
        .iter()
        .map(|loc| format!("{},{}", loc.x, loc.y))
        .collect::<Vec<String>>()
        .join(";");
    let url = format!(
        "https://api.mapbox.com/directions-matrix/v1/mapbox/driving-traffic/{}?annotations=distance,duration&access_token={}",
        coords, access_token
    );
    let response = reqwest::get(&url).await?;
    let response_text = response.text().await?;
    let matrix_response: MatrixResponse = serde_json::from_str(&response_text)?;
    Ok((matrix_response.distances, matrix_response.durations))
}

fn compute_haversine_matrices(locations: &[Location]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = locations.len();
    let mut distance = vec![vec![0.0; n]; n];
    let mut duration = vec![vec![0.0; n]; n];

    for i in 0..n {
        let lat1 = locations[i].y.to_radians();
        let lon1 = locations[i].x.to_radians();
        for j in i + 1..n {
            let lat2 = locations[j].y.to_radians();
            let lon2 = locations[j].x.to_radians();
            let dlat = lat2 - lat1;
            let dlon = lon2 - lon1;

            let a =
                (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
            let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
            let meters = EARTH_RADIUS_METERS * c;

            distance[i][j] = meters;
            distance[j][i] = meters;

            let time_seconds = if FALLBACK_SPEED_METERS_PER_SECOND > 0.0 {
                meters / FALLBACK_SPEED_METERS_PER_SECOND
            } else {
                meters
            };
            duration[i][j] = time_seconds;
            duration[j][i] = time_seconds;
        }
    }

    (distance, duration)
}

// --- Hybrid Metaheuristic Solver ---
pub async fn solve_vrp(problem: Problem) -> Solution {
    solve_vrp_with_config(problem, HybridConfig::default()).await
}

pub async fn solve_vrp_with_config(problem: Problem, config: HybridConfig) -> Solution {
    solve_vrp_with_callbacks(problem, config, None, None, None).await
}

pub async fn solve_vrp_with_callbacks(
    problem: Problem,
    config: HybridConfig,
    progress: Option<ProgressCallback>,
    should_cancel: Option<CancelCallback>,
    logger: Option<RunRecorder>,
) -> Solution {
    solve_vrp_internal(problem, config, progress, should_cancel, logger).await
}

async fn solve_vrp_internal(
    problem: Problem,
    config: HybridConfig,
    mut progress: Option<ProgressCallback>,
    should_cancel: Option<CancelCallback>,
    logger: Option<RunRecorder>,
) -> Solution {
    let mut logger = logger;
    let start_time = Instant::now();

    if problem.locations.is_empty() || problem.vehicles.is_empty() {
        return fallback_solution(problem);
    }

    let num_vehicles = problem.vehicles.len();
    let locations = &problem.locations;

    let location_points: Vec<[f64; 2]> = locations.iter().map(|loc| [loc.x, loc.y]).collect();
    let observations =
        Array::from_shape_vec((locations.len(), 2), location_points.concat()).unwrap();
    let dataset = Dataset::from(observations);

    let model = KMeans::params(num_vehicles)
        .fit(&dataset)
        .expect("KMeans failed to fit");
    let predictions = model.predict(&dataset);

    let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); num_vehicles];
    for (i, &cluster_idx) in predictions.iter().enumerate() {
        clusters[cluster_idx].push(i + 1);
    }

    let cancel_cb_ref = should_cancel.as_ref();

    let mut layered_config = LayeredConfig::default();

    if let Ok(path) = std::env::var("MOVE_REGRESSOR_PATH") {
        layered_config.neural_model_path = Some(PathBuf::from(path));
    }

    let mut coordinator = LayeredCoordinator::new(layered_config.clone());
    if let Some(model_path) = layered_config.neural_model_path.clone() {
        if let Ok(predictor) = MoveRegressorPredictor::load(model_path) {
            coordinator = coordinator.with_neural_predictor(Arc::new(predictor));
        }
    }
    let planning_units = coordinator.estimated_work_units(&clusters);
    let progress_total = config.iterations + planning_units;
    let progress_total_effective = progress_total.max(1);

    if should_abort(cancel_cb_ref) {
        emit_progress(&mut progress, progress_total_effective, progress_total_effective);
        return fallback_solution(problem);
    }

    let all_locations = std::iter::once(problem.depot.clone())
        .chain(problem.locations.iter().cloned())
        .collect::<Vec<_>>();

    let (mut distance_matrix, mut duration_matrix) = compute_haversine_matrices(&all_locations);

    if all_locations.len() <= MAPBOX_MATRIX_LIMIT {
        if should_abort(cancel_cb_ref) {
            emit_progress(&mut progress, progress_total_effective, progress_total_effective);
            return fallback_solution(problem);
        }
        if let Ok((distances, durations)) = get_matrix_data(&all_locations).await {
            distance_matrix = distances;
            duration_matrix = durations;
        }
    } else {
        let chunk_capacity = MAPBOX_MATRIX_LIMIT.saturating_sub(1);
        if chunk_capacity > 0 {
            for cluster_nodes in &clusters {
                if should_abort(cancel_cb_ref) {
                    emit_progress(&mut progress, progress_total_effective, progress_total_effective);
                    return fallback_solution(problem);
                }
                if cluster_nodes.is_empty() {
                    continue;
                }

                for chunk in cluster_nodes.chunks(chunk_capacity) {
                    if should_abort(cancel_cb_ref) {
                        emit_progress(&mut progress, progress_total_effective, progress_total_effective);
                        return fallback_solution(problem);
                    }
                    let mut subset = Vec::with_capacity(chunk.len() + 1);
                    subset.push(problem.depot.clone());
                    for &node_idx in chunk {
                        subset.push(problem.locations[node_idx - 1].clone());
                    }

                    if should_abort(cancel_cb_ref) {
                        emit_progress(&mut progress, progress_total_effective, progress_total_effective);
                        return fallback_solution(problem);
                    }
                    if let Ok((distances, durations)) = get_matrix_data(&subset).await {
                        let global_indices = std::iter::once(0usize)
                            .chain(chunk.iter().copied())
                            .collect::<Vec<_>>();
                        for (i_local, &i_global) in global_indices.iter().enumerate() {
                            for (j_local, &j_global) in global_indices.iter().enumerate() {
                                distance_matrix[i_global][j_global] = distances[i_local][j_local];
                                duration_matrix[i_global][j_global] = durations[i_local][j_local];
                            }
                        }
                    }
                }
            }
        }
    }

    let context = ProblemContext::new(&problem, distance_matrix, duration_matrix);
    let mut rng = StdRng::from_entropy();

    let mut layered_progress = {
        let progress_cb = progress.as_mut().map(|cb| &mut **cb as &mut dyn FnMut(usize, usize));
        ProgressTracker::new(progress_cb, progress_total_effective, 0)
    };

    let layered_result = construct_initial_solution(
        &problem,
        &context,
        &clusters,
        &mut rng,
        &coordinator,
        &mut layered_progress,
        cancel_cb_ref,
    );

    let layered_progress_units = layered_progress.current();
    drop(layered_progress);

    let (initial_state, layered_metrics) = match layered_result {
        Some(value) => value,
        None => {
            if let Some(cb) = progress.as_mut() {
                (**cb)(progress_total_effective, progress_total_effective);
            }
            return fallback_solution(problem);
        }
    };

    if let Some(recorder) = logger.as_mut() {
        for entry in &layered_metrics {
            recorder.record_event(&RunEvent::LayeredPhase {
                cluster_index: entry.cluster_index,
                heuristic_scale: entry.heuristic_scale,
                layer_penalty: entry.layer_penalty,
                ga_params: entry.ga_params,
                best_cost: entry.best_cost,
            });
        }
        let snapshot = logging::snapshot_state(&initial_state, &context, &problem.vehicles);
        recorder.record_event(&RunEvent::RunStart {
            instance_fingerprint: fingerprint_problem(&problem),
            vehicles: problem.vehicles.len(),
            customers: problem.locations.len(),
            config,
            initial_state: snapshot,
        });
    }

    let progress_cb = progress
        .as_mut()
        .map(|cb| &mut **cb as &mut dyn FnMut(usize, usize));

    let (optimized_state, iterations_run) = hybrid_optimize(
        initial_state,
        &context,
        &config,
        &mut rng,
        &problem.vehicles,
        progress_cb,
        should_cancel.as_ref(),
        logger.as_mut(),
        layered_progress_units,
        progress_total_effective,
    );

    if let Some(recorder) = logger.as_mut() {
        recorder.record_event(&RunEvent::RunComplete {
            iterations_run,
            best_cost: optimized_state.total_duration,
            runtime_ms: Some(start_time.elapsed().as_millis()),
        });
        recorder.finalize();
    }

    build_solution(problem, &context, optimized_state)
}

fn construct_initial_solution(
    problem: &Problem,
    ctx: &ProblemContext,
    clusters: &[Vec<usize>],
    rng: &mut StdRng,
    coordinator: &LayeredCoordinator,
    tracker: &mut ProgressTracker,
    should_cancel: Option<&CancelCallback>,
) -> Option<(SolutionState, Vec<LayeredTelemetry>)> {
    let mut routes = Vec::with_capacity(problem.vehicles.len());
    let mut unassigned = Vec::new();

    let cancel_fn = should_cancel.map(|cb| &**cb as &dyn Fn() -> bool);
    let LayeredPlan {
        per_vehicle,
        summary_cost: _,
        telemetry,
    } = coordinator.plan_routes(ctx, clusters, rng, tracker, cancel_fn)?;
    let mut per_vehicle = per_vehicle;

    for (vehicle_idx, vehicle) in problem.vehicles.iter().enumerate() {
        if let Some(cb) = cancel_fn {
            if cb() {
                return None;
            }
        }
        let cluster_nodes = clusters.get(vehicle_idx).cloned().unwrap_or_default();
        let mut remaining_capacity = vehicle.capacity;
        let mut assigned = Vec::new();

        for node_idx in cluster_nodes {
            let demand = ctx.demand(node_idx);
            if demand <= remaining_capacity {
                assigned.push(node_idx);
                remaining_capacity -= demand;
            } else {
                unassigned.push(node_idx);
            }
        }

        let assigned_lookup: std::collections::HashSet<_> = assigned.iter().copied().collect();
        let mut order = per_vehicle
            .get_mut(vehicle_idx)
            .map(std::mem::take)
            .unwrap_or_default()
            .into_iter()
            .filter(|node| assigned_lookup.contains(node))
            .collect::<Vec<_>>();

        if order.is_empty() {
            order = nearest_neighbor_order(&assigned, ctx);
        }

        let load = compute_route_load(&order, ctx);
        let (distance, duration) = compute_route_metrics(&order, ctx);

        routes.push(RouteState {
            vehicle_idx,
            customers: order,
            load,
            distance,
            duration,
        });
    }

    let mut state = SolutionState::new(routes, unassigned);
    distribute_unassigned(&mut state, ctx);
    Some((state, telemetry))
}

fn nearest_neighbor_order(customers: &[usize], ctx: &ProblemContext) -> Vec<usize> {
    if customers.is_empty() {
        return Vec::new();
    }

    let mut order = Vec::with_capacity(customers.len());
    let mut unvisited = customers.to_vec();
    let mut current = 0usize;

    while !unvisited.is_empty() {
        let mut best_idx = 0;
        let mut best_duration = f64::MAX;
        for (i, &candidate) in unvisited.iter().enumerate() {
            let duration = ctx.duration[current][candidate];
            if duration < best_duration {
                best_duration = duration;
                best_idx = i;
            }
        }
        let next = unvisited.swap_remove(best_idx);
        order.push(next);
        current = next;
    }

    order
}

fn compute_route_metrics(customers: &[usize], ctx: &ProblemContext) -> (f64, f64) {
    if customers.is_empty() {
        return (0.0, 0.0);
    }

    let mut distance = ctx.distance[0][customers[0]];
    let mut duration = ctx.duration[0][customers[0]];

    for window in customers.windows(2) {
        let from = window[0];
        let to = window[1];
        distance += ctx.distance[from][to];
        duration += ctx.duration[from][to];
    }

    let last = *customers.last().unwrap();
    distance += ctx.distance[last][0];
    duration += ctx.duration[last][0];
    (distance, duration)
}

fn compute_route_load(customers: &[usize], ctx: &ProblemContext) -> u32 {
    customers.iter().map(|idx| ctx.demand(*idx)).sum()
}

fn distribute_unassigned(state: &mut SolutionState, ctx: &ProblemContext) {
    let unassigned = std::mem::take(&mut state.unassigned);
    let mut remaining = Vec::new();
    for node_idx in unassigned {
        if !insert_node_into_best_route(node_idx, state, ctx) {
            remaining.push(node_idx);
        }
    }
    state.unassigned = remaining;
}

fn insert_node_into_best_route(
    node_idx: usize,
    state: &mut SolutionState,
    ctx: &ProblemContext,
) -> bool {
    let demand = ctx.demand(node_idx);
    let mut best_choice: Option<(usize, Vec<usize>, f64, f64, f64)> = None;

    for (route_idx, route) in state.routes.iter().enumerate() {
        let capacity = ctx.capacity_for_vehicle(route.vehicle_idx);
        if route.load + demand > capacity {
            continue;
        }

        let len = route.customers.len();

        if len == 0 {
            let candidate = vec![node_idx];
            let (distance, duration) = compute_route_metrics(&candidate, ctx);
            let delta_duration = duration - route.duration;
            if best_choice
                .as_ref()
                .map(|choice| delta_duration + EPSILON < choice.4)
                .unwrap_or(true)
            {
                best_choice = Some((route_idx, candidate, distance, duration, delta_duration));
            }
            continue;
        }

        for insert_pos in 0..=len {
            let mut candidate = route.customers.clone();
            candidate.insert(insert_pos, node_idx);
            let (distance, duration) = compute_route_metrics(&candidate, ctx);
            let delta_duration = duration - route.duration;

            if best_choice
                .as_ref()
                .map(|choice| delta_duration + EPSILON < choice.4)
                .unwrap_or(true)
            {
                best_choice = Some((route_idx, candidate, distance, duration, delta_duration));
            }
        }
    }

    if let Some((route_idx, candidate, distance, duration, _)) = best_choice {
        let route = &mut state.routes[route_idx];
        let old_distance = route.distance;
        let old_duration = route.duration;
        route.customers = candidate;
        route.distance = distance;
        route.duration = duration;
        route.load += demand;

        state.total_distance = state.total_distance - old_distance + distance;
        state.total_duration = state.total_duration - old_duration + duration;
        true
    } else {
        false
    }
}

fn hybrid_optimize(
    mut state: SolutionState,
    ctx: &ProblemContext,
    config: &HybridConfig,
    rng: &mut StdRng,
    vehicles: &[Vehicle],
    mut progress: Option<&mut dyn FnMut(usize, usize)>,
    should_cancel: Option<&CancelCallback>,
    mut logger: Option<&mut RunRecorder>,
    progress_offset: usize,
    progress_total: usize,
) -> (SolutionState, usize) {
    local_search(&mut state, ctx);
    let mut best = state.clone();
    let mut current = state;
    let mut temperature = config.initial_temperature.max(config.minimum_temperature);
    let mut iterations_completed = 0usize;

    if let Some(cb) = progress.as_mut() {
        (**cb)(progress_offset, progress_total);
    }

    for iteration in 0..config.iterations {
        if let Some(cancel) = should_cancel {
            if cancel() {
                break;
            }
        }

        let (mut candidate, move_kind) = random_neighbor(&current, ctx, rng);
        local_search(&mut candidate, ctx);

        let delta = candidate.total_duration - current.total_duration;
        let (accept, acceptance_probability) = if delta < -EPSILON {
            (true, 1.0)
        } else if delta.abs() <= EPSILON {
            (true, 1.0)
        } else if temperature > config.minimum_temperature {
            let acceptance = (-delta / temperature).exp().clamp(0.0, 1.0);
            (rng.gen::<f64>() < acceptance, acceptance)
        } else {
            (false, 0.0)
        };

        if let Some(recorder) = logger.as_mut() {
            recorder.record_event(&RunEvent::Iteration {
                iteration,
                temperature,
                acceptance_probability,
                current_cost: current.total_duration,
                best_cost: best.total_duration,
            });
        }

        if accept {
            current = candidate.clone();
            if current.total_duration + EPSILON < best.total_duration {
                best = current.clone();
            }

            if let Some(recorder) = logger.as_mut() {
                let snapshot = logging::snapshot_state(&current, ctx, vehicles);
                let operator = move_kind.clone();
                recorder.record_event(&RunEvent::MoveAccepted {
                    iteration,
                    operator,
                    delta_cost: delta,
                    current_cost: current.total_duration,
                    best_cost: best.total_duration,
                    state: snapshot,
                });
            }
        }

        temperature = (temperature * config.cooling_factor).max(config.minimum_temperature);

        if let Some(cb) = progress.as_mut() {
            let current_progress = progress_offset.saturating_add(iteration + 1);
            (**cb)(current_progress, progress_total);
        }

        iterations_completed = iteration + 1;
    }

    (best, iterations_completed)
}

fn local_search(state: &mut SolutionState, ctx: &ProblemContext) {
    loop {
        if two_opt_first_improvement(state, ctx) {
            continue;
        }
        if or_opt_first_improvement(state, ctx) {
            continue;
        }
        if swap_first_improvement(state, ctx) {
            continue;
        }
        break;
    }
}

fn two_opt_first_improvement(state: &mut SolutionState, ctx: &ProblemContext) -> bool {
    for route_idx in 0..state.routes.len() {
        if state.routes[route_idx].customers.len() < 2 {
            continue;
        }

        let original_customers = state.routes[route_idx].customers.clone();
        let route_len = original_customers.len();

        for i in 0..route_len - 1 {
            for j in i + 1..route_len {
                let mut candidate = original_customers.clone();
                candidate[i..=j].reverse();

                let (new_distance, new_duration) = compute_route_metrics(&candidate, ctx);
                let old_distance = state.routes[route_idx].distance;
                let old_duration = state.routes[route_idx].duration;

                if new_duration + EPSILON < old_duration {
                    state.routes[route_idx].customers = candidate;
                    state.routes[route_idx].distance = new_distance;
                    state.routes[route_idx].duration = new_duration;
                    state.total_distance = state.total_distance - old_distance + new_distance;
                    state.total_duration = state.total_duration - old_duration + new_duration;
                    return true;
                }
            }
        }
    }
    false
}

fn or_opt_first_improvement(state: &mut SolutionState, ctx: &ProblemContext) -> bool {
    let route_count = state.routes.len();

    for from_idx in 0..route_count {
        let from_customers = state.routes[from_idx].customers.clone();
        let from_len = from_customers.len();
        if from_len == 0 {
            continue;
        }

        for segment_len in 1..=usize::min(3, from_len) {
            for start in 0..=from_len - segment_len {
                let segment = from_customers[start..start + segment_len].to_vec();
                let segment_demand: u32 = segment.iter().map(|idx| ctx.demand(*idx)).sum();

                for to_idx in 0..route_count {
                    let same_route = from_idx == to_idx;

                    if same_route && from_len == segment_len {
                        continue;
                    }

                    if !same_route {
                        let capacity = ctx.capacity_for_vehicle(state.routes[to_idx].vehicle_idx);
                        if state.routes[to_idx].load + segment_demand > capacity {
                            continue;
                        }
                    }

                    if same_route {
                        let mut base_route = from_customers.clone();
                        base_route.drain(start..start + segment_len);

                        for insert_pos in 0..=base_route.len() {
                            if insert_pos == start {
                                continue;
                            }

                            let mut new_route = base_route.clone();
                            for (offset, &node) in segment.iter().enumerate() {
                                new_route.insert(insert_pos + offset, node);
                            }

                            let (new_distance, new_duration) =
                                compute_route_metrics(&new_route, ctx);
                            let old_distance = state.routes[from_idx].distance;
                            let old_duration = state.routes[from_idx].duration;

                            if new_duration + EPSILON < old_duration {
                                state.routes[from_idx].customers = new_route;
                                state.routes[from_idx].distance = new_distance;
                                state.routes[from_idx].duration = new_duration;
                                state.total_distance =
                                    state.total_distance - old_distance + new_distance;
                                state.total_duration =
                                    state.total_duration - old_duration + new_duration;
                                return true;
                            }
                        }
                    } else {
                        let mut new_from = from_customers.clone();
                        new_from.drain(start..start + segment_len);

                        let base_to = state.routes[to_idx].customers.clone();
                        for insert_pos in 0..=base_to.len() {
                            let mut new_to = base_to.clone();
                            for (offset, &node) in segment.iter().enumerate() {
                                new_to.insert(insert_pos + offset, node);
                            }

                            let (new_from_distance, new_from_duration) =
                                compute_route_metrics(&new_from, ctx);
                            let (new_to_distance, new_to_duration) =
                                compute_route_metrics(&new_to, ctx);

                            let old_from_distance = state.routes[from_idx].distance;
                            let old_from_duration = state.routes[from_idx].duration;
                            let old_to_distance = state.routes[to_idx].distance;
                            let old_to_duration = state.routes[to_idx].duration;

                            let new_total_duration =
                                state.total_duration - old_from_duration - old_to_duration
                                    + new_from_duration
                                    + new_to_duration;

                            if new_total_duration + EPSILON < state.total_duration {
                                state.routes[from_idx].customers = new_from.clone();
                                state.routes[from_idx].distance = new_from_distance;
                                state.routes[from_idx].duration = new_from_duration;
                                state.routes[from_idx].load -= segment_demand;

                                state.routes[to_idx].customers = new_to;
                                state.routes[to_idx].distance = new_to_distance;
                                state.routes[to_idx].duration = new_to_duration;
                                state.routes[to_idx].load += segment_demand;

                                state.total_distance =
                                    state.total_distance - old_from_distance - old_to_distance
                                        + new_from_distance
                                        + new_to_distance;
                                state.total_duration = new_total_duration;
                                return true;
                            }
                        }
                    }
                }
            }
        }
    }

    false
}

fn swap_first_improvement(state: &mut SolutionState, ctx: &ProblemContext) -> bool {
    let route_count = state.routes.len();

    for i in 0..route_count {
        for j in i + 1..route_count {
            if state.routes[i].customers.is_empty() || state.routes[j].customers.is_empty() {
                continue;
            }

            let capacity_i = ctx.capacity_for_vehicle(state.routes[i].vehicle_idx);
            let capacity_j = ctx.capacity_for_vehicle(state.routes[j].vehicle_idx);

            for pos_i in 0..state.routes[i].customers.len() {
                let customer_i = state.routes[i].customers[pos_i];
                let demand_i = ctx.demand(customer_i);

                for pos_j in 0..state.routes[j].customers.len() {
                    let customer_j = state.routes[j].customers[pos_j];
                    let demand_j = ctx.demand(customer_j);

                    let new_load_i = state.routes[i].load - demand_i + demand_j;
                    let new_load_j = state.routes[j].load - demand_j + demand_i;

                    if new_load_i > capacity_i || new_load_j > capacity_j {
                        continue;
                    }

                    let mut new_route_i = state.routes[i].customers.clone();
                    let mut new_route_j = state.routes[j].customers.clone();
                    new_route_i[pos_i] = customer_j;
                    new_route_j[pos_j] = customer_i;

                    let (new_i_distance, new_i_duration) = compute_route_metrics(&new_route_i, ctx);
                    let (new_j_distance, new_j_duration) = compute_route_metrics(&new_route_j, ctx);

                    let old_i_distance = state.routes[i].distance;
                    let old_i_duration = state.routes[i].duration;
                    let old_j_distance = state.routes[j].distance;
                    let old_j_duration = state.routes[j].duration;

                    let new_total_duration = state.total_duration - old_i_duration - old_j_duration
                        + new_i_duration
                        + new_j_duration;

                    if new_total_duration + EPSILON < state.total_duration {
                        state.routes[i].customers = new_route_i;
                        state.routes[i].distance = new_i_distance;
                        state.routes[i].duration = new_i_duration;
                        state.routes[i].load = new_load_i;

                        state.routes[j].customers = new_route_j;
                        state.routes[j].distance = new_j_distance;
                        state.routes[j].duration = new_j_duration;
                        state.routes[j].load = new_load_j;

                        state.total_distance =
                            state.total_distance - old_i_distance - old_j_distance
                                + new_i_distance
                                + new_j_distance;
                        state.total_duration = new_total_duration;
                        return true;
                    }
                }
            }
        }
    }

    false
}

fn random_neighbor(
    current: &SolutionState,
    ctx: &ProblemContext,
    rng: &mut StdRng,
) -> (SolutionState, MoveKind) {
    const MAX_ATTEMPTS: usize = 32;

    for _ in 0..MAX_ATTEMPTS {
        let mut candidate = current.clone();
        let applied = match rng.gen_range(0..3) {
            0 => apply_random_two_opt(&mut candidate, ctx, rng),
            1 => apply_random_or_opt(&mut candidate, ctx, rng),
            _ => apply_random_swap(&mut candidate, ctx, rng),
        };

        if let Some(kind) = applied {
            return (candidate, kind);
        }
    }

    let mut fallback = current.clone();
    if let Some((route_idx, route)) = fallback
        .routes
        .iter_mut()
        .enumerate()
        .find(|(_, r)| r.customers.len() > 1)
    {
        let old_distance = route.distance;
        let old_duration = route.duration;
        route.customers.shuffle(rng);
        let (distance, duration) = compute_route_metrics(&route.customers, ctx);
        route.distance = distance;
        route.duration = duration;
        fallback.total_distance = fallback.total_distance - old_distance + distance;
        fallback.total_duration = fallback.total_duration - old_duration + duration;
        return (fallback, MoveKind::Shuffle { route_idx });
    }

    (fallback, MoveKind::Shuffle { route_idx: 0 })
}

fn apply_random_two_opt(
    state: &mut SolutionState,
    ctx: &ProblemContext,
    rng: &mut StdRng,
) -> Option<MoveKind> {
    let route_indices: Vec<usize> = state
        .routes
        .iter()
        .enumerate()
        .filter(|(_, route)| route.customers.len() >= 2)
        .map(|(idx, _)| idx)
        .collect();

    if route_indices.is_empty() {
        return None;
    }

    let route_idx = *route_indices.choose(rng).unwrap();
    let len = state.routes[route_idx].customers.len();
    if len < 2 {
        return None;
    }

    let i = rng.gen_range(0..len - 1);
    let j = rng.gen_range(i + 1..len);

    let mut candidate = state.routes[route_idx].customers.clone();
    candidate[i..=j].reverse();

    if candidate == state.routes[route_idx].customers {
        return None;
    }

    let old_distance = state.routes[route_idx].distance;
    let old_duration = state.routes[route_idx].duration;
    let (distance, duration) = compute_route_metrics(&candidate, ctx);

    state.routes[route_idx].customers = candidate;
    state.routes[route_idx].distance = distance;
    state.routes[route_idx].duration = duration;
    state.total_distance = state.total_distance - old_distance + distance;
    state.total_duration = state.total_duration - old_duration + duration;
    Some(MoveKind::TwoOpt { route_idx, i, j })
}

fn apply_random_or_opt(
    state: &mut SolutionState,
    ctx: &ProblemContext,
    rng: &mut StdRng,
) -> Option<MoveKind> {
    let available_routes: Vec<usize> = state
        .routes
        .iter()
        .enumerate()
        .filter(|(_, route)| !route.customers.is_empty())
        .map(|(idx, _)| idx)
        .collect();

    if available_routes.is_empty() {
        return None;
    }

    let from_idx = *available_routes.choose(rng).unwrap();
    let from_len = state.routes[from_idx].customers.len();
    let segment_len = rng.gen_range(1..=usize::min(3, from_len));
    let start = rng.gen_range(0..=from_len - segment_len);
    let segment = state.routes[from_idx].customers[start..start + segment_len].to_vec();
    let segment_demand: u32 = segment.iter().map(|idx| ctx.demand(*idx)).sum();

    let mut to_indices: Vec<usize> = (0..state.routes.len()).collect();
    to_indices.shuffle(rng);

    for to_idx in to_indices {
        let same_route = from_idx == to_idx;

        if !same_route {
            let capacity = ctx.capacity_for_vehicle(state.routes[to_idx].vehicle_idx);
            if state.routes[to_idx].load + segment_demand > capacity {
                continue;
            }
        }

        if same_route {
            let mut base_route = state.routes[from_idx].customers.clone();
            base_route.drain(start..start + segment_len);

            if base_route.is_empty() {
                continue;
            }

            let mut positions: Vec<usize> = (0..=base_route.len()).collect();
            positions.shuffle(rng);

            for insert_pos in positions {
                if insert_pos == start {
                    continue;
                }

                let mut new_route = base_route.clone();
                for (offset, &node) in segment.iter().enumerate() {
                    new_route.insert(insert_pos + offset, node);
                }

                if new_route == state.routes[from_idx].customers {
                    continue;
                }

                let old_distance = state.routes[from_idx].distance;
                let old_duration = state.routes[from_idx].duration;
                let (distance, duration) = compute_route_metrics(&new_route, ctx);

                state.routes[from_idx].customers = new_route;
                state.routes[from_idx].distance = distance;
                state.routes[from_idx].duration = duration;
                state.total_distance = state.total_distance - old_distance + distance;
                state.total_duration = state.total_duration - old_duration + duration;
                return Some(MoveKind::OrOpt {
                    from_route: from_idx,
                    to_route: from_idx,
                    start,
                    segment_len,
                    insert_pos,
                });
            }
        } else {
            let mut from_route = state.routes[from_idx].customers.clone();
            from_route.drain(start..start + segment_len);
            let to_route = state.routes[to_idx].customers.clone();

            let mut positions: Vec<usize> = (0..=to_route.len()).collect();
            positions.shuffle(rng);

            for insert_pos in positions {
                let new_from = from_route.clone();
                let mut new_to = to_route.clone();
                for (offset, &node) in segment.iter().enumerate() {
                    new_to.insert(insert_pos + offset, node);
                }

                let (from_distance, from_duration) = compute_route_metrics(&new_from, ctx);
                let (to_distance, to_duration) = compute_route_metrics(&new_to, ctx);

                let old_from_distance = state.routes[from_idx].distance;
                let old_from_duration = state.routes[from_idx].duration;
                let old_to_distance = state.routes[to_idx].distance;
                let old_to_duration = state.routes[to_idx].duration;
                let old_from_load = state.routes[from_idx].load;
                let old_to_load = state.routes[to_idx].load;

                state.routes[from_idx].customers = new_from;
                state.routes[from_idx].distance = from_distance;
                state.routes[from_idx].duration = from_duration;
                state.routes[from_idx].load = old_from_load - segment_demand;

                state.routes[to_idx].customers = new_to;
                state.routes[to_idx].distance = to_distance;
                state.routes[to_idx].duration = to_duration;
                state.routes[to_idx].load = old_to_load + segment_demand;

                state.total_distance = state.total_distance - old_from_distance - old_to_distance
                    + from_distance
                    + to_distance;
                state.total_duration = state.total_duration - old_from_duration - old_to_duration
                    + from_duration
                    + to_duration;
                return Some(MoveKind::OrOpt {
                    from_route: from_idx,
                    to_route: to_idx,
                    start,
                    segment_len,
                    insert_pos,
                });
            }
        }
    }

    None
}

fn apply_random_swap(
    state: &mut SolutionState,
    ctx: &ProblemContext,
    rng: &mut StdRng,
) -> Option<MoveKind> {
    if state.routes.len() < 2 {
        return None;
    }

    let mut route_indices: Vec<usize> = (0..state.routes.len()).collect();
    route_indices.shuffle(rng);

    for window in route_indices.windows(2) {
        let i = window[0];
        let j = window[1];

        if state.routes[i].customers.is_empty() || state.routes[j].customers.is_empty() {
            continue;
        }

        let capacity_i = ctx.capacity_for_vehicle(state.routes[i].vehicle_idx);
        let capacity_j = ctx.capacity_for_vehicle(state.routes[j].vehicle_idx);

        let pos_i = rng.gen_range(0..state.routes[i].customers.len());
        let pos_j = rng.gen_range(0..state.routes[j].customers.len());

        let customer_i = state.routes[i].customers[pos_i];
        let customer_j = state.routes[j].customers[pos_j];
        let demand_i = ctx.demand(customer_i);
        let demand_j = ctx.demand(customer_j);

        let new_load_i = state.routes[i].load - demand_i + demand_j;
        let new_load_j = state.routes[j].load - demand_j + demand_i;

        if new_load_i > capacity_i || new_load_j > capacity_j {
            continue;
        }

        let mut new_route_i = state.routes[i].customers.clone();
        let mut new_route_j = state.routes[j].customers.clone();
        new_route_i[pos_i] = customer_j;
        new_route_j[pos_j] = customer_i;

        let (new_i_distance, new_i_duration) = compute_route_metrics(&new_route_i, ctx);
        let (new_j_distance, new_j_duration) = compute_route_metrics(&new_route_j, ctx);

        let old_i_distance = state.routes[i].distance;
        let old_i_duration = state.routes[i].duration;
        let old_j_distance = state.routes[j].distance;
        let old_j_duration = state.routes[j].duration;

        state.routes[i].customers = new_route_i;
        state.routes[i].distance = new_i_distance;
        state.routes[i].duration = new_i_duration;
        state.routes[i].load = new_load_i;

        state.routes[j].customers = new_route_j;
        state.routes[j].distance = new_j_distance;
        state.routes[j].duration = new_j_duration;
        state.routes[j].load = new_load_j;

        state.total_distance = state.total_distance - old_i_distance - old_j_distance
            + new_i_distance
            + new_j_distance;
        state.total_duration = state.total_duration - old_i_duration - old_j_duration
            + new_i_duration
            + new_j_duration;
        return Some(MoveKind::Swap {
            route_a: i,
            pos_a: pos_i,
            route_b: j,
            pos_b: pos_j,
        });
    }

    None
}

fn build_solution(problem: Problem, ctx: &ProblemContext, state: SolutionState) -> Solution {
    let depot_id = ctx.depot_id.clone();

    let mut routes = Vec::with_capacity(problem.vehicles.len());
    for (route_state, vehicle) in state.routes.iter().zip(problem.vehicles.iter()) {
        let mut locations = Vec::with_capacity(route_state.customers.len() + 2);
        locations.push(depot_id.clone());
        for &customer_idx in &route_state.customers {
            locations.push(ctx.id_for(customer_idx).to_string());
        }
        locations.push(depot_id.clone());

        routes.push(Route {
            vehicle_id: vehicle.id.clone(),
            locations,
            total_distance: route_state.distance,
            total_duration: route_state.duration,
        });
    }

    Solution {
        routes,
        unassigned_locations: state
            .unassigned
            .into_iter()
            .map(|idx| ctx.id_for(idx).to_string())
            .collect(),
        total_cost: state.total_duration,
    }
}

fn fallback_solution(problem: Problem) -> Solution {
    let depot_id = problem.depot.id.clone();
    let routes = problem
        .vehicles
        .iter()
        .map(|vehicle| Route {
            vehicle_id: vehicle.id.clone(),
            locations: vec![depot_id.clone(), depot_id.clone()],
            total_distance: 0.0,
            total_duration: 0.0,
        })
        .collect::<Vec<_>>();

    Solution {
        routes,
        unassigned_locations: problem.locations.iter().map(|loc| loc.id.clone()).collect(),
        total_cost: 0.0,
    }
}
