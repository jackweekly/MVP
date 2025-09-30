use super::bayes::{BayesianOptimizer, ParameterBounds};
use super::graph::{LayerId, LayeredGraph};
use super::heuristics::{HeuristicModel, NeuralBlendHeuristic, WeightedDurationHeuristic};
use super::local::{a_star, bidirectional_dijkstra, PathResult};
use super::meta::{GeneticAlgorithm, GeneticHyperParams};
use super::nn::{EdgeFeatures, EdgePredictor};
use super::pso::{optimize_weights, PsoHyperParams};
use crate::ProblemContext;
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

fn should_abort(cancel: Option<&dyn Fn() -> bool>) -> bool {
    cancel.map(|cb| cb()).unwrap_or(false)
}

pub struct ProgressTracker<'a> {
    current: usize,
    total: usize,
    cb: Option<&'a mut dyn FnMut(usize, usize)>,
}

impl<'a> ProgressTracker<'a> {
    pub fn new(cb: Option<&'a mut dyn FnMut(usize, usize)>, total: usize, start: usize) -> Self {
        Self {
            current: start,
            total,
            cb,
        }
    }

    pub fn emit(&mut self) {
        if let Some(cb) = self.cb.as_deref_mut() {
            cb(self.current, self.total);
        }
    }

    pub fn advance(&mut self, steps: usize) {
        self.current = self.current.saturating_add(steps);
        self.emit();
    }

    pub fn current(&self) -> usize {
        self.current
    }
}

#[derive(Clone, Debug)]
pub struct LayeredTelemetry {
    pub cluster_index: usize,
    pub heuristic_scale: f64,
    pub layer_penalty: f64,
    pub ga_params: GeneticHyperParams,
    pub best_cost: f64,
}

#[derive(Clone, Debug)]
pub struct LayeredPlan {
    pub per_vehicle: Vec<Vec<usize>>,
    #[allow(dead_code)]
    pub summary_cost: f64,
    pub telemetry: Vec<LayeredTelemetry>,
}

#[derive(Clone, Debug)]
pub struct LayeredConfig {
    pub enabled: bool,
    pub ga_base: GeneticHyperParams,
    pub pso_params: PsoHyperParams,
    pub heuristic_scale_bounds: (f64, f64),
    pub layer_penalty_bounds: (f64, f64),
    pub bayes_iterations: usize,
    pub bayes_exploration: f64,
    pub neural_weight: f64,
    pub neural_model_path: Option<PathBuf>,
}

impl Default for LayeredConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            ga_base: GeneticHyperParams {
                population_size: 24,
                crossover_rate: 0.8,
                mutation_rate: 0.15,
                elite_count: 4,
                generations: 50,
            },
            pso_params: PsoHyperParams::default(),
            heuristic_scale_bounds: (0.5, 1.5),
            layer_penalty_bounds: (0.0, 60.0),
            bayes_iterations: 15,
            bayes_exploration: 0.6,
            neural_weight: 0.35,
            neural_model_path: Some(PathBuf::from("ml/move_regressor.json")),
        }
    }
}

pub struct LayeredCoordinator {
    config: LayeredConfig,
    neural_predictor: Option<Arc<dyn EdgePredictor>>, 
}

impl LayeredCoordinator {
    pub fn new(config: LayeredConfig) -> Self {
        Self {
            config,
            neural_predictor: None,
        }
    }

    pub fn with_neural_predictor(mut self, predictor: Arc<dyn EdgePredictor>) -> Self {
        self.neural_predictor = Some(predictor);
        self
    }

    pub fn estimated_work_units(&self, clusters: &[Vec<usize>]) -> usize {
        let per_cluster = 1 + self.config.bayes_iterations.max(1);
        clusters
            .iter()
            .filter(|cluster| !cluster.is_empty())
            .count()
            .saturating_mul(per_cluster)
    }

    pub fn plan_routes<R: Rng + ?Sized>(
        &self,
        ctx: &ProblemContext,
        clusters: &[Vec<usize>],
        rng: &mut R,
        tracker: &mut ProgressTracker,
        cancel: Option<&dyn Fn() -> bool>,
    ) -> Option<LayeredPlan> {
        if should_abort(cancel) {
            return None;
        }

        if !self.config.enabled {
            return Some(LayeredPlan {
                per_vehicle: clusters.to_vec(),
                summary_cost: 0.0,
                telemetry: Vec::new(),
            });
        }

        let durations = Arc::new(ctx.duration_matrix().clone());
        let distances = Arc::new(ctx.distance_matrix().clone());
        let layers = self.assign_layers(clusters, durations.len());
        let graph = LayeredGraph::from_dense_costs(layers, durations.as_ref());

        let mut per_vehicle = Vec::with_capacity(clusters.len());
        let mut total_cost = 0.0;
        let mut telemetry = Vec::with_capacity(clusters.len());

        tracker.emit();

        for (cluster_idx, nodes) in clusters.iter().enumerate() {
            if should_abort(cancel) {
                return None;
            }
            if nodes.is_empty() {
                per_vehicle.push(Vec::new());
                continue;
            }

            let result = self.optimize_cluster(
                cluster_idx,
                nodes,
                &graph,
                distances.clone(),
                durations.clone(),
                rng,
                tracker,
                cancel,
            )?;
            total_cost += result.cost;
            telemetry.push(LayeredTelemetry {
                cluster_index: cluster_idx,
                heuristic_scale: result.weights.0,
                layer_penalty: result.weights.1,
                ga_params: result.ga_params,
                best_cost: result.cost,
            });
            per_vehicle.push(result.nodes);
        }

        Some(LayeredPlan {
            per_vehicle,
            summary_cost: total_cost,
            telemetry,
        })
    }

    fn assign_layers(&self, clusters: &[Vec<usize>], node_count: usize) -> Vec<LayerId> {
        let mut layers = vec![LayerId(0); node_count];
        for (layer_idx, cluster) in clusters.iter().enumerate() {
            for &node in cluster {
                if node < node_count {
                    layers[node] = LayerId(layer_idx + 1);
                }
            }
        }
        layers
    }

    fn optimize_cluster<R: Rng + ?Sized>(
        &self,
        _cluster_idx: usize,
        nodes: &[usize],
        graph: &LayeredGraph,
        distances: Arc<Vec<Vec<f64>>>,
        durations: Arc<Vec<Vec<f64>>>,
        rng: &mut R,
        tracker: &mut ProgressTracker,
        cancel: Option<&dyn Fn() -> bool>,
    ) -> Option<ClusterResult> {
        if should_abort(cancel) {
            return None;
        }
        let bounds = vec![
            self.config.heuristic_scale_bounds,
            self.config.layer_penalty_bounds,
        ];

        let samples = self.sample_pairs(nodes, rng);
        let weights = optimize_weights(
            2,
            &bounds,
            self.config.pso_params,
            |candidate| self.heuristic_fit(candidate, &samples, &durations, graph),
            rng,
        );

        tracker.advance(1);
        if should_abort(cancel) {
            return None;
        }

        let base_heuristic = WeightedDurationHeuristic::new(durations.clone(), weights[0], weights[1]);
        let heuristic: Arc<dyn HeuristicModel> = if let Some(predictor) = self.neural_predictor.clone() {
            let predictor_clone = predictor.clone();
            let durations_clone = durations.clone();
            let distances_clone = distances.clone();
            let layer_lookup: Vec<LayerId> = (0..graph.node_count()).map(|idx| graph.layer_of(idx)).collect();
            let degrees: Vec<u32> = (0..graph.node_count())
                .map(|idx| graph.neighbors(idx).len() as u32)
                .collect();
            Arc::new(NeuralBlendHeuristic::new(
                base_heuristic.clone(),
                move |from, to| {
                    let features = EdgeFeatures {
                        from,
                        to,
                        distance: distances_clone[from][to],
                        duration: durations_clone[from][to],
                        layer_same: layer_lookup[from] == layer_lookup[to],
                        degree_from: degrees[from],
                        degree_to: degrees[to],
                    };
                    predictor_clone.predict(&features)
                },
                self.config.neural_weight,
            ))
        } else {
            Arc::new(base_heuristic.clone())
        };

        let mut bayes = BayesianOptimizer::new(self.config.bayes_exploration);
        let ga_bounds = [
            ParameterBounds { min: 0.4, max: 0.95 }, // crossover
            ParameterBounds { min: 0.01, max: 0.35 }, // mutation
            ParameterBounds { min: 0.05, max: 0.4 },  // elite ratio
        ];

        let mut best_sequence = nodes.to_vec();
        let mut best_cost = f64::INFINITY;
        let mut best_ga_params = self.tuned_ga_base(nodes.len());

        for _ in 0..self.config.bayes_iterations.max(1) {
            if should_abort(cancel) {
                return None;
            }
            let sample = bayes.suggest(&ga_bounds, rng);
            let base_params = self.tuned_ga_base(nodes.len());
            let candidate_params = self.build_ga_params(&sample, base_params);
            let ga = GeneticAlgorithm::new(candidate_params);
            let fitness = |ordering: &[usize]| -> f64 {
                self.evaluate_sequence(ordering, heuristic.clone(), graph)
            };
            let candidate = ga.run(nodes, rng, &fitness);
            let candidate_cost = fitness(&candidate);
            bayes.observe(sample.clone(), candidate_cost);

            if candidate_cost + f64::EPSILON < best_cost {
                best_cost = candidate_cost;
                best_sequence = candidate;
                best_ga_params = candidate_params;
            }

            tracker.advance(1);
            if should_abort(cancel) {
                return None;
            }
        }

        Some(ClusterResult {
            nodes: best_sequence,
            cost: best_cost,
            weights: (weights[0], weights[1]),
            ga_params: best_ga_params,
        })
    }

    fn tuned_ga_base(&self, cluster_size: usize) -> GeneticHyperParams {
        let scale = (cluster_size as f64 / 10.0).clamp(0.5, 2.0);
        let pop = (self.config.ga_base.population_size as f64 * scale).round() as usize;
        let generations =
            (self.config.ga_base.generations as f64 * scale.clamp(0.75, 1.5)).round() as usize;
        GeneticHyperParams {
            population_size: pop.max(6),
            crossover_rate: self.config.ga_base.crossover_rate,
            mutation_rate: self.config.ga_base.mutation_rate,
            elite_count: self.config.ga_base.elite_count.max(1),
            generations: generations.max(10),
        }
    }

    fn build_ga_params(&self, sampled: &[f64], base: GeneticHyperParams) -> GeneticHyperParams {
        let elite_ratio = sampled.get(2).copied().unwrap_or(0.1);
        let elite = ((base.population_size as f64) * elite_ratio).round() as usize;
        GeneticHyperParams {
            population_size: base.population_size,
            crossover_rate: sampled.get(0).copied().unwrap_or(base.crossover_rate),
            mutation_rate: sampled.get(1).copied().unwrap_or(base.mutation_rate),
            elite_count: elite.clamp(1, base.population_size.saturating_sub(1)),
            generations: base.generations,
        }
    }

    fn heuristic_fit(
        &self,
        candidate: &[f64],
        samples: &[(usize, usize)],
        durations: &Arc<Vec<Vec<f64>>>,
        graph: &LayeredGraph,
    ) -> f64 {
        let heuristic = WeightedDurationHeuristic::new(durations.clone(), candidate[0], candidate[1]);
        samples
            .iter()
            .map(|&(a, b)| {
                let predicted = heuristic.estimate(a, b, graph);
                let actual = durations[a][b];
                (predicted - actual).abs()
            })
            .sum::<f64>()
            / samples.len().max(1) as f64
    }

    fn evaluate_sequence(
        &self,
        ordering: &[usize],
        heuristic: Arc<dyn HeuristicModel>,
        graph: &LayeredGraph,
    ) -> f64 {
        if ordering.is_empty() {
            return 0.0;
        }

        let mut total_cost = 0.0;
        let mut current = 0usize; // depot index in context matrices

        for &next in ordering {
            if current == next {
                continue;
            }

            let path = a_star(graph, current, next, heuristic.as_ref())
                .or_else(|| bidirectional_dijkstra(graph, current, next));

            match path {
                Some(PathResult { cost, .. }) => {
                    total_cost += cost;
                    current = next;
                }
                None => return f64::INFINITY,
            }
        }

        if let Some(PathResult { cost, .. }) =
            a_star(graph, current, 0, heuristic.as_ref())
                .or_else(|| bidirectional_dijkstra(graph, current, 0))
        {
            total_cost += cost;
        } else {
            return f64::INFINITY;
        }

        total_cost
    }

    fn sample_pairs<R: Rng + ?Sized>(&self, nodes: &[usize], rng: &mut R) -> Vec<(usize, usize)> {
        let mut set = HashSet::new();
        set.insert((0, nodes[0]));
        for &node in nodes {
            set.insert((0, node));
        }
        for _ in 0..nodes.len() {
            let a = *nodes.choose(rng).unwrap_or(&nodes[0]);
            let b = *nodes.choose(rng).unwrap_or(&nodes[0]);
            if a != b {
                set.insert((a, b));
            }
        }
        set.into_iter().collect()
    }
}

struct ClusterResult {
    nodes: Vec<usize>,
    cost: f64,
    weights: (f64, f64),
    ga_params: GeneticHyperParams,
}
