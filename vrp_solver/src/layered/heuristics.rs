use super::graph::LayeredGraph;
use std::sync::Arc;

pub trait HeuristicModel: Send + Sync {
    fn estimate(&self, from: usize, to: usize, graph: &LayeredGraph) -> f64;
}

#[derive(Clone)]
pub struct WeightedDurationHeuristic {
    durations: Arc<Vec<Vec<f64>>>,
    scale: f64,
    layer_penalty: f64,
}

impl WeightedDurationHeuristic {
    pub fn new(durations: Arc<Vec<Vec<f64>>>, scale: f64, layer_penalty: f64) -> Self {
        Self {
            durations,
            scale,
            layer_penalty,
        }
    }
}

impl HeuristicModel for WeightedDurationHeuristic {
    fn estimate(&self, from: usize, to: usize, graph: &LayeredGraph) -> f64 {
        let raw = self.durations[from][to];
        let penalty = if graph.layer_of(from) == graph.layer_of(to) {
            0.0
        } else {
            self.layer_penalty
        };

        raw * self.scale + penalty
    }
}

pub struct NeuralBlendHeuristic<H> {
    base: WeightedDurationHeuristic,
    predictor: H,
    nn_weight: f64,
}

impl<H> NeuralBlendHeuristic<H>
where
    H: Fn(usize, usize) -> f64 + Send + Sync,
{
    pub fn new(base: WeightedDurationHeuristic, predictor: H, nn_weight: f64) -> Self {
        Self {
            base,
            predictor,
            nn_weight,
        }
    }
}

impl<H> HeuristicModel for NeuralBlendHeuristic<H>
where
    H: Fn(usize, usize) -> f64 + Send + Sync,
{
    fn estimate(&self, from: usize, to: usize, graph: &LayeredGraph) -> f64 {
        let base = self.base.estimate(from, to, graph);
        let nn = (self.predictor)(from, to);
        base * (1.0 - self.nn_weight) + nn * self.nn_weight
    }
}
