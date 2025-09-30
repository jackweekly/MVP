use rand::Rng;
use rand_distr::{Distribution, Normal};

#[derive(Clone, Copy, Debug)]
pub struct ParameterBounds {
    pub min: f64,
    pub max: f64,
}

impl ParameterBounds {
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        rng.gen_range(self.min..=self.max)
    }

    pub fn clamp(&self, value: f64) -> f64 {
        value.clamp(self.min, self.max)
    }
}

#[derive(Clone, Debug)]
pub struct BayesianOptimizer {
    observations: Vec<(Vec<f64>, f64)>,
    exploration: f64,
}

impl BayesianOptimizer {
    pub fn new(exploration: f64) -> Self {
        Self {
            observations: Vec::new(),
            exploration,
        }
    }

    pub fn observe(&mut self, params: Vec<f64>, score: f64) {
        self.observations.push((params, score));
    }

    pub fn suggest<R: Rng + ?Sized>(
        &self,
        bounds: &[ParameterBounds],
        rng: &mut R,
    ) -> Vec<f64> {
        if self.observations.len() < bounds.len() + 1 {
            return bounds.iter().map(|range| range.sample(rng)).collect();
        }

        let (best_params, _best_score) = self
            .observations
            .iter()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .cloned()
            .unwrap();

        let std_dev = self.estimate_std();
        bounds
            .iter()
            .enumerate()
            .map(|(idx, range)| {
                let mean = best_params[idx];
                let variation = std_dev * self.exploration;
                let normal = Normal::new(mean, variation.max(1e-6))
                    .unwrap_or_else(|_| Normal::new(mean, 1e-6).unwrap());
                let candidate = normal.sample(rng);
                range.clamp(candidate)
            })
            .collect()
    }

    fn estimate_std(&self) -> f64 {
        if self.observations.len() < 2 {
            return 1.0;
        }
        let mean = self
            .observations
            .iter()
            .map(|(_, score)| *score)
            .sum::<f64>()
            / self.observations.len() as f64;
        let variance = self
            .observations
            .iter()
            .map(|(_, score)| (*score - mean).powi(2))
            .sum::<f64>()
            / (self.observations.len() as f64 - 1.0);
        variance.sqrt().max(1e-6)
    }

    pub fn best_observation(&self) -> Option<&(Vec<f64>, f64)> {
        self.observations
            .iter()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    }
}
