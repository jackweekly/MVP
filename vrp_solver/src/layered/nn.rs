use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

#[derive(Clone, Copy, Debug)]
pub struct EdgeFeatures {
    pub from: usize,
    pub to: usize,
    pub distance: f64,
    pub duration: f64,
    pub layer_same: bool,
    pub degree_from: u32,
    pub degree_to: u32,
}

pub trait EdgePredictor: Send + Sync {
    fn predict(&self, features: &EdgeFeatures) -> f64;
}

#[derive(Deserialize)]
struct DenseLayer {
    weights: Vec<Vec<f64>>,
    bias: Vec<f64>,
}

#[derive(Deserialize)]
struct MoveRegressorModel {
    layers: Vec<DenseLayer>,
    #[serde(default)]
    feature_mean: Option<Vec<f64>>,
    #[serde(default)]
    feature_std: Option<Vec<f64>>,
}

impl MoveRegressorModel {
    fn forward(&self, mut input: Vec<f64>) -> f64 {
        if let (Some(mean), Some(std)) = (&self.feature_mean, &self.feature_std) {
            for (idx, value) in input.iter_mut().enumerate() {
                let mu = mean.get(idx).copied().unwrap_or(0.0);
                let sigma = std.get(idx).copied().unwrap_or(1.0).max(1e-6);
                *value = (*value - mu) / sigma;
            }
        }

        let mut activations = input;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let mut next = vec![0.0; layer.bias.len()];
            for (out_idx, bias) in layer.bias.iter().enumerate() {
                let mut sum = *bias;
                if let Some(weights) = layer.weights.get(out_idx) {
                    for (in_idx, weight) in weights.iter().enumerate() {
                        let value = activations.get(in_idx).copied().unwrap_or(0.0);
                        sum += weight * value;
                    }
                }
                if layer_idx + 1 == self.layers.len() {
                    next[out_idx] = sum;
                } else {
                    next[out_idx] = sum.max(0.0);
                }
            }
            activations = next;
        }

        activations.into_iter().next().unwrap_or(0.0)
    }
}

pub struct MoveRegressorPredictor {
    model: Option<MoveRegressorModel>,
    cache_path: Option<PathBuf>,
    cache: Mutex<HashMap<(usize, usize), f64>>,
}

impl MoveRegressorPredictor {
    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let path_ref = path.as_ref();
        if let Ok(file) = File::open(path_ref) {
            let model: MoveRegressorModel = serde_json::from_reader(file).unwrap_or_else(|_| MoveRegressorModel {
                layers: Vec::new(),
                feature_mean: None,
                feature_std: None,
            });
            Ok(Self {
                model: Some(model),
                cache_path: Some(path_ref.with_extension("cache.json")),
                cache: Mutex::new(HashMap::new()),
            })
        } else {
            Ok(Self {
                model: None,
                cache_path: None,
                cache: Mutex::new(HashMap::new()),
            })
        }
    }

    fn run_model(&self, features: Vec<f64>) -> f64 {
        if let Some(model) = &self.model {
            model.forward(features)
        } else {
            // Without a trained model fall back to zero residual so the base heuristic dominates.
            0.0
        }
    }

    pub fn persist_cache(&self) {
        if let Some(path) = &self.cache_path {
            if let Ok(cache) = self.cache.lock() {
                if let Some(parent) = path.parent() {
                    if !parent.as_os_str().is_empty() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                }
                if let Ok(file) = File::create(path) {
                    let _ = serde_json::to_writer(file, &*cache);
                }
            }
        }
    }
}

impl EdgePredictor for MoveRegressorPredictor {
    fn predict(&self, features: &EdgeFeatures) -> f64 {
        if let Ok(mut cache) = self.cache.lock() {
            let key = (features.from, features.to);
            if let Some(value) = cache.get(&key) {
                return *value;
            }
            let input = vec![
                features.distance,
                features.duration,
                if features.layer_same { 1.0 } else { 0.0 },
                features.degree_from as f64,
                features.degree_to as f64,
            ];
            let value = self.run_model(input);
            cache.insert(key, value);
            value
        } else {
            let input = vec![
                features.distance,
                features.duration,
                if features.layer_same { 1.0 } else { 0.0 },
                features.degree_from as f64,
                features.degree_to as f64,
            ];
            self.run_model(input)
        }
    }
}

impl Drop for MoveRegressorPredictor {
    fn drop(&mut self) {
        self.persist_cache();
    }
}
