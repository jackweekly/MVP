use rand::Rng;

#[derive(Clone, Copy, Debug)]
pub struct PsoHyperParams {
    pub particles: usize,
    pub inertia: f64,
    pub cognitive: f64,
    pub social: f64,
    pub iterations: usize,
}

impl Default for PsoHyperParams {
    fn default() -> Self {
        Self {
            particles: 20,
            inertia: 0.7,
            cognitive: 1.4,
            social: 1.4,
            iterations: 50,
        }
    }
}

pub fn optimize_weights<R, F>(
    dimension: usize,
    bounds: &[(f64, f64)],
    params: PsoHyperParams,
    mut evaluate: F,
    rng: &mut R,
) -> Vec<f64>
where
    R: Rng + ?Sized,
    F: FnMut(&[f64]) -> f64,
{
    let particles = params.particles.max(4);
    let iterations = params.iterations.max(1);

    let mut positions: Vec<Vec<f64>> = (0..particles)
        .map(|_| random_vector(dimension, bounds, rng))
        .collect();
    let mut velocities: Vec<Vec<f64>> = vec![vec![0.0; dimension]; particles];
    let mut personal_best = positions.clone();
    let mut personal_best_score: Vec<f64> = personal_best
        .iter()
        .map(|position| evaluate(position))
        .collect();

    let mut global_best_idx = personal_best_score
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    for _ in 0..iterations {
        for i in 0..particles {
            let position = &mut positions[i];
            let velocity = &mut velocities[i];
            let best_position = &personal_best[i];
            let global_best_position = &personal_best[global_best_idx];

            for d in 0..dimension {
                let r_p = rng.gen::<f64>();
                let r_g = rng.gen::<f64>();
                velocity[d] = params.inertia * velocity[d]
                    + params.cognitive * r_p * (best_position[d] - position[d])
                    + params.social * r_g * (global_best_position[d] - position[d]);

                position[d] = clamp(position[d] + velocity[d], bounds[d]);
            }

            let score = evaluate(position);
            if score + f64::EPSILON < personal_best_score[i] {
                personal_best_score[i] = score;
                personal_best[i] = position.clone();
                if score + f64::EPSILON < personal_best_score[global_best_idx] {
                    global_best_idx = i;
                }
            }
        }
    }

    personal_best[global_best_idx].clone()
}

fn random_vector<R: Rng + ?Sized>(dimension: usize, bounds: &[(f64, f64)], rng: &mut R) -> Vec<f64> {
    (0..dimension)
        .map(|i| {
            let (min, max) = bounds.get(i).copied().unwrap_or((0.0, 1.0));
            rng.gen_range(min..=max)
        })
        .collect()
}

fn clamp(value: f64, (min, max): (f64, f64)) -> f64 {
    value.max(min).min(max)
}
