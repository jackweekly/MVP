use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};

pub type FitnessFn<'a> = &'a dyn Fn(&[usize]) -> f64;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct GeneticHyperParams {
    pub population_size: usize,
    pub crossover_rate: f64,
    pub mutation_rate: f64,
    pub elite_count: usize,
    pub generations: usize,
}

impl GeneticHyperParams {
    pub fn clamp(self) -> Self {
        Self {
            population_size: self.population_size.max(4),
            crossover_rate: self.crossover_rate.clamp(0.0, 1.0),
            mutation_rate: self.mutation_rate.clamp(0.0, 1.0),
            elite_count: self.elite_count.min(self.population_size.saturating_sub(1)),
            generations: self.generations.max(1),
        }
    }
}

pub struct GeneticAlgorithm {
    params: GeneticHyperParams,
}

impl GeneticAlgorithm {
    pub fn new(params: GeneticHyperParams) -> Self {
        Self {
            params: params.clamp(),
        }
    }

    pub fn run<R: Rng + ?Sized>(
        &self,
        items: &[usize],
        rng: &mut R,
        fitness: FitnessFn,
    ) -> Vec<usize> {
        if items.is_empty() {
            return Vec::new();
        }

        let mut population = self.initial_population(items, rng);
        let mut best = population[0].clone();
        let mut best_score = fitness(&best);

        for _ in 0..self.params.generations {
            population.sort_by(|a, b| fitness(a).partial_cmp(&fitness(b)).unwrap());
            let elite = population[..self.params.elite_count.max(1)].to_vec();

            if let Some((candidate, fitness_value)) = elite
                .iter()
                .map(|chromosome| (chromosome, fitness(chromosome)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            {
                if fitness_value + f64::EPSILON < best_score {
                    best = candidate.clone();
                    best_score = fitness_value;
                }
            }

            let mut next_population = elite;

            while next_population.len() < self.params.population_size {
                let parents = self.select_parents(&population, fitness, rng);
                let mut child = if rng.gen::<f64>() < self.params.crossover_rate {
                    self.order_crossover(&parents.0, &parents.1, rng)
                } else {
                    parents.0.clone()
                };

                if rng.gen::<f64>() < self.params.mutation_rate {
                    self.swap_mutation(&mut child, rng);
                }

                next_population.push(child);
            }

            population = next_population;
        }

        best
    }

    fn initial_population<R: Rng + ?Sized>(
        &self,
        items: &[usize],
        rng: &mut R,
    ) -> Vec<Vec<usize>> {
        let mut population = Vec::with_capacity(self.params.population_size);
        let mut base = items.to_vec();
        for _ in 0..self.params.population_size {
            base.shuffle(rng);
            population.push(base.clone());
        }
        population
    }

    fn select_parents<'a, R: Rng + ?Sized>(
        &self,
        population: &'a [Vec<usize>],
        fitness: FitnessFn,
        rng: &mut R,
    ) -> (&'a Vec<usize>, &'a Vec<usize>) {
        let tournament_size = (population.len() / 5).max(2);
        let parent_a = self.tournament(population, fitness, tournament_size, rng);
        let parent_b = self.tournament(population, fitness, tournament_size, rng);
        (parent_a, parent_b)
    }

    fn tournament<'a, R: Rng + ?Sized>(
        &self,
        population: &'a [Vec<usize>],
        fitness: FitnessFn,
        tournament_size: usize,
        rng: &mut R,
    ) -> &'a Vec<usize> {
        let mut best: Option<(&Vec<usize>, f64)> = None;
        for _ in 0..tournament_size {
            if let Some(candidate) = population.choose(rng) {
                let score = fitness(candidate);
                if best
                    .as_ref()
                    .map(|(_, best_score)| score + f64::EPSILON < *best_score)
                    .unwrap_or(true)
                {
                    best = Some((candidate, score));
                }
            }
        }
        best.map(|(chromosome, _)| chromosome).unwrap_or(&population[0])
    }

    fn order_crossover<R: Rng + ?Sized>(
        &self,
        parent_a: &[usize],
        parent_b: &[usize],
        rng: &mut R,
    ) -> Vec<usize> {
        if parent_a.len() < 2 {
            return parent_a.to_vec();
        }

        let len = parent_a.len();
        let idx1 = rng.gen_range(0..len);
        let idx2 = rng.gen_range(idx1..len);

        let mut child = vec![usize::MAX; len];
        child[idx1..=idx2].copy_from_slice(&parent_a[idx1..=idx2]);

        let mut position = (idx2 + 1) % len;
        for gene in parent_b {
            if child.contains(gene) {
                continue;
            }
            child[position] = *gene;
            position = (position + 1) % len;
        }

        for gene in child.iter_mut() {
            if *gene == usize::MAX {
                *gene = parent_a[0];
            }
        }

        child
    }

    fn swap_mutation<R: Rng + ?Sized>(&self, chromosome: &mut [usize], rng: &mut R) {
        if chromosome.len() < 2 {
            return;
        }
        let a = rng.gen_range(0..chromosome.len());
        let mut b = rng.gen_range(0..chromosome.len());
        if a == b {
            b = (b + 1) % chromosome.len();
        }
        chromosome.swap(a, b);
    }
}
