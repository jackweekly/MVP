use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LayerId(pub usize);

#[derive(Debug, Clone)]
pub struct Edge {
    pub target: usize,
    pub cost: f64,
}

#[derive(Clone)]
pub struct LayeredGraph {
    adjacency: Vec<Vec<Edge>>,
    layers: Vec<LayerId>,
}

impl LayeredGraph {
    pub fn new(layers: Vec<LayerId>, adjacency: Vec<Vec<Edge>>) -> Self {
        debug_assert!(layers.len() == adjacency.len());
        Self { adjacency, layers }
    }

    pub fn from_dense_costs(layers: Vec<LayerId>, costs: &[Vec<f64>]) -> Self {
        let adjacency = costs
            .iter()
            .enumerate()
            .map(|(idx, row)| {
                row.iter()
                    .enumerate()
                    .filter_map(|(j, &cost)| {
                        if idx == j || !cost.is_finite() {
                            None
                        } else {
                            Some(Edge { target: j, cost })
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        Self::new(layers, adjacency)
    }

    pub fn node_count(&self) -> usize {
        self.adjacency.len()
    }

    pub fn layer_of(&self, node: usize) -> LayerId {
        self.layers[node]
    }

    pub fn neighbors(&self, node: usize) -> &[Edge] {
        &self.adjacency[node]
    }
}

impl fmt::Debug for LayeredGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LayeredGraph")
            .field("nodes", &self.node_count())
            .finish()
    }
}
