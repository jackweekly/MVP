use super::graph::LayeredGraph;
use super::heuristics::HeuristicModel;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

#[derive(Debug, Clone)]
pub struct PathResult {
    pub cost: f64,
    pub nodes: Vec<usize>,
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct QueueState {
    node: usize,
    cost: f64,
    priority: f64,
}

impl Eq for QueueState {}

impl Ord for QueueState {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .priority
            .partial_cmp(&self.priority)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.node.cmp(&self.node))
    }
}

impl PartialOrd for QueueState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub fn a_star(
    graph: &LayeredGraph,
    start: usize,
    goal: usize,
    heuristic: &dyn HeuristicModel,
) -> Option<PathResult> {
    if start == goal {
        return Some(PathResult {
            cost: 0.0,
            nodes: vec![start],
        });
    }

    let mut open = BinaryHeap::new();
    open.push(QueueState {
        node: start,
        cost: 0.0,
        priority: 0.0,
    });

    let mut came_from: HashMap<usize, usize> = HashMap::new();
    let mut g_score: HashMap<usize, f64> = HashMap::new();
    g_score.insert(start, 0.0);

    while let Some(current) = open.pop() {
        if current.node == goal {
            return Some(reconstruct_path(current.node, start, &came_from, current.cost));
        }

        for edge in graph.neighbors(current.node) {
            let tentative = current.cost + edge.cost;
            let entry = g_score.entry(edge.target).or_insert(f64::INFINITY);
            if tentative + f64::EPSILON < *entry {
                *entry = tentative;
                came_from.insert(edge.target, current.node);
                let h = heuristic.estimate(edge.target, goal, graph);
                open.push(QueueState {
                    node: edge.target,
                    cost: tentative,
                    priority: tentative + h,
                });
            }
        }
    }

    None
}

pub fn bidirectional_dijkstra(graph: &LayeredGraph, start: usize, goal: usize) -> Option<PathResult> {
    if start == goal {
        return Some(PathResult {
            cost: 0.0,
            nodes: vec![start],
        });
    }

    let mut forward_front = BinaryHeap::new();
    let mut backward_front = BinaryHeap::new();

    let mut forward_dist: HashMap<usize, f64> = HashMap::new();
    let mut backward_dist: HashMap<usize, f64> = HashMap::new();

    let mut forward_prev: HashMap<usize, usize> = HashMap::new();
    let mut backward_prev: HashMap<usize, usize> = HashMap::new();

    forward_front.push(QueueState {
        node: start,
        cost: 0.0,
        priority: 0.0,
    });
    backward_front.push(QueueState {
        node: goal,
        cost: 0.0,
        priority: 0.0,
    });

    forward_dist.insert(start, 0.0);
    backward_dist.insert(goal, 0.0);

    let mut best_cost = f64::INFINITY;
    let mut meeting_node: Option<usize> = None;
    let mut visited_forward: HashSet<usize> = HashSet::new();
    let mut visited_backward: HashSet<usize> = HashSet::new();

    while let (Some(f_state), Some(b_state)) = (forward_front.pop(), backward_front.pop()) {
        if f_state.cost + b_state.cost >= best_cost {
            break;
        }

        if visited_forward.insert(f_state.node) {
            for edge in graph.neighbors(f_state.node) {
                let tentative = f_state.cost + edge.cost;
                if tentative + f64::EPSILON < *forward_dist.get(&edge.target).unwrap_or(&f64::INFINITY)
                {
                    forward_dist.insert(edge.target, tentative);
                    forward_prev.insert(edge.target, f_state.node);
                    forward_front.push(QueueState {
                        node: edge.target,
                        cost: tentative,
                        priority: tentative,
                    });
                }

                if let Some(back_cost) = backward_dist.get(&edge.target) {
                    let total = tentative + back_cost;
                    if total + f64::EPSILON < best_cost {
                        best_cost = total;
                        meeting_node = Some(edge.target);
                    }
                }
            }
        }

        if visited_backward.insert(b_state.node) {
            for edge in graph.neighbors(b_state.node) {
                let tentative = b_state.cost + edge.cost;
                if tentative + f64::EPSILON
                    < *backward_dist.get(&edge.target).unwrap_or(&f64::INFINITY)
                {
                    backward_dist.insert(edge.target, tentative);
                    backward_prev.insert(edge.target, b_state.node);
                    backward_front.push(QueueState {
                        node: edge.target,
                        cost: tentative,
                        priority: tentative,
                    });
                }

                if let Some(forward_cost) = forward_dist.get(&edge.target) {
                    let total = tentative + forward_cost;
                    if total + f64::EPSILON < best_cost {
                        best_cost = total;
                        meeting_node = Some(edge.target);
                    }
                }
            }
        }
    }

    meeting_node.map(|mid| reconstruct_bidirectional_path(mid, start, goal, &forward_prev, &backward_prev, best_cost))
}

fn reconstruct_path(
    mut current: usize,
    start: usize,
    came_from: &HashMap<usize, usize>,
    cost: f64,
) -> PathResult {
    let mut path = VecDeque::new();
    path.push_front(current);
    while let Some(&prev) = came_from.get(&current) {
        current = prev;
        path.push_front(current);
        if current == start {
            break;
        }
    }
    PathResult {
        cost,
        nodes: path.into_iter().collect(),
    }
}

fn reconstruct_bidirectional_path(
    meeting: usize,
    start: usize,
    goal: usize,
    forward_prev: &HashMap<usize, usize>,
    backward_prev: &HashMap<usize, usize>,
    cost: f64,
) -> PathResult {
    let mut path = VecDeque::new();
    let mut current = meeting;
    path.push_front(current);

    while current != start {
        if let Some(&prev) = forward_prev.get(&current) {
            current = prev;
            path.push_front(current);
        } else {
            break;
        }
    }

    current = meeting;
    while current != goal {
        if let Some(&next) = backward_prev.get(&current) {
            current = next;
            path.push_back(current);
        } else {
            break;
        }
    }

    PathResult {
        cost,
        nodes: path.into_iter().collect(),
    }
}
