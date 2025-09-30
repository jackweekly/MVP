use clap::{Parser, Subcommand};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use vrp_solver::{
    default_logger_path, fingerprint_problem, HybridConfig, LogFormat, LoggerConfig, RunEvent,
    RunRecorder, Solution,
};

#[derive(Parser)]
#[command(
    name = "vrp-dataset",
    version,
    about = "Utility helpers for VRP solver datasets"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Execute the solver across a batch of VRP instances and record logs.
    Collect {
        /// Path to a JSON array of VRP `Problem` definitions.
        #[arg(long)]
        input: PathBuf,
        /// Directory to write JSONL run logs into.
        #[arg(long, default_value = "logs")]
        output: PathBuf,
        /// Override the number of hybrid iterations (defaults to solver default).
        #[arg(long)]
        iterations: Option<usize>,
        /// Flush interval for the log writer.
        #[arg(long, default_value_t = 100usize)]
        flush_interval: usize,
    },
    /// Summarize a JSONL run log file (counts, moves, iterations).
    Summarize {
        #[arg(long)]
        input: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Command::Collect {
            input,
            output,
            iterations,
            flush_interval,
        } => collect_runs(input, output, iterations, flush_interval).await?,
        Command::Summarize { input } => summarize_log(input)?,
    }

    Ok(())
}

async fn collect_runs(
    input: PathBuf,
    output: PathBuf,
    iterations: Option<usize>,
    flush_interval: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(&input)?;
    let problems: Vec<vrp_solver::Problem> = serde_json::from_reader(file)?;

    println!("Loaded {} problem instances", problems.len());

    for (idx, problem) in problems.into_iter().enumerate() {
        let fingerprint = fingerprint_problem(&problem);
        let log_path = default_logger_path(&output, &format!("{}_{}", fingerprint, idx));
        let config = HybridConfig {
            iterations: iterations.unwrap_or(HybridConfig::default().iterations),
            ..HybridConfig::default()
        };
        let logger_config = LoggerConfig {
            output_path: log_path.clone(),
            format: LogFormat::JsonLines,
            flush_interval,
        };
        let recorder: RunRecorder = logger_config.ensure_writer()?;

        let solution: Solution =
            vrp_solver::solve_vrp_with_callbacks(problem, config, None, None, Some(recorder)).await;

        println!(
            "[{idx}] logged run to {} | total_duration={:.2}",
            log_path.display(),
            solution.total_cost
        );
    }

    Ok(())
}

fn summarize_log(path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(&path)?;
    let reader = BufReader::new(file);

    let mut run_starts = 0usize;
    let mut move_accepts = 0usize;
    let mut iterations = 0usize;

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let event: RunEvent = serde_json::from_str(&line)?;
        match event {
            RunEvent::RunStart { .. } => run_starts += 1,
            RunEvent::Iteration { .. } => iterations += 1,
            RunEvent::MoveAccepted { .. } => move_accepts += 1,
            RunEvent::LayeredPhase { .. } => {}
            RunEvent::RunComplete { .. } => {}
        }
    }

    println!("Runs: {}", run_starts);
    println!("Iterations logged: {}", iterations);
    println!("Moves accepted: {}", move_accepts);

    Ok(())
}
