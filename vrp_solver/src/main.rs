use axum::{routing::post, Json, Router};
use std::net::SocketAddr;
// Import your existing structs and solver function from your library
use vrp_solver::{solve_vrp, Problem, Solution};

// This is the main entry point for your API application
#[tokio::main]
async fn main() {
    // Define the application's routes
    let app = Router::new().route("/optimize", post(run_optimization));

    // Define the address and port to run the server on
    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    println!("🚀 Listening on http://{}", addr);

    // Run the server
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

// This is the handler function for the /optimize endpoint
async fn run_optimization(Json(problem): Json<Problem>) -> Json<Solution> {
    // Axum's Json extractor automatically deserializes the incoming JSON
    // into your `Problem` struct.

    // Here, you would call your more complex solver logic.
    // For now, let's create a placeholder solution.
    // NOTE: You'll need to adapt your `solve_vrp` function to work with
    // structs directly instead of JSON strings, or adapt this handler.

    let solution = solve_vrp(problem).await;

    // The return `Json(solution)` automatically serializes your `Solution`
    // struct back into a JSON response.
    Json(solution)
}
