import json
from vrp_solver import solve_vrp # Import the function from your Rust library

def main():
    """
    Defines a simple Vehicle Routing Problem and solves it using the Rust solver.
    """
    # 1. Define the problem as a Python dictionary
    problem_data = {
        "depot": {"id": "depot", "x": 0.0, "y": 0.0},
        "locations": [
            {"id": "loc_1", "x": 1.0, "y": 5.0},
            {"id": "loc_2", "x": -3.0, "y": -3.0},
            {"id": "loc_3", "x": 5.0, "y": 2.0},
            {"id": "loc_4", "x": -2.0, "y": 4.0},
        ],
        "vehicles": [
            {"id": "vehicle_1", "capacity": 10},
            {"id": "vehicle_2", "capacity": 10},
        ]
    }

    # 2. Convert the dictionary to a JSON string
    problem_json = json.dumps(problem_data)

    print("--- Sending problem to Rust solver ---")
    print(problem_json)

    # 3. Call the Rust function
    try:
        solution_json = solve_vrp(problem_json)
        solution = json.loads(solution_json)

        print("\n--- Received solution from Rust solver ---")
        for i, route in enumerate(solution.get("routes", [])):
            vehicle_id = route.get("vehicle_id")
            path = " -> ".join(route.get("locations", []))
            distance = route.get("total_distance", 0)
            print(f"Route #{i+1} ({vehicle_id}): {path} (Distance: {distance:.2f})")
        
        print(f"\nTotal Cost: {solution.get('total_cost', 0):.2f}")

    except Exception as e:
        print(f"\n--- An error occurred ---")
        print(e)


if __name__ == "__main__":
    main()
