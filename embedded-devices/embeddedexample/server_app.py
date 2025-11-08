"""embeddedexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from embeddedexample.task import Net

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]

    # Load global model
    global_model = Net()

    # Load old model weights if available
    try:
        state_dict = torch.load("final_model.pt")
        global_model.load_state_dict(state_dict)
        print("Loaded existing model weights from 'final_model.pt'.")
    except FileNotFoundError:
        print("No existing model weights found. Initializing new model.")

    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_evaluate=fraction_evaluate)
    for i in range(10):
        # Start strategy, running FedAvg for `num_rounds`
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            num_rounds=num_rounds,
        )

        # Save final model to disk
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")
