# server.py
import flwr as fl

if __name__ == "__main__":
    # Start Flower server
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=3  # require all 3 clients
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy
    )
