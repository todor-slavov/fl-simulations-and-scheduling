import hydra
from omegaconf import DictConfig, OmegaConf
from dataset import prepare_dataset, prepare_dataset_nonIID, prepare_dataset_nonIID_varying_sizes
from client import generate_client_fn
import flwr as fl
from server import get_on_fit_config, get_evaluate_fn
from schedule import read_schedule, get_non_IID_schedule, get_IID_schedule

import matplotlib.pyplot as plt
import csv
import pickle
import h5py
from dataclasses import dataclass

from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg
from async_server import AsyncServer
from async_strategy import AsynchronousStrategy

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    ## 1. Parse config

    print(OmegaConf.to_yaml(cfg))

    ## 2. Prepare dataset
    trainloaders, validationloaders, testloader = prepare_dataset_nonIID_varying_sizes(cfg.num_clients, cfg.batch_size)
    # trainloaders, validationloaders, testloader = prepare_dataset(cfg.num_clients, cfg.batch_size)
    print(len(trainloaders), len(trainloaders[0].dataset))

    ## 3. Define your clients
    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes)

    ## 4. Define your strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.0,  # in simulation, since all clients are available at all times, we can just use `min_fit_clients` to control exactly how many clients we want to involve during fit
        min_fit_clients=cfg.num_clients_per_round_fit,  # number of clients to sample for fit()
        fraction_evaluate=0.0,  # similar to fraction_fit, we don't need to use this argument.
        min_evaluate_clients=cfg.num_clients_per_round_eval,  # number of clients to sample for evaluate()
        min_available_clients=cfg.num_clients,  # total clients in the simulation
        on_fit_config_fn=get_on_fit_config(
            {
                "lr": 0.005,
                "momentum": 0.9,
                "local_epochs": 2,
                "server_age": 0 
            }
        ),  # a function to execute to obtain the configuration to send to the clients during fit()
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader),
    )  #

    # schedule = read_schedule('schedule.txt')
    # print(schedule)
    # schedule = get_non_IID_schedule(175)
    # schedule = [8, 7, 2, 9, 5, 4, 0, 1, 3, 8, 7, 2, 9, 6, 5, 0, 4, 3, 1, 8, 2, 9, 7, 5, 6, 0, 3, 4, 1, 8, 9, 2, 7, 5, 6, 3, 0, 1, 4, 8, 9, 2, 7, 5, 6, 3, 1, 0, 9, 8, 2, 4, 7, 5, 6, 3, 1, 9, 0, 2, 8, 7, 4, 5, 3, 6, 9, 1, 2, 0, 8, 7, 5, 4, 3, 9, 6, 1, 2, 8, 0, 7, 5, 3, 9, 4, 6, 1, 2, 8, 7, 0, 3, 9, 5, 6, 4, 2, 8, 1, 7, 3, 9, 0, 5, 6, 2, 8, 4, 1, 7, 9, 3, 5, 0, 2, 6, 8, 1, 4, 9, 7, 3, 5, 2, 0, 6, 8, 1, 9, 7, 3, 4, 2, 5, 0, 6, 8, 9, 1, 7, 3, 2, 4, 5, 6, 0, 9, 8, 1, 7, 2, 3, 5, 4, 6, 9, 8, 0, 7, 2, 1, 3, 5, 6, 9, 4, 8, 7, 0, 2, 3, 1, 5, 9, 6, 8, 7, 4]
    # schedule = [7, 8, 9, 5, 0, 1, 2, 3, 4, 7, 9, 2, 5, 0, 6, 8, 4, 8, 3, 1, 9, 2, 7, 0, 5, 6, 3, 1, 9, 4, 8, 7, 6, 2, 3, 0, 5, 4, 8, 2, 9, 1, 6, 5, 7, 0, 1, 3, 9, 2, 4, 8, 5, 7, 6, 1, 3, 9, 0, 8, 4, 2, 5, 7, 3, 1, 6, 9, 8, 0, 2, 7, 5, 9, 3, 4, 2, 6, 0, 8, 1, 3, 7, 5, 9, 4, 1, 8, 6, 2, 3, 7, 0, 5, 4, 9, 2, 8, 7, 1, 9, 3, 5, 6, 0, 6, 4, 8, 7, 9, 1, 3, 5, 2, 0, 8, 1, 6, 2, 4, 3, 9, 2, 5, 0, 7, 1, 6, 8, 9, 3, 7, 4, 2, 5, 0, 6, 8, 9, 1, 7, 3, 4, 2, 6, 5, 9, 8, 0, 1, 7, 2, 5, 3, 9, 8, 4, 0, 7, 6, 1, 2, 5, 9, 4, 8, 3, 6, 0, 3, 2, 1, 9, 5, 7, 8, 4, 6, 7]
    schedule = [7, 8, 9, 2, 4, 0, 1, 5, 7, 8, 2, 5, 0, 4, 6, 9, 3, 8, 2, 7, 1, 5, 3, 9, 3, 1, 6, 4, 0, 9, 8, 2, 6, 5, 1, 7, 3, 0, 8, 4, 5, 6, 9, 3, 0, 1, 7, 9, 2, 4, 8, 2, 5, 7, 9, 3, 2, 1, 7, 6, 8, 0, 5, 4, 1, 3, 6, 2, 0, 7, 9, 8, 5, 3, 6, 1, 8, 2, 4, 9, 0, 5, 9, 3, 4, 7, 2, 1, 8, 6, 0, 9, 3, 7, 2, 8, 6, 5, 3, 7, 0, 1, 6, 4, 2, 9, 8, 4, 7, 1, 5, 9, 2, 6, 5, 8, 3, 4, 0, 9, 5, 7, 3, 1, 6, 1, 2, 8, 3, 4, 7, 0, 9, 0, 5, 9, 8, 1, 6, 2, 7, 2, 3, 6, 0, 4, 5, 9, 8, 2, 3, 4, 5, 1, 6, 9, 7, 0, 8, 3, 2, 1, 7, 9, 6, 5, 7, 4, 0, 2, 8, 3, 6, 8, 7, 1, 9, 4, 5]
    schedule = get_IID_schedule(180)
    print(schedule)

    ## 5. Start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,  # a function that spawns a particular client
        num_clients=cfg.num_clients,  # total number of clients
        config=fl.server.ServerConfig(
            num_rounds=cfg.num_rounds
        ),  # minimal config for the server loop telling the number of rounds in FL
        # strategy=strategy,  # our strategy of choice
        client_resources={
            "num_cpus": 4,
            "num_gpus": 0.0,
        },  # (optional) controls the degree of parallelism of your simulation.
        # Lower resources per client allow for more clients to run concurrently
        # (but need to be set taking into account the compute/memory footprint of your run)
        # `num_cpus` is an absolute number (integer) indicating the number of threads a client should be allocated
        # `num_gpus` is a ratio indicating the portion of gpu memory that a client needs.
        
        server=AsyncServer(strategy=strategy, max_workers=10, client_manager=SimpleClientManager(), \
            async_strategy=AsynchronousStrategy(total_samples=len(trainloaders[0].dataset), alpha=0.1, staleness_alpha=0.5, fedasync_mixing_alpha=0.5, fedasync_a=0.9, num_clients=cfg.num_clients, async_aggregation_strategy="fedasync", use_staleness=True, use_sample_weighing=False), \
            num_clients=cfg.num_clients, schedule=schedule, with_schedule=True)
        
        #server=AsyncServer(strategy=fl.server.strategy.FedAvg(), client_manager=AsyncClientManager(), base_conf_dict={},async_strategy=AsynchronousStrategy(total_samples=len(trainloaders[0].dataset), alpha=0.5, staleness_alpha=0.5, fedasync_mixing_alpha=0.5, fedasync_a=0.5, num_clients=cfg.num_clients, async_aggregation_strategy="fedasync", use_staleness=False, use_sample_weighing=False))
    )

    ## 6. Save results

if __name__ == "__main__":
    eval: bool = False
    if not eval:
        main()
    else:
        trainloaders, validationloaders, testloader = prepare_dataset_nonIID(10, 64)
                



# 
# 1. 0.5922493681550126
# 2. 0.6920808761583824
# 3. 0.723251895534962
# 4. 0.7687447346251053
# 5. 0.5602358887952822

# 
# 1. 0.7434709351305813
# 2. 0.7624262847514743
# 3. 0.8133951137320977
# 4. 0.7716933445661331
# 5. 0.7413647851727043