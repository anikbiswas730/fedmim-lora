"""
FedMIM-LoRA: main federated training loop.

Run with:  python train.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

import config
from src.aggregation import extract_trainable_state, linear_aggregation
from src.checkpoint import load_resume_state, save_round_checkpoints
from src.client import train_client
from src.data import HFCifar100Wrapper, MIMDataCollator, generate_dirichlet_partitions
from src.model import initialize_global_model


def run_fedmim_lora():
    print("--- Bootstrapping FedMIM-LoRA Simulation ---")

    print("Loading CIFAR-100 via Hugging Face datasets...")
    full_dataset = HFCifar100Wrapper()

    print(f"Partitioning across {config.NUM_CLIENTS} clients "
          f"(Dirichlet alpha={config.DIRICHLET_ALPHA})...")
    client_datasets = generate_dirichlet_partitions(
        full_dataset, config.NUM_CLIENTS, config.DIRICHLET_ALPHA, config.NUM_CLASSES
    )

    collator = MIMDataCollator(num_patches=config.NUM_PATCHES, mask_ratio=config.MASKING_RATIO)
    client_loaders = [
        DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collator)
        for ds in client_datasets
    ]

    print("Instantiating global model (ViT-MIM + Fixed-A LoRA)...")
    models = {config.DEVICE_0: initialize_global_model(config.DEVICE_0)}
    if config.DEVICE_1 != config.DEVICE_0:
        # Force the second device's model to reuse the SAME lora_A as the
        # first, instead of an independently random one -- required for
        # the Fixed-A aggregation guarantee to hold across devices.
        shared_lora_a_state = {
            k: v.cpu() for k, v in models[config.DEVICE_0].state_dict().items() if "lora_A" in k
        }
        models[config.DEVICE_1] = initialize_global_model(
            config.DEVICE_1, shared_lora_a_state=shared_lora_a_state
        )
    else:
        models[config.DEVICE_1] = models[config.DEVICE_0]

    global_state = extract_trainable_state(models[config.DEVICE_0])

    global_state, loss_history, start_round, best_loss = load_resume_state(
        config.RESUME_CHECKPOINT_PATH, global_state, models
    )

    num_selected = max(1, int(config.NUM_CLIENTS * config.FRACTION_FIT))

    for round_num in range(start_round, config.NUM_ROUNDS):
        print(f"\n--- Global Round {round_num + 1}/{config.NUM_ROUNDS} ---")
        selected_clients = np.random.choice(range(config.NUM_CLIENTS), num_selected, replace=False)

        client_updates, client_weights, round_losses = [], [], []

        for idx, client_id in enumerate(selected_clients):
            target_device = config.DEVICE_0 if idx % 2 == 0 else config.DEVICE_1

            local_state, loss = train_client(
                model=models[target_device],
                dataloader=client_loaders[client_id],
                global_state=global_state,
                device=target_device,
            )

            client_updates.append(local_state)
            client_weights.append(len(client_datasets[client_id]))
            round_losses.append(loss)

            print(f"  -> Client {client_id} done | MSE Loss: {loss:.4f}")

        print("  -> Aggregating (Fixed-A linear averaging of lora_B)...")
        global_state = linear_aggregation(client_updates, client_weights)
        round_avg_loss = float(np.mean(round_losses))
        loss_history.append(round_avg_loss)
        print(f"  -> Global avg loss: {round_avg_loss:.4f}")

        best_loss = save_round_checkpoints(
            config.CHECKPOINT_DIR, round_num, global_state, loss_history, round_avg_loss, best_loss
        )

    print("\n--- Training complete ---")
    final_model = models[config.DEVICE_0]
    final_model.load_state_dict(global_state, strict=False)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    final_model.save_pretrained(config.OUTPUT_DIR)
    print(f"Final model saved to {config.OUTPUT_DIR}")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker="o")
    plt.xlabel("Global Round")
    plt.ylabel("Average Client Loss")
    plt.title("FedMIM-LoRA Training Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png")

    return loss_history


if __name__ == "__main__":
    run_fedmim_lora()
