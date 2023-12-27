from train import run_sbm, load_config
import sys
import torch

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train.py <config_file_path>")
        sys.exit(1)

    config = load_config(sys.argv[1])
    config.update({"device": "cuda" if torch.cuda.is_available() else "cpu"})
    config.update({"device": "cuda" if torch.cuda.is_available() else "cpu"})
    config.get("model").update({"gnn_type": "graphsage"})

    for gating in [0, 1.0, 2.0, 3.0]:
        print("Testing with gradient gating parameter p = " + str(gating))
        config.get("model").update({"gradient_gating": gating})
        config.get("logger").update({"run_name": "gating_p="+str(gating)})
        run_sbm(config)
        