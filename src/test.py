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

    # for gating in [0, 1.0, 2.0, 3.0]:
    #     print("Testing with gradient gating parameter p = " + str(gating))
    #     config.get("model").update({"gradient_gating": gating})
    #     config.get("logger").update({"run_name": "gating_p="+str(gating)})
    #     run_sbm(config)

    # for k_hop in [3,8,16,32,64]:
    #     config.get("model").update({"k_hop": k_hop})
    #     config.get("logger").update({"run_name": "k_hop="+str(k_hop)})
    #     print(config)
    #     run_sbm(config)

    for k_hop in [16, 3, 5, 8]:
        for gating in [1.0, 2.0]:
                print("Testing with gradient gating parameter p = " + str(gating) + "and k_hop=" + str(k_hop))
                config.get("model").update({"gradient_gating_p": gating})
                config.get("model").update({"k_hop": k_hop})
                config.get("logger").update({"run_name": "gating_p="+str(gating)+",k_hop="+str(k_hop)})
                run_sbm(config) 
