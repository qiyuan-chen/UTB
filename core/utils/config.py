import argparse

def get_config():
    parser = argparse.ArgumentParser(description="Tabular Data Predication (UTB)")
    
    """
    Saving & loading of the model.
    """
    parser.add_argument("--save_dir", type=str, default="./saved_models")
    parser.add_argument("-sn", "--save_name", type=str, default="mlp")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--load_path", type=str)
    parser.add_argument("-o", "--overwrite", action="store_true", default=True)
    parser.add_argument(
        "--use_tensorboard",
        action="store_true",
        help="Use tensorboard to plot and save curves",
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Use wandb to plot and save curves"
    )
    parser.add_argument(
        "--use_aim", action="store_true", help="Use aim to plot and save curves"
    )
    
    """
    Data Configurations
    """

    ## standard setting configurations
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("-ds", "--dataset", type=str, default="iris")
    parser.add_argument("-nc", "--num_classes", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=1)
    