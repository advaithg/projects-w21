import argparse
import os
import time
import torch

import constants
from datasets.StartingDataset import StartingDataset
from networks.StartingNetwork import ConvNet
from train_functions.starting_train import starting_train
from networks.FineTune import initialize_model


SUMMARIES_PATH = "training_summaries"
num_classes  = 5


def main():
    images_dir = "./cassava-leaf-disease-classification/train_images"

    # Get command line arguments    
    args = parse_arguments()
    hyperparameters = {"epochs": args.epochs, "batch_size": args.batch_size, "weight_decay": args.weight_decay}

    # Create path for training summaries
    label = f"cassava__{int(time.time())}"
    summary_path = f"{SUMMARIES_PATH}/{label}"
    os.makedirs(summary_path, exist_ok=True)

    # TODO: Add GPU support. This line of code might be helpful.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #print(torch.cuda.get_device_name(0))

    print("Summary path:", summary_path)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)

    # Initalize dataset and model. Then train the model!
    train_dataset = StartingDataset(images_dir, 'train.csv')
    val_dataset = StartingDataset(images_dir, 'val.csv')
    model = ConvNet(num_classes, args.network, args.p_dropout)
    #train_dataset.__showitem__(0)
    #print(model)


    starting_train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        hyperparameters=hyperparameters,
        n_eval=args.n_eval,
        summary_path=summary_path,
        device=device,
        usePretrained = args.use_pretrained
    )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=constants.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=constants.BATCH_SIZE)
    parser.add_argument("--n_eval", type=int, default=constants.N_EVAL)
    parser.add_argument("--use_pretrained", type = bool, default = False)
    parser.add_argument("--network", type=str, default = "resnet18" )
    parser.add_argument("--weight_decay", type=float, default = 0.00001)
    parser.add_argument("--p_dropout", type=float, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
