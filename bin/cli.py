from datetime import datetime
from pathlib import Path

import typer
import torch
import git

from ser.train import train as run_train
from ser.constants import RESULTS_DIR
from ser.data import train_dataloader, val_dataloader, test_dataloader
from ser.params import Params, save_params
from ser.transforms import transforms, normalize

main = typer.Typer()


@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    epochs: int = typer.Option(
        5, "-e", "--epochs", help="Number of epochs to run for."
    ),
    batch_size: int = typer.Option(
        1000, "-b", "--batch-size", help="Batch size for dataloader."
    ),
    learning_rate: float = typer.Option(
        0.01, "-l", "--learning-rate", help="Learning rate for the model."
    ),
):
    """Run the training algorithm."""
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    # wraps the passed in parameters
    params = Params(name, epochs, batch_size, learning_rate, sha)

    # setup device to run on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup run
    fmt = "%Y-%m-%dT%H-%M"
    timestamp = datetime.strftime(datetime.utcnow(), fmt)
    run_path = RESULTS_DIR / name / timestamp
    run_path.mkdir(parents=True, exist_ok=True)

    # Save parameters for the run
    save_params(run_path, params)

    # Train!
    run_train(
        run_path,
        params,
        train_dataloader(params.batch_size, transforms(normalize)),
        val_dataloader(params.batch_size, transforms(normalize)),
        device,
    )

from ser.generate_art import generate_ascii_art
from ser.model import Net
from ser.run_summary import print_summary

@main.command()
def infer(
    run_name: str = typer.Option(
        ...,"-r","--run",help="The name of the run to infer from, corresponding to the folder name within 'results' folder."
    ),
):
    # find better way of defining run path
    # assumes we are in the "ser" directory where "/results" is accessible 
    run_path = Path("results/"+run_name)
    label = 6

    # print summary of experiment / run 
    print_summary(run_path, run_name)

    # select image to run inference for
    print("Selecting image to run inference for...")
    dataloader = test_dataloader(1, transforms(normalize))
    images, labels = next(iter(dataloader))
    while labels[0].item() != label:
        images, labels = next(iter(dataloader))

    # load the model
    print("Loading model...")
    model = Net()
    state_dict = torch.load(run_path / f"{run_name}_model.pt")
    model.load_state_dict(state_dict)

    # run inference
    print("Running inference...")
    model.eval()
    output = model(images)
    pred = output.argmax(dim=1, keepdim=True)[0].item()
    confidence = max(list(torch.exp(output)[0]))
    pixels = images[0][0]
    
    print("Inference complete. Now generating ascii art.")

    print(generate_ascii_art(pixels))
    print(f"This is a {pred}")
    print("Confidence: " + str(confidence))



