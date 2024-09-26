import typer
from dataclasses import dataclass

from ser.train import training_loop
from ser.data import dataloaders
from ser.utils import write_json, save_model

main = typer.Typer()

@dataclass
class Params:
    name: str
    epochs: int
    batch_size: float
    learning_rate: float

@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    epochs: int = typer.Option(
        2,"-e", "--epochs", help="Number of epochs for training."
    ),
    batch_size: int = typer.Option(
        1000,"-bs","--batch_size", help="Batch size for training."
    ),
    learning_rate: float = typer.Option(
        0.01,"-lr","--learning_rate", help="Learning rate for training."
    ),
):
    print(f"Running experiment {name}")

    parameters = Params(name, epochs, batch_size, learning_rate)
    
    # initialise dataloaders
    training_dataloader, validation_dataloader = dataloaders(batch_size)

    # train model
    trained_model = training_loop(epochs,learning_rate,training_dataloader, validation_dataloader)

    # save parameters to json
    write_json(parameters.name, parameters)

    # save model, required parent directory to be made 
    save_model(parameters.name, trained_model)
    


@main.command()
def infer():
    print("This is where the inference code will go")
