# Binary SNN Attack

This is the responsory for the paper "Exploring Vulnerabilities in Spiking Neural Networks: Direct Adversarial Attacks on Raw Event Data".

## Requirements

```
torch ==  2.0.0+cu118
torchvision == 0.15.1+cu118
spikingjelly == 0.0.0.0.14
snntorch == 0.6.4
wandb == 0.16.1
numpy == 1.23.5
```

To install all the requirements, run:

```bash
pip install -r requirements.txt
```

or you can choose to install them one by one.

## File Structure

- `data/`: contains the datasets used in the paper.
- `model/`: contains the models used in the paper.
- `attacks/`: contains the attacks used in the paper.
- `config/`: the configuration file to set the parameters.
- `utils/`: the utils folder.
- `main.py`: the main file to run the attack.
- `attacker.py`: the attacker class files.
- `requirements.txt`: the requirements file.
- `train.py`: train code.
- `README.md`: this file.

## Usage

The config and arguments are managed using [Hydra](https://hydra.cc/). To set the parameters, edit the `config.py` file, or use the command line arguments, details are in the `config/readme.md` file.
Before you can run the code, you may need a wandb account. If you don't use wandb, you can set it up in the main.py file
os.environ["WANDB_MODE"] = "disabled", which is set to "online" if used.
