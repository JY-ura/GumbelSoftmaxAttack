import os

os.environ["WANDB_MODE"] = "online"
from pathlib import Path
from typing import Callable, Tuple

import hydra
import torch
from ignite.utils import manual_seed
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader

import wandb
from utils.instatiate_utils import get_dataloaders
from utils.modified_model import _get_net_params, get_model
from utils.train_utils import get_lr_scheduler, get_optimizer


def get_functional(model_name: str):
    if model_name != "spikformer":
        from spikingjelly.activation_based import functional

        return functional
    else:
        from spikingjelly.clock_driven import functional

        return functional


def acc_fn(logits, labels):
    return torch.mean((torch.argmax(logits, dim=-1) == labels).float())


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


@hydra.main(config_path="config", config_name="train_network", version_base=None)
def main(_cfg: DictConfig):
    manual_seed(_cfg["seed"])
    cfg: dict = OmegaConf.to_container(_cfg, resolve=True)  # type:ignore
    # pprint(cfg)
    wandb.init(
        project=cfg["project"],
        config=cfg,
        entity=cfg["entity"],
        name=cfg["name"],
        tags=[cfg["tags"]],
    )
    train_data, val_data, _ = get_dataloaders(
        **cfg["dataset"],  # type:ignore
        transform=cfg["transform"],
    )
    # imgs, labels = next(iter(train_data))
    # print(imgs)
    # print(torch.sum((imgs == 1).float()))
    device = torch.device(
        f"cuda:{cfg['gpu_idx']}" if torch.cuda.is_available() else "cpu"
    )

    model = get_model(
        **cfg["model"]  # type:ignore
    )
    model.to(device)
    functional = get_functional(cfg["model"]["name"])

    train(
        train_loader=train_data,
        val_loader=val_data,
        model=model,
        device=device,
        cfg=cfg,
        functional=functional,
    )


def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    cfg: dict,
    functional,
) -> None:
    optimizer = get_optimizer(params=model.parameters(), **cfg["optimizer"])
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    best_acc = 0.6
    lr_scheduler = get_lr_scheduler(optimizer=optimizer, **cfg["scheduler"])

    for epoch in range(cfg["epoch"]):
        train_loss, train_acc = train_one_step(
            model, optimizer, loss_fn, acc_fn, train_loader, device, functional
        )
        eval_loss, eval_acc = eval_one_step(
            model, loss_fn, acc_fn, val_loader, device, functional
        )
        lr_scheduler.step()
        print(f"epoch {epoch} train loss: {train_loss} train acc: {train_acc}")
        print(f"epoch {epoch} eval loss: {eval_loss} eval acc: {eval_acc}")

        wandb.log({"train/loss": train_loss})
        wandb.log({"train/acc": train_acc})
        wandb.log({"eval/loss": eval_loss})
        wandb.log({"eval/acc": eval_acc})

        if eval_acc > best_acc:
            best_acc = eval_acc

            dir_name = (
                f"./models/train/{cfg['model']['name']}_{cfg['model']['num_layers']}"
            )

            # create dir if not exist
            if not Path(dir_name).exists():
                Path(dir_name).mkdir(parents=True)

            model_name = f"{cfg['dataset']['name']}_{cfg['model']['name']}_{cfg['model']['num_layers']}_{cfg['model']['dnn_act']}"
            torch.save(
                model.state_dict(),
                f"{dir_name}/{model_name}_best_{best_acc}.pth",
            )
            print(f"save best model in {dir_name}/best.pth")
            print(f"best acc: {best_acc}")
            wandb.log({"best_acc": best_acc})

    wandb.finish()


def train_one_step(
    model: nn.Module,
    optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    acc_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data_loader: DataLoader,
    device: torch.device,
    functional,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0
    total_acc = 0

    for batch_idx, (data, label) in enumerate(data_loader):
        functional.reset_net(model)
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        # data.shape T,batch_size,channel, H, W
        output = model(data).mean(0)  # take mean over the time dimension
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()
        acc = acc_fn(output, label)
        # update metrics
        total_loss += loss.item()
        total_acc += acc.item()

        if batch_idx % 100 == 0:
            print(f"batch {batch_idx} loss: {loss.item()} acc: {acc.item()}")

    total_loss /= len(data_loader)
    total_acc /= len(data_loader)

    return total_loss, total_acc


def eval_one_step(
    model: torch.nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    acc_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data_loader: DataLoader,
    device: torch.device,
    functional,
) -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_acc = 0

        for batch_idx, (data, label) in enumerate(data_loader):
            functional.reset_net(model)
            data = data.to(device)
            label = label.to(device)
            output = model(data).mean(0)
            loss = loss_fn(output, label)
            acc = acc_fn(output, label)
            # update metrics
            total_loss += loss.item()
            total_acc += acc.item()

    total_loss /= len(data_loader)
    total_acc /= len(data_loader)

    return total_loss, total_acc


if __name__ == "__main__":
    main()
