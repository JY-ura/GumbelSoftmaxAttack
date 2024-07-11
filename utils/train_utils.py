from typing import Any, Callable, Dict, Optional, Sequence, Union

import torch
from ignite.engine import Engine, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Metric
from spikingjelly.activation_based import functional, layer, neuron, surrogate
from spikingjelly.activation_based.functional import reset_net
from torch import nn
from torch.cuda.amp import grad_scaler
from torch.optim import SGD, Adam

optimizer_dict = {"sgd": SGD, "adam": Adam}
lr_fun_dict = {
    "CosALR": torch.optim.lr_scheduler.CosineAnnealingLR,
    "StepLR": torch.optim.lr_scheduler.StepLR,
}


def get_lr_scheduler(optimizer, **kwargs):
    name = kwargs["name"]
    del kwargs["name"]
    return lr_fun_dict[name](optimizer, **kwargs)


def get_optimizer(params, **kwargs):
    name = kwargs["name"]
    del kwargs["name"]
    return optimizer_dict[name](params=params, **kwargs)


def get_snn_trainer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Union[Callable, torch.nn.Module],
    device: Optional[Union[str, torch.device]] = None,
    amp_mode: bool = False,
    scaler: Optional[grad_scaler.GradScaler] = None,
    output_transform: Callable[
        [Any, Any, Any, torch.Tensor], Any
    ] = lambda x, y, y_pred, loss: loss.item(),
) -> Engine:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if amp_mode:
        assert scaler is not None, "scaler must be setted if amp is True"

    def update(engine: Engine, batch: Sequence):
        model.train()
        reset_net(model)
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        with torch.cuda.amp.autocast(amp_mode):  # type: ignore
            y_pred = model(x).mean(0)
            loss = loss_fn(y_pred, y)

        if scaler:
            scaler.scale(loss).backward()  # type: ignore
            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()
            optimizer.step()

        return output_transform(x, y, y_pred, loss)

    return Engine(update)


def get_snn_evaluator(
    model: nn.Module,
    metrics: Dict[str, Metric],
    device: Optional[Union[str, torch.device]] = None,
    amp_mode: bool = False,
    output_transform: Callable[[Any, Any, Any], Any] = lambda x, y, y_pred: (y_pred, y),
) -> Engine:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    def evaluate(engine: Engine, batch: Sequence):
        model.eval()
        with torch.no_grad():
            reset_net(model)
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            with torch.cuda.amp.autocast(amp_mode):  # type: ignore
                y_pred = model(x).mean(0)

        return output_transform(x, y, y_pred)

    engine = Engine(evaluate)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine
