from functools import partial
from typing import Optional, Type

from spikingjelly.activation_based import functional as activation_functional
from spikingjelly.activation_based import layer, neuron, surrogate
from spikingjelly.activation_based.model import sew_resnet, spiking_resnet, spiking_vgg

from timm.models import create_model
from torch import nn
from torchvision import models


snn_model_dict = {
    "resnet_18": spiking_resnet.spiking_resnet18,
    "vgg_11": spiking_vgg.spiking_vgg11_bn,
    "sew_resnet_34": sew_resnet.sew_resnet34,
}

dnn_model_dict = {
    "dnn_resnet_18": models.resnet18,
    "dnn_vgg_11": models.vgg11,
}

activation_fn_dict = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
}

neuron_list = {
    "LIF": neuron.LIFNode,
    "ParametricLIF": neuron.ParametricLIFNode,
}


def _get_net_params(dict):
    if "sew" not in dict["model_name"]:
        del dict["cnf"]
        if "vgg" in dict["model_name"]:
            del dict["zero_init_residual"]

        if "dnn" in dict["model_name"]:
            del dict["spiking_neuron"]
            del dict["surrogate_function"]
            del dict["detach_reset"]

    del dict["model_name"]
    return dict


def get_model(
    name: str,
    in_channels: int,
    num_class: int,
    num_layers: int,
    neuron: str,
    cnf: str,
    zero_init_residual: bool = True,
    pretrain: bool = True,
    dnn_act: Optional[str] = None,
    **kwargs,
):  # type: ignore
    """Creates a modified version of a pre-trained resnet model for spiking neural network simulations.

    Args:
        model_name (str): Name of the model to modify.
        in_channels (int): Input channel of the network.
        num_class (int): Number of the dataset.

    Returns:
        nn.Module: A modified spiking neural network model.
    """
    if name == "spikformer":
        model = create_model(
            model_name=name,
            pretrained=True,
            drop_rate=0.0,
            drop_path_rate=0.1,
            drop_block_rate=None,
            num_class=num_class,
        )

    else:
        try:
            if "dnn" in name.lower():
                model_cls = dnn_model_dict[f"{name.lower()}_{num_layers}"]
                conv2d_layer_class = nn.Conv2d
                linear_layer_class = nn.Linear

            else:
                model_cls = snn_model_dict[f"{name.lower()}_{num_layers}"]
                conv2d_layer_class = layer.Conv2d
                linear_layer_class = layer.Linear

        except KeyError:
            raise KeyError(f"Model {name} is not supported.")

        model_params_dict = {
            "model_name": name,
            "pretrained": pretrain,
            "progress": False,
            "spiking_neuron": neuron_list[neuron],
            "surrogate_function": surrogate.ATan(),
            "detach_reset": True,
            "cnf": cnf,
            "zero_init_residual": zero_init_residual,
        }

        new_params_dict = _get_net_params(model_params_dict)

        model = model_cls(**new_params_dict)

        if in_channels != 3:
            if "vgg" in name.lower():
                feature = model.features[0]
                model.features[0] = conv2d_layer_class(
                    in_channels=in_channels,
                    out_channels=feature.out_channels,
                    kernel_size=feature.kernel_size,
                    stride=feature.stride,
                    padding=feature.padding,
                )

            elif "resnet" in name.lower():
                conv1 = model.conv1
                model.conv1 = conv2d_layer_class(
                    in_channels=in_channels,
                    out_channels=conv1.out_channels,
                    kernel_size=conv1.kernel_size,  # type: ignore
                    stride=conv1.stride,
                    padding=conv1.padding,  # type: ignore
                )

        if num_class != 1000:
            if "vgg" in name.lower():
                model.classifier[6] = linear_layer_class(
                    in_features=model.classifier[6].in_features,
                    out_features=num_class,
                )
                print(model.classifier[6])
            else:
                model.fc = linear_layer_class(
                    in_features=model.fc.in_features, out_features=num_class
                )

        if dnn_act is not None:
            print(f"Replace spiking neuron with {dnn_act} activation function")
            activation_cls = activation_fn_dict[dnn_act]
            _replace_spiking_neuron(model, activation_cls)

        activation_functional.set_step_mode(model, "m")
        activation_functional.set_backend(model, "cupy")

    return model


def _replace_in_all_layer_inplace(model: nn.Module):
    for module in model.modules():
        if hasattr(module, "inplace"):
            setattr(module, "inplace", False)
    return model


def _replace_in_place_relu(model: nn.Module):
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False


def _replace_spiking_neuron(model: nn.Module, activation_cls: Type[nn.Module]):
    """Replace all spiking neurons in the model with DNN neurons.

    Args:
        model (nn.Module): SNN network.
        activation_cls (Type[nn.Module]): DNN neuron class.
    """
    for name, module in list(model.named_children()):
        if isinstance(module, neuron.BaseNode):
            if "." in name:
                # Split the name for non-top-level modules
                # print(f"replace {name} with {activation_cls}")
                parent_name, child_name = name.rsplit(".", 1)
                parent_module = dict(model.named_modules())[parent_name]
                setattr(parent_module, child_name, activation_cls())
            else:
                # Directly set attribute for top-level modules
                # print(f"replace {name} with {activation_cls}")
                setattr(model, name, activation_cls())
        else:
            # Recursively apply the function to child modules
            _replace_spiking_neuron(module, activation_cls)
