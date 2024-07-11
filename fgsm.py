import os
from pprint import pprint

from utils.fgsm import get_attack_indices_values
from utils.instatiate_utils import get_model

os.environ["WANDB_MODE"] = "disabled"
import time

import hydra
import torch
from ignite.utils import manual_seed
from omegaconf import OmegaConf

import wandb
from attacker import FGSMAttacker
from attacks.ours.probability_space.frame_generator import FrameGenerator
from utils.general_utils import *
from utils.instatiate_utils import get_dataloaders
from utils.metrics import Metrics, TotalMetrics
from utils.optimizer import CosineAnnealing, get_lr_scheduler, get_optimizer


@hydra.main(config_path="./config", config_name="default", version_base=None)
def main(dictconfig: DictConfig):
    cfg: dict = OmegaConf.to_container(dictconfig, resolve=True)  # type:ignore
    # pprint(cfg, depth=3)
    manual_seed(cfg["seed"])
    cfg["use_grad_scaling"] = cfg["use_grad_scaling"] if cfg["use_amp"] else False

    # init one run for all image
    wandb.init(
        config=cfg,
        project=cfg["project"],
        entity=cfg["entity"],
        name=cfg["name"],
    )
    wandb.define_metric("*", step_metric="global_step")

    # prepare device, dataset and model
    device = get_device(cfg["gpu_idx"])
    _, _, test_data = get_dataloaders(
        **cfg["dataset"],
        transform=cfg["transform"],
    )
    model = get_model(cfg, device)

    frame_processor = FrameGenerator(
        split_by=cfg["dataset"]["split_by"],
        frame_number=cfg["dataset"]["frame_number"],
        frame_size=cfg["dataset"]["frame_size"],
    )

    (
        correct_num,
        correct_events,
        true_labels,
        true_values,
        true_indices,
        frame_list,
    ) = remove_misclassification(cfg, test_data, model, frame_processor, device)
    target_labels = get_target_label(cfg, true_labels, device)
    auxiliary_event_dict: dict = generate_auxiliary_samples_for_each_sample(
        cfg, true_labels, correct_events
    )
    # print("true_labels:", true_labels)

    # Prepare a sample for each sample to be attacked.
    metrics = Metrics(total_num_correct=correct_num)
    start = time.time()

    for i, (
        event,
        true_value,
        target_label,
        true_indice,
        true_label,
        frame,
    ) in enumerate(
        zip(
            correct_events,
            true_values,
            target_labels,
            true_indices,
            true_labels,
            frame_list,
        )
    ):
        true_value = true_value.to(device)  # shape [1,5159]

        config = {
            "event": event,
            "true_values": true_value,
            "true_indice": true_indice,
            "true_label": true_label,
            "target_label": target_label,
            "auxiliary_event_dict": auxiliary_event_dict,
            "cfg": cfg,
            "device": device,
            "num_class": cfg["dataset"]["num_class"],
        }
        attack_true_values, attack_true_indices = get_attack_indices_values(
            config=config,
        )
        # attack_true_indices = true_indice
        # attack_true_values = true_value
        # initial attacker
        attacker = FGSMAttacker(
            cfg=cfg,
            model=model,
            true_indice=attack_true_indices,
            device=device,
            target_label=target_label,
            orginal_value=attack_true_values,
            epsilon=cfg["attack"]["epsilon"],
            frame_processor=frame_processor,
        )

        attack_metrics = TotalMetrics()

        one_step_metrics, results = attacker.attack_one_step(epoch=1)

        # judge if success
        preds = torch.argmax(results["logits"], dim=-1)
        if is_success(preds, target_label, cfg["targeted"]):
            print(f"[succ] Image:{i}, target:{target_label.item()}, true:{true_label}")
            metrics.update_metrics(
                cfg=cfg,
                preds=preds.cpu(),
                target_label=target_label.cpu(),
                results=results,
                true_value=attack_true_values.cpu(),
                iters=1,
                i=i,
                one_step_metrics=one_step_metrics,
                true_frame=frame,
                true_label=true_label,
            )
            metrics.event_number_orginal_list.append(event["p"].shape[0])

        else:
            pass
            # print(
            #     f"[failed] Image:{i}, target:{target_label.item()}, true: {true_label}"
            # )

        # update attack_metrics
        attack_metrics.update_metrics(
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            **one_step_metrics,
        )

        attack_metrics.upload(
            step=1,
            name="",
            image_id=i,
        )
        attack_metrics.reset()

        model.zero_grad()
        torch.cuda.empty_cache()

        torch.cuda.synchronize()

    end = time.time()
    metrics.finalize_metrics(start=start, end=end, cfg=cfg)


if __name__ == "__main__":
    main()
