### The configuration file is structured as follows:

- `attack/`: Attack Method Configure the folder.
- `dataset/`: DVS datasets configuration folder.
- `model/`: Models configuration folder.
- `optimizer/`: Optimizer configuration folder.
- `scheduler/`: Learning rate scheduler configuration folder.
- `transform/`: Data augmentation profile.
- `default`: Attack configuration.
- `train_network`: Train configuration.

### run command:
```
python main.py attack=[ours,fgsm] dataset=[cifar10-dvs,gesture-dvs,nmnist] model=[resnet,sew_resnet,vgg] attack.sample_num=40 attack.init_alpha_mode=[target,default,random_add_indices] model.model_path=... targeted=[True,False] optimizer=[adam,sgd] optimizer.lr=1 kappa=0.1 gpu_idx=0
```