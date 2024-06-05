## Training Commands
Example training commands with possible combinations

Train multimodal with vgg16
```bash
python train.py --model multimodal --data_root path/to/root --pretrain_epoch 200 
--image_embedding vgg16 --classifier_epoch 100 --finetune_epoch 100 
--train_pipeline --use_wandb --augmentation --scheduler cos --l2regularization 0.001 
--batch 128 --mlp_hidden_dim 256 --mlp_output_dim 64 --word_embedding_dim 100 --ntxnet_alpha 0.5
```

 

Train multimodal with vgg1
```bash 
python train.py --model multimodal --data_root path/to/root --pretrain_epoch 200
 --image_embedding vgg11 --classifier_epoch 100 --finetune_epoch 100 
 --train_pipeline --use_wandb
--augmentation --scheduler cos --l2regularization 0.001 
--batch 128 --mlp_hidden_dim 256 --mlp_output_dim 64 --word_embedding_dim 100 --ntxnet_alpha 0.5
```

Train multimodal with resnet18
```bash 
python train.py --data_root path/to/root --pretrain_epoch 200
 --image_embedding resnet18 --classifier_epoch 100 --finetune_epoch 100 
--model multimodal --train_pipeline --use_wandb
--augmentation --scheduler cos --l2regularization 0.001 
--batch 128 --mlp_hidden_dim 256 --mlp_output_dim 64 --word_embedding_dim 100 --ntxnet_alpha 0.5
```

Train multimodal with resnet50
```bash 
python train.py --data_root path/to/root --pretrain_epoch 200
 --image_embedding resnet50 --classifier_epoch 100 --finetune_epoch 100 
--model multimodal --train_pipeline --use_wandb
--augmentation --scheduler cos --l2regularization 0.001 
--batch 128 --mlp_hidden_dim 256 --mlp_output_dim 64 --word_embedding_dim 100 --ntxnet_alpha 0.5
```



Train vgg11
```bash 
train.py --model vgg --data_root data/FER2013 --vgg_config vgg11 --pretrain_epoch
300  --use_wandb --scheduler cos --dropout 0.25 --l2regularization
0.001 --augmentation --use_wandb --batch 256
```

Train vgg16
```bash 
train.py --model vgg --data_root data/FER2013 --vgg_config vgg16 --pretrain_epoch
300  --use_wandb --scheduler cos --dropout 0.25 --l2regularization
0.001 --augmentation --use_wandb --batch 256
```

Train resnet18
```bash 
train.py --model resnet --data_root data/FER2013 --resnet_config resnet18 --pretrain_epoch
300  --use_wandb --scheduler cos --dropout 0.25 --l2regularization
0.001 --augmentation --use_wandb --batch 256
```

Train resnet50
```bash 
train.py --model resnet --data_root data/FER2013 --resnet_config resnet50 --pretrain_epoch
300  --use_wandb --scheduler cos --dropout 0.25 --l2regularization
0.001 --augmentation --use_wandb --batch 256
```






