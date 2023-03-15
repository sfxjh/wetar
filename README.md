# WETAR - Weighting Examples Towards Adversarial Robustness

The code implemetation of ACL 2022 _findings_ : Towards Adversarially Robust Text Classifiers by Learning to Reweight Clean Examples

## Environment
```
conda env create -f env.yaml
```

## Train
```
python main.py --mode 'train' --use_benchmark 'False' --training_type 'l2rew' --sgd_lr '2e-5' --batch_size 32 --epsilon '1e-8' --valid_batch_size 32 --augment_ratio 0.5 --dataset_name 'agnews' --epoch_update_valid 2
```

## Attack
```
python main.py --mode 'attack' --use_benchmark 'False' --training_type 'l2rew' --sgd_lr '2e-5' --batch_size 32 --epsilon '1e-8' --valid_batch_size 32 --augment_ratio 0.5 --dataset_name 'agnews' --attack_numbers 1000 --epoch_update_valid 2 
```

## Attack Certain Epoch
```
python main.py --mode 'attack_epoch' --use_benchmark 'False' --training_type 'l2rew' --sgd_lr '2e-5' --batch_size 32 --epsilon '1e-8' --valid_batch_size 32 --augment_ratio 0.5 --dataset_name 'agnews' --attack_numbers 1000 --epoch_update_valid 2 --attack_epoch 2
```


