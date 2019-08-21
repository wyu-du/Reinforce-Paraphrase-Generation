# A Unified Reinforcement Learning Framework for Pointer Generator Model

### Training
Train the standard pointer-generator model with supervised learning from scratch
```
python train.py
```
You also need to modify the path where the model will be saved. 
```
vim config.py
log_root = os.path.join(root_dir, "Reinforce-PG/log_twitter")
```


### Decoding
Use greedy decoding to generate sentences on test set:
```
python decode.py ../log_twitter/best_model/model_best_XXXXX
```