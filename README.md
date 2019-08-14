# RL-pointer-generator
pointer generator with reinforcement learning, pytorch version

### Training
Train the standard pointer-generator model with supervised learning from scratch
```
python train.py
```
You also need to modify the path where the model will be saved. 
```
vim config.py
log_root = os.path.join(root_dir, "RL-pointer-generator/model_path")
```


### Decoding
Use greedy decoding to generate sentences on test set:
```
python decode.py ../model_path/best_model/model_best_XXXXX
```