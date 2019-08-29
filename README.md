# A Unified Reinforcement Learning Framework for Pointer Generator Model
This repository contains the data and code for the paper ["An Empirical Comparison on Imitation Learning and Reinforcement Learning for Paraphrase Generation"](https://arxiv.org/abs/1908.10835).

## Useage
### Training
1. Model Setting: modify the path where the model will be saved.
```
vim config.py
log_root = os.path.join(root_dir, "Reinforce-Paraphrase-Generation/log_twitter")
```

2. Pre-train: train the standard pointer-generator model with supervised learning from scratch.
```
python train.py
```

3. Fine-tune: modify the training mode and the path where the fine-tuned model will be saved.
```
vim config.py
log_root = os.path.join(root_dir, "Reinforce-Paraphrase-Generation/log_rl")
mode = "RL"
```
Fine tune the pointer-generator model with REINFORCE algorithm.
```
python train.py -m ../log_twitter/best_model/model_best_XXXXX
```


### Decoding & Evaluation
1. Decoding: first, specify the model path.
```
vim config.py
log_root = os.path.join(root_dir, "Reinforce-Paraphrase-Generation/log_twitter")
```
Second, apply beam search to generate sentences on test set:
```
python decode.py ../log_twitter/best_model/model_best_XXXXX
```

2. Evaluation: 
	- The average BLEU score will show up automatically in the terminal after finishing decoding.
	
	- If you want to get the ROUGE scores, you should first intall `pyrouge`, here is the [guidance](https://ireneli.eu/2018/01/11/working-with-rouge-1-5-5-evaluation-metric-in-python/). Then, you can uncomment the code snippet specified in `utils.py` and `decode.py`. Finally, run `decode.py` to get the ROUGE scores.