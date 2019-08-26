# A Unified Reinforcement Learning Framework for Pointer Generator Model


### Training
1. Setting

Modify the path where the model will be saved.
```
vim config.py
log_root = os.path.join(root_dir, "Reinforce-PG/log_twitter")
```

2. Pre-train

Train the standard pointer-generator model with supervised learning from scratch
```
python train.py
```

3. Fine-tune

Modify the training mode, and the path will the fine-tuned model will be saved.
```
vim config.py
log_root = os.path.join(root_dir, "Reinforce-PG/log_rl")
mode = "RL"
```
Fine tune the pointer-generator model with REINFORCE algorithm.
```
python train.py -m ../log_twitter/best_model/model_best_XXXXX
```


### Decoding & Evaluation
First, specify the model path.
```
vim config.py
log_root = os.path.join(root_dir, "Reinforce-PG/log_twitter")
```
Second, apply beam search to generate sentences on test set:
```
python decode.py ../log_twitter/best_model/model_best_XXXXX
```
The average BLEU score will show up in the terminal after finishing decoding.

If you want to get the ROUGE score, you should first intall `pyrouge`, here is the [guidance](http://kavita-ganesan.com/rouge-howto/#.XWQ28ZNKjBI).