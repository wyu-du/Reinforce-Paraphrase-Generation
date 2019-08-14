import os

root_dir = os.path.expanduser("~")
#root_dir = os.path.join(root_dir, "Desktop")

print_interval = 100
save_model_iter = 500

train_data_path = os.path.join(root_dir, "Reinforce-PG/data/quora/chunked/train_*")
eval_data_path = os.path.join(root_dir, "Reinforce-PG/data/quora/chunked/val_*")
decode_data_path = os.path.join(root_dir, "Reinforce-PG/data/quora/chunked/test_*")
vocab_path = os.path.join(root_dir, "Reinforce-PG/data/quora/vocab")
cluster_path = os.path.join(root_dir, "Reinforce-PG/data/quora/paths")
log_root = os.path.join(root_dir, "Reinforce-PG/log_test")

# Hyperparameters
hidden_dim= 256
emb_dim= 128
batch_size= 8
sample_size= 4
max_enc_steps= 20
max_dec_steps= 20
beam_size= 8
min_dec_steps= 5
vocab_size= 5000
alpha = 1.0
beta = 1.0
k1 = 0.9999
k2 = 3000.

max_iterations = 5000000
lr = 1e-5
pointer_gen = True
is_coverage = False
lr_coverage = 0.15
cov_loss_wt = 1.0
max_grad_norm = 2.0
rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4
eps = 1e-12
use_gpu = True