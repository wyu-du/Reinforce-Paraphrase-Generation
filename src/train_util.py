from torch.autograd import Variable
import numpy as np
import torch
import config
import data
from utils import rouge_2


def get_input_from_batch(batch, use_cuda):
    batch_size = len(batch.enc_lens)
    enc_batch = Variable(torch.from_numpy(batch.enc_batch).long())
    enc_padding_mask = Variable(torch.from_numpy(batch.enc_padding_mask)).float()
    enc_lens = batch.enc_lens
    extra_zeros = None
    enc_batch_extend_vocab = None

    if config.pointer_gen:
        enc_batch_extend_vocab = Variable(torch.from_numpy(batch.enc_batch_extend_vocab).long())
        # max_art_oovs is the max over all the article oov list in the batch
        if batch.max_art_oovs > 0:
            extra_zeros = Variable(torch.zeros((batch_size, batch.max_art_oovs)))

    c_t_1 = Variable(torch.zeros((batch_size, 2 * config.hidden_dim)))

    coverage = None
    if config.is_coverage:
        coverage = Variable(torch.zeros(enc_batch.size()))

    if use_cuda:
        enc_batch = enc_batch.cuda()
        enc_padding_mask = enc_padding_mask.cuda()
        c_t_1 = c_t_1.cuda()
        
    if enc_batch_extend_vocab is not None:
        enc_batch_extend_vocab = enc_batch_extend_vocab.cuda()
    if extra_zeros is not None:
        extra_zeros = extra_zeros.cuda()
    if coverage is not None:
        coverage = coverage.cuda()
    return enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage


def get_output_from_batch(batch, use_cuda):
    dec_batch = Variable(torch.from_numpy(batch.dec_batch).long())
    dec_padding_mask = Variable(torch.from_numpy(batch.dec_padding_mask)).float()
    dec_lens = batch.dec_lens
    max_dec_len = np.max(dec_lens)
    dec_lens_var = Variable(torch.from_numpy(dec_lens)).float()

    target_batch = Variable(torch.from_numpy(batch.target_batch)).long()

    if use_cuda:
        dec_batch = dec_batch.cuda()
        dec_padding_mask = dec_padding_mask.cuda()
        dec_lens_var = dec_lens_var.cuda()
        target_batch = target_batch.cuda()
    return dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch


def compute_reward(batch, decode_batch, vocab, mode, use_cuda):
    target_sents = batch.original_abstracts  # list of string
    decode_batch = decode_batch.cpu().numpy()  # B x S x L
    output_ids = decode_batch[:, :, 1:]
    all_rewards = torch.zeros((config.batch_size, config.sample_size)) # B x S
    if use_cuda: all_rewards = all_rewards.cuda()
    for i in range(config.batch_size):
        for j in range(config.sample_size):
            words = data.outputids2words(list(output_ids[i,j,:]), vocab,
                                       (batch.art_oovs[i] if config.pointer_gen else None))
            # Remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = words.index(data.STOP_DECODING)
                words = words[:fst_stop_idx]
            except ValueError:
                words = words
            decode_sent = ' '.join(words)
            all_rewards[i, j] = rouge_2(target_sents[i], decode_sent)
    batch_avg_reward = torch.mean(all_rewards, dim=1, keepdim=True)  # B x 1
    
    ones = torch.ones((config.batch_size, config.sample_size))
    if use_cuda: ones = ones.cuda()
    if mode == 'MLE':
        return ones, torch.zeros(1)
    else:
        batch_avg_reward = batch_avg_reward * ones   # B x S
        if torch.equal(all_rewards, batch_avg_reward):
            all_rewards = all_rewards
        else:
            all_rewards = all_rewards - batch_avg_reward
        
        for i in range(config.batch_size):
            for j in range(config.sample_size):
                if all_rewards[i,j] < 0:
                    all_rewards[i, j] = 0
        return all_rewards, batch_avg_reward.mean()


def gen_preds(samples_batch, use_cuda):
    latest_batch = torch.ones((config.batch_size, config.sample_size), dtype=torch.long)
    if use_cuda: latest_batch = latest_batch.cuda()
    for i in range(config.batch_size):
        for j in range(config.sample_size):
            if samples_batch[i,j] < config.vocab_size:
                latest_batch[i,j] = samples_batch[i,j]
            else:
                latest_batch[i,j] = 0 # UNK_TOKEN
    return latest_batch