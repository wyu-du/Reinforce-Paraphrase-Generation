from __future__ import unicode_literals, print_function, division

import time
import sys
import numpy as np
import torch

import config
from batcher import Batcher
from data import Vocab
from train_util import get_input_from_batch, get_output_from_batch
from model import Model


use_cuda = config.use_gpu and torch.cuda.is_available()


class Evaluate(object):
    def __init__(self, model_file_path):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.eval_data_path, self.vocab, mode='eval',
                               batch_size=config.batch_size, single_pass=True)
        self.model_file_path = model_file_path
        time.sleep(5)

        self.model = Model(model_file_path, is_eval=True)


    def eval_one_batch(self, batch):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch, use_cuda)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, use_cuda)
            
        with torch.no_grad():
            encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
            s_t_1 = self.model.reduce_state(encoder_hidden)
    
            step_losses = []
            for di in range(min(max_dec_len, config.max_dec_steps)):
                y_t_1 = dec_batch[:, di]  # Teacher forcing
                final_dist, s_t_1, c_t_1,attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1,
                                                            encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                            extra_zeros, enc_batch_extend_vocab, coverage, di)
                
                target = target_batch[:, di]     
                gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
                step_loss = -torch.log(gold_probs + config.eps)
                if config.is_coverage:
                    step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                    step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                    coverage = next_coverage
    
                step_mask = dec_padding_mask[:, di]
                step_loss = step_loss * step_mask
                step_losses.append(step_loss)
                
        sum_step_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_step_losses / dec_lens_var
        loss = torch.mean(batch_avg_loss)
        return loss.item()


    def run_eval(self):
        batch = self.batcher.next_batch()
        loss_list = []
        while batch is not None:
            loss = self.eval_one_batch(batch)
            loss_list.append(loss)
            batch = self.batcher.next_batch()
        return np.mean(loss_list)
    

if __name__ == '__main__':
    model_filename = sys.argv[1]
    eval_processor = Evaluate(model_filename)
    eval_processor.run_eval()
