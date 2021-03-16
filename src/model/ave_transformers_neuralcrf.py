# 
# @author: Allan
#

import torch
import torch.nn as nn

from src.model.module.bilstm_encoder import BiLSTMEncoder
from src.model.module.linear_crf_inferencer import LinearCRF
from src.model.module.linear_encoder import LinearEncoder
from src.model.module.attention import Attention
from src.model.embedder import TransformersEmbedder
from typing import Tuple
from overrides import overrides

from src.data.data_utils import START_TAG, STOP_TAG, PAD


class AVETransformersCRF(nn.Module):

    def __init__(self, config):
        super(AVETransformersCRF, self).__init__()
        self.embedder = TransformersEmbedder(transformer_model_name=config.embedder_type,
                                             parallel_embedder=config.parallel_embedder)
        if config.hidden_dim > 0:
            assert config.hidden_dim == self.embedder.get_output_dim()
            self.encoder = BiLSTMEncoder(input_dim=self.embedder.get_output_dim(),
                                         hidden_dim=config.hidden_dim, drop_lstm=config.dropout)
            self.attr_encoder = BiLSTMEncoder(input_dim=self.embedder.get_output_dim(),
                                              hidden_dim=config.hidden_dim, drop_lstm=config.dropout)
            self.ln = nn.LayerNorm(config.hidden_dim * 2)
            self.hidden2tag = nn.Linear(config.hidden_dim*2, config.label_size)
        else:
            self.encoder = None
            self.ln = nn.LayerNorm(self.embedder.get_output_dim() * 2)
            self.hidden2tag = nn.Linear(self.embedder.get_output_dim()*2, config.label_size)
        self.attention = Attention()
        self.dropout = nn.Dropout(config.dropout)
        self.inferencer = LinearCRF(label_size=config.label_size, label2idx=config.label2idx, add_iobes_constraint=config.add_iobes_constraint,
                                    idx2labels=config.idx2labels)
        self.pad_idx = config.label2idx[PAD]


    @overrides
    def forward(self,
                words: torch.Tensor,
                attr_words: torch.Tensor,
                word_seq_lens: torch.Tensor,
                attr_word_seq_lens: torch.Tensor,
                orig_to_tok_index: torch.Tensor,
                attr_orig_to_tok_index: torch.Tensor,
                input_mask: torch.Tensor,
                attr_input_mask: torch.Tensor,
                labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate the negative loglikelihood.
        :param words: (batch_size x max_seq_len)
        :param word_seq_lens: (batch_size)
        :param context_emb: (batch_size x max_seq_len x context_emb_size)
        :param chars: (batch_size x max_seq_len x max_char_len)
        :param char_seq_lens: (batch_size x max_seq_len)
        :param labels: (batch_size x max_seq_len)
        :return: the total negative log-likelihood loss
        """
        word_rep, cls_rep = self.embedder(words, orig_to_tok_index, input_mask)
        attr_word_rep, attr_cls_rep = self.embedder(attr_words, attr_orig_to_tok_index, attr_input_mask)
        if self.encoder is None:
            features = word_rep
            attr_features = attr_cls_rep
        else:
            features, _ = self.encoder(word_rep, word_seq_lens)
            attr_features, (attr_h, attr_c) = self.attr_encoder(attr_word_rep, attr_word_seq_lens)
            attr_features = torch.cat([attr_h[-2], attr_hidden[-1]], dim=-1)
        attention_features = self.attention(features, attr_features)
        features = torch.cat([features, attention_features], dim=-1)
        features = self.ln(features)
        features = self.dropout(features)
        all_scores = self.hidden2tag(features)
        batch_size = word_rep.size(0)
        sent_len = word_rep.size(1)
        dev_num = word_rep.get_device()
        curr_dev = torch.device(f"cuda:{dev_num}") if dev_num >= 0 else torch.device("cpu")
        maskTemp = torch.arange(1, sent_len + 1, dtype=torch.long, device=curr_dev).view(1, sent_len).expand(batch_size, sent_len)
        mask = torch.le(maskTemp, word_seq_lens.view(batch_size, 1).expand(batch_size, sent_len))
        unlabed_score, labeled_score =  self.inferencer(all_scores, word_seq_lens, labels, mask)
        bestScores, decodeIdx = self.inferencer.decode(all_scores, word_seq_lens)
        return unlabed_score - labeled_score, bestScores, decodeIdx

    def decode(self,
               words: torch.Tensor,
               attr_words: torch.Tensor,
               word_seq_lens: torch.Tensor,
               attr_word_seq_lens: torch.Tensor,
               orig_to_tok_index: torch.Tensor,
               attr_orig_to_tok_index: torch.Tensor,
               input_mask: torch.Tensor,
               attr_input_mask: torch.Tensor,
               **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode the batch input
        :param batchInput:
        :return:
        """
        word_rep, cls_rep = self.embedder(words, orig_to_tok_index, input_mask)
        attr_word_rep, attr_cls_rep = self.embedder(attr_words, attr_orig_to_tok_index, attr_input_mask)
        #print("t bert:", word_rep.size(), cls_rep.size())
        #print("a bert:", attr_word_rep.size(), attr_cls_rep.size())
        if self.encoder is None:
            features = word_rep
            attr_features = attr_cls_rep
        else:
            features, _ = self.encoder(word_rep, word_seq_lens)
            #print("t lstm:", features.size())
            attr_features, (attr_h, attr_c) = self.attr_encoder(attr_word_rep, attr_word_seq_lens)
            #print("a lstm:", attr_h.size(), attr_c.size())
            attr_features = torch.cat([attr_h[-2], attr_hidden[-1]], dim=-1)
        attention_features = self.attention(features, attr_features)
        #print("attn out:", attention_features.size())
        features = torch.cat([features, attention_features], dim=-1)
        #print("out:", features.size())
        features = self.ln(features)
        features = self.dropout(features)
        all_scores = self.hidden2tag(features)
        #print("logits:", all_scores.size())
        bestScores, decodeIdx = self.inferencer.decode(all_scores, word_seq_lens)
        return bestScores, decodeIdx
