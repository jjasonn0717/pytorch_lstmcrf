# 
# @author: Allan
#

from tqdm import tqdm
from typing import List, Dict, Tuple
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from transformers import PreTrainedTokenizer
import collections
import numpy as np
import spacy
import json
from src.data.data_utils import convert_iobes, build_label_idx, check_all_labels_in_dict, s3_readline

from src.data import AVEInstance

AVEFeature = collections.namedtuple('AVEFeature', 'input_ids attr_input_ids attention_mask attr_attention_mask token_type_ids attr_token_type_ids orig_to_tok_index attr_orig_to_tok_index word_seq_len attr_word_seq_len label_ids')
AVEFeature.__new__.__defaults__ = (None,) * 6


WORD_TOKENIZER = spacy.load("en_core_web_lg", exclude=["tok2vec", "parser", "ner"])


def tokenize_per_word(words, tokenizer):
    orig_to_tok_index = []
    tokens = []
    for i, word in enumerate(words):
        """
        Note: by default, we use the first wordpiece token to represent the word
        If you want to do something else (e.g., use last wordpiece to represent), modify them here.
        """
        orig_to_tok_index.append(len(tokens))
        ## tokenize the word into word_piece / BPE
        ## NOTE: adding a leading space is important for BART/GPT/Roberta tokenization.
        ## Related GitHub issues:
        ##      https://github.com/huggingface/transformers/issues/1196
        ##      https://github.com/pytorch/fairseq/blob/master/fairseq/models/roberta/hub_interface.py#L38-L56
        ##      https://github.com/ThilinaRajapakse/simpletransformers/issues/458
        word_tokens = tokenizer.tokenize(" " + word)
        for sub_token in word_tokens:
            tokens.append(sub_token)
    return tokens, orig_to_tok_index


def convert_instances_to_feature_tensors(instances: List[AVEInstance],
                                         tokenizer: PreTrainedTokenizer,
                                         label2idx: Dict[str, int]) -> List[AVEFeature]:
    features = []
    # max_candidate_length = -1

    for idx, inst in enumerate(instances):
        words = inst.ori_words
        attr_words = inst.ori_attr_words
        tokens, orig_to_tok_index = tokenize_per_word(words, tokenizer)
        attr_tokens, attr_orig_to_tok_index = tokenize_per_word(attr_words, tokenizer)
        labels = inst.labels
        label_ids = [label2idx[label] for label in labels] if labels else [-100] * len(words)
        input_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + tokens + [tokenizer.sep_token])
        attr_input_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + attr_tokens + [tokenizer.sep_token])
        segment_ids = [0] * len(input_ids)
        attr_segment_ids = [0] * len(attr_input_ids)
        input_mask = [1] * len(input_ids)
        attr_input_mask = [1] * len(attr_input_ids)

        if idx < 3:
            print("*** Example ***")
            print("tokens_title:", " ".join([str(x) for x in tokens]))
            print("input_ids_title:", " ".join([str(x) for x in input_ids]))
            print("input_mask_title:", " ".join([str(x) for x in input_mask]))
            print("segment_ids_title:", " ".join([str(x) for x in segment_ids]))
            print("orig_to_tok_index_title:", " ".join([str(x) for x in orig_to_tok_index]))
            print("tokens_attr:", " ".join([str(x) for x in attr_tokens]))
            print("input_ids_attr:", " ".join([str(x) for x in attr_input_ids]))
            print("input_mask_attr:", " ".join([str(x) for x in attr_input_mask]))
            print("segment_ids_attr:", " ".join([str(x) for x in attr_segment_ids]))
            print("orig_to_tok_index_attr:", " ".join([str(x) for x in attr_orig_to_tok_index]))
            print("label_ids:", " ".join([str(x) for x in label_ids]))

        features.append(AVEFeature(input_ids=input_ids,
                                   attr_input_ids=attr_input_ids,
                                   attention_mask=input_mask,
                                   attr_attention_mask=attr_input_mask,
                                   orig_to_tok_index=orig_to_tok_index,
                                   attr_orig_to_tok_index=attr_orig_to_tok_index,
                                   token_type_ids=segment_ids,
                                   attr_token_type_ids=attr_segment_ids,
                                   word_seq_len=len(orig_to_tok_index),
                                   attr_word_seq_len=len(attr_orig_to_tok_index),
                                   label_ids=label_ids))
    return features


def normalize_attribute(attr):
    if attr == 'size_value':
        attr = 'value_for_size'
    elif attr == 'size_uom':
        attr = 'unit_of_measurement_for_size'
    elif attr == 'unit_count':
        attr = 'count_for_unit'
    elif attr == 'unit_uom':
        attr = 'unit_of_measurement_for_unit'
    attr = " ".join(attr.split('_'))
    return attr


class TransformersAVEDataset(Dataset):

    def __init__(self, file: str,
                 tokenizer: PreTrainedTokenizer,
                 is_train: bool,
                 use_s3: int,
                 sents: List[Tuple[List[str], List[str]]] = None,
                 label2idx: Dict[str, int] = None,
                 number: int = -1):
        """
        sents: we use sentences if we want to build dataset from sentences directly instead of file
        """
        ## read all the instances. sentences and labels
        insts = self.read_txt(file=file, number=number, use_s3=use_s3) if sents is None else self.read_from_sentences(sents)
        self.insts = insts
        if is_train:
            # assert label2idx is None
            if label2idx is not None:
                print(f"[WARNING] YOU ARE USING EXTERNAL label2idx, WHICH IS NOT BUILT FROM TRAINING SET.")
                self.label2idx = label2idx
            else:
                print(f"[Data Info] Using the training set to build label index")
                ## build label to index mapping. e.g., B-PER -> 0, I-PER -> 1
                idx2labels, label2idx = build_label_idx(insts)
                self.idx2labels = idx2labels
                self.label2idx = label2idx
        else:
            assert label2idx is not None ## for dev/test dataset we don't build label2idx
            self.label2idx = label2idx
            # check_all_labels_in_dict(insts=insts, label2idx=self.label2idx)
        self.insts_ids = convert_instances_to_feature_tensors(insts, tokenizer, label2idx)
        self.tokenizer = tokenizer

    def read_from_sentences(self, sents: List[Tuple[List[str], List[str]]]):
        """
        sents = [(['word_a', 'word_b'], [attr_a, attr_b]), (['word_aaa', 'word_bccc', 'word_ccc'], [attr_aaa, attr_bbb, attr_ccc])]
        """
        insts = []
        for sent, attr in sents:
            insts.append(Instance(words=sent, ori_words=sent, attr_words=attr))
        return insts


    def read_txt(self, file: str = None, number: int = -1, use_s3: int = 0) -> List[AVEInstance]:
        print(f"[Data Info] Reading file: {file}, labels will be converted to IOBES encoding")
        print(f"[Data Info] Modify src/data/transformers_dataset.read_txt function if you have other requirements")
        if file is None and s3_file is None:
            raise ValueError("No file to read")
        if use_s3 == 0:
            f = open(file, 'r', encoding='utf-8')
            line_iter = f.readlines()
        else:
            line_iter = s3_readline(file)
        all_attr_set = set()
        insts = []
        for line in tqdm(line_iter):
            line = line.rstrip()
            line = json.loads(line)
            words = line['tokens']
            if len(words) == 0:
                continue
            ori_words = line['tokens']
            attr2anns = collections.defaultdict(list)
            for ann in line['annotation']:
                attr = normalize_attribute(ann['label'][0])
                if attr == 'ignore':
                    continue
                attr2anns[attr].append(ann)
                all_attr_set.add(attr)
            for attr in attr2anns:
                labels = ['O'] * len(words)
                attr_words = [tok.text for tok in WORD_TOKENIZER(attr)]
                ori_attr_words = [tok for tok in attr_words]
                for ann in attr2anns[attr]:
                    assert all([l == 'O' for l in labels[ann['points'][0]['tok_start']:ann['points'][0]['tok_end']+1]])
                    value_words = words[ann['points'][0]['tok_start']:ann['points'][0]['tok_end']+1]
                    if len(value_words) == 1:
                        #labels[ann['points'][0]['tok_start']] = 'S-a'
                        labels[ann['points'][0]['tok_start']] = 'B-a'
                    else:
                        labels[ann['points'][0]['tok_start']] = 'B-a'
                        labels[ann['points'][0]['tok_start']+1:ann['points'][0]['tok_end']+1] = ['I-a']*(len(value_words)-1)

                labels = convert_iobes(labels)
                assert len(words) == len(ori_words) and len(words) == len(labels)
                insts.append(AVEInstance(words=words, ori_words=ori_words, labels=labels, attr_words=attr_words, ori_attr_words=ori_attr_words))
                if len(insts) == number:
                    break
        print("number of sentences: {}".format(len(insts)))
        print("all attributes:")
        print(repr(list(all_attr_set)))
        if use_s3 == 0:
            f.close()
        return insts

    def __len__(self):
        return len(self.insts_ids)

    def __getitem__(self, index):
        return self.insts_ids[index]

    def collate_fn(self, batch:List[AVEFeature]):
        word_seq_len = [len(feature.orig_to_tok_index) for feature in batch]
        attr_word_seq_len = [len(feature.attr_orig_to_tok_index) for feature in batch]
        max_seq_len = max(word_seq_len)
        max_attr_seq_len = max(attr_word_seq_len)
        max_wordpiece_length = max([len(feature.input_ids) for feature in batch])
        max_attr_wordpiece_length = max([len(feature.attr_input_ids) for feature in batch])
        for i, feature in enumerate(batch):
            padding_length = max_wordpiece_length - len(feature.input_ids)
            attr_padding_length = max_attr_wordpiece_length - len(feature.attr_input_ids)
            input_ids = feature.input_ids + [self.tokenizer.pad_token_id] * padding_length
            attr_input_ids = feature.attr_input_ids + [self.tokenizer.pad_token_id] * attr_padding_length
            mask = feature.attention_mask + [0] * padding_length
            attr_mask = feature.attr_attention_mask + [0] * attr_padding_length
            type_ids = feature.token_type_ids + [self.tokenizer.pad_token_type_id] * padding_length
            attr_type_ids = feature.attr_token_type_ids + [self.tokenizer.pad_token_type_id] * attr_padding_length
            padding_word_len = max_seq_len - len(feature.orig_to_tok_index)
            attr_padding_word_len = max_attr_seq_len - len(feature.attr_orig_to_tok_index)
            orig_to_tok_index = feature.orig_to_tok_index + [0] * padding_word_len
            attr_orig_to_tok_index = feature.attr_orig_to_tok_index + [0] * attr_padding_word_len
            label_ids = feature.label_ids + [0] * padding_word_len

            batch[i] = AVEFeature(input_ids=np.asarray(input_ids),
                                  attr_input_ids=np.asarray(attr_input_ids),
                                  attention_mask=np.asarray(mask), token_type_ids=np.asarray(type_ids),
                                  attr_attention_mask=np.asarray(attr_mask), attr_token_type_ids=np.asarray(attr_type_ids),
                                  orig_to_tok_index=np.asarray(orig_to_tok_index),
                                  attr_orig_to_tok_index=np.asarray(attr_orig_to_tok_index),
                                  word_seq_len=feature.word_seq_len,
                                  attr_word_seq_len=feature.attr_word_seq_len,
                                  label_ids=np.asarray(label_ids))
        results = AVEFeature(*(default_collate(samples) for samples in zip(*batch)))
        return results


## testing code to test the dataset
# from transformers import *
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# dataset = TransformersNERDataset(file= "data/conll2003_sample/train.txt",tokenizer=tokenizer, is_train=True)
# from torch.utils.data import DataLoader
# train_dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=2, collate_fn=dataset.collate_fn)
# print(len(train_dataloader))
# for batch in train_dataloader:
#     # print(batch.input_ids.size())
#     print(batch.input_ids)
#     pass
