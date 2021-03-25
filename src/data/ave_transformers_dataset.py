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


def tokenize_per_word(words, tokenizer, is_wp_tokenized):
    if is_wp_tokenized:
        orig_to_tok_index = list(range(len(words)))
        tokens = words
    else:
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
                                         label2idx: Dict[str, int],
                                         wp_level: int) -> List[AVEFeature]:
    features = []
    # max_candidate_length = -1

    for idx, inst in enumerate(instances):
        words = inst.ori_words
        attr_words = inst.ori_attr_words
        tokens, orig_to_tok_index = tokenize_per_word(words, tokenizer, is_wp_tokenized=wp_level)
        attr_tokens, attr_orig_to_tok_index = tokenize_per_word(attr_words, tokenizer, is_wp_tokenized=wp_level)
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


def make_word_level_instance(words, ori_words, attr_words, anns):
    labels = ['O'] * len(words)
    ori_attr_words = [tok for tok in attr_words]
    for ann in anns:
        assert len(ann['points']) == 1
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
    return AVEInstance(words=words, ori_words=ori_words, labels=labels, attr_words=attr_words, ori_attr_words=ori_attr_words)


def wp_tokenize_with_mapping(text, wp_tokenizer):
    wp_res = wp_tokenizer.encode_plus(" " + text, add_special_tokens=False, return_offsets_mapping=True)
    wp_tokens = wp_tokenizer.convert_ids_to_tokens(wp_res['input_ids'])
    wp_offsets = wp_res['offset_mapping'] # will start from 1, since there is a leading space. exlusive
    wp_offsets = [(s, e-1) for s, e in wp_res['offset_mapping']] # make it inclusive
    wp_offset_starts = [s-1 for s, e in wp_offsets] # make it start from 0
    wp_offset_ends = [e-1 for s, e in wp_offsets] # make it start from 0
    return wp_tokens, wp_offset_starts, wp_offset_ends


def char_span_to_token_span(char_s, char_e, offset_starts, offset_ends):
    try :
        token_s = offset_starts.index(char_s)
    except ValueError:
        token_s = None
    try:
        token_e = offset_ends.index(char_e)
    except ValueError:
        token_e = None
    return token_s, token_e


def make_wordpiece_level_instance(product_str, attr_str, anns, wp_tokenizer):
    # get wp tokens and offsets
    words, words_offset_starts, words_offset_ends = wp_tokenize_with_mapping(product_str, wp_tokenizer)
    ori_words = [tok for tok in words]
    attr_words, attr_words_offest_starts, attr_words_offset_ends = wp_tokenize_with_mapping(attr_str, wp_tokenizer)
    ori_attr_words = [tok for tok in attr_words]
    if anns is not None:
        labels = ['O'] * len(words)
        for ann in anns:
            assert len(ann['points']) == 1
            # find the span of attribute values in wp_tokens
            wp_start, wp_end = char_span_to_token_span(ann['points'][0]['start'],
                                                       ann['points'][0]['end'],
                                                       words_offset_starts,
                                                       words_offset_ends)
            assert wp_start is not None and wp_end is not None
            assert wp_start <= wp_end
            assert all([l == 'O' for l in labels[wp_start:wp_end+1]])
            value_words = words[wp_start:wp_end+1]
            if len(value_words) == 1:
                #labels[wp_start] = 'S-a'
                labels[wp_start] = 'B-a'
            else:
                labels[wp_start] = 'B-a'
                labels[wp_start+1:wp_end+1] = ['I-a']*(len(value_words)-1)

        labels = convert_iobes(labels)
        assert len(words) == len(ori_words) and len(words) == len(labels)
        return AVEInstance(words=words, ori_words=ori_words, labels=labels, attr_words=attr_words, ori_attr_words=ori_attr_words)
    else:
        return AVEInstance(words=words, ori_words=ori_words, attr_words=attr_words, ori_attr_words=ori_attr_words)


class TransformersAVEDataset(Dataset):

    def __init__(self, file: str,
                 tokenizer: PreTrainedTokenizer,
                 is_train: bool,
                 use_s3: int,
                 wp_level: int,
                 sents: List[Tuple[List[str], List[str]]] = None,
                 label2idx: Dict[str, int] = None,
                 number: int = -1):
        """
        sents: we use sentences if we want to build dataset from sentences directly instead of file
        """
        self.tokenizer = tokenizer
        self.wp_level = wp_level
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
        self.insts_ids = convert_instances_to_feature_tensors(insts, tokenizer, label2idx, wp_level)

    def read_from_sentences(self, sents: List[Tuple[List[str], List[str]]]):
        """
        sents = [(['word_a', 'word_b'], [attr_a, attr_b]), (['word_aaa', 'word_bccc', 'word_ccc'], [attr_aaa, attr_bbb, attr_ccc])]
        """
        insts = []
        for sent, attr in sents:
            if self.wp_level:
                if type(sent) == list:
                    sent = " ".join(sent)
                if type(attr) == list:
                    attr = " ".join(attr)
                attr = normalize_attribute(attr)
                ave_inst = make_wordpiece_level_instance(sent, attr, anns=None, wp_tokenizer=self.tokenizer)
                insts.append(ave_inst)
            else:
                if type(sent) == str:
                    sent = [tok.text for tok in WORD_TOKENIZER(sent)]
                if type(attr) == str:
                    attr = normalize_attribute(attr)
                    attr = [tok.text for tok in WORD_TOKENIZER(attr)]
                insts.append(AVEInstance(words=sent, ori_words=sent, attr_words=attr, ori_attr_words=attr))
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
        all_attr_counter = collections.defaultdict(int)
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
                assert len(ann['label']) >= 1
                assert len(set([normalize_attribute(l) for l in ann['label']])) == len(ann['label'])
                for attr in ann['label']:
                    attr = normalize_attribute(attr)
                    attr2anns[attr].append(ann)
                    all_attr_counter[attr] += 1
            for attr in attr2anns:
                if self.wp_level:
                    ave_inst = make_wordpiece_level_instance(line['content'], attr, attr2anns[attr], self.tokenizer)
                else:
                    attr_words = [tok.text for tok in WORD_TOKENIZER(attr)]
                    ave_inst = make_word_level_instance(words, ori_words, attr_words, attr2anns[attr])
                insts.append(ave_inst)
                if len(insts) == number:
                    break
        print("number of sentences: {}".format(len(insts)))
        print("all attributes:")
        print(repr({k: v for k, v in all_attr_counter.items()}))
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
