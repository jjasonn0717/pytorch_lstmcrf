
import pickle

from src.model import AVETransformersCRF
import torch
from termcolor import colored
from src.config import context_models
from src.data import TransformersAVEDataset
from src.data.data_utils import iobes_to_spans
from src.data.data_utils import s3_upload_file, s3_download_file
from src.data.data_utils import s3_upload_file, s3_download_file
from typing import List, Union, Tuple
import tarfile
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import tempfile


class TransformersAVEPredictor:

    def __init__(self, model_archived_file:str,
                 cuda_device: str = "cpu"):
        """
        model_archived_file: ends with "tar.gz"
        OR
        directly use the model folder patth
        """
        device = torch.device(cuda_device)
        if model_archived_file.endswith("tar.gz"):
            tar = tarfile.open(model_archived_file)
            self.conf = pickle.load(tar.extractfile('model/config.conf')) ## config file
            self.model = AVETransformersCRF(self.conf)
            self.model.load_state_dict(torch.load(tar.extractfile('model/lstm_crf.m'), map_location=device)) ## model file
        else:
            folder_name = model_archived_file
            assert os.path.isdir(folder_name)
            f = open(folder_name + "/config.conf", 'rb')
            self.conf = pickle.load(f)
            f.close()
            self.model = AVETransformersCRF(self.conf)
            self.model.load_state_dict(torch.load(f"{folder_name}/lstm_crf.m", map_location=device))
        self.conf.device = device
        self.model.to(device)
        self.model.eval()

        print(colored(f"[Data Info] Tokenizing the instances using '{self.conf.embedder_type}' tokenizer", "blue"))
        self.tokenizer = context_models[self.conf.embedder_type]["tokenizer"].from_pretrained(self.conf.embedder_type)

    def predict(self, sents:  List[Tuple[List[str], List[str]]], batch_size = -1):
        batch_size = len(sents) if batch_size == -1 else batch_size

        dataset = TransformersAVEDataset(file=None, sents=sents, tokenizer=self.tokenizer, label2idx=self.conf.label2idx, is_train=False, use_s3=0, wp_level=self.conf.wp_level)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)

        all_predictions = []
        for batch_id, batch in tqdm(enumerate(loader, 0), desc="--evaluating batch", total=len(loader)):
            one_batch_insts = dataset.insts[batch_id * batch_size:(batch_id + 1) * batch_size]
            batch_max_scores, batch_max_ids = self.model.decode(words=batch.input_ids.to(self.conf.device),
                                                                attr_words=batch.attr_input_ids.to(self.conf.device),
                                                                word_seq_lens=batch.word_seq_len.to(self.conf.device),
                                                                attr_word_seq_lens=batch.attr_word_seq_len.to(self.conf.device),
                                                                orig_to_tok_index=batch.orig_to_tok_index.to(self.conf.device),
                                                                attr_orig_to_tok_index=batch.attr_orig_to_tok_index.to(self.conf.device),
                                                                input_mask=batch.attention_mask.to(self.conf.device),
                                                                attr_input_mask=batch.attr_attention_mask.to(self.conf.device))

            for idx in range(len(batch_max_ids)):
                length = batch.word_seq_len[idx]
                prediction = batch_max_ids[idx][:length].tolist()
                prediction = prediction[::-1]
                prediction = [self.conf.idx2labels[l] for l in prediction]
                one_batch_insts[idx].prediction = prediction
                prediction_spans = iobes_to_spans(prediction)
                obj = {
                    'words': one_batch_insts[idx].words,
                    'attr_words': one_batch_insts[idx].attr_words,
                    'prediction': prediction,
                    'prediction_spans': prediction_spans
                }
                #all_predictions.append(prediction)
                all_predictions.append(obj)
        return all_predictions


if __name__ == '__main__':
    sents = [
        ("vitamin e3 & d2 8% reduced fat milk", "vitamins"),
        ("vitamim c & d 8% reduced fat milk", "vitamins"),
        ("vitamin a & d & d3 8% reduced fat milk", "vitamins"),
        ("grade a 8% reduced fat milk", "grade_a"),
        ("grade a 8% reduced fat milk", "fat_content"),
        ("grade a 16% reduced fat milk", "fat_content"),
        ("grade a 16.2% reduced fat milk", "fat_content"),
        ("grade a 3.25% reduced fat milk", "fat_content"),
        ("grade a 1.7% reduced fat milk", "fat_content"),
        ("grade a 0.7% reduced fat milk", "fat_content"),
        ("monkey mee™ grade a 0.7% reduced fat milk", "brand"),
        ("monkey mee™ 0.7% reduced fat milk", "brand"),
        ("left field™ farms 0.7% reduced fat milk", "brand"),
        ("left field farms™ 0.7% reduced fat milk", "brand"),
        ("grade a reduced fat 16.234% milk", "fat_content"),
        ("grade a reduced fat 16.2% milk", "fat_content"),
        ("grade a reduced fat 16.2% milk", "lactose_content"),
        ("grade a reduced fat 16.2% milk", "grass_fed"),
        ("grade a reduced fat 16% milk", "fat_content"),
        ("grade a reduced fat milk", "grade_a"),
        ("grade a reduced fat orange milk", "flavor"),
        ("grade a reduced fat banana milk", "flavor"),
        ("grade a reduced fat bana milk", "flavor"),
        ("grade a reduced fat watermelon milk", "flavor"),
        ("grade a reduced fat lime milk", "flavor"),
        ("grade a reduced fat lemon milk", "flavor"),
        ("grade a reduced fat peach milk", "flavor"),
        ("grade a reduced fat cherry milk", "flavor"),
        ("grade a reduced fat tomato milk", "flavor"),
        ("grade a reduced fat plum milk", "flavor"),
        ("grade a reduced fat coconut milk", "flavor"),
        ("grade a reduced fat orange milk", "source"),
        ("grade a reduced fat banana milk", "source"),
        ("grade a reduced fat bana milk", "source"),
        ("grade a reduced fat watermelon milk", "source"),
        ("grade a reduced fat lime milk", "source"),
        ("grade a reduced fat lemon milk", "source"),
        ("grade a reduced fat peach milk", "source"),
        ("grade a reduced fat cherry milk", "source"),
        ("grade a reduced fat tomato milk", "source"),
        ("grade a reduced fat plum milk", "source"),
        ("grade a reduced fat coconut milk", "source"),
        ("plum wine", "flavor"),
        ("plum wine", "taxonomy"),
        ("orange wine", "taxonomy"),
        ("grade a reduced fat milk", "fat_content"),
        ("vanilla almond bliss", 'brand'),
        ("Horizon Organic Whole Shelf-Stable Milk Boxes, 8 Oz., 12 Count", 'taxonomy'),
        ("vitamin a, milk fat 2% yogurt", 'taxonomy'),
        ("vitamin a, milk fat 2% yogurt", 'fat_content'),
        ("vitamin a, 2% milk fat yogurt", 'taxonomy'),
        ("vitamin a, 2% milk fat yogurt", 'fat_content'),
        ("vitamin a, 2% milkfat yogurt", 'fat_content'),
        ("vitamin a, 2% milk-fat yogurt", 'fat_content'),
        ("vitamin a, 0% milkfat yogurt", 'fat_content'),
        ("vitamin a, 1% milkfat yogurt", 'fat_content'),
        ("vitamin a, 3.25% milk fat yogurt", 'fat_content'),
        ("vitamin a, 3.25%milk fat yogurt", 'fat_content'),
        ("vitamin a, 3.25%milkfat yogurt", 'fat_content'),
        ("vitamin a, 3.25 percent milk fat yogurt", 'fat_content'),
        ("vitamin a, 1.25 percent milk fat yogurt", 'fat_content'),
        ("vitamin a, 1 1/2% milkfat yogurt", 'fat_content'),
        ("vitamin a, 1 1/2% reduced fat yogurt", 'fat_content'),
        ("vitamin a, 3.25% milkfat yogurt", 'fat_content'),
        ("vitamin a, 3.5% milkfat yogurt", 'fat_content'),
        ("vitamin a, 3.5% milk fat yogurt", 'fat_content'),
        ("Wonder® Classic White Bread 20 oz. Loaf", "brand"),
        ("Wonder® Classic White Bread 20 oz. Loaf", "size_uom"),
        ("Wonder® Classic White Bread 20 oz. Loaf", "taxonomy"),
        ("Wonder™ Classic White Bread 20 oz. Loaf", "taxonomy"),
        ("monkey mates™ Classic White Bread 20 oz. Loaf", "taxonomy"),
        ("Light + Fit Nonfat Gluten-Free Key Lime Greek Yogurt, 5.3 Oz.", "taxonomy"),
        ("Light + Fit Nonfat Gluten-Free Key Lime Greek Yogurt, 5.3 Oz.", "fat_content"),
        ("Light + Fit Nonfat Gluten-Free Key Lime Greek Yogurt, 5.3 Oz.", "size_uom"),
        ("Light + Fit Nonfat Gluten-Free Key Lime Greek Yogurt, 5.3 Oz.", "flavor"),
        ("Light + Fit Nonfat Gluten-Free Key Lime Greek Yogurt, 5.3 Oz.", "region_origin"),
        ("Light + Fit Nonfat Gluten-Free Key Lime Greek Yogurt, 5.3 Oz.", "source"),
        ("Light + Fit Nonfat Gluten-Free Key Lime Greek Yogurt, 5.3 Oz.", "Gluten_free"),
        ("Light + Fit Nonfat Gluten-Free Key Lime Greek Yogirt, 5.3 Oz.", "taxonomy"),
        ("Light + Fit Nonfat Gluten-Free Key Lime Greek Yoga, 5.3 Oz.", "taxonomy"),
        ("grade b reduced fat milk", "grade_a"),
        ("grade a reduced fat molk", "taxonomy"),
        ("grade a reduced fat mild", "taxonomy"),
        ("Great Value Mini Vanilla Flavored Ice Cream Sandwiches, 36.8 oz, 16 Count", "taxonomy"),
        ("Great Value Mini Vanilla Flavored Ice Cream Sandwiches, 36.8 oz, 16 Count", "brand"),
        ("Great Value Mini Vanilla Flavored Ice Cream Sandwiches, 36.8 oz, 16 Count", "flavor"),
        ("Great Value Mini Vanilla Flavored Ice Cream Sandwiches, 36.8 oz, 16 Count", "size_uom"),
        ("Great Value Mini Vanilla Flavored Ice Cream Sandwiches, 36.8 oz, 16 Count", "size_uom"),
        ("OUTSHINE Watermelon Frozen Fruit Bars, 6 Ct. Box | Gluten Free | Non GMO", "taxonomy"),
        ("OUTSHINE Watermelon Frozen Fruit Bars, 6 Ct. Box | Gluten Free | Non GMO", "flavor"),
        ("OUTSHINE Watermelon Frozen Fruit Bars, 6 Ct. Box | Gluten Free | Non GMO", "flavor"),
        ("OUTSHINE Watermelon Frozen Fruit Bars, 6 Ct. Box | Gluten Free | Non GMO", "flavour"),
        ("HAAGEN-DAZS Ice Cream, Coffee, 14 Fl. Oz. Cup | No GMO Ingredients | No rBST | Gluten Free", "taxonomy"),
        ("HAAGEN-DAZS Ice Cream, Coffee, 14 Fl. Oz. Cup | No GMO Ingredients | No rBST | Gluten Free", "brand"),
        ("HAAGEN-DAZS Ice Cream, Coffee, 14 Fl. Oz. Cup | No GMO Ingredients | No rBST | Gluten Free", "flavor"),
        ("HAAGEN-DAZS Ice Cream, watermelon, 14 Fl. Oz. Cup | No GMO Ingredients | No rBST | Gluten Free", "flavor"),
        ("HAAGEN-DAZS Ice Cream, lemon, 14 Fl. Oz. Cup | No GMO Ingredients | No rBST | Gluten Free", "flavor"),
        ("HAAGEN-DAZS Ice Cream, lemon, 14 Fl. Oz. Cup | No GMO Ingredients | No rBST | Gluten Free", "flavour"),
        ("HAAGEN-DAZS Ice Cream, orange, 14 Fl. Oz. Cup | No GMO Ingredients | No rBST | Gluten Free", "flavour"),
        ("HAAGEN-DAZS Ice Cream, strawberry, 14 Fl. Oz. Cup | No GMO Ingredients | No rBST | Gluten Free", "flavour"),
        ("HAAGEN-DAZS Ice Cream, Lime, 14 Fl. Oz. Cup | No GMO Ingredients | No rBST | Gluten Free", "brand"),
        ("HAAGEN-DAZS Ice Cream, Lime, 14 Fl. Oz. Cup | No GMO Ingredients | No rBST | Gluten Free", "flavor"),
        ("HAAGEN-DAZS Ice Cream, Lime, 14 Fl. Oz. Cup | No GMO Ingredients | No rBST | Gluten Free", "taxonomy"),
        ("HAAGEN-DAZS Ice Cream, Lime, 14 Fl. Oz. Cup | No GMO Ingredients | No rBST | Gluten Free", "gluten_free"),
        ("HAAGEN-DAZS Ice Cream, Lime, 14 Fl. Oz. Cup | No GMO Ingredients | No rBST | ", "gluten_free"),
        ("HAAGEN-DAZS Ice Cream, Lime, 14 Fl. Oz. Cup | No GMO Ingredients | No rBST | Gluten Free", "non_gmo"),
        ("HAAGEN-DAZS Ice Cream, Lime, 14 Fl. Oz. Cup | Non GMO Ingredients | No rBST | Gluten Free", "non_gmo"),
        ("HAAGEN-DAZS Ice Cream, Lime, 14 Fl. Oz. Cup | No GMO Ingredients | No rBST | ", "non_gmo"),
        ("HAAGEN-DAZS Ice Cream, Lime, 14 Fl. Oz. Cup | No GMO Ingredients | No rBST | Gluten Free", "rbst_free"),
        ("HAAGEN-DAZS Ice Cream, Lime, 14 Fl. Oz. Cup | No GMO Ingredients | No rBST | ", "rbst_free"),
        ("HAAGEN-DAZS Ice Cream, Lime, 14 Fl. Oz. Cup | No GMO Ingredients | rBST free | ", "rbst_free"),
        ("HAAGEN-DAZS lemon milk", "flavour"),
        ("HAAGEN-DAZS lime milk", "flavour"),
        ("HAAGEN-DAZS lemon milk, 14 Fl. Oz. Cup | No GMO Ingredients | No rBST | Gluten Free", "flavor"),
        ("Yellow Peaches, each", "taxonomy"),
        ("Yellow Peaches, each", "flavor"),
        ("Yellow Peaches, each", "color"),
        ("Oreo Chocolate Sandwich Cookies", "taxonomy"),
        ("Oreo Chocolate Sandwich Cookies", "brand"),
        ("Cheetah Chocolate Sandwich Cookies", "brand"),
        ("Chocolate Sandwich Cookies, Oreo", "brand"),
        ("Oreo Chocolate Sandwich Cookies", "flavor"),
        ("Doritos Nacho Flavored Tortilla Chips", "taxonomy"),
        ("Doritos Nacho Flavored Tortilla Chips", "brand"),
        ("Boritos Nacho Flavored Tortilla Chips", "brand"),
        ("Nacho Flavored Doritos Tortilla Chips", "brand"),
        ("Nacho Flavored Tortilla Chips, Doritos", "brand"),
        ("Doritos Nacho Flavored Tortilla Chips", "flavor"),
        ("Cheetos Cheese Flavored Snacks", "brand"),
        ("Tire Flavored Snacks", "brand"),
    ]
    if False:#args.use_s3:
        #s3_path = "slin/catalog/models/ave_model_test/model.tar.gz"
        s3_path = "slin/catalog/models/ave_model_test_iobeconstraint/model.tar.gz"
        temp_dir = tempfile.mkdtemp(dir='/tmp')
        model_path = os.path.join(temp_dir, os.path.basename(s3_path))
        s3_download_file(s3_path, model_path)
    else:
        #model_path = "test_model/model"
        #model_path = "/tmp/tmpt9rw2w_c/model.tar.gz"
        #model_path = "./ave_model_test_iobeconstraint/model"
        model_path = "/tmp/tmpiw1ftsl_/model.tar.gz"
    device = "cpu" # cpu, cuda:0, cuda:1
    ## or model_path = "test_model/model.tar.gz"
    predictor = TransformersAVEPredictor(model_path, cuda_device=device)
    prediction = predictor.predict(sents)
    for p in prediction:
        #print(p)
        print("Product name:", " ".join(p['words']))
        print(f"{' '.join(p['attr_words'])}:", " ".join(f"[{' '.join(p['words'][s:e+1])}]" for s, e in p['prediction_spans']))
        print()
