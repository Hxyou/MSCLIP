from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import logging
import os
import pprint
import numpy as np
import torch
import torch.nn.parallel
from torch.utils.collect_env import get_pretty_env_info
# These 2 lines are a wordaround for "Too many open files error". Refer: https://github.com/pytorch/pytorch/issues/11201
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import _init_paths
from utils.comm import comm
from utils.utils import create_logger
from config import config
from config import update_config
import logging
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from torchvision import transforms
import torchvision.models
from PIL import Image
#########################################
# The following 2 lines are to solve PIL "IOError: image file truncated" with big images.
# Refer to https://stackoverflow.com/questions/12984426/python-pil-ioerror-image-file-truncated-with-big-images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from dataset.languages import SimpleTokenizer
from dataset.prompts.constants import ALL_CLASSES_DICT
from dataset.prompts.constants import ALL_TEMPLATES_DICT
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_auc_score
from models import clip_openai_pe_res_v1


TRANSFER_NAME = {'oxford-flower-102':'flower102-tf', 'fgvc-aircraft-2013b':'fgvc-aircraft-2013b-variants102'}

def parse_args():
    parser = argparse.ArgumentParser(
        description='Test a classification model.')

    parser.add_argument('--ds',
                        required=True,
                        help='Evaluation dataset configure file name.',
                        type=str)

    parser.add_argument('--model',
                        required=True,
                        help='Evaluation model configure file name',
                        type=str)


    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)



    args = parser.parse_args()
    return args

def get_dataloader(dataset, val_split=0, batch_size_per_gpu=32, workers=6, pin_memory=True):
    if val_split == 0:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size_per_gpu,
            shuffle=False,
            num_workers=workers,
            pin_memory=pin_memory,
            sampler=None,
            drop_last=False,
        )
        return data_loader
    else:
        def train_val_dataset(dataset, val_split):
            train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
            datasets = {}
            datasets['train'] = Subset(dataset, train_idx)
            datasets['val'] = Subset(dataset, val_idx)
            return datasets
        datasets = train_val_dataset(dataset, val_split)
        train_loader = torch.utils.data.DataLoader(
            datasets['train'],
            batch_size=batch_size_per_gpu,
            shuffle=True,
            num_workers=workers,
            pin_memory=pin_memory,
            sampler=None,
            drop_last=False,
        )
        val_loader = torch.utils.data.DataLoader(
            datasets['val'],
            batch_size=batch_size_per_gpu,
            shuffle=True,
            num_workers=workers,
            pin_memory=pin_memory,
            sampler=None,
            drop_last=False,
        )
        return train_loader, val_loader


def multilabel_to_vec(indices, n_classes):
    vec = np.zeros(n_classes)
    for x in indices:
        vec[x] = 1
    return vec


def multiclass_to_int(indices):
    return indices[0]


@torch.no_grad()
def zeroshot_classifier(classnames, templates, tokenizer, model, action=None):
    zeroshot_weights = []
    for classname in tqdm(classnames):
        texts = [template.format(classname) for template in templates]
        texts = tokenizer(texts).cuda()
        class_embeddings = model.encode_text(texts, action=action)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()

    return zeroshot_weights

def mAP_11points(y_label, y_pred_proba):
    precision, recall, _ = precision_recall_curve(y_label, y_pred_proba)
    recall_thresholds = np.linspace(1, 0, 11, endpoint=True).tolist()
    precision_sum = 0
    recall_idx = 0
    precision_tmp = 0
    for threshold in recall_thresholds:
        while recall_idx < len(recall) and threshold <= recall[recall_idx]:
            precision_tmp = max(precision_tmp, precision[recall_idx])
            recall_idx += 1
        precision_sum += precision_tmp
    return precision_sum/11

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def zero_shot():
    args = parse_args()
    args.cfg = args.ds
    update_config(config, args)
    args.cfg = args.model
    update_config(config, args)
    config.defrost()
    config.NAME = ""
    config.freeze()

    final_output_dir = create_logger(config, args.cfg, 'zero_shot_{}'.format(config.MODEL.PRETRAINED_MODEL.split('/')[-2]))
    if comm.is_main_process():
        logging.info("=> collecting env info (might take some time)")
        logging.info("\n" + get_pretty_env_info())
        logging.info(pprint.pformat(args))
        logging.info(config)
        logging.info("=> saving logging info into: {}".format(final_output_dir))

    # >>>>>>>>>>>>>> Load data <<<<<<<<<<<<<<
    transform_CLIP = transforms.Compose([
            transforms.Resize(224, interpolation=Image.BICUBIC),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.INPUT.MEAN, std=config.INPUT.STD),
        ])
    if config.DATASET.DATASET == "voc2007classification":
        from evaluation.dataset import Voc2007Classification
        test_dataloader = get_dataloader(Voc2007Classification(config.DATASET.ROOT, image_set="test", transform=transform_CLIP))
    elif config.DATASET.DATASET == "hatefulmemes":
        from evaluation.dataset import HatefulMemes
        test_dataloader = get_dataloader(HatefulMemes(config.DATASET.ROOT, image_set="val", transform=transform_CLIP))
    else:
        test_dataloader = get_dataloader(
            torchvision.datasets.ImageFolder(os.path.join(config.DATASET.ROOT, config.DATASET.TEST_SET),
                                                 transform=transform_CLIP))
    # >>>>>>>>>>>>>> Load Model <<<<<<<<<<<<<<
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = clip_openai_pe_res_v1.get_clip_model(config)
    model_file = config.MODEL.PRETRAINED_MODEL
    logging.info('=> load model file: {}'.format(model_file))
    state_dict = torch.load(model_file, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    logging.info('=> switch to eval mode')
    model_without_ddp = model.module if hasattr(model, 'module') else model
    model_without_ddp.eval()


    # >>>>>>>>>>>>>> compute zero-shot weight <<<<<<<<<<<<<<
    tokenobj = SimpleTokenizer()
    classnames = ''
    templates = ''
    if config.DATASET.DATASET in TRANSFER_NAME:
        prompt_template_name = TRANSFER_NAME[config.DATASET.DATASET]
    else:
        prompt_template_name = config.DATASET.DATASET
    if prompt_template_name in ALL_CLASSES_DICT:
        classnames = ALL_CLASSES_DICT[prompt_template_name]
        templates = ALL_TEMPLATES_DICT[prompt_template_name]
    else:
        raise ValueError('Can not find prompt for dataset: {}'.format(config.DATASET.DATASET))
    logging.info('=> Start to build zeroshot classifier')
    zeroshot_weights = zeroshot_classifier(
        classnames, templates, tokenobj, model_without_ddp
    )
    top1 = AverageMeter()
    logging.info('=> Start to inference')
    if config.TEST.METRIC == '11point_mAP' or config.TEST.METRIC == 'mean-per-class' or config.TEST.METRIC == 'roc_auc':
        total_logits = []
        total_y = []
    for _, batch in enumerate(tqdm(test_dataloader)):
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
        if isinstance(y, list):
            assert len(y) == 1
            y = y[0]
        if device == torch.device('cuda'):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

        features_image = model_without_ddp.encode_image(x)
        logits = 100. * features_image @ zeroshot_weights

        if config.TEST.METRIC == '11point_mAP':
            total_logits.append(logits.detach().cpu())
            total_y.append(y.detach().cpu())
        elif config.TEST.METRIC == 'mean-per-class' or config.TEST.METRIC == 'roc_auc':
            total_logits.append(logits.detach().cpu())
            total_y.append(y.detach().cpu())
        else:
            prec1 = accuracy(logits, y, (1,))[0]
            top1.update(prec1, x.size(0))

    logging.info('=> synchronize...')
    comm.synchronize()
    if config.TEST.METRIC == '11point_mAP':
        logits = torch.cat(total_logits, dim=0)
        y = torch.cat(total_y, dim=0)
        mAP_sum = 0
        for class_i in range(y.shape[1]):
            mAP_sum += mAP_11points(y[:, class_i].detach().cpu(), logits[:, class_i].detach().cpu())
        top1_acc = mAP_sum * 100 / y.shape[1]
    elif config.TEST.METRIC == 'mean-per-class':
        logits = torch.cat(total_logits, dim=0)
        y = torch.cat(total_y, dim=0)
        pred_label = logits.argmax(-1)
        top1_acc = balanced_accuracy_score(y, pred_label)
        top1_acc = top1_acc * 100
    elif config.TEST.METRIC == 'roc_auc':
        logits = torch.cat(total_logits, dim=0)
        y = torch.cat(total_y, dim=0)
        # import pdb
        # pdb.set_trace()
        # Added for hateful meme
        top1_acc = roc_auc_score(y, logits[:, 1], multi_class="ovr")
        top1_acc = top1_acc * 100
    else:
        top1_acc = top1.avg

    msg = '=> {dataset}% TEST:\t' \
        'Error@1 {error1:.3f}%\t' \
        '{metric}@1 {top1:.3f}%\t'.format(dataset=config.DATASET.DATASET,
        metric=config.TEST.METRIC, top1=top1_acc, error1=100-top1_acc
        )

    logging.info(msg)


if __name__ == "__main__":
    zero_shot()
