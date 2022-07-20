import os
import json
from PIL import Image
import torch
from torchvision import transforms
import clip
from dataset.languages.simple_tokenizer import SimpleTokenizer



class Voc2007Classification(torch.utils.data.Dataset):
    def __init__(self, data_root, image_set="train", transform=None):
        """
        Pascal voc2007 training/validation data: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
        test data: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
        """
        self.data_root = self._update_path(data_root, image_set)
        self.transform = transform
        self.labels = self._read_annotation(image_set)
        self.images = list(self.labels.keys())

    def _update_path(self, data_root, image_set):
        if image_set == "train" or image_set == "val":
            data_root += "train/VOCdevkit/VOC2007"
        elif image_set == "test":
            data_root += "test/VOCdevkit 2/VOC2007"
        else:
            raise Exception("Incorrect image set!")
        return data_root

    def __getitem__(self, index):
        img_path = os.path.join(self.data_root, 'JPEGImages/'+self.images[index]+'.jpg')
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        label = self.labels[self.images[index]]
        label = torch.LongTensor(label)
        return image,label

    def __len__(self):
        return len(self.images)

    def _read_annotation(self, image_set="train"):
        """
        Annotation interpolation, refer to: 
        http://host.robots.ox.ac.uk/pascal/VOC/voc2007/htmldoc/voc.html#SECTION00093000000000000000
        """
        object_categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', \
                            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', \
                            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        annotation_folder = os.path.join(self.data_root, "ImageSets/Main/")
        files = [file_name for file_name in os.listdir(annotation_folder) if file_name.endswith("_"+image_set+".txt")]
        labels_all = dict()
        for file_name in files:
            label_str = file_name.split("_")[0]
            label_int = object_categories.index(label_str)
            with open(annotation_folder+"/"+file_name, "r") as fread:
                for line in fread.readlines():
                    index = line[:6]
                    if index not in labels_all.keys():
                        labels_all[index] = [0]*len(object_categories)
                    flag = 1
                    if line[7:9] and int(line[7:9]) != 1:
                        flag = -1
                    if flag == 1:
                        labels_all[index][label_int] = 1
        return labels_all


class HatefulMemes(torch.utils.data.Dataset):    
    def __init__(self, data_root, image_set="train", transform=None, tokenizer=None, context_length=77):
        """
        Facebook Hateful Memes: Phase 1 dataset: https://www.drivendata.org/competitions/64/hateful-memes/data/
        """
        self.data_root = data_root
        self.transform = transform
        self.images = self._read_annotation(image_set)
        self.tokenizer = tokenizer
        self.context_length = context_length

    def __getitem__(self, index):
        img_path = os.path.join(self.data_root, self.images[index]["image_file"])
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        label = self.images[index]["label"]
        label = torch.tensor(label)
        text = self.images[index]["text"]
        if self.tokenizer:
            text = self.tokenizer(text, context_length=self.context_length)[0,:]
        return image, label

    def __len__(self):
        return len(self.images)

    def _read_annotation(self, image_set="train"):
        """
        Annotation interpolation, refer to: 
        https://www.drivendata.org/competitions/64/hateful-memes/page/206/
        """
        if image_set == "train":
            label_file = os.path.join(self.data_root, "train.jsonl")
        elif image_set == "val":
            label_file = os.path.join(self.data_root, "dev_seen.jsonl")
        else:
            raise Exception(f"Incorrect image_set value: {image_set}!")
        image_records = []
        with open(label_file, "r") as fread_file:
            for line in fread_file.readlines():
                record = json.loads(line)
                image_records.append({"image_file":record["img"], "text":record["text"], "label": record["label"]})
        return image_records


class ChestXRay8(torch.utils.data.Dataset):
    def __init__(self, data_root, image_set="train", transform=None):
        """
        ChestX-ray dataset: https://paperswithcode.com/dataset/chestx-ray8
        """
        self.data_root = data_root
        self.transform = transform
        self.image_set = image_set
        self.labels = self._read_annotation()
        self.images = self._read_split_file()

    def __getitem__(self, index):
        img_path = os.path.join(self.data_root, "images", self.images[index])
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        label = self.labels[self.images[index]]
        label = torch.LongTensor(label)
        return image,label

    def __len__(self):
        return len(self.images)

    def _read_split_file(self):
        if self.image_set == "train":
            split_file = "train_val_list.txt"
        elif self.image_set == "test":
            split_file = "test_list.txt"
        else:
            raise Exception("Incorrect image set!")
        file_list = []
        with open(os.path.join(self.data_root, split_file), "r") as fread:
            for line in fread.readlines():
                file_list.append(line.replace("\n", ""))
        return file_list

    def _read_annotation(self):
        """
        Annotation interpolation, refer to: 
        http://host.robots.ox.ac.uk/pascal/VOC/voc2007/htmldoc/voc.html#SECTION00093000000000000000
        """
        object_categories = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',\
            'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', \
            'Pleural_Thickening', 'Pneumonia', 'Pneumothorax', 'No Finding']
        annotation_file = os.path.join(self.data_root, "Data_Entry_2017_v2020.csv")
        image2labels = dict()
        with open(annotation_file, "r") as fread:
            for i, line in enumerate(fread.readlines()):
                if i==0:
                    continue
                image_name, labels_raw, _, _, _, _, _, _, _, _, _= line.split(",")
                labels = labels_raw.split('|')
                labels_int = [0]*(len(object_categories) - 1)
                for label in labels:
                    if label == "No Finding":
                        continue
                    labels_int[object_categories.index(label)] = 1
                image2labels[image_name] = labels_int
        return image2labels
