import os
import random
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from encoder_alt import DataEncoder
from bbox import BoundingBox, Annotation
from augmentation_tools import AugMethod3D

def normalize(image, threshold=(-1000., 400.)):
    noise = np.random.randint(-76, 76, image.shape)
    image += noise
    image = (image - threshold[0]) / (threshold[1] - threshold[0])
    image = np.clip(image, 0, 1)

    return image

class SPHDataset(data.Dataset):
    def __init__(self, scan_path, scan_list, label_df, threshold, transform=False):
        '''
        Args:
          scan_path: (str) scan images path
          label_file: (str) label file 
          threshold: (tuple) (min, max) threshold to convert HU to gray
          transform: (boolean) image transform, in 3d nodule scan, only consider "flip"
        '''
        self.scan_path = scan_path
        self.label_df = label_df
        self.scan_list = scan_list
        self.threshold = threshold
        self.transform = transform
        self.encoder = DataEncoder()

    def __getitem__(self, idx):
        '''
        Return:
        image: (tensor) image array after normalized and transformed
        boxes: (tensor) bbox coords list -> [#obj, 6]
        labels: (tensor) class label list -> [#obj,]
        '''
        sample_id = self.scan_list[idx]
 	#print(sample_id)
        patient_df = self.label_df[self.label_df["sample_id"] == sample_id]
        image_path = os.path.join(self.scan_path, sample_id+".npy")
        #print(os.path.join(self.scan_path, sample_id+".npy"))
        image = np.load(image_path)
        image = normalize(image, self.threshold)
        #image_shape = image.shape
        if self.transform:
            axis = ["d", "h", "w"]
            flip_dim = random.sample(axis, 1)
            image = AugMethod3D(image).flip(trans_position=flip_dim[0])
            labels = Annotation(image_shape=image.shape, 
                                label_dataframe=patient_df, 
                                flip_dim=flip_dim[0])
            loc_tsr, cls_tsr = labels.build_annotations()
        else:
            labels = Annotation(image_shape=image.shape,
                                label_dataframe=patient_df)
            loc_tsr, cls_tsr = labels.build_annotations()

        image = np.expand_dims(image, 0)
        image_tsr = torch.from_numpy(image.copy())
        sample = {"id":sample_id, "image": image_tsr, "boxes": loc_tsr, "classes": cls_tsr}

        return sample

    def __len__(self):
        return len(self.scan_list)

    def collate_fn(self, batch):
        '''
        Pad images and encode targets

        Args:
          batch: (list) of samples

        Returns:
          Padded images, stacked cls_targets, stacked loc_targets.
        '''
        ids = [x["id"] for x in batch]
        imgs = [x["image"] for x in batch]
        boxes = [x["boxes"] for x in batch]
        labels = [x["classes"] for x in batch]

        max_d = max([im.size(1) for im in imgs])
        max_h = max([im.size(2) for im in imgs])
        max_w = max([im.size(3) for im in imgs])
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 1, max_d, max_h, max_w)

        loc_targets = []
        cls_targets = []
        dict_list = []
        for i in range(num_imgs):
            gt_dict = {}
            im = imgs[i]
            im_d, im_h, im_w = im.size(1), im.size(2), im.size(3)
            inputs[i,:, :im_d, :im_h, :im_w] = im
            loc_target, cls_target = self.encoder.encode(boxes[i], labels[i], input_size=(max_d, max_h, max_w))
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
            gt_dict["id"] = ids[i]
            gt_dict["boxes"] = boxes[i]
            gt_dict["labels"] = labels[i]
            dict_list.append(gt_dict)

        #print(inputs.size())

        loc_targets = torch.stack(loc_targets)
        cls_targets = torch.stack(cls_targets)
        #print(loc_targets.size())
        #print(cls_targets.size())

        return dict_list, inputs, loc_targets, cls_targets


if __name__ == '__main__':
    import config
    cfg = config.Config()
    scan_path = cfg.crop_128_samples
    label_df = pd.read_csv(cfg.crop_128_label)
    scan_list = label_df["sample_id"].drop_duplicates().tolist()
    #print(type(scan_list))
    threshold = cfg.norm_threshold
    dataset = SPHDataset(scan_path=scan_path, scan_list=scan_list, label_df=label_df, threshold=threshold)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1, 
                                 collate_fn=dataset.collate_fn)
    

    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(dataloader):
        print(inputs.size())
        print(loc_targets.size(), type(loc_targets))
        print(cls_targets.size(), type(cls_targets))
