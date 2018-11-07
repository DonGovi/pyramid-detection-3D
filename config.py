import os

class Config(object):
    def __init__(self):
        super(Config, self).__init__()
        self.root = "/home/youkun/sph_samples/"
        self.resampled_data = os.path.join(self.root, "resample/resample_data/")
        self.annt_dir = os.path.join(self.root, "annotations/")
        self.resample_label_file = os.path.join(self.annt_dir, "bbox_label.csv")
        self.crop_root = "/home/youkun/sph_samples/ais_and_mia/"
        self.crop_64_samples = os.path.join(self.crop_root, "samples/")
        self.crop_64_label = os.path.join(self.crop_root, "samples_train_aug.csv")
        self.crop_128_samples = os.path.join(self.crop_root, "samples_128/samples/")
        self.crop_128_label = os.path.join(self.crop_root, "samples_128/samples_bbox.csv")
        self.test_label = "/home/youkun/sph_samples/slic_samples/bbox_test_label.csv"
        self.test_path = os.path.join(self.root, "test/")

        self.batch_size = 8
        self.trainval_ratio = 5.
        self.learning_rate_start = 0.01
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.lr_decay_epochs = [600]
        self.input_size = 64
        self.norm_threshold = (-1000., 400.)
        self.num_epochs = 500 
        self.pre_train_epochs = 1 
        self.sample_save = "/home/youkun/ais_and_mia/samples/"
        self.backbone = "resnet50"
        self.num_cls = 2
        self.checkpoints_path = "/home/youkun/pyramid_detection/checkpoints/"
