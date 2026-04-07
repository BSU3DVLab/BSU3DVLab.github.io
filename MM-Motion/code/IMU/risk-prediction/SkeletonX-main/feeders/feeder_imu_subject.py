import numpy as np

from torch.utils.data import Dataset

from feeders import tools

class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False, class_group=None, train_indices_info_path=None):
        """
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param train_indices_info_path: Path to subject information file
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.load_data()
        self.train_indices_info_path = train_indices_info_path
        self.DASP_same_action_cnt = 0
        self.SADP_same_performer_cnt = 0
        
        if normalization:
            self.get_mean_map()

        # calculate valid frames
        self.valid_frames = np.zeros(len(self.data))
        for i in range(len(self.data)):
            self.valid_frames[i] = np.sum(self.data[i].sum(0).sum(-1).sum(-1) != 0)
        self.valid_frames = self.valid_frames.astype(int)

        # print samples per class
        cnt = np.zeros(16)
        for label in self.label:
            cnt[label] += 1
        print("[Dataset] Split: {}. Samples per class: {}".format(self.split, cnt))

        # Simple subject information for IMU data (since we don't have real subject data)
        # For now, we'll assign each sample to a default subject
        self.performer = np.zeros(len(self.data), dtype=int)
        
        # Build quick access dicts
        self.performer_cls_dict = {0: {}}
        self.cls_performer_dict = {}
        
        for cls in set(self.label):
            self.performer_cls_dict[0][cls] = list(np.where(self.label == cls)[0])
            self.cls_performer_dict[cls] = {0: list(np.where(self.label == cls)[0])}

    def load_data(self):
        # data: N C T V M   (already in correct format)
        npz_data = np.load(self.data_path, mmap_mode='r' if self.use_mmap else None)
        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        elif self.split == 'aux':
            self.data = npz_data['x_train']  # Use train data for aux since we don't have separate aux data
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['aux_' + str(i) for i in range(len(self.data))]
        elif self.split == 'anchor':
            self.data = npz_data['x_train']  # Use train data for anchor since we don't have separate anchor data
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['anchor_' + str(i) for i in range(len(self.data))]
        elif self.split == 'eval':
            self.data = npz_data['x_train']  # Use train data for eval since we don't have separate eval data
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['eval_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def data_aug(self, data_numpy, index):
        # For simplicity, we'll just return the original data
        return data_numpy

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = self.valid_frames[index]
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
        return data_numpy, label, index

    def get_subject(self, index):
        return self.performer[index]

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def get_pair_data(self, indexes, mode='SASP', in_performer=None, in_labels=None):
        # For simplicity, we'll just return the same data for both pairs
        data_list = []
        label_list = []
        for index in indexes:
            data, label = self.__getitem__(index)
            data_list.append(data)
            label_list.append(label)
        return data_list[0], label_list[0], data_list[0], label_list[0]

    def output_pair_sample_stat(self) -> str:
        return f"DASP same action: {self.DASP_same_action_cnt}, SADP same performer: {self.SADP_same_performer_cnt}"