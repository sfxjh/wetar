from datasets import load_dataset
import os
from utils.public import check_and_create_path
import numpy as np



def dataset_download():
    path = '/home/xujh/Diff-level-adv-training/dataset'
    dataset_name = 'snli'
    path = os.path.join(path, dataset_name)
    dataset_dict = load_dataset(dataset_name)
    for k in dataset_dict.keys():
        dataset_dict[k].to_csv(os.path.join(path, k+'.csv'))


def dataset_loader():
    dataset_type = ['train.csv', 'dev.csv', 'test.csv']
    type = ['train', 'validation', 'test']
    # type = ['train', 'validation_matched', 'test_matched']
    validation_portion = 0.1
    datasets_dir = '/home/xujh/Diff-level-adv-training/dataset'

    dataset_name = 'snli' # glue, sst2  imdb ag_news
    dataset_dict = load_dataset(dataset_name)
    file = os.path.join(datasets_dir, dataset_name)
    check_and_create_path(file)


    if ('validation' in dataset_dict.keys() or 'validation_matched' in dataset_dict.keys()) and (dataset_name != 'snli') :
        for i in range(len(dataset_type)):
            file = os.path.join(datasets_dir, dataset_name, dataset_type[i])
            dataset = dataset_dict[type[i]]
            dataset.to_csv(file)
    else:
        train = dataset_dict[type[0]]
        data = train.train_test_split(test_size=validation_portion)
        train = data[type[0]]
        valid = data[type[2]]
        test = dataset_dict[type[2]]
        li = [train, valid, test]
        for i in range(len(li)):
            file = os.path.join(datasets_dir, dataset_name, dataset_type[i])
            dataset = li[i]
            dataset.to_csv(file)


def valid_batch_manager(instances, batch_size, ratio):
    length = len(instances)
    np.random.shuffle(instances)
    aug_num_batch = int(batch_size * ratio / (1 + ratio))
    ori_num_batch = batch_size - aug_num_batch
    batch_num = int(np.ceil(length / ori_num_batch))
    ori_instances = []
    need_aug_instances = []
    for i in range(batch_num):
        if (i+1)*ori_num_batch >= length:
            temp_instances = instances[i*ori_num_batch:]
            aug_num = int((length-i*ori_num_batch) * ratio)
            need_aug_instances += np.random.choice(temp_instances, size=(aug_num,), replace=False).tolist()
        else:
            temp_instances = instances[i*ori_num_batch:(i+1)*ori_num_batch%length]
            need_aug_instances += np.random.choice(temp_instances, size=(aug_num_batch,), replace=False).tolist()
        ori_instances.append(temp_instances)
    return ori_instances, need_aug_instances, aug_num_batch
    


if __name__ == '__main__':
    dataset_loader()