import os
import pickle
from torch.utils import data
from dataset.cityscapes import CityscapesDataSet, CityscapesTrainInform, CityscapesValDataSet, CityscapesTestDataSet
from dataset.camvid import CamVidDataSet, CamVidValDataSet, CamVidTrainInform, CamVidTestDataSet
from dataset.ade20k import ADE20KSegmentation
from torchvision import transforms
from dataset.distributed import make_data_sampler, make_batch_data_sampler


def build_dataset_train(dataset, input_size, batch_size, train_type, random_scale, random_mirror, num_workers, args):
    data_dir = os.path.join('/media/sdb/datasets/segment/', dataset)
    dataset_list = dataset + '_trainval_list.txt'
    train_data_list = os.path.join(data_dir, dataset + '_' + train_type + '_list.txt')
    val_data_list = os.path.join(data_dir, dataset + '_val' + '_list.txt')
    inform_data_file = os.path.join('./dataset/inform/', dataset + '_inform.pkl')

    if dataset == "cityscapes":
        # inform_data_file collect the information of mean, std and weigth_class
        if not os.path.isfile(inform_data_file):
            print("%s is not found" % (inform_data_file))
            dataCollect = CityscapesTrainInform(data_dir, 19, train_set_file=dataset_list,
                                                    inform_data_file=inform_data_file)
            datas = dataCollect.collectDataAndSave()
            if datas is None:
                print("error while pickling data. Please check.")
                exit(-1)
        else:
            print("find file: ", str(inform_data_file))
            datas = pickle.load(open(inform_data_file, "rb"))

        trainLoader = data.DataLoader(
            CityscapesDataSet(data_dir, train_data_list, crop_size=input_size, scale=random_scale,
                              mirror=random_mirror, mean=datas['mean']),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True)

        valLoader = data.DataLoader(
            CityscapesValDataSet(data_dir, val_data_list, f_scale=1, mean=datas['mean']),
            batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=True,
            drop_last=True)
    elif dataset == "camvid":
        # inform_data_file collect the information of mean, std and weigth_class
        if not os.path.isfile(inform_data_file):
            print("%s is not found" % (inform_data_file))
            dataCollect = CamVidTrainInform(data_dir, 11, train_set_file=dataset_list, inform_data_file=inform_data_file)
            datas = dataCollect.collectDataAndSave()
            if datas is None:
                print("error while pickling data. Please check.")
                exit(-1)
        else:
            print("find file: ", str(inform_data_file))
            datas = pickle.load(open(inform_data_file, "rb"))

        trainLoader = data.DataLoader(
            CamVidDataSet(data_dir, train_data_list, crop_size=input_size, scale=random_scale,
                          mirror=random_mirror, mean=datas['mean']),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True)

        valLoader = data.DataLoader(
            CamVidValDataSet(data_dir, val_data_list, f_scale=1, mean=datas['mean']),
            batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=True)
    elif dataset == "ade20k":
        inform_data_file = os.path.join('./dataset/inform/', 'cityscapes_inform.pkl')
        datas = pickle.load(open(inform_data_file, "rb"))

        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        data_kwargs = {'transform': input_transform, 'base_size': 520,
                       'crop_size': 480, 'encode': False}
        train_dataset = ADE20KSegmentation(split='train', mode='train', **data_kwargs)
        val_dataset = ADE20KSegmentation(split='val', mode='val', **data_kwargs)

        train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=False)
        train_batch_sampler = make_batch_data_sampler(train_sampler, batch_size)
        val_sampler = make_data_sampler(val_dataset, shuffle=False, distributed=False)
        val_batch_sampler = make_batch_data_sampler(val_sampler, batch_size)

        trainLoader = data.DataLoader(dataset=train_dataset,
                                 batch_sampler=train_batch_sampler,
                                 num_workers=args.num_workers,
                                 pin_memory=True)
        valLoader = data.DataLoader(dataset=val_dataset,
                                     batch_sampler=val_batch_sampler,
                                     num_workers=args.num_workers,
                                     pin_memory=True)
    else:
        raise NotImplementedError(
            "This repository now supports datasets: cityscapes, camvid and ade20k, %s is not included" % dataset)
    return datas, trainLoader, valLoader


def build_dataset_test(dataset, num_workers, none_gt=False):
    data_dir = os.path.join('/media/sdb/datasets/segment/', dataset)
    dataset_list = os.path.join(dataset, '_trainval_list.txt')
    test_data_list = os.path.join(data_dir, dataset + '_test' + '_list.txt')
    inform_data_file = os.path.join('./dataset/inform/', dataset + '_inform.pkl')

    # inform_data_file collect the information of mean, std and weigth_class
    if not os.path.isfile(inform_data_file):
        print("%s is not found" % (inform_data_file))
        if dataset == "cityscapes":
            dataCollect = CityscapesTrainInform(data_dir, 19, train_set_file=dataset_list,
                                                inform_data_file=inform_data_file)
        elif dataset == 'camvid':
            dataCollect = CamVidTrainInform(data_dir, 11, train_set_file=dataset_list,
                                            inform_data_file=inform_data_file)
        else:
            raise NotImplementedError(
                "This repository now supports two datasets: cityscapes and camvid, %s is not included" % dataset)

        datas = dataCollect.collectDataAndSave()
        if datas is None:
            print("error while pickling data. Please check.")
            exit(-1)
    else:
        print("find file: ", str(inform_data_file))
        datas = pickle.load(open(inform_data_file, "rb"))

    if dataset == "cityscapes":
        # for cityscapes, if test on validation set, set none_gt to False
        # if test on the test set, set none_gt to True
        if none_gt:
            testLoader = data.DataLoader(
                CityscapesTestDataSet(data_dir, test_data_list, mean=datas['mean']),
                batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
        else:
            test_data_list = os.path.join(data_dir, dataset + '_val' + '_list.txt')
            testLoader = data.DataLoader(
                CityscapesValDataSet(data_dir, test_data_list, mean=datas['mean']),
                batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

        return datas, testLoader

    elif dataset == "camvid":

        testLoader = data.DataLoader(
            CamVidValDataSet(data_dir, test_data_list, mean=datas['mean']),
            batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

        return datas, testLoader