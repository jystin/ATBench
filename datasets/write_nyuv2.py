import os
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.io import loadmat

def write_txt(f, list_ids):
    f.write('\n'.join(list_ids))
    f.close()

def extract_data(root):
    """
    extract images and labels.
    :param root:
    :return:
    """
    print('Extracting images and labels from nyu_depth_v2_labeled.mat...')
    data = h5py.File(os.path.join(root, 'nyu_depth_v2_labeled.mat'))
    images = np.array(data['images'])
    depths = np.array(data['depths'])
    raw_depths = np.array(data['rawDepths'])
    print(f'images shape: {images.shape}')
    num_img = images.shape[0]
    print(f'image number: {num_img}')

    images_dir = os.path.join(root, 'images')
    depths_dir = os.path.join(root, 'depths')
    raw_depths_dir = os.path.join(root, "raw_depths")
    if not os.path.isdir(images_dir):
        os.makedirs(images_dir)
    if not os.path.isdir(depths_dir):
        os.makedirs(depths_dir)
    if not os.path.isdir(raw_depths_dir):
        os.makedirs(raw_depths_dir)


    # max = depths.max()
    # depths = depths / max * 255  # (0 - 255)
    # depths = depths.transpose((0, 2, 1))  # (B, h, w)

    depths = depths.transpose((0, 2, 1))  # (B, h, w)
    depths = depths * 1000.0
    raw_depths = raw_depths.transpose((0, 2, 1))  # (B, h, w)
    raw_depths = raw_depths * 1000.0

    bar = tqdm(range(num_img))
    for i in bar:
        img = images[i]
        r = Image.fromarray(img[0]).convert('L')
        g = Image.fromarray(img[1]).convert('L')
        b = Image.fromarray(img[2]).convert('L')
        img = Image.merge('RGB', (r, g, b))
        img = img.transpose(Image.ROTATE_270)
        img = np.array(img)
        image_black_boundary = np.zeros((480, 640, 3), dtype=np.uint8)
        image_black_boundary[7:474, 7:632, :] = img[7:474, 7:632, :]
        img = Image.fromarray(image_black_boundary)
        img.save(os.path.join(images_dir, str(i) + '.jpg'), optimize=True)

        # depths_img = Image.fromarray(np.uint16(depths[i]))
        # depths_img = depths_img.transpose(Image.FLIP_LEFT_RIGHT)
        # iconpath = os.path.join(depths_dir, str(i) + '.png')
        # depths_img.save(iconpath, 'PNG', optimize=True)
        #
        # raw_depths_img = Image.fromarray(np.uint16(raw_depths[i]))
        # raw_depths_img = raw_depths_img.transpose(Image.FLIP_LEFT_RIGHT)
        # iconpath = os.path.join(raw_depths_dir, str(i) + '.png')
        # raw_depths_img.save(iconpath, 'PNG', optimize=True)



def split(root):
    print('Generating training and validation split from split.mat...')
    split_file = loadmat(os.path.join(root, 'splits.mat'))
    train_images = tuple([int(x) for x in split_file["trainNdxs"]])
    test_images = tuple([int(x) for x in split_file["testNdxs"]])
    print("%d training images" % len(train_images))
    print("%d test images" % len(test_images))

    train_ids = [str(i - 1) for i in train_images]
    test_ids = [str(i - 1) for i in test_images]

    train_list_file = open(os.path.join(root, 'train.txt'), 'a')
    write_txt(train_list_file, train_ids)

    test_list_file = open(os.path.join(root, 'val.txt'), 'a')
    write_txt(test_list_file, test_ids)


def labels_40(root):
    print('Extracting labels with 40 classes from labels40.mat...')
    data = loadmat(os.path.join(root, 'labels40.mat'))
    labels = np.array(data['labels40'])
    print(f'labels shape: {labels.shape}')

    path_converted = os.path.join(root, 'labels40')
    if not os.path.isdir(path_converted):
        os.makedirs(path_converted)

    bar = tqdm(range(labels.shape[2]))
    for i in bar:
        label = np.array(labels[:, :, i].transpose((1, 0)))
        label_img = Image.fromarray(np.uint8(label))
        label_img = label_img.transpose(Image.ROTATE_270)
        label_img.save(os.path.join(path_converted, str(i) + '.png'), optimize=True)


def main():
    root = "../xdecoder_data/depth_datasets/nyuv2"
    extract_data(root)
    # split(root)
    # labels_40(root)


if __name__ == '__main__':
    main()
    # img = Image.open('../xdecoder_data/nyu_v2/depths/0.png')
    # img2 = Image.open('../xdecoder_data/nyu_v2/raw_depths/0.png')
    # img = np.array(img)
    # img2 = np.array(img2)
    # max = img.max()
    # img = img / max * 255  # (0 - 255)
    # img2 = img2 / max * 255  # (0 - 255)
    # Image.fromarray(np.uint8(img)).show()
    # print(img[240])
    # print(img2[240])





