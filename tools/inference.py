##############################################################
# % Author: Castle
# % Date:14/01/2023
###############################################################
import argparse
import os
import numpy as np
# import cv2
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

import torch
print(torch.cuda.is_available())

from tools import mybuilder as builder
from utils.config import cfg_from_yaml_file
from utils import misc
from datasets.io import IO
from datasets.data_transforms import Compose
import cv2
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_config', 
        help = 'yaml config file')
    parser.add_argument(
        'model_checkpoint', 
        help = 'pretrained weight')
    parser.add_argument('--pc_root', type=str, default='', help='Pc root')
    parser.add_argument('--pc', type=str, default='', help='Pc file')   
    parser.add_argument(
        '--save_vis_img',
        action='store_true',
        default=False,
        help='whether to save img of complete point cloud') 
    parser.add_argument(
        '--out_pc_root',
        type=str,
        default='',
        help='root of the output pc file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--num_workers', type=int, default=4) 
    parser.add_argument('--experimental', action='store_true', default=False, help='Flag if the input cloud is from experimental data -- Default == False')
    parser.add_argument('--save_img_path', type=str, default='', help='Where to save output image')
    parser.add_argument('--n_imgs', type=int, default=20, help='sets the number of images saved -- default == first 20 events')
    args = parser.parse_args()

    # assert args.save_vis_img or (args.out_pc_root != '')
    assert args.model_config is not None
    assert args.model_checkpoint is not None
    # assert (args.pc != '') or (args.pc_root != '')

    return args

def inference_single(model, pc_path, args, config, root=None):
    if root is not None:
        pc_file = os.path.join(root, pc_path)
    else:
        pc_file = pc_path
    # read single point cloud
    pc_ndarray = IO.get(pc_file).astype(np.float32)
    # transform it according to the model 
    if config.dataset.train._base_['NAME'] == 'ShapeNet':
        # normalize it to fit the model on ShapeNet-55/34
        centroid = np.mean(pc_ndarray, axis=0)
        pc_ndarray = pc_ndarray - centroid
        m = np.max(np.sqrt(np.sum(pc_ndarray**2, axis=1)))
        pc_ndarray = pc_ndarray / m

    transform = Compose([{
        'callback': 'UpSamplePoints',
        'parameters': {
            'n_points': 2048
        },
        'objects': ['input']
    }, {
        'callback': 'ToTensor',
        'objects': ['input']
    }])
    
    pc_ndarray_normalized = transform({'input': pc_ndarray})
    # inference
    ret = model(pc_ndarray_normalized['input'].unsqueeze(0).to(args.device.lower()))
    dense_points = ret[-1].squeeze(0).detach().cpu().numpy()

    if config.dataset.train._base_['NAME'] == 'ShapeNet':
        # denormalize it to adapt for the original input
        dense_points = dense_points * m
        dense_points = dense_points + centroid

    if args.out_pc_root != '':
        target_path = os.path.join(args.out_pc_root, os.path.splitext(pc_path)[0])
        os.makedirs(target_path, exist_ok=True)

        np.save(os.path.join(target_path, 'fine.npy'), dense_points)
        # if args.save_vis_img:
        #     input_img = misc.get_ptcloud_img(pc_ndarray_normalized['input'].numpy())
        #     dense_img = misc.get_ptcloud_img(dense_points)
        #     cv2.imwrite(os.path.join(target_path, 'input.jpg'), input_img)
        #     cv2.imwrite(os.path.join(target_path, 'fine.jpg'), dense_img)
    
    return

# def add_points_count_to_images(pc, base_img_path):
#     # Loop through the specific range of images
#     for i in range(30):  # Since you want to go from 0000 to 0029
#         # Construct the full image path for the current event, adjusting the format as necessary
#         img_path = f"{base_img_path}event{i:04d}.png"
        
#         # Load the image
#         img = cv2.imread(img_path)
#         if img is None:
#             print(f"Failed to load image from {img_path}. Please check the file path and integrity.")
#             continue  # Skip to the next image if the current one couldn't be loaded

#         # Add text to the image
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         bottomLeftCornerOfText = (10, 50)
#         fontScale = 1
#         fontColor = (255, 255, 255)
#         lineType = 2

#         text = f'Points: {len(pc)}'  # You may want to adjust how pc is passed if it varies per image
#         cv2.putText(img, text, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

#         # Save the modified image
#         cv2.imwrite(img_path, img)



# def my_inference(model, args, config):

#     with torch.no_grad():
        
#         _, data_loader = builder.dataset_builder(args, config.dataset.test, test=True)

#         for idx, (feats, labels) in enumerate(data_loader):

#             if idx == args.n_imgs:
#                 break

#             partial = feats.cuda()
#             gt = labels.cuda()

#             ret = model(partial)

#             input_pc = partial.squeeze().detach().cpu().numpy()
#             output_pc = ret[-1].squeeze().detach().cpu().numpy()
#             gt_pc = gt.squeeze().detach().cpu().numpy()

#             # misc.better_img(input_pc, idx)
#             # misc.better_img(output_pc, idx, out=True)
#             # misc.better_img(gt_pc, idx, gt=True)

#             if args.experimental:
#                 misc.experimental_img(input_pc, output_pc, idx, args.save_img_path, config.dataset.test.partial.path)
                
#                 # Assuming `output_img_path` is where you save the generated image
#                 # and `output_pc` is the generated point cloud used for the image
#                 print(args.save_img_path)

#                 print("double image printed")
#             else:
#                 misc.triplet_img(input_pc, output_pc, gt_pc, idx, args.save_img_path, config.dataset.test.partial.path)
                
#                 # and `output_pc` is the generated point cloud used for the image
#                 print(args.save_img_path)
#                 print("triple image printed")


#     return


from utils import misc
# Other imports remain the same...

def add_points_count_to_images(pc, img_path):
    # Load the image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image from {img_path}. Please check the file path and integrity.")
        return

    # Add text to the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 50)
    fontScale = 100
    fontColor = (255, 255, 255)
    lineType = 2

    text = f'Points: {len(pc)}'
    cv2.putText(img, text, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

    # Save the modified image
    cv2.imwrite(img_path, img)
    print(f"Updated image saved to {img_path}")

def my_inference(model, args, config):
    with torch.no_grad():
        _, data_loader = builder.dataset_builder(args, config.dataset.test, test=True)

        for idx, (feats, labels) in enumerate(data_loader):
            if idx == args.n_imgs:
                break

            partial = feats.cuda()
            gt = labels.cuda()

            ret = model(partial)

            input_pc = partial.squeeze().detach().cpu().numpy()
            output_pc = ret[-1].squeeze().detach().cpu().numpy()  # This is your output point cloud
            gt_pc = gt.squeeze().detach().cpu().numpy()

            # Save or update the image path here as needed
            img_path = f"{args.save_img_path}event{idx:04d}.png"  # Make sure this path is correct
            print(img_path)
            # Assuming you save an image for each output point cloud here
            # Now, add the point count to each saved image
            add_points_count_to_images(output_pc, img_path)



def main():
    args = get_args()
    args.distributed = False

    # init config
    config = cfg_from_yaml_file(args.model_config)
    # build model
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.model_checkpoint)
    base_model.to(args.device.lower())
    base_model.eval()

    # if args.pc_root != '':
    #     pc_file_list = os.listdir(args.pc_root)
    #     for pc_file in pc_file_list:
    #         inference_single(base_model, pc_file, args, config, root=args.pc_root)
    # else:
    #     inference_single(base_model, args.pc, args, config)

    my_inference(base_model, args, config)
    # print(args.experimental)

    output_img_base_path = "test_image1s"  # Make sure this base path is correctly defined
    pc = np.random.rand(100, 3)  # Example point cloud data, replace with actual data relevant to your use case



if __name__ == '__main__':
    main()