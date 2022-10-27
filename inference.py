import torch
import argparse
from models import build_DABDETR, build_dab_deformable_detr, build_dab_deformable_detr_deformable_encoder_only
from util.utils import clean_state_dict
from util.slconfig import SLConfig
from util.visualizer import COCOVisualizer
from util import box_ops
import os
import time

def get_args_parser():
    parser = argparse.ArgumentParser('DAB-DETR', add_help=False)

    # about dn args
    parser.add_argument('--use_dn', action="store_true",
                        help="use denoising training.")
    parser.add_argument('--scalar', default=5, type=int,
                        help="number of dn groups")
    parser.add_argument('--label_noise_scale', default=0.2, type=float,
                        help="label noise ratio to flip")
    parser.add_argument('--box_noise_scale', default=0.4, type=float,
                        help="box noise scale to shift and scale")

    # about lr
    parser.add_argument('--lr', default=1e-4, type=float, 
                        help='learning rate')
    parser.add_argument('--lr_backbone', default=1e-5, type=float, 
                        help='learning rate for backbone')

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--drop_lr_now', action="store_true", help="load checkpoint and drop for 12epoch setting")
    parser.add_argument('--save_checkpoint_interval', default=10, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--modelname', '-m', type=str, required=True, choices=['dn_dab_detr', 'dn_dab_deformable_detr',
                                                                    'dn_dab_deformable_detr_deformable_encoder_only'])
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--pe_temperatureH', default=20, type=int, 
                        help="Temperature for height positional encoding.")
    parser.add_argument('--pe_temperatureW', default=20, type=int, 
                        help="Temperature for width positional encoding.")
    parser.add_argument('--batch_norm_type', default='FrozenBatchNorm2d', type=str, 
                        choices=['SyncBatchNorm', 'FrozenBatchNorm2d', 'BatchNorm2d'], help="batch norm type for backbone")

    # * Transformer
    parser.add_argument('--return_interm_layers', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--backbone_freeze_keywords', nargs="+", type=str, 
                        help='freeze some layers in backbone. for catdet5.')
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true', 
                        help="Using pre-norm in the Transformer blocks.")    
    parser.add_argument('--num_select', default=300, type=int, 
                        help='the number of predictions selected for evaluation')
    parser.add_argument('--transformer_activation', default='prelu', type=str)
    parser.add_argument('--num_patterns', default=0, type=int, 
                        help='number of pattern embeddings. See Anchor DETR for more details.')
    parser.add_argument('--random_refpoints_xy', action='store_true', 
                        help="Random init the x,y of anchor boxes and freeze them.")

    # for DAB-Deformable-DETR
    parser.add_argument('--two_stage', default=False, action='store_true', 
                        help="Using two stage variant for DAB-Deofrmable-DETR")
    parser.add_argument('--num_feature_levels', default=4, type=int, 
                        help='number of feature levels')
    parser.add_argument('--dec_n_points', default=4, type=int, 
                        help="number of deformable attention sampling points in decoder layers")
    parser.add_argument('--enc_n_points', default=4, type=int, 
                        help="number of deformable attention sampling points in encoder layers")


    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float, 
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--cls_loss_coef', default=1, type=float, 
                        help="loss coefficient for cls")
    parser.add_argument('--mask_loss_coef', default=1, type=float, 
                        help="loss coefficient for mask")
    parser.add_argument('--dice_loss_coef', default=1, type=float, 
                        help="loss coefficient for dice")
    parser.add_argument('--bbox_loss_coef', default=5, type=float, 
                        help="loss coefficient for bbox L1 loss")
    parser.add_argument('--giou_loss_coef', default=2, type=float, 
                        help="loss coefficient for bbox GIOU loss")
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--focal_alpha', type=float, default=0.25, 
                        help="alpha for focal loss")


    # dataset parameters
    parser.add_argument('--dataset_file', default='ford2')
    # parser.add_argument('--coco_path', type=str, required=True)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true', 
                        help="Using for debug only. It will fix the size of input images to the maximum.")


    # Traing utils
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--note', default='', help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+', 
                        help="A list of keywords to ignore when loading pretrained models.")
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help="eval only. w/o Training.")
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--debug', action='store_true', 
                        help="For debug only. It will perform only a few steps during trainig and val.")
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--save_results', action='store_true', 
                        help="For eval only. Save the outputs for all images.")
    parser.add_argument('--save_log', action='store_true', 
                        help="If save the training prints to the log file.")

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    return parser

def build_model_main(args):
    if args.modelname.lower() == 'dn_dab_detr':
        model, criterion, postprocessors = build_DABDETR(args)
    elif args.modelname.lower() == 'dn_dab_deformable_detr':
        model, criterion, postprocessors = build_dab_deformable_detr(args)
    elif args.modelname.lower() == 'dn_dab_deformable_detr_deformable_encoder_only':
        model, criterion, postprocessors = build_dab_deformable_detr_deformable_encoder_only(args)
    else:
        raise NotImplementedError

    return model, criterion, postprocessors


# model_config_path = "config.json" # change the path of the model config
# model_checkpoint_path = "checkpoint_optimized_44.7ap.pth" # change the path of the model checkpoint
parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()

model_checkpoint_path = "/mnt/workspace/users/ymjian/test/exps/fine_tune_test/image_based/dn_d_detr_from_scratch/checkpoint.pth"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model, criterion, postprocessors = build_model_main(args)

checkpoint = torch.load(model_checkpoint_path)
model.load_state_dict(checkpoint['model'])
model = model.cuda()
# model.eval()
model = model.to(device)
print("device model: ", device)

# n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print("parameters:", n_parameters)

# model.eval()
# id2name = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
id2name = {1: 'Car', 2: 'Truck', 3: 'Pedestrian', 4: 'Pedestrian_With_Object', 5: 'Trash Cans',
        6: 'Construction_Vehicle', 7: 'Bus', 8: 'Traffic_cones', 9: 'Wheeled_Pedestrian', 10: 'Bicycle',
        11: 'Cart', 12: 'Trailer', 13: 'Emergency_Vehicle', 14: 'Wheelchair', 15: 'Traffic_Barrel',
        16: 'Robots', 17: 'Tram or Subway', 18: 'Train', 19: 'Motorcycle', 20: 'Stroller',
        21: 'Animal', 22: 'Rickshaw', 23: 'Semi_truck', 24: 'Trolley', 25: 'Towed_Object', 26: 'Vehicle_Towing'}
vslzr = COCOVisualizer()

from PIL import Image
import datasets.transforms as T
# import torchvision.transforms as T

image_folder = "/mnt/workspace/users/ymjian/Shuttle_Deepen_Data_ImgBased_front_cam/time_measure"
files = os.listdir(image_folder)
number_of_images = 0
detected_objects = 0
total_t = 0

for file in files:

    img_path = os.path.join(image_folder, file)
    image = Image.open(img_path).convert("RGB")
    # image
    # transform images
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image, _ = transform(image, None)
    # image = image.unsqueeze(0)
    # print("image: ", image.shape)
    # print("device image: ", device)
    # image.to(device)
    image = image.cuda()
    # image.to(device)
    # print("image type: ", image.type())

    # predict images
    start = time.time()
    # print("device: ", image.device())
    output, _ = model(image[None],0)
    # print("type output: ", output.keys()) # type output:  dict_keys(['pred_logits', 'pred_boxes', 'aux_outputs'])
    # print("out ", output["pred_logits"].type())
    # print("out ", output["pred_boxes"].type())
    output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]))[0]
    # visualize outputs
    thershold = 0.3 # set a thershold

    scores = output['scores']
    labels = output['labels']
    boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
    select_mask = scores > thershold

    box_label = [id2name[int(item)] for item in labels[select_mask]]
    pred_dict = {
        'boxes': boxes[select_mask],
        'size': torch.Tensor([image.shape[1], image.shape[2]]),
        'box_label': box_label
    }

    total_t += time.time() - start
    number_of_images += 1

    # vslzr.visualize(image, pred_dict, savedir=None)
print("total time: ", total_t)
print("total image: ", number_of_images)
print("FPS: ", number_of_images/total_t)