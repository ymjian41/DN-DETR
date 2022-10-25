from operator import mod
from jax import checkpoint
import torch

def prepare_no_class(model):
    input = 'model_pt_files/' + model + '.pth'
    checkpoint = torch.load(input, map_location='cpu')
    del checkpoint["model"]["class_embed.0.weight"]
    del checkpoint["model"]["class_embed.0.bias"]

    del checkpoint["model"]["class_embed.1.weight"]
    del checkpoint["model"]["class_embed.1.bias"]

    del checkpoint["model"]["class_embed.2.weight"]
    del checkpoint["model"]["class_embed.2.bias"]

    del checkpoint["model"]["class_embed.3.weight"]
    del checkpoint["model"]["class_embed.3.bias"]

    del checkpoint["model"]["class_embed.4.weight"]
    del checkpoint["model"]["class_embed.4.bias"]

    del checkpoint["model"]["class_embed.5.weight"]
    del checkpoint["model"]["class_embed.5.bias"]

    del checkpoint["model"]["label_enc.weight"]
    
    output = 'model_pt_files/' + model + '_no-class-head.pth'
    torch.save(checkpoint, output)

# if args.dataset_file == 'ford': num_classes = 15
def prepare_resized_class(model, num_classes):
    input = 'model_pt_files/' + model + '.pth'
    checkpoint = torch.load(input, map_location='cpu')

    checkpoint["model"]["class_embed.0.weight"].resize_(num_classes, 256)
    checkpoint["model"]["class_embed.0.bias"].resize_(num_classes)

    checkpoint["model"]["class_embed.1.weight"].resize_(num_classes, 256)
    checkpoint["model"]["class_embed.1.bias"].resize_(num_classes)

    checkpoint["model"]["class_embed.2.weight"].resize_(num_classes, 256)
    checkpoint["model"]["class_embed.2.bias"].resize_(num_classes)

    checkpoint["model"]["class_embed.3.weight"].resize_(num_classes, 256)
    checkpoint["model"]["class_embed.3.bias"].resize_(num_classes)

    checkpoint["model"]["class_embed.4.weight"].resize_(num_classes, 256)
    checkpoint["model"]["class_embed.4.bias"].resize_(num_classes)

    checkpoint["model"]["class_embed.5.weight"].resize_(num_classes, 256)
    checkpoint["model"]["class_embed.5.bias"].resize_(num_classes)

    # adjust the number of query to amount we need
    checkpoint["model"]["query_embed.weight"].resize_(50, 512)
    
    output = 'model_pt_files/' + model + '_resized-class-head.pth'
    torch.save(checkpoint, output)

def viewModel(model):
    pthfile = 'model_pt_files/' + model + '.pth'
    checkpoint = torch.load(pthfile, map_location='cpu')

    # model, optimizer, lr_scheduler, epoch, args
    # for k in checkpoint.keys():
    #     print(k)

    for k, v in checkpoint["model"].items():
        print(k) # layer name
        print(v.shape) # layer shape

    # check shape
    # print(checkpoint["model"]["transformer.decoder.layers.5.self_attn.out_proj.weight"].shape)

def model_num_params(path):
    checkpoint = torch.load(path)
    model = checkpoint["model"]
    # print()
    # model.load_state_dict(model)
    print(model.keys())

    print('type: ', type(model))

# viewModel('r50_deformable_detr-checkpoint')
# viewModel('r50_deformable_detr-checkpoint_no-class-head')

# viewModel('exps/R50_fully_vis/checkpoint')

# viewModel('dn_deformable_detr_no-class-head')
# prepare_no_class('checkpoint0049')
model_num_params('/mnt/workspace/users/ymjian/test/exps/fine_tune_test/image_based/dn_d_detr_from_scratch/checkpoint.pth')

print('done')