import argparse

parser = argparse.ArgumentParser(description='Fusion')

# Seed
parser.add_argument('--seed', type=int, default=1, help='random seed')

# gpu device
parser.add_argument('--gpu_num', type=int, default=6,  help='gpu number')

parser.add_argument('--llava_device', type=str, default='cuda:2', help='gpu id for llava')
parser.add_argument('--devices', type=str, default='cuda:1', help='gpu id for fusion')
# parser.add_argument('--devices', type=str, default=[0, 1], nargs='+', help='gpu id for fusion')
parser.add_argument('--local_rank', type=int, default=1, help='gpu id for ddp')
parser.add_argument('--train_ddp', type=bool, default=False)

# Data Acquisition
parser.add_argument('--train_ir_path', type=str, default='/u2/yg/data/M3FD/M3FD_Detection/ir',
                    help='training dataset path')
parser.add_argument('--train_vis_path', type=str, default='/u2/yg/data/M3FD/M3FD_Detection/vi',
                    help='training dataset path')

parser.add_argument('--llava_path', type=str, default='/u2/llm_model/llava-v1.6-mistral-7b/',
                    help='path for pretrained llava model')
parser.add_argument("--load_8bit_llava", action='store_true', default=False)
parser.add_argument('--text_save', type=str, default='./generated_text/M3FD_Detection',
                    help='folder to save the pre-generated description text')

parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')

# network saving and loading parameters
parser.add_argument('--model_save', type=str, default='./model_save', help='trained model to save')
parser.add_argument('--crop', type=bool, default=False)
parser.add_argument('--resize', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
parser.add_argument('--load_size', type=int, default=512, help='scale images to this size')
parser.add_argument('--crop_size', type=int, default=128, help='crop images')
parser.add_argument('--resize_size', type=int, default=320, help='crop images')
parser.add_argument('--feature_num', type=int, default=24, help='number of features of MDA')

# training parameters
parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate for adam')
parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

# loss function
parser.add_argument('--loss_mse', type=float, default=0.2, help='weights for the mse in the loss function')
parser.add_argument('--loss_clip', type=float, default=0.2, help='weights for the clip in the loss function')


args = parser.parse_args(args=[])
