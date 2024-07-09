import argparse

parser = argparse.ArgumentParser(description='Fusion')

# Seed
parser.add_argument('--seed', type=int, default=1, help='random seed')

# gpu device
parser.add_argument('--gpu_num', type=int, default=2,  help='gpu number')

parser.add_argument('--llava_device', type=str, default='cuda:0', help='gpu id for llava')

# parser.add_argument('--devices', type=int, default=[1, 2, 3, 4, 5], nargs='+', help='gpu id for fusion')
parser.add_argument('--devices', type=str, default='cuda:1', help='gpu id for fusion')

# Data Acquisition
parser.add_argument('--train_ir_path', type=str, default='/u2/yg/data/M3FD/M3FD_Detection/ir',
                    help='training dataset path')
parser.add_argument('--train_vis_path', type=str, default='/u2/yg/data/M3FD/M3FD_Detection/vi',
                    help='training dataset path')

parser.add_argument('--data_test_path', type=str, default='/data/yg/data/PQA-MEF/datasets/test_set',
                    help='MEF dataset path')

parser.add_argument('--train_ddp', type=bool, default=False)

parser.add_argument('--llava_path', type=str, default='/u2/llm_model/llava-v1.6-mistral-7b/',
                    help='path for pretrained llava model')
parser.add_argument("--load_8bit_llava", action='store_true', default=False)

parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')

# visdom and HTML visualization parameters
parser.add_argument('--display_freq', type=int, default=59456, help='frequency of showing training results on screen')
parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
# network saving and loading parameters

parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
parser.add_argument('--load_size', type=int, default=512, help='scale images to this size')
parser.add_argument('--crop_size', type=int, default=128, help='crop images')
parser.add_argument('--resize_size', type=int, default=256, help='crop images')

parser.add_argument('--model_save', type=str, default='./model_save', help='trained model to save')

parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')

# training parameters
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate for adam')
parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

parser.add_argument('--feature_num', type=int, default=32, help='number of features of MDA')

parser.add_argument('--loss_mse', type=float, default=0.5, help='weights for the mse in the loss function')
parser.add_argument('--loss_clip', type=float, default=0.5, help='weights for the clip in the loss function')
parser.add_argument('--crop', type=bool, default=False)
parser.add_argument('--resize', type=bool, default=True)

args = parser.parse_args(args=[])
