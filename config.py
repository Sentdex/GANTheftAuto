"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/GameGAN_code.
Authors: Seung Wook Kim, Yuhao Zhou, Jonah Philion, Antonio Torralba, Sanja Fidler
"""
from optparse import OptionParser, OptionGroup

def init_parser():
    '''
    '''
    usage = """
        Usage of this tool.
        $ python main.py [--train]
    """
    parser = OptionParser(usage=usage)

    parser.add_option('--train', action='store_true', default=True)
    parser.add_option('--play', action='store_true', default=False)
    parser.add_option('--port', action='store', type=int, default=8888,
                      help='')
    parser.add_option('--skip', action='store', type=int, default=1,
                      help='')

    parser.add_option('--saved_model', type=str, default=None)
    parser.add_option('--saved_optim', type=str, default=None)
    parser.add_option('--data', action='store', type=str, default=None)
    parser.add_option('--warm_up', action='store', type=int, default=10)


    # training settings
    train_param = OptionGroup(parser, 'training hyperparameters')
    train_param.add_option('--encoder_chan_multiplier', action='store', type=int, default=1)
    train_param.add_option('--gpu', action='store', type=int, default=0,
            help='Whether to use GPU for training')
    train_param.add_option('--num_gpu', action='store', type=int, default=1,
                           help='Whether to use GPU for training')

    train_param.add_option('--optimizer', action='store', type='choice', default='adam',
            help='which optimizer used for training',
            choices=['adam', 'sgd', 'rmsprop'])
    train_param.add_option('--lr', action='store', type=float, default=1e-4,
            help='learning rate')
    train_param.add_option('--override_lr', type=float, default=-1)

    train_param.add_option('--bs', action='store', type=int, default=64,
            help='batch size')
    train_param.add_option('--nep', action='store', type=int, default=10000,
            help='max number of epochs')
    train_param.add_option('--memory_dim', action='store', type=int, default=512,
                         help='# of iters for training netD')
    train_param.add_option('--memory_h', action='store', type=int, default=441,
                           help='# of iters for training netD')

    gan_param = OptionGroup(parser, 'hyperparameters for training GANs')
    gan_param.add_option('--z', action='store', type=int, default=32,
                         help='# of iters for training netD')
    gan_param.add_option('--Diters', action='store', type=int, default=1,
            help='# of iters for training netD')
    gan_param.add_option('--nfilterG', action='store', type=int, default=64,
                         help='# of iters for training netD')

    gan_param.add_option('--nfilterD', action='store', type=int, default=16,
                         help='# of iters for training netD')

    gan_param.add_option('--hidden_dim', action='store', type=int, default=512,
                         help='# of iters for training netD')

    gan_param.add_option('--LAMBDA', action='store', type=float, default=1.0,
            help='Lambda value for W-GAN')
    gan_param.add_option('--LAMBDA_temporal', action='store', type=float, default=10.0,
                         help='Lambda value for W-GAN')

    gan_param.add_option('--lrD', action='store', type=float, default=1e-4,
            help='learning rate for D')
    gan_param.add_option('--lrG_temporal', action='store', type=float, default=1e-4,
            help='learning rate for G')
    gan_param.add_option('--lrG_graphic', action='store', type=float, default=1e-4,
                         help='learning rate for G')
    gan_param.add_option('--recon_loss_multiplier', action='store', type=float, default=0.05)
    gan_param.add_option('--override_recon_loss_multiplier', action='store', type=float, default=-1)
    gan_param.add_option('--rev_recon_loss_multiplier', action='store', type=float, default=0.05)
    gan_param.add_option('--gen_info_loss_multiplier', action='store', type=float, default=30.0)
    gan_param.add_option('--disc_info_loss_multiplier', action='store', type=float, default=10.0)
    gan_param.add_option('--disc_content_loss_multiplier', action='store', type=float, default=1.0)
    gan_param.add_option('--disc_single_frame_loss_multiplier', action='store', type=float, default=1.0)
    gan_param.add_option('--disc_action_loss_multiplier', action='store', type=float, default=1.0)
    gan_param.add_option('--disc_loss_multiplier', action='store', type=float, default=1.0)
    gan_param.add_option('--gen_content_loss_multiplier', action='store', type=float, default=1.0)
    gan_param.add_option('--gen_single_frame_loss_multiplier', action='store', type=float, default=1.0)
    gan_param.add_option('--gen_action_loss_multiplier', action='store', type=float, default=1.0)
    gan_param.add_option('--feature_loss_multiplier', action='store', type=float, default=10.0)
    gan_param.add_option('--reg_type',  type=str, default='real',
                         help='GP reg type')
    gan_param.add_option('--img_size', action='store', type=int, default=64,
                         help='# of iters for training netD')
    gan_param.add_option('--num_steps', action='store', type=int, default=15,
                         help='# of iters for training netD')
    gan_param.add_option('--override_num_steps', action='store', type=int, default=-1,
                         help='# of iters for training netD')
    gan_param.add_option('--num_components', action='store', type=int, default=1,
                         help='# of iters for training netD')
    gan_param.add_option('--do_memory', action='store_true', default=False)

    gan_param.add_option('--gan_type', type=str, default='standard')
    gan_param.add_option('--temperature', type=float, action='store', default=1.0)
    gan_param.add_option('--do_temporal', action='store_true', default=True)


    align_param = OptionGroup(parser, 'whether to enforce feature to be the same')
    align_param.add_option('--final_l1', action='store_true', default=False)
    align_param.add_option('--final_l2', action='store_true', default=True)
    align_param.add_option('--rev_final_l1', action='store_true', default=False)
    align_param.add_option('--rev_final_l2', action='store_true', default=True)

    align_param.add_option('--gan_loss', action='store_true', default=True)
    align_param.add_option('--cycle_loss', action='store_true', default=False)


    vis_param = OptionGroup(parser, 'logger parameters')
    vis_param.add_option('--log_dir', type=str, default='logs/test')

    train_param.add_option('--seed', action='store', type=int, default=10000,
                           help='random seed')
    train_param.add_option('--input_detach', action='store_true', default=True)

    train_param.add_option('--fix_graphic', action='store_true', default=False)
    train_param.add_option('--override_fix_graphic', action='store_true', default=False)
    train_param.add_option('--att_dim', action='store', type=int, default=512)
    train_param.add_option('--fixed_v_dim', action='store', type=int, default=512)

    train_param.add_option('--penultimate_tanh', action='store_true', default=True)
    train_param.add_option('--fine_mask', action='store_true', default=False)
    train_param.add_option('--disc_features', action='store_true', default=True)
    train_param.add_option('--disc_temporal_features', action='store_true', default=False)
    train_param.add_option('--recon_loss_min_multiplier', action='store', type=float, default=-1)
    train_param.add_option('--warmup_decay_epoch', action='store', type=int, default=10)
    train_param.add_option('--cycle_start_epoch', action='store', type=int, default=0)
    train_param.add_option('--min_warmup', action='store', type=int, default=0)
    train_param.add_option('--feature_l2', action='store_true', default=False)
    train_param.add_option('--mem_use_h', action='store_true', default=False)
    train_param.add_option('--config_temporal', type=int, default=18)
    train_param.add_option('--override_config_temporal', type=int, default=-1)

    train_param.add_option('--alpha_loss_multiplier', action='store', type=float, default=-1)

    train_param.add_option('--temporal_hierarchy', action='store_true', default=True)
    train_param.add_option('--temporal_hierarchy_epoch', action='store', type=int, default=0)
    train_param.add_option('--lr_decay_epoch', action='store', type=int, default=100000000)

    train_param.add_option('--nfilterD_temp', action='store', type=int, default=64)
    train_param.add_option('--spade_index', action='store', type=int, default=-1)
    train_param.add_option('--softmax_kernel', action='store_true', default=False)
    train_param.add_option('--sigmoid_maps', action='store_true', default=False)
    train_param.add_option('--no_in', action='store_true', default=True)
    train_param.add_option('--normalize_mean', action='store_true', default=True)
    train_param.add_option('--rev_multiply_map', action='store_true', default=False)
    train_param.add_option('--permute_color', action='store_true', default=False)
    train_param.add_option('--latent_only', action='store_true', default=False)

    train_param.add_option('--alpha_T', action='store', type=float, default=0.1)
    train_param.add_option('--save_epoch', action='store', type=int, default=10)
    train_param.add_option('--foreground_img', type=str, default='')
    train_param.add_option('--background_img', type=str, default='')
    train_param.add_option('--endstate_classifier_path', type=str, default='')

    train_param.add_option('--initial_screen', type=str, default='')
    train_param.add_option('--D_temp_mode', type=str, default='sn')
    train_param.add_option('--dec_saved_model', type=str, default='')

    train_param.add_option('--action_space', action='store', type=int, default=10)
    train_param.add_option('--num_channel', action='store', type=int, default=3)
    train_param.add_option('--num_task', action='store', type=int, default=8)

    train_param.add_option('--no_attention', action='store_true', default=True)

    train_param.add_option('--reset_lstm', action='store_true', default=False)
    train_param.add_option('--base_temperature', action='store', type=float, default=0.1)
    train_param.add_option('--state_dim_multiplier', action='store', type=int, default=1)


    train_param.add_option('--free_dynamic_component', action='store_true', default=False)
    train_param.add_option('--end_bias', action='store', type=float, default=0.0)

    train_param.add_option('--simple_blocks', action='store_true', default=False)
    train_param.add_option('--standard_gan_loss', action='store_true', default=False)
    train_param.add_option('--min_lr', action='store', type=float, default=0.00001)

    return parser
