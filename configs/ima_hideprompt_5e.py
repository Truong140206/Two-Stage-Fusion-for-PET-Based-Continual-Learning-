import argparse


def get_args_parser(subparsers):
    subparsers.add_argument('--batch-size', default=24, type=int, help='Batch size per device')
    subparsers.add_argument('--epochs', default=5, type=int)

    # Model parameters
    subparsers.add_argument('--original_model', default='vit_base_patch16_224', type=str, metavar='OMODEL',
                            help='Name of original model to train')
    subparsers.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL',
                            help='Name of model to train')
    subparsers.add_argument('--input-size', default=224, type=int, help='images input size')
    subparsers.add_argument('--pretrained', default=True, help='Load pretrained model or not')
    subparsers.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
    subparsers.add_argument('--drop-path', type=float, default=0.0, metavar='PCT', help='Drop path rate (default: 0.)')

    # Optimizer parameters
    subparsers.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adam"')
    subparsers.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                            help='Optimizer Epsilon (default: 1e-8)')
    subparsers.add_argument('--opt-betas', default=(0.9, 0.999), type=float, nargs='+', metavar='BETA',
                            help='Optimizer Betas (default: (0.9, 0.999), use opt default)')
    subparsers.add_argument('--clip-grad', type=float, default=1.0, metavar='NORM',
                            help='Clip gradient norm (default: None, no clipping)')
    subparsers.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    subparsers.add_argument('--weight-decay', type=float, default=0.0, help='weight decay (default: 0.0)')
    subparsers.add_argument('--reinit_optimizer', type=bool, default=True, help='reinit optimizer (default: True)')

    # Learning rate schedule parameters
    subparsers.add_argument('--sched', default='step', type=str, metavar='SCHEDULER',
                            help='LR scheduler (default: "constant"')
    subparsers.add_argument('--lr', type=float, default=0.03, metavar='LR', help='learning rate (default: 0.03)')
    subparsers.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                            help='learning rate noise on/off epoch percentages')
    subparsers.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                            help='learning rate noise limit percent (default: 0.67)')
    subparsers.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                            help='learning rate noise std-dev (default: 1.0)')
    subparsers.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                            help='warmup learning rate (default: 1e-6)')
    subparsers.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                            help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    subparsers.add_argument('--decay-epochs', type=float, default=30, metavar='N', help='epoch interval to decay LR')
    subparsers.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                            help='epochs to warmup LR, if scheduler supports')
    subparsers.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                            help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    subparsers.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                            help='patience epochs for Plateau LR scheduler (default: 10')
    subparsers.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                            help='LR decay rate (default: 0.1)')
    subparsers.add_argument('--unscale_lr', type=bool, default=True, help='scaling lr by batch size (default: True)')

    # Augmentation parameters
    subparsers.add_argument('--color-jitter', type=float, default=None, metavar='PCT',
                            help='Color jitter factor (default: 0.3)')
    subparsers.add_argument('--aa', type=str, default=None, metavar='NAME',
                            help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    subparsers.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    subparsers.add_argument('--train-interpolation', type=str, default='bicubic',
                            help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # * Random Erase params
    subparsers.add_argument('--reprob', type=float, default=0.0, metavar='PCT',
                            help='Random erase prob (default: 0.25)')
    subparsers.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
    subparsers.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')

    # Data parameters
    subparsers.add_argument('--data-path', default='/local_datasets/', type=str, help='dataset path')
    subparsers.add_argument('--dataset', default='Split-Imagenet-A', type=str, help='dataset name')
    subparsers.add_argument('--shuffle', default=False, help='shuffle the data order')
    subparsers.add_argument('--output_dir', default='./output', help='path where to save, empty for no saving')
    subparsers.add_argument('--device', default='cuda', help='device to use for training / testing')
    subparsers.add_argument('--seed', default=42, type=int)
    subparsers.add_argument('--eval', action='store_true', help='Perform evaluation only')
    subparsers.add_argument('--num_workers', default=4, type=int)
    subparsers.add_argument('--pin-mem', action='store_true',
                            help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    subparsers.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                            help='')
    subparsers.set_defaults(pin_mem=True)

    # distributed training parameters
    subparsers.add_argument('--world_size', default=1, type=int,
                            help='number of distributed processes')
    subparsers.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Continual learning parameters
    subparsers.add_argument('--num_tasks', default=10, type=int, help='number of sequential tasks')
    subparsers.add_argument('--train_mask', default=True, type=bool, help='if using the class mask at training')
    subparsers.add_argument('--task_inc', default=False, type=bool, help='if doing task incremental')

    # G-Prompt parameters
    subparsers.add_argument('--use_g_prompt', default=False, type=bool, help='if using G-Prompt')
    subparsers.add_argument('--g_prompt_length', default=5, type=int, help='length of G-Prompt')
    subparsers.add_argument('--g_prompt_layer_idx', default=[], type=int, nargs="+",
                            help='the layer index of the G-Prompt')
    subparsers.add_argument('--use_prefix_tune_for_g_prompt', default=False, type=bool,
                            help='if using the prefix tune for G-Prompt')

    # E-Prompt parameters
    subparsers.add_argument('--use_e_prompt', default=True, type=bool, help='if using the E-Prompt')
    subparsers.add_argument('--e_prompt_layer_idx', default=[0, 1, 2, 3, 4], type=int, nargs="+",
                            help='the layer index of the E-Prompt')
    subparsers.add_argument('--use_prefix_tune_for_e_prompt', default=True, type=bool,
                            help='if using the prefix tune for E-Prompt')
    subparsers.add_argument('--larger_prompt_lr', action='store_true', help='if using larger prompt lr')

    # Use prompt pool in L2P to implement E-Prompt
    subparsers.add_argument('--prompt_pool', default=True, type=bool, )
    subparsers.add_argument('--size', default=10, type=int, )
    subparsers.add_argument('--length', default=20, type=int, )
    subparsers.add_argument('--top_k', default=1, type=int, )
    subparsers.add_argument('--initializer', default='uniform', type=str, )
    subparsers.add_argument('--prompt_key', default=False, type=bool, )
    subparsers.add_argument('--prompt_key_init', default='uniform', type=str)
    subparsers.add_argument('--use_prompt_mask', default=True, type=bool)
    subparsers.add_argument('--mask_first_epoch', default=False, type=bool)
    subparsers.add_argument('--shared_prompt_pool', default=True, type=bool)
    subparsers.add_argument('--shared_prompt_key', default=False, type=bool)
    subparsers.add_argument('--batchwise_prompt', default=False, type=bool)
    subparsers.add_argument('--embedding_key', default='cls', type=str)
    subparsers.add_argument('--predefined_key', default='', type=str)
    subparsers.add_argument('--pull_constraint', default=True)
    subparsers.add_argument('--pull_constraint_coeff', default=1.0, type=float)
    subparsers.add_argument('--same_key_value', default=False, type=bool)

    # ViT parameters
    subparsers.add_argument('--global_pool', default='token', choices=['token', 'avg'], type=str,
                            help='type of global pooling for final sequence')
    subparsers.add_argument('--head_type', default='token', choices=['token', 'gap', 'prompt', 'token+prompt'],
                            type=str, help='input type of classification head')
    subparsers.add_argument('--freeze', default=['blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed'], nargs='*',
                            type=list, help='freeze part in backbone model')


    # CA parameters
    subparsers.add_argument('--crct_epochs', default=30, type=int)
    subparsers.add_argument('--crct_use_all_samples', action='store_true', help='use every balanced synthetic replay sample during classifier correction')
    subparsers.add_argument('--crct_balanced_batches', action='store_true', help='interleave classes so each classifier-correction batch is approximately balanced')
    subparsers.add_argument('--train_inference_task_only', action='store_true')
    subparsers.add_argument('--original_model_mlp_structure', default=[2], type=int, nargs='*')
    subparsers.add_argument('--ca_lr', default=0.005, type=float)
    subparsers.add_argument('--weight_decay', default=5e-4, type=float)
    subparsers.add_argument('--milestones', default=[10], type=int)
    subparsers.add_argument('--trained_original_model', default='', type=str)
    subparsers.add_argument('--prompt_momentum', default=0, type=float)
    subparsers.add_argument('--not_train_ca', action='store_true')
    subparsers.add_argument('--ca_storage_efficient_method', default='multi-centroid', choices=['covariance', 'multi-centroid', 'variance'], type=str)
    subparsers.add_argument('--n_centroids', default=2, type=int)
    subparsers.add_argument('--cfs_sampling', action='store_true', help='use CFS-selected Gaussian samples for CRCT')
    subparsers.add_argument('--cfs_epochs', default=50, type=int)
    subparsers.add_argument('--cfs_lr', default=0.01, type=float)
    subparsers.add_argument('--cfs_momentum', default=0.9, type=float)
    subparsers.add_argument('--cfs_hidden_dim', default=512, type=int)
    subparsers.add_argument('--cfs_batch_size', default=256, type=int)
    subparsers.add_argument('--cfs_train_max_samples', default=1024, type=int)
    subparsers.add_argument('--cfs_candidate_multiplier', default=3, type=int)
    subparsers.add_argument('--cfs_tau', default=1.0, type=float)
    subparsers.add_argument('--cfs_init_strategy', default='random', choices=['random', 'mean'], type=str, help='initial candidate selection strategy for CFS diversity sampling')
    subparsers.add_argument('--cfs_boundary_replay', action='store_true', help='mix diverse CFS samples with samples near the current classifier boundary')
    subparsers.add_argument('--cfs_boundary_ratio', default=0.5, type=float)
    subparsers.add_argument('--cfs_boundary_multiplier', default=3, type=int)
    subparsers.add_argument('--cfs_boundary_density_quantile', default=0.9, type=float)
    subparsers.add_argument('--cfs_boundary_target_side', action='store_true', help='prefer in-distribution boundary samples still classified on the target side')
    subparsers.add_argument('--cfs_distribution_filter', action='store_true', help='filter CFS Gaussian candidates by class feature distribution before diversity selection')
    subparsers.add_argument('--cfs_filter_multiplier', default=3, type=int)
    subparsers.add_argument('--cfs_filter_cosine_weight', default=0.0, type=float)
    subparsers.add_argument('--cfs_paper_style', action='store_true', help='use paper-style multi-step CFS selection')
    subparsers.add_argument('--cfs_selection_ratio', default=0.5, type=float)
    subparsers.add_argument('--cfs_selection_steps', default=5, type=int)
    subparsers.add_argument('--cfs_step_candidates', default=0, type=int)
    subparsers.add_argument('--semantic_distill', action='store_true', help='use semantic-aware weighting for relation distillation')
    subparsers.add_argument('--semantic_backend', default='hash', choices=['hash', 'clip', 'auto'], type=str)
    subparsers.add_argument('--semantic_clip_model', default='ViT-B-16', type=str)
    subparsers.add_argument('--semantic_clip_pretrained', default='openai', type=str)
    subparsers.add_argument('--semantic_clip_templates', default='a photo of a {}.', type=str)
    subparsers.add_argument('--semantic_dim', default=512, type=int)
    subparsers.add_argument('--semantic_class_name_file', default='', type=str)
    subparsers.add_argument('--semantic_alpha', default=0.05, type=float)
    subparsers.add_argument('--semantic_floor', default=0.2, type=float)
    subparsers.add_argument('--semantic_sharpness', default=1.0, type=float)
    # Misc parameters
    subparsers.add_argument('--semantic_top_k', default=5, type=int)
    subparsers.add_argument('--semantic_mode', default='adaptive_gate', choices=['adaptive_gate', 'topk_mix', 'weight'], type=str)
    subparsers.add_argument('--semantic_projection', action='store_true', help='use semantic projection to generate CRCT features from related classes')
    subparsers.add_argument('--semantic_projection_ratio', default=0.25, type=float)
    subparsers.add_argument('--semantic_projection_top_k', default=5, type=int)
    subparsers.add_argument('--semantic_projection_strength', default=1.0, type=float)
    subparsers.add_argument('--semantic_projection_mode', default='mean_shift', choices=['mean_shift', 'paper', 'covariance_transfer'], type=str)
    subparsers.add_argument('--semantic_projection_alpha', default=0.1, type=float)
    subparsers.add_argument('--semantic_cov_transfer_min_scale', default=0.5, type=float)
    subparsers.add_argument('--semantic_cov_transfer_max_scale', default=2.0, type=float)
    subparsers.add_argument('--semantic_projection_preserve_norm', action='store_true')
    subparsers.add_argument('--semantic_projection_filter', action='store_true', help='filter semantic projected CRCT features by target feature distribution')
    subparsers.add_argument('--semantic_projection_filter_multiplier', default=3, type=int)
    subparsers.add_argument('--semantic_projection_filter_cosine_weight', default=0.1, type=float)
    subparsers.add_argument('--semantic_feature_adapter', action='store_true', help='align semantic embeddings to HRM feature means before semantic projection')
    subparsers.add_argument('--semantic_adapter_dim', default=512, type=int)
    subparsers.add_argument('--semantic_adapter_ridge', default=0.01, type=float)
    subparsers.add_argument('--semantic_adapter_blend', default=1.0, type=float)
    subparsers.add_argument('--semantic_adapter_min_classes', default=5, type=int)
    subparsers.add_argument('--print_freq', type=int, default=10, help='The frequency of printing')
