import argparse


def set_base_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="../data/sample", help='数据文件地址')
    parser.add_argument('--model_path', type=str, default="../resources", help='模型位置')
    parser.add_argument('--img_size', type=int, default=256, help="height")
    parser.add_argument('--network', type=str, default='MoEA', help='网络解构分类')
    parser.add_argument('--model', type=str, default='', help='网络下的子模型')
    return parser


def set_train_args():
    # Adam训练完记得使用SGD，lr0.01再跑一次
    parser = set_base_args()
    parser.add_argument('--epoch_num', type=int, default=64, help='epoch num')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='数据读取线程数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--l2_weight', type=float, default=1e-4, help='l2正则')
    parser.add_argument('--opt', type=str, default='Adam', help='优化器')
    parser.add_argument('--neg_scale', type=int, default=4, help='负样本比例')
    parser.add_argument("--local_rank", type=int, default=-1, help='提供单机多卡，DDP支持')
    parser.add_argument('--depth_map_size', type=int, default=64, help='深度图大小')
    parser.add_argument('--mask_map_size', type=int, default=32, help='标签图大小，应对少爷问题')

    args = parser.parse_args()
    if args.model == '':
        args.model = args.network

    return args


def set_test_args():
    parser = set_base_args()
    parser.add_argument('--threshold', type=float, default=-1, help='threshold')
    parser.add_argument("--infer_type", type=str, default="sdepth", help="推理类型，仅仅在生产onnx时使用，可以选择depth、score、None")
    parser.add_argument("--min_color_check", type=int, default=4, help="如果6张图中，颜色验证通过的图片数量小于min_color_check，就认为样本不通过")
    args = parser.parse_args()
    if args.model == '':
        args.model = args.network
    return args
