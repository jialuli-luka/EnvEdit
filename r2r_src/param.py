import argparse
import os
import torch


class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="")

        # General
        self.parser.add_argument('--iters', type=int, default=100000)
        self.parser.add_argument('--name', type=str, default='default')
        self.parser.add_argument('--train', type=str, default='speaker')

        # Data preparation
        self.parser.add_argument('--maxInput', type=int, default=80, help="max input instruction")
        self.parser.add_argument('--maxDecode', type=int, default=120, help="max input instruction")
        self.parser.add_argument('--maxAction', type=int, default=20, help='Max Action sequence')
        self.parser.add_argument('--batchSize', type=int, default=64)
        self.parser.add_argument('--ignoreid', type=int, default=-100)
        self.parser.add_argument('--feature_size', type=int, default=512)
        self.parser.add_argument("--loadOptim",action="store_const", default=False, const=True)

        # Load the model from
        self.parser.add_argument("--speaker", default=None)
        self.parser.add_argument("--listener", default=None)
        self.parser.add_argument("--load", type=str, default=None)

        # More Paths from
        self.parser.add_argument("--aug", default=None)

        # Listener Model Config
        self.parser.add_argument("--zeroInit", dest='zero_init', action='store_const', default=False, const=True)
        self.parser.add_argument("--mlWeight", dest='ml_weight', type=float, default=0.05)
        self.parser.add_argument("--teacherWeight", dest='teacher_weight', type=float, default=1.)
        self.parser.add_argument("--accumulateGrad", dest='accumulate_grad', action='store_const', default=False, const=True)
        self.parser.add_argument("--features", type=str, default='imagenet')

        # Env Dropout Param
        self.parser.add_argument('--featdropout', type=float, default=0.3)

        # SSL configuration
        self.parser.add_argument("--selfTrain", dest='self_train', action='store_const', default=False, const=True)

        # Submision configuration
        self.parser.add_argument("--candidates", type=int, default=1)
        self.parser.add_argument("--paramSearch", dest='param_search', action='store_const', default=False, const=True)
        self.parser.add_argument("--submit", action='store_const', default=False, const=True)
        self.parser.add_argument("--beam", action="store_const", default=False, const=True)
        self.parser.add_argument("--alpha", type=float, default=0.5)

        # Training Configurations
        self.parser.add_argument('--optim', type=str, default='rms')    # rms, adam
        self.parser.add_argument('--lr', type=float, default=0.0001, help="The learning rate")
        self.parser.add_argument('--decay', dest='weight_decay', type=float, default=0.)
        self.parser.add_argument('--dropout', type=float, default=0.5)
        self.parser.add_argument('--feedback', type=str, default='sample',
                            help='How to choose next position, one of ``teacher``, ``sample`` and ``argmax``')
        self.parser.add_argument('--teacher', type=str, default='final',
                            help="How to get supervision. one of ``next`` and ``final`` ")
        self.parser.add_argument('--epsilon', type=float, default=0.1)

        # Model hyper params:
        self.parser.add_argument('--rnnDim', dest="rnn_dim", type=int, default=512)
        self.parser.add_argument('--wemb', type=int, default=256)
        self.parser.add_argument('--aemb', type=int, default=64)
        self.parser.add_argument('--proj', type=int, default=512)
        self.parser.add_argument("--fast", dest="fast_train", action="store_const", default=False, const=True)
        self.parser.add_argument("--valid", action="store_const", default=False, const=True)
        self.parser.add_argument("--candidate", dest="candidate_mask",
                                 action="store_const", default=False, const=True)

        self.parser.add_argument("--bidir", type=bool, default=True)    # This is not full option
        self.parser.add_argument("--encode", type=str, default="word")  # sub, word, sub_ctx
        self.parser.add_argument("--subout", dest="sub_out", type=str, default="tanh")  # tanh, max
        self.parser.add_argument("--attn", type=str, default="soft")    # soft, mono, shift, dis_shift

        self.parser.add_argument("--angleFeatSize", dest="angle_feat_size", type=int, default=128)

        # A2C
        self.parser.add_argument("--gamma", default=0.9, type=float)
        self.parser.add_argument("--normalize", dest="normalize_loss", default="total", type=str, help='batch or total')

        # EnvEdit
        self.parser.add_argument("--aug_env", nargs='*', default=None)
        self.parser.add_argument("--train_env", default='aug')
        self.parser.add_argument("--valid_env", default='aug')
        self.parser.add_argument("--log_every", type=int, default=100)
        self.parser.add_argument("--aug_method", default='alternative') # alternative, together, specify
        self.parser.add_argument("--speaker_lr", default=1e-4, type=float)
        self.parser.add_argument("--speaker_aug", default=None)
        self.parser.add_argument("--feature_extract", default=None)
        self.parser.add_argument("--seed", default=42, type=int)
        self.parser.add_argument("--division", default='equal')
        self.parser.add_argument("--style_embedding", default=None)
        self.parser.add_argument("--style", default='first')
        self.parser.add_argument("--style_embedding_aug", default=None)
        self.parser.add_argument("--bt_weight", default=0.6, type=float)
        self.parser.add_argument("--eval_environment", action='store_const', default=False, const=True)

        # RxR dataset
        self.parser.add_argument("--dataset", default='R2R')
        self.parser.add_argument("--language", default='multi')
        self.parser.add_argument("--bert_dim", default=768, type=int)
        self.parser.add_argument("--bert_decay", default=1e-2, type=float)
        self.parser.add_argument("--bert_lr", default=4e-5, type=float)
        self.parser.add_argument("--ndtw", dest="ndtw", action="store_const", default=False, const=True)

        # Ensemble
        self.parser.add_argument("--ensemble", nargs='*', default=None)

        self.args = self.parser.parse_args()

        if self.args.optim == 'rms':
            print("Optimizer: Using RMSProp")
            self.args.optimizer = torch.optim.RMSprop
        elif self.args.optim == 'adam':
            print("Optimizer: Using Adam")
            self.args.optimizer = torch.optim.Adam
        elif self.args.optim == 'sgd':
            print("Optimizer: sgd")
            self.args.optimizer = torch.optim.SGD
        else:
            assert False

param = Param()
args = param.args
args.TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
args.TRAINVAL_VOCAB = 'tasks/R2R/data/trainval_vocab.txt'

args.IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
args.CANDIDATE_FEATURES = 'img_features/ResNet-152-candidate.tsv'
args.features_fast = 'img_features/ResNet-152-imagenet-fast.tsv'
args.log_dir = 'snap/%s' % args.name

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
DEBUG_FILE = open(os.path.join('snap', args.name, "debug.log"), 'w')

