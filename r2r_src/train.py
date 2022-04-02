
import torch

import os
import time
import json
import numpy as np
from collections import defaultdict
from speaker import Speaker

from utils import read_vocab,write_vocab,build_vocab,Tokenizer,padding_idx,timeSince, read_img_features
import utils
from env import R2RBatch
from agent import Seq2SeqAgent
from agent_RxR import Seq2SeqAgentRxR
from eval import Evaluation
from param import args

import random

import warnings
warnings.filterwarnings("ignore")

from tensorboardX import SummaryWriter

from transformers import BertTokenizer


log_dir = 'snap/%s' % args.name
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
TRAINVAL_VOCAB = 'tasks/R2R/data/trainval_vocab.txt'

IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
PLACE365_FEATURES = 'img_features/ResNet-152-places365.tsv'

if args.features == 'imagenet':
    features = IMAGENET_FEATURES

if args.fast_train:
    name, ext = os.path.splitext(features)
    features = name + "-fast" + ext

feedback_method = args.feedback # teacher or sample

print(args)


def train_speaker(train_env, tok, n_iters, log_every=500, val_envs={}):
    writer = SummaryWriter(logdir=log_dir)
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)
    speaker = Speaker(train_env, listner, tok)

    if args.fast_train:
        log_every = 40

    best_bleu = defaultdict(lambda: 0)
    best_loss = defaultdict(lambda: 1232)
    for idx in range(0, n_iters, log_every):
        interval = min(log_every, n_iters - idx)

        # Train for log_every interval
        speaker.env = train_env
        speaker.train(interval)   # Train interval iters

        print()
        print("Iter: %d" % idx)

        # Evaluation
        for env_name, (env, evaluator) in val_envs.items():
            if 'train' in env_name: # Ignore the large training set for the efficiency
                continue

            print("............ Evaluating %s ............." % env_name)
            speaker.env = env
            path2inst, loss, word_accu, sent_accu = speaker.valid()
            path_id = next(iter(path2inst.keys()))
            print("Inference: ", tok.decode_sentence(path2inst[path_id]))
            print("GT: ", evaluator.gt[str(path_id)]['instructions'])
            bleu_score, precisions = evaluator.bleu_score(path2inst)

            # Tensorboard log
            writer.add_scalar("bleu/%s" % (env_name), bleu_score, idx)
            writer.add_scalar("loss/%s" % (env_name), loss, idx)
            writer.add_scalar("word_accu/%s" % (env_name), word_accu, idx)
            writer.add_scalar("sent_accu/%s" % (env_name), sent_accu, idx)
            writer.add_scalar("bleu4/%s" % (env_name), precisions[3], idx)

            # Save the model according to the bleu score
            if bleu_score > best_bleu[env_name]:
                best_bleu[env_name] = bleu_score
                print('Save the model with %s BEST env bleu %0.4f' % (env_name, bleu_score))
                speaker.save(idx, os.path.join(log_dir, 'state_dict', 'best_%s_bleu' % env_name))

            if loss < best_loss[env_name]:
                best_loss[env_name] = loss
                print('Save the model with %s BEST env loss %0.4f' % (env_name, loss))
                speaker.save(idx, os.path.join(log_dir, 'state_dict', 'best_%s_loss' % env_name))

            # Screen print out
            print("Bleu 1: %0.4f Bleu 2: %0.4f, Bleu 3 :%0.4f,  Bleu 4: %0.4f" % tuple(precisions))


def train(train_env, tok, n_iters, log_every=500, val_envs={}, aug_env=None):
    writer = SummaryWriter(logdir=log_dir)
    if args.dataset == 'R2R':
        listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)
    elif args.dataset == 'RxR':
        listner = Seq2SeqAgentRxR(train_env, "", tok, args.maxAction)

    speaker = None
    if args.self_train:
        speaker = Speaker(train_env, listner, tok)
        if args.speaker is not None:
            print("Load the speaker at iter %d from %s." % (speaker.load(args.speaker), args.speaker))


    start_iter = 0
    if args.load is not None:
        print("LOAD THE listener from %s" % args.load)
        start_iter = listner.load(os.path.join(args.load))

    start = time.time()

    best_val = {'val_seen': {"accu": 0., "state":"", 'update':False},
                'val_unseen': {"accu": 0., "state":"", 'update':False}}

    best_val_ndtw = {'val_seen': {"accu": 0., "state": "", 'update': False},
                     'val_unseen': {"accu": 0., "state": "", 'update': False}}

    if args.fast_train:
        log_every = 40

    print("Start training")
    for idx in range(start_iter, start_iter+n_iters, log_every):

        listner.logs = defaultdict(list)
        interval = min(log_every, n_iters-idx)
        iter = idx + interval

        # Train for log_every interval
        if aug_env is None:     # The default training process
            if args.accumulate_grad:
                listner.env = train_env
                for _ in range(interval):
                    listner.zero_grad()
                    listner.accumulate_gradient(feedback_method)
                    listner.accumulate_gradient(feedback_method)
                    listner.optim_step()
            else:
                listner.env = train_env
                listner.train(interval, feedback=feedback_method)  # Train interval iters
        else:
            if args.accumulate_grad:
                for _ in range(interval // 2):
                    listner.zero_grad()
                    listner.env = train_env

                    # Train with GT data
                    args.ml_weight = 0.2
                    listner.accumulate_gradient(feedback_method)
                    listner.env = aug_env

                    # Train with Back Translation
                    args.ml_weight = 0.6        # Sem-Configuration
                    listner.accumulate_gradient(feedback_method, speaker=speaker)
                    listner.optim_step()
            else:
                for _ in range(interval // 2):
                    # Train with GT data
                    listner.env = train_env
                    args.ml_weight = 0.2
                    listner.train(1, feedback=feedback_method)

                    # Train with Back Translation
                    listner.env = aug_env
                    args.ml_weight = 0.6
                    listner.train(1, feedback=feedback_method, speaker=speaker)

        # Log the training stats to tensorboard
        total = max(sum(listner.logs['total']), 1)
        length = max(len(listner.logs['critic_loss']), 1)
        critic_loss = sum(listner.logs['critic_loss']) / total #/ length / args.batchSize
        entropy = sum(listner.logs['entropy']) / total #/ length / args.batchSize
        predict_loss = sum(listner.logs['us_loss']) / max(len(listner.logs['us_loss']), 1)
        writer.add_scalar("loss/critic", critic_loss, idx)
        writer.add_scalar("policy_entropy", entropy, idx)
        writer.add_scalar("loss/unsupervised", predict_loss, idx)
        writer.add_scalar("total_actions", total, idx)
        writer.add_scalar("max_length", length, idx)
        print("total_actions", total)
        print("max_length", length)

        if (iter - start_iter) % log_every == 0:
            # Run validation
            loss_str = ""
            for env_name, (env, evaluator) in val_envs.items():
                listner.env = env

                # Get validation loss under the same conditions as training
                iters = None if args.fast_train or env_name != 'train' else 20     # 20 * 64 = 1280

                # Get validation distance from goal under test evaluation conditions
                listner.test(use_dropout=False, feedback='argmax', iters=iters)
                result = listner.get_results()
                score_summary, _ = evaluator.score(result)
                loss_str += ", %s " % env_name
                for metric,val in score_summary.items():
                    if metric in ['success_rate']:
                        writer.add_scalar("accuracy/%s" % env_name, val, idx)
                        if env_name in best_val:
                            if val > best_val[env_name]['accu']:
                                best_val[env_name]['accu'] = val
                                best_val[env_name]['update'] = True
                    if metric in ['ndtw']:
                        writer.add_scalar("ndtw/%s" % env_name, val, idx)
                        if env_name in best_val_ndtw:
                            if val > best_val_ndtw[env_name]['accu']:
                                best_val_ndtw[env_name]['accu'] = val
                                best_val_ndtw[env_name]['update'] = True
                    loss_str += ', %s: %.3f' % (metric, val)

            for env_name in best_val:
                if best_val[env_name]['update']:
                    best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                    best_val[env_name]['update'] = False
                    listner.save(idx, os.path.join("snap", args.name, "state_dict", "best_%s" % (env_name)))

            for env_name in best_val_ndtw:
                if best_val_ndtw[env_name]['update']:
                    best_val_ndtw[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                    best_val_ndtw[env_name]['update'] = False
                    listner.save(idx, os.path.join("snap", args.name, "state_dict", "best_%s_ndtw" % (env_name)))

            print(('%s (%d %d%%) %s' % (timeSince(start, float(iter)/n_iters),
                                                 iter, float(iter)/n_iters*100, loss_str)))

        if iter % 5000 == 0:
            print("BEST RESULT TILL NOW")
            for env_name in best_val:
                print(env_name, best_val[env_name]['state'])
            for env_name in best_val_ndtw:
                print(env_name, best_val_ndtw[env_name]['state'])

        if iter % 50000 == 0:
            listner.save(idx, os.path.join("snap", args.name, "state_dict", "Iter_%06d" % (iter)))

    listner.save(idx, os.path.join("snap", args.name, "state_dict", "LAST_iter%d" % (idx)))

    return best_val['val_unseen']['accu']


def valid(train_env, tok, val_envs={}):
    if args.dataset == 'R2R':
        agent = Seq2SeqAgent(train_env, "", tok, args.maxAction)
    elif args.dataset == 'RxR':
        agent = Seq2SeqAgentRxR(train_env, "", tok, args.maxAction)

    print("Loaded the listener model at iter %d from %s" % (agent.load(args.load), args.load))

    for env_name, (env, evaluator) in val_envs.items():
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        agent.test(use_dropout=False, feedback='argmax', iters=iters)
        result = agent.get_results()

        if env_name != 'test':
            score_summary, _ = evaluator.score(result)
            loss_str = "Env name: %s" % env_name
            for metric,val in score_summary.items():
                loss_str += ', %s: %.4f' % (metric, val)
            print(loss_str)

        sr = []
        ndtw = []
        sdtw = []
        spl = []
        weights = []

        if args.eval_environment and env_name == 'val_unseen':
            score_summary, scores = evaluator.score_environment(result)
            loss_str = "Env name: %s" % env_name
            for scan, value in score_summary.items():
                loss_str = loss_str + '/n' + scan
                for metric, val in value.items():
                    if metric == 'success_rate':
                        sr.append(val)
                    elif metric == 'ndtw':
                        ndtw.append(val)
                    elif metric == 'sdtw':
                        sdtw.append(val)
                    elif metric == 'spl':
                        spl.append(val)
                    loss_str += ', %s: %.4f' % (metric, val)
                weights.append(len(scores[scan]['nav_errors']))
            print(loss_str)
            mean_sr = np.average(np.array(sr), weights=weights)
            std_sr = np.average((np.array(sr) - mean_sr) ** 2, weights=weights)
            mean_ndtw = np.average(np.array(ndtw), weights=weights)
            std_ndtw = np.average((np.array(ndtw) - mean_ndtw) ** 2, weights=weights)
            mean_sdtw = np.average(np.array(sdtw), weights=weights)
            std_sdtw = np.average((np.array(sdtw) - mean_sdtw) ** 2, weights=weights)
            mean_spl = np.average(np.array(spl), weights=weights)
            std_spl = np.average((np.array(spl) - mean_spl) ** 2, weights=weights)
            print("SR:", mean_sr, std_sr ** (1 / 2))
            print("SPL:", mean_spl, std_spl ** (1 / 2))
            print("NDTW:", mean_ndtw, std_ndtw ** (1 / 2))
            print("SDTW:", mean_sdtw, std_sdtw ** (1 / 2))
            print(weights)

        if args.submit:
            json.dump(
                result,
                open(os.path.join(log_dir, "submit_%s.json" % env_name), 'w'),
                sort_keys=True, indent=4, separators=(',', ': ')
            )


def validensemble(train_env, tok, val_envs={}):
    agent1 = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    print("Loaded the listener model at iter %d from %s" % (agent1.load(args.ensemble[0]), args.ensemble[0]))

    agent2 = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    print("Loaded the listener model at iter %d from %s" % (agent2.load(args.ensemble[1]), args.ensemble[1]))

    agent3 = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    print("Loaded the listener model at iter %d from %s" % (agent3.load(args.ensemble[2]), args.ensemble[2]))

    for env_name, (env, evaluator) in val_envs.items():
        agent1.logs = defaultdict(list)
        agent1.env = env
        agent2.env = env
        agent3.env = env

        iters = None
        agent1.ensemble_test(use_dropout=False, feedback='argmax', iters=iters, agent2=agent2, agent3=agent3)
        results = agent1.get_results()

        if env_name != 'test':
            score_summary, _ = evaluator.score(results)
            loss_str = "Env name: %s" % env_name
            for metric, val in score_summary.items():
                loss_str += ', %s: %.4f' % (metric, val)
            print(loss_str)

        if args.submit:
            json.dump(
                results,
                open(os.path.join(log_dir, "submit_%s.json" % env_name), 'w'),
                sort_keys=True, indent=4, separators=(',', ': ')
            )

def setup():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    # Check for vocabs
    if not os.path.exists(TRAIN_VOCAB):
        write_vocab(build_vocab(splits=['train']), TRAIN_VOCAB)
    if not os.path.exists(TRAINVAL_VOCAB):
        write_vocab(build_vocab(splits=['train','val_seen','val_unseen']), TRAINVAL_VOCAB)


def train_val():
    ''' Train on the training set, and validate on seen and unseen splits. '''
    setup()
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    if args.dataset == 'R2R':
        tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)
    elif args.dataset == 'RxR':
        tok = BertTokenizer.from_pretrained('bert-base-multilingual-cased', cache_dir="./cache/")

    if args.feature_extract:
        extract = args.feature_extract
        feat_dict = read_img_features(extract)
        featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])
    else:
        feat_dict = read_img_features(features)
        featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])

    train_env = R2RBatch(feat_dict, batch_size=args.batchSize, splits=['train'], tokenizer=tok)
    from collections import OrderedDict

    val_env_names = ['val_unseen', 'val_seen']
    if args.submit:
        val_env_names.append('test')
    else:
        pass

    val_env_names.append("train")

    val_envs = OrderedDict(
        ((split,
          (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split], tokenizer=tok),
           Evaluation([split], featurized_scans, tok))
          )
         for split in val_env_names
         )
    )

    if args.train == 'listener':
        train(train_env, tok, args.iters, val_envs=val_envs)
    elif args.train == 'validlistener':
        valid(train_env, tok, val_envs=val_envs)
    elif args.train == 'validensemble':
        validensemble(train_env, tok, val_envs=val_envs)
    elif args.train == 'speaker':
        train_speaker(train_env, tok, args.iters, val_envs=val_envs)
    elif args.train == 'validspeaker':
        valid_speaker(tok, train_env, val_envs)
    else:
        assert False


def valid_speaker(tok, train_env, val_envs):
    import tqdm
    listner = Seq2SeqAgent(None, "", tok, args.maxAction)
    speaker = Speaker(None, listner, tok)
    speaker.load(args.load)

    for env_name, (env, evaluator) in val_envs.items():
        if env_name == 'train':
            continue
        print("............ Evaluating %s ............." % env_name)
        speaker.env = env
        path2inst, loss, word_accu, sent_accu = speaker.valid(wrapper=tqdm.tqdm)
        path_id = next(iter(path2inst.keys()))
        print("Inference: ", tok.decode_sentence(path2inst[path_id]))
        print("GT: ", evaluator.gt[path_id]['instructions'])
        pathXinst = list(path2inst.items())
        name2score = evaluator.lang_eval(pathXinst, no_metrics={'METEOR'})
        score_string = " "
        for score_name, score in name2score.items():
            score_string += "%s_%s: %0.4f " % (env_name, score_name, score)
        print("For env %s" % env_name)
        print(score_string)
        print("Average Length %0.4f" % utils.average_length(path2inst))


def train_val_augment():
    """
    Train the listener with the augmented data
    """
    setup()

    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)

    # Load the env img features
    if args.feature_extract:
        extract = args.feature_extract
        feat_dict = read_img_features(extract)
        featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])
    else:
        feat_dict = read_img_features(features)
        featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])

    # Load the augmentation data
    aug_path = args.aug

    # Create the training environment
    train_env = R2RBatch(feat_dict, batch_size=args.batchSize,
                         splits=['train'], tokenizer=tok)
    aug_env   = R2RBatch(feat_dict, batch_size=args.batchSize,
                         splits=[aug_path], tokenizer=tok, name='aug')

    # Printing out the statistics of the dataset
    stats = train_env.get_statistics()
    print("The training data_size is : %d" % train_env.size())
    print("The average instruction length of the dataset is %0.4f." % (stats['length']))
    print("The average action length of the dataset is %0.4f." % (stats['path']))
    stats = aug_env.get_statistics()
    print("The augmentation data size is %d" % aug_env.size())
    print("The average instruction length of the dataset is %0.4f." % (stats['length']))
    print("The average action length of the dataset is %0.4f." % (stats['path']))

    # Setup the validation data
    val_envs = {split: (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split],
                                 tokenizer=tok), Evaluation([split], featurized_scans, tok))
                for split in ['train', 'val_seen', 'val_unseen']}

    # Start training
    train(train_env, tok, args.iters, val_envs=val_envs, aug_env=aug_env)


def train_aug_env_path(train_env, tok, n_iters, log_every=100, val_envs={}, aug_env=None):
    writer = SummaryWriter(logdir=log_dir)
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    speaker = None
    if args.self_train:
        speaker = Speaker(train_env, listner, tok)
        if args.speaker is not None:
            print("Load the speaker at iter %d from %s." % (speaker.load(args.speaker), args.speaker))
        if args.speaker_aug is not None:
            speaker_aug = Speaker(train_env, listner, tok)
            print("Load the speaker for augmented environment at iter %d from %s." % (speaker_aug.load(args.speaker_aug), args.speaker_aug))
        else:
            speaker_aug = speaker

    start_iter = 0
    if args.load is not None:
        print("LOAD THE listener from %s" % args.load)
        start_iter = listner.load(os.path.join(args.load))

    start = time.time()

    best_val = {'val_seen': {"accu": 0., "state":"", 'update':False},
                'val_unseen': {"accu": 0., "state":"", 'update':False}}
    if args.fast_train:
        log_every = 40

    print("Start training")
    for idx in range(start_iter, start_iter+n_iters, log_every):
        listner.logs = defaultdict(list)
        interval = min(log_every, n_iters-idx)
        iter = idx + interval

        # Train for log_every interval
        if args.accumulate_grad:
            for _ in range(interval // 4):
                listner.zero_grad()
                listner.env = train_env
                listner.env.feature_specify = 'original'

                # Train with GT data and original environment
                args.ml_weight = 0.2
                listner.accumulate_gradient(feedback_method)

                # Train with GT data and augment environment
                listner.env.feature_specify = 'aug'
                listner.accumulate_gradient(feedback_method)

                listner.env = aug_env
                listner.env.feature_specify = 'original'

                # Train with Back Translation and original environment
                args.ml_weight = args.bt_weight        # Sem-Configuration
                listner.accumulate_gradient(feedback_method, speaker=speaker)

                # Train with Back Translation and augment environment
                listner.env.feature_specify = 'aug'
                listner.accumulate_gradient(feedback_method, speaker=speaker_aug)
                listner.optim_step()
        else:
            for _ in range(interval // 2):
                # Train with GT data
                listner.env = train_env
                args.ml_weight = 0.2
                listner.train(1, feedback=feedback_method)

                # Train with Back Translation
                listner.env = aug_env
                args.ml_weight = args.bt_weight
                listner.train(1, feedback=feedback_method, speaker=speaker)

        # Log the training stats to tensorboard
        total = max(sum(listner.logs['total']), 1)
        length = max(len(listner.logs['critic_loss']), 1)
        critic_loss = sum(listner.logs['critic_loss']) / total #/ length / args.batchSize
        entropy = sum(listner.logs['entropy']) / total #/ length / args.batchSize
        predict_loss = sum(listner.logs['us_loss']) / max(len(listner.logs['us_loss']), 1)
        writer.add_scalar("loss/critic", critic_loss, idx)
        writer.add_scalar("policy_entropy", entropy, idx)
        writer.add_scalar("loss/unsupervised", predict_loss, idx)
        writer.add_scalar("total_actions", total, idx)
        writer.add_scalar("max_length", length, idx)
        print("total_actions", total)
        print("max_length", length)

        # Run validation
        loss_str = ""
        for env_name, (env, evaluator) in val_envs.items():
            listner.env = env

            # Get validation loss under the same conditions as training
            iters = None if args.fast_train or env_name != 'train' else 20     # 20 * 64 = 1280

            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=iters)
            result = listner.get_results()
            score_summary, _ = evaluator.score(result)
            loss_str += ", %s " % env_name
            for metric,val in score_summary.items():
                if metric in ['success_rate']:
                    writer.add_scalar("accuracy/%s" % env_name, val, idx)
                    if env_name in best_val:
                        if val > best_val[env_name]['accu']:
                            best_val[env_name]['accu'] = val
                            best_val[env_name]['update'] = True
                loss_str += ', %s: %.3f' % (metric, val)

        for env_name in best_val:
            if best_val[env_name]['update']:
                best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                best_val[env_name]['update'] = False
                listner.save(idx, os.path.join("snap", args.name, "state_dict", "best_%s" % (env_name)))

        print(('%s (%d %d%%) %s' % (timeSince(start, float(iter)/n_iters),
                                             iter, float(iter)/n_iters*100, loss_str)))

        if iter % 1000 == 0:
            print("BEST RESULT TILL NOW")
            for env_name in best_val:
                print(env_name, best_val[env_name]['state'])

        if iter % 50000 == 0:
            listner.save(idx, os.path.join("snap", args.name, "state_dict", "Iter_%06d" % (iter)))

    listner.save(idx, os.path.join("snap", args.name, "state_dict", "LAST_iter%d" % (idx)))


def train_val_augenv():
    """
    Train the listener with the augmented environment
    """
    setup()
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    if args.dataset == 'R2R':
        tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)
    elif args.dataset == 'RxR':
        tok = BertTokenizer.from_pretrained('bert-base-multilingual-cased', cache_dir="./cache/")

    # Load the env img features
    if args.feature_extract:
        extract = args.feature_extract
        feat_dict = read_img_features(extract)
        featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])
    else:
        feat_dict = read_img_features(features)
        featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])

    # Load the augmented environment
    feat_aug = []
    print(args.aug_env)
    for i in range(len(args.aug_env)):
        aug_env = args.aug_env[i]
        feat_aug.append(read_img_features(aug_env))

    # Create the training environment
    if args.train_env == 'aug':
        train_env = R2RBatch(feat_aug, batch_size=args.batchSize, splits=['train'], tokenizer=tok)
    elif args.train_env == 'both':
        train_env = R2RBatch([feat_dict, feat_aug], batch_size=args.batchSize, splits=['train'], tokenizer=tok)
    else:
        train_env = R2RBatch(feat_dict, batch_size=args.batchSize, splits=['train'], tokenizer=tok)

    # Setup the validation data
    if args.valid_env == 'aug':
        val_envs = {split: (R2RBatch(feat_aug, batch_size=args.batchSize, splits=[split],
                                     tokenizer=tok), Evaluation([split], featurized_scans, tok))
                    for split in ['train', 'val_seen', 'val_unseen']}
    elif args.valid_env == 'both':
        val_envs = {split: (R2RBatch([feat_dict, feat_aug], batch_size=args.batchSize, splits=[split],
                                     tokenizer=tok), Evaluation([split], featurized_scans, tok))
                    for split in ['train', 'val_seen', 'val_unseen']}
    else:
        val_envs = {split: (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split],
                                     tokenizer=tok), Evaluation([split], featurized_scans, tok))
                    for split in ['train', 'val_seen', 'val_unseen']}

    # Start training
    if args.train == 'augenv':
        return train(train_env, tok, args.iters, log_every= args.log_every, val_envs=val_envs)
    elif args.train == 'augenvaugpath':
        aug_path = args.aug
        if args.train_env == 'aug':
            aug_env = R2RBatch(feat_aug, batch_size=args.batchSize, splits=[aug_path], tokenizer=tok)
        elif args.train_env == 'both':
            aug_env = R2RBatch([feat_dict, feat_aug], batch_size=args.batchSize, splits=[aug_path], tokenizer=tok)
        else:
            aug_env = R2RBatch(feat_dict, batch_size=args.batchSize, splits=[aug_path], tokenizer=tok)
        train_aug_env_path(train_env, tok, args.iters, log_every=args.log_every, val_envs=val_envs, aug_env=aug_env)
    elif args.train == 'speaker':
        train_speaker(train_env, tok, args.iters, log_every = args.log_every, val_envs=val_envs)


if __name__ == "__main__":
    if args.train in ['rlspeaker', 'validspeaker',
                          'listener', 'validlistener', 'validensemble']:
        train_val()
    elif args.train == 'auglistener':
        train_val_augment()
    elif args.train in ['augenv', 'augenvaugpath', 'speaker']:
        score = train_val_augenv()
    else:
        assert False
