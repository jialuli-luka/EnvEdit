
import json
import os
import sys
import numpy as np
import random
import math
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from env import R2RBatch
from utils import padding_idx, add_idx, Tokenizer
import utils
import model_RxR as model
import param
from param import args
from collections import defaultdict

from transformers import BertModel, BertConfig, AdamW, get_linear_schedule_with_warmup
from torch.nn import DataParallel as DP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaseAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.losses = [] # For learning agents
    
    def write_results(self):
        output = [{'instr_id':k, 'trajectory': v} for k,v in self.results.items()]
        with open(self.results_path, 'w') as f:
            json.dump(output, f)

    def get_results(self):
        output = [{'instr_id': k, 'trajectory': v[0], 'path_id':v[1]} for k, v in self.results.items()]
        return output

    def rollout(self, **args):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self, iters=None, **kwargs):
        self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                for traj in self.rollout(**kwargs):
                    self.loss = 0
                    self.results[traj['instr_id']] = [traj['path'], traj['path_id']]
        else:   # Do a full round
            while True:
                for traj in self.rollout(**kwargs):
                    if traj['instr_id'] in self.results:
                        looped = True
                    else:
                        self.loss = 0
                        self.results[traj['instr_id']] = [traj['path'], traj['path_id']]

                if looped:
                    break

class Seq2SeqAgentRxR(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''

    # For now, the agent can't pick which forward move to make - just the one in the middle
    env_actions = {
      'left': (0,-1, 0), # left
      'right': (0, 1, 0), # right
      'up': (0, 0, 1), # up
      'down': (0, 0,-1), # down
      'forward': (1, 0, 0), # forward
      '<end>': (0, 0, 0), # <end>
      '<start>': (0, 0, 0), # <start>
      '<ignore>': (0, 0, 0)  # <ignore>
    }

    def __init__(self, env, results_path, tok, episode_len=20, last_epoch=-1, eval=None):
        super(Seq2SeqAgentRxR, self).__init__(env, results_path)
        self.tok = tok
        self.episode_len = episode_len
        self.feature_size = self.env.feature_size

        # For NDTW reward
        self.eval = eval

        # Models
        model_config = BertConfig.from_pretrained("bert-base-multilingual-cased", return_dict=True, cache_dir="./cache/")
        self.encoder = BertModel.from_pretrained("bert-base-multilingual-cased", config=model_config, cache_dir="./cache/").to(device)
        self.encoder_pos = model.EncoderMBert(args.bert_dim).to(device)
        self.decoder = model.AttnDecoderLSTM(args.aemb, args.bert_dim, args.dropout, feature_size=self.feature_size + args.angle_feat_size).to(device)
        self.critic = model.Critic().to(device)
        self.models = (self.encoder, self.encoder_pos, self.decoder, self.critic)

        # Optimizers
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        encoder_parameters = [
            {
                "params": [p for n, p in self.encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.bert_decay,
            },
            {"params": [p for n, p in self.encoder.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        self.encoder_optimizer = AdamW(encoder_parameters, lr=args.bert_lr)
        self.encoder_scheduler = get_linear_schedule_with_warmup(self.encoder_optimizer, 0.2*args.iters, args.iters, last_epoch=last_epoch)
        self.decoder_optimizer = args.optimizer(self.decoder.parameters(), lr=args.lr)
        self.critic_optimizer = args.optimizer(self.critic.parameters(), lr=args.lr)
        encoder_pos_parameters = [
            {
                "params": [p for n, p in self.encoder_pos.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.bert_decay,
            },
            {"params": [p for n, p in self.encoder_pos.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        self.encoder_pos_optimizer = AdamW(encoder_pos_parameters, lr=args.bert_lr)
        self.encoder_pos_scheduler = get_linear_schedule_with_warmup(self.encoder_pos_optimizer, 0.2*args.iters, args.iters, last_epoch=last_epoch)
        self.optimizers = (self.encoder_optimizer, self.decoder_optimizer, self.critic_optimizer, self.encoder_pos_optimizer)

        # Evaluations
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignoreid, size_average=False)

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)

    def _sort_batch(self, obs):
        ''' Extract instructions from a list of observations and sort by descending
            sequence length (to enable PyTorch packing). '''

        seq_tensor = []
        seq_lengths = []
        attention_mask = []
        for ob in obs:
            seq_tensor.append(ob['instr_encoding'])
            seq_lengths.append(ob['seq_length'])
            attention_mask.append(ob['seq_mask'])
        seq_tensor = torch.from_numpy(np.array(seq_tensor))
        seq_lengths = torch.from_numpy(np.array(seq_lengths))
        attention_mask = torch.from_numpy(np.array(attention_mask))
        mask = utils.length2mask(seq_lengths, args.maxInput)    #attention_mask = 1-mask

        return Variable(seq_tensor, requires_grad=False).long().to(device), \
               mask.to(device),  attention_mask.to(device), \
               list(seq_lengths)

    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        features = np.empty((len(obs), args.views, self.feature_size + args.angle_feat_size), dtype=np.float32)
        for i, ob in enumerate(obs):
            features[i, :, :] = ob['feature']   # Image feat
        return Variable(torch.from_numpy(features), requires_grad=False).to(device)

    def _candidate_variable(self, obs):
        candidate_leng = [len(ob['candidate']) + 1 for ob in obs]       # +1 is for the end
        candidate_feat = np.zeros((len(obs), max(candidate_leng), self.feature_size + args.angle_feat_size), dtype=np.float32)
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            for j, c in enumerate(ob['candidate']):
                candidate_feat[i, j, :] = c['feature']                         # Image feat
        return Variable(torch.from_numpy(candidate_feat), requires_grad=False).to(device), candidate_leng

    def get_input_feat(self, obs):
        input_a_t = np.zeros((len(obs), args.angle_feat_size), np.float32)
        for i, ob in enumerate(obs):
            input_a_t[i] = utils.angle_feature(ob['heading'], ob['elevation'])
        input_a_t = torch.from_numpy(input_a_t).to(device)

        f_t = self._feature_variable(obs)      # Image features from obs
        candidate_feat, candidate_leng = self._candidate_variable(obs)

        return input_a_t, f_t, candidate_feat, candidate_leng

    def _teacher_action_(self, obs, ended):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = args.ignoreid
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:   # Next view point
                        a[i] = k
                        break
                else:   # Stop here
                    assert ob['teacher'] == ob['viewpoint']         # The teacher action should be "STAY HERE"
                    a[i] = len(ob['candidate'])
        return torch.from_numpy(a).to(device)

    def _teacher_action(self, obs, ended):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = args.ignoreid
            else:
                if ob['viewpoint'] == ob['path'][-1]:
                    a[i] = len(ob['candidate'])
                    assert ob['teacher'] == ob['viewpoint']
                elif ob['viewpoint'] in ob['path']:
                    step = ob['path'].index(ob['viewpoint'])
                    for k, candidate in enumerate(ob['candidate']):
                        if candidate['viewpointId'] == ob['path'][step + 1]:
                            a[i] = k
                            break
                    else:
                        assert False
                else:
                    for k, candidate in enumerate(ob['candidate']):
                        if candidate['viewpointId'] == ob['teacher']:  # Next view point
                            a[i] = k
                            break
                    else:  # Stop here
                        assert ob['teacher'] == ob['viewpoint']  # The teacher action should be "STAY HERE"
                        a[i] = len(ob['candidate'])

        return torch.from_numpy(a).to(device)

    def make_equiv_action(self, a_t, perm_obs, perm_idx=None, traj=None):
        """
        Interface between Panoramic view and Egocentric view 
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        def take_action(i, idx, name):
            if type(name) is int:       # Go to the next view
                self.env.env.sims[idx].makeAction(name, 0, 0)
            else:                       # Adjust
                self.env.env.sims[idx].makeAction(*self.env_actions[name])
            state = self.env.env.sims[idx].getState()
            if traj is not None and state.location.viewpointId != traj[i]['path'][-1][0]:
                traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))
        if perm_idx is None:
            perm_idx = range(len(perm_obs))
        for i, idx in enumerate(perm_idx):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                select_candidate = perm_obs[i]['candidate'][action]
                src_point = perm_obs[i]['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point ) // 12   # The point idx started from 0
                trg_level = (trg_point ) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, idx, 'up')
                    src_level += 1
                while src_level > trg_level:    # Tune down
                    take_action(i, idx, 'down')
                    src_level -= 1
                while self.env.env.sims[idx].getState().viewIndex != trg_point:    # Turn right until the target
                    take_action(i, idx, 'right')
                assert select_candidate['viewpointId'] == \
                       self.env.env.sims[idx].getState().navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, idx, select_candidate['idx'])

    def rollout(self, train_ml=None, train_rl=True, reset=True, speaker=None):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether use RL in training
        :param reset:       Reset the environment
        :param speaker:     Speaker used in back translation.
                            If the speaker is not None, use back translation.
                            O.w., normal training
        :return:
        """
        if self.feedback == 'teacher' or self.feedback == 'argmax':
            train_rl = False

        if reset:
            # Reset env
            obs = np.array(self.env.reset())
        else:
            obs = np.array(self.env._get_obs())

        batch_size = len(obs)

        if speaker is not None:         # Trigger the self_train mode!
            noise = self.decoder.drop_env(torch.ones(self.feature_size).to(device))
            batch = self.env.batch.copy()
            speaker.env = self.env
            insts = speaker.infer_batch(featdropmask=noise)     # Use the same drop mask in speaker

            # Create fake environments with the generated instruction
            boss = np.ones((batch_size, 1), np.int64) * self.tok.word_to_index['<BOS>']  # First word is <BOS>
            insts = np.concatenate((boss, insts), 1)
            for i, (datum, inst) in enumerate(zip(batch, insts)):
                if inst[-1] != self.tok.word_to_index['<PAD>']: # The inst is not ended!
                    inst[-1] = self.tok.word_to_index['<EOS>']
                datum.pop('instructions')
                datum.pop('instr_encoding')
                datum['instructions'] = self.tok.decode_sentence(inst)
                datum['instr_encoding'] = inst
            obs = np.array(self.env.reset(batch))

        seq, seq_mask, attention_mask, seq_lengths = self._sort_batch(obs)

        text_features = self.encoder(seq, attention_mask=attention_mask)
        ctx, h_t, c_t = self.encoder_pos(text_features.last_hidden_state)   # batch_size, sequence_length, hidden_size
        ctx_mask = seq_mask

        # Init the reward shaping
        last_dist = np.zeros(batch_size, np.float32)
        last_ndtw = np.zeros(batch_size, np.float32)
        for i, ob in enumerate(obs):   # The init distance from the view point to the target
            last_dist[i] = ob['distance']
            if train_rl and args.ndtw:
                last_ndtw[i] = self.eval.ndtw(ob['scan'], [(ob['viewpoint'], ob['heading'], ob['elevation'])], ob['path'])

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
            'path_id': ob['path_id'],
            'gt': ob['path']
        } for ob in obs]

        # For test result submission
        visited = [set() for _ in obs]

        # Initialization the tracking state
        ended = np.array([False] * batch_size)   # Indices match permuation of the model, not env

        # Init the logs
        rewards = []
        hidden_states = []
        policy_log_probs = []
        masks = []
        entropys = []
        ml_loss = 0.

        h1 = h_t
        for t in range(self.episode_len):
            # print(t)
            input_a_t, f_t, candidate_feat, candidate_leng = self.get_input_feat(obs)

            if speaker is not None:       # Apply the env drop mask to the feat
                candidate_feat[..., :-args.angle_feat_size] *= noise
                f_t[..., :-args.angle_feat_size] *= noise

            h_t, c_t, logit, h1, _, _ = self.decoder(input_a_t, f_t, candidate_feat,
                                                   h_t, h1, c_t,
                                                   ctx, ctx_mask,
                                                   already_dropfeat=(speaker is not None))

            hidden_states.append(h_t)

            # Mask outputs where agent can't move forward
            # Here the logit is [b, max_candidate]
            candidate_mask = utils.length2mask(candidate_leng)
            # if args.submit:     # Avoding cyclic path
            #     for ob_id, ob in enumerate(obs):
            #         visited[ob_id].add(ob['viewpoint'])
            #         for c_id, c in enumerate(ob['candidate']):
            #             if c['viewpointId'] in visited[ob_id]:
            #                 candidate_mask[ob_id][c_id] = 1
            logit.masked_fill_(candidate_mask, -float('inf'))

            # Supervised training
            target = self._teacher_action(obs, ended)
            ml_loss += self.criterion(logit, target)

            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target                # teacher forcing
            elif self.feedback == 'argmax': 
                _, a_t = logit.max(1)        # student forcing - argmax
                a_t = a_t.detach()
                log_probs = F.log_softmax(logit, 1)                              # Calculate the log_prob here
                policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))   # Gather the log_prob for each batch
            elif self.feedback == 'sample':
                probs = F.softmax(logit, 1)    # sampling an action from model
                c = torch.distributions.Categorical(probs)
                self.logs['entropy'].append(c.entropy().sum().item())      # For log
                entropys.append(c.entropy())                                # For optimization
                a_t = c.sample().detach()
                policy_log_probs.append(c.log_prob(a_t))
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Prepare environment action
            # NOTE: Env action is in the perm_obs space
            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if next_id == (candidate_leng[i]-1) or next_id == args.ignoreid or ended[i]:    # The last action is <end>
                    cpu_a_t[i] = -1             # Change the <end> and ignore action to -1

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, obs, None, traj)
            obs = np.array(self.env._get_obs())

            # Calculate the mask and reward
            if train_rl:
                dist = np.zeros(batch_size, np.float32)
                reward = np.zeros(batch_size, np.float32)
                mask = np.ones(batch_size, np.float32)
                ndtw_score = np.zeros(batch_size, np.float32)
                for i, ob in enumerate(obs):
                    dist[i] = ob['distance']
                    if args.ndtw:
                        ndtw_score[i] = self.eval.ndtw(ob['scan'], traj[i]['path'], ob['path'])
                    else:
                        ndtw_score[i] = 0
                    if ended[i]:            # If the action is already finished BEFORE THIS ACTION.
                        reward[i] = 0.
                        mask[i] = 0.
                    else:       # Calculate the reward
                        action_idx = cpu_a_t[i]
                        if action_idx == -1:        # If the action now is end
                            if dist[i] < 3:         # Correct
                                reward[i] = 2. + ndtw_score[i] * 2.0
                            else:                   # Incorrect
                                reward[i] = -2.
                        else:                       # The action is not end
                            reward[i] = - (dist[i] - last_dist[i])      # Change of distance
                            ndtw_reward = ndtw_score[i] - last_ndtw[i]
                            if reward[i] > 0:                           # Quantification
                                reward[i] = 1 + ndtw_reward
                            elif reward[i] < 0:
                                reward[i] = -1 + ndtw_reward
                            else:
                                raise NameError("The action doesn't change the move")
                            # Miss the target penalty
                            if (last_dist[i] <= 1.0) and (dist[i] - last_dist[i] > 0.0):
                                reward[i] -= (1.0 - last_dist[i]) * 2.0
                rewards.append(reward)
                masks.append(mask)
                last_dist[:] = dist
                last_ndtw[:] = ndtw_score
            # Update the finished actions
            # -1 means ended or ignored (already ended)
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all(): 
                break

        if train_rl:
            # Last action in A2C
            input_a_t, f_t, candidate_feat, candidate_leng = self.get_input_feat(obs)

            if speaker is not None:
                candidate_feat[..., :-args.angle_feat_size] *= noise
                f_t[..., :-args.angle_feat_size] *= noise
            last_h_, _, _, _, _, _ = self.decoder(input_a_t, f_t, candidate_feat,
                                            h_t, h1, c_t,
                                            ctx, ctx_mask,
                                            speaker is not None)
            rl_loss = 0.

            # NOW, A2C!!!
            # Calculate the final discounted reward
            last_value__ = self.critic(last_h_).detach()    # The value esti of the last state, remove the grad for safety
            discount_reward = np.zeros(batch_size, np.float32)  # The inital reward is zero
            for i in range(batch_size):
                if not ended[i]:        # If the action is not ended, use the value function as the last reward
                    discount_reward[i] = last_value__[i]

            length = len(rewards)
            total = 0
            for t in range(length-1, -1, -1):
                discount_reward = discount_reward * args.gamma + rewards[t]   # If it ended, the reward will be 0
                mask_ = Variable(torch.from_numpy(masks[t]), requires_grad=False).to(device)
                clip_reward = discount_reward.copy()
                r_ = Variable(torch.from_numpy(clip_reward), requires_grad=False).to(device)
                v_ = self.critic(hidden_states[t])
                a_ = (r_ - v_).detach()

                # r_: The higher, the better. -ln(p(action)) * (discount_reward - value)
                rl_loss += (-policy_log_probs[t] * a_ * mask_).sum()
                rl_loss += (((r_ - v_) ** 2) * mask_).sum() * 0.5     # 1/2 L2 loss
                if self.feedback == 'sample':
                    rl_loss += (- 0.01 * entropys[t] * mask_).sum()
                self.logs['critic_loss'].append((((r_ - v_) ** 2) * mask_).sum().item())

                total = total + np.sum(masks[t])
            self.logs['total'].append(total)

            # Normalize the loss function
            if args.normalize_loss == 'total':
                rl_loss /= total
            elif args.normalize_loss == 'batch':
                rl_loss /= batch_size
            else:
                assert args.normalize_loss == 'none'

            self.loss += rl_loss

        if train_ml is not None:
            self.logs['us_loss'].append(ml_loss * train_ml / batch_size)
            self.loss += ml_loss * train_ml / batch_size

        if type(self.loss) is int:  # For safety, it will be activated if no losses are added
            self.losses.append(0.)
        else:
            self.losses.append(self.loss.item() / self.episode_len)    # This argument is useless.

        return traj

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, iters=None):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        if use_dropout:
            self.encoder.train()
            self.decoder.train()
            self.critic.train()
            self.encoder_pos.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
            self.critic.eval()
            self.encoder_pos.eval()
        super(Seq2SeqAgentRxR, self).test(iters)

    def zero_grad(self):
        self.loss = 0.
        self.losses = []
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            optimizer.zero_grad()

    def accumulate_gradient(self, feedback='teacher', **kwargs):
        if feedback == 'teacher':
            self.feedback = 'teacher'
            self.rollout(train_ml=args.teacher_weight, train_rl=False, **kwargs)
        elif feedback == 'sample':
            self.feedback = 'teacher'
            self.rollout(train_ml=args.ml_weight, train_rl=False, **kwargs)
            self.feedback = 'sample'
            self.rollout(train_ml=None, train_rl=True, **kwargs)
        else:
            assert False

    def optim_step(self):
        self.loss.backward()

        torch.nn.utils.clip_grad_norm(self.decoder.parameters(), 40.)
        torch.nn.utils.clip_grad_norm(self.encoder_pos.parameters(), 40.)
        torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 40.)
        self.encoder_optimizer.step()
        self.encoder_scheduler.step()
        self.encoder_pos_optimizer.step()
        self.encoder_pos_scheduler.step()
        self.decoder_optimizer.step()
        self.critic_optimizer.step()

    def train(self, n_iters, feedback='teacher', **kwargs):
        ''' Train for a given number of iterations '''
        self.feedback = feedback

        self.decoder.train()
        self.critic.train()
        self.encoder.train()
        self.encoder_pos.train()

        self.losses = []
        for iter in range(1, n_iters + 1):

            self.encoder_pos_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            self.encoder_optimizer.zero_grad()

            self.loss = 0
            if feedback == 'teacher':
                self.feedback = 'teacher'
                self.rollout(train_ml=args.teacher_weight, train_rl=False, **kwargs)
            elif feedback == 'sample':
                if args.ml_weight != 0:
                    self.feedback = 'teacher'
                    self.rollout(train_ml=args.ml_weight, train_rl=False, **kwargs)
                self.feedback = 'sample'
                self.rollout(train_ml=None, train_rl=True, **kwargs)
            else:
                assert False

            self.loss.backward()

            torch.nn.utils.clip_grad_norm(self.decoder.parameters(), 40.)
            torch.nn.utils.clip_grad_norm(self.encoder_pos.parameters(), 40.)
            torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 40.)
            self.encoder_optimizer.step()
            self.encoder_scheduler.step()
            self.encoder_pos_optimizer.step()
            self.encoder_pos_scheduler.step()
            self.decoder_optimizer.step()
            self.critic_optimizer.step()

    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        all_tuple = [("encoder", self.encoder, self.encoder_optimizer),
                     ("decoder", self.decoder, self.decoder_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        all_tuple.append(("encoder_pos", self.encoder_pos, self.encoder_pos_optimizer))
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)
        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
            state.update(states[name]['state_dict'])
            model.load_state_dict(state)
            if args.loadOptim:
                optimizer.load_state_dict(states[name]['optimizer'])

        all_tuple = [("encoder", self.encoder, self.encoder_optimizer),
                     ("decoder", self.decoder, self.decoder_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]

        all_tuple.append(("encoder_pos", self.encoder_pos, self.encoder_pos_optimizer))
        for param in all_tuple:
            recover_state(*param)
        return states['encoder']['epoch'] - 1

