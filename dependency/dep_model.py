from __future__ import absolute_import, division, print_function

import copy
import math

import numpy as np
import torch
from torch import nn
from modules import BertModel, ZenModel, BertTokenizer, Biaffine, MLP, LayerNormalization, PositionalEncoding
from transformers_xlnet import XLNetModel, XLNetTokenizer
from util import eisner, ZenNgramDict, ispunct
from dep_helper import save_json, load_json
import os
import subprocess

DEFAULT_HPARA = {
    'max_seq_length': 128,
    'use_bert': False,
    'use_xlnet': False,
    'use_zen': False,
    'do_lower_case': False,
    'use_pos': False,
    'mlp_dropout': 0.33,
    'n_mlp_arc': 500,
    'n_mlp_rel': 100,
    'use_biaffine': True,
    #
    'use_encoder': False,
    'num_layers': 3,
    'd_model': 1024,
    'num_heads': 8,
    'd_ff': 2048,
}


class DependencyParser(nn.Module):

    def __init__(self, labelmap, hpara, model_path, word2id=None, pos2id=None, from_pretrained=True):
        super().__init__()
        self.labelmap = labelmap
        self.hpara = hpara
        self.num_labels = len(self.labelmap) + 1
        self.max_seq_length = self.hpara['max_seq_length']
        self.arc_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.rel_criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.use_biaffine = self.hpara['use_biaffine']
        self.use_pos = hpara['use_pos']
        self.use_encoder = hpara['use_encoder']

        if hpara['use_zen']:
            raise ValueError()

        self.tokenizer = None
        self.bert = None
        self.xlnet = None
        self.zen = None
        self.zen_ngram_dict = None

        if self.hpara['use_bert']:
            self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=self.hpara['do_lower_case'])
            if from_pretrained:
                self.bert = BertModel.from_pretrained(model_path, cache_dir='')
            else:
                from modules import CONFIG_NAME, BertConfig
                config_file = os.path.join(model_path, CONFIG_NAME)
                config = BertConfig.from_json_file(config_file)
                self.bert = BertModel(config)
            hidden_size = self.bert.config.hidden_size
            self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        elif self.hpara['use_xlnet']:
            self.tokenizer = XLNetTokenizer.from_pretrained(model_path, do_lower_case=self.hpara['do_lower_case'])
            if from_pretrained:
                self.xlnet = XLNetModel.from_pretrained(model_path)
                state_dict = torch.load(os.path.join(model_path, 'pytorch_model.bin'))
                key_list = list(state_dict.keys())
                reload = False
                for key in key_list:
                    if key.find('xlnet.') > -1:
                        reload = True
                        state_dict[key[key.find('xlnet.') + len('xlnet.'):]] = state_dict[key]
                    state_dict.pop(key)
                if reload:
                    self.xlnet.load_state_dict(state_dict)
            else:
                config, model_kwargs = XLNetModel.config_class.from_pretrained(model_path, return_unused_kwargs=True)
                self.xlnet = XLNetModel(config)

            hidden_size = self.xlnet.config.hidden_size
            self.dropout = nn.Dropout(self.xlnet.config.summary_last_dropout)
        elif self.hpara['use_zen']:
            self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=self.hpara['do_lower_case'])
            self.zen_ngram_dict = ZenNgramDict(model_path, tokenizer=self.zen_tokenizer)
            self.zen = ZenModel.from_pretrained(model_path, cache_dir='')
            hidden_size = self.zen.config.hidden_size
            self.dropout = nn.Dropout(self.zen.config.hidden_dropout_prob)
        else:
            raise ValueError()

        if self.use_pos:
            self.pos2id = pos2id
            if self.use_encoder:
                self.pos_embedding = nn.Embedding(len(self.pos2id), int(self.hpara['d_model']))
            else:
                self.pos_embedding = nn.Embedding(len(self.pos2id), hidden_size)
                self.layer_norm_encoder = LayerNormalization(hidden_size)
                self.pos_norm = LayerNormalization(hidden_size)
        else:
            self.pos2id = None
            self.pos_embedding = None

        if self.use_encoder:
            self.linear = MLP(n_in=hidden_size, n_hidden=(self.hpara['d_model']), dropout=0.2)
            self.linear_dep = MLP(n_in=hidden_size, n_hidden=(self.hpara['d_model']), dropout=0.2)
            self.layer_norm_encoder = LayerNormalization(self.hpara['d_model'])
            self.layer_norm_dep = LayerNormalization(self.hpara['d_model'])
            self.positional_embedding = PositionalEncoding(self.hpara['d_model'], dropout=0.2)
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.hpara['d_model'], nhead=self.hpara['num_heads'],
                                                       dim_feedforward=self.hpara['d_ff'])
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.hpara['num_layers'])

            hidden_size = self.hpara['d_model']
        else:
            self.linear_dep = None
            self.layer_norm_dep = None
            self.positional_embedding = None
            self.encoder = None

        self.word2id = None

        if self.use_biaffine:
            self.mlp_arc_h = MLP(n_in=hidden_size,
                                 n_hidden=self.hpara['n_mlp_arc'],
                                 dropout=self.hpara['mlp_dropout'])
            self.mlp_arc_d = MLP(n_in=hidden_size,
                                 n_hidden=self.hpara['n_mlp_arc'],
                                 dropout=self.hpara['mlp_dropout'])
            self.mlp_rel_h = MLP(n_in=hidden_size,
                                 n_hidden=self.hpara['n_mlp_rel'],
                                 dropout=self.hpara['mlp_dropout'])
            self.mlp_rel_d = MLP(n_in=hidden_size,
                                 n_hidden=self.hpara['n_mlp_rel'],
                                 dropout=self.hpara['mlp_dropout'])

            self.arc_attn = Biaffine(n_in=self.hpara['n_mlp_arc'],
                                     bias_x=True,
                                     bias_y=False)
            self.rel_attn = Biaffine(n_in=self.hpara['n_mlp_rel'],
                                     n_out=self.num_labels,
                                     bias_x=True,
                                     bias_y=True)
        else:
            self.linear_arc = nn.Linear(hidden_size, hidden_size, bias=False)
            self.rel_classifier_1 = nn.Linear(hidden_size, self.num_labels, bias=False)
            self.rel_classifier_2 = nn.Linear(hidden_size, self.num_labels, bias=False)
            self.bias = nn.Parameter(torch.tensor(self.num_labels, dtype=torch.float), requires_grad=True)
            nn.init.zeros_(self.bias)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, valid_ids=None,
                attention_mask_label=None,
                word_ids=None, pos_ids=None,
                input_ngram_ids=None, ngram_position_matrix=None,
                ):

        if self.bert is not None:
            sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        elif self.xlnet is not None:
            transformer_outputs = self.xlnet(input_ids, token_type_ids, attention_mask=attention_mask)
            sequence_output = transformer_outputs[0]
        elif self.zen is not None:
            sequence_output, _ = self.zen(input_ids, input_ngram_ids=input_ngram_ids,
                                          ngram_position_matrix=ngram_position_matrix,
                                          token_type_ids=token_type_ids, attention_mask=attention_mask,
                                          output_all_encoded_layers=False)
        else:
            raise ValueError()

        batch_size, _, feat_dim = sequence_output.shape
        max_len = attention_mask_label.shape[1]
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=sequence_output.dtype, device=input_ids.device)
        for i in range(batch_size):
            try:
                temp = sequence_output[i][valid_ids[i] == 1]
            except Exception:
                import pdb
                pdb.set_trace()
            sent_len = attention_mask_label[i].sum()
            # valid_output[i][:temp.size(0)] = temp
            valid_output[i][:sent_len] = temp[:sent_len]

        if self.encoder is not None:
            encoder_input = self.linear(valid_output)
            if self.pos_embedding is not None:
                pos_embedding = self.pos_embedding(pos_ids)
                encoder_input = encoder_input + pos_embedding
            encoder_input = self.layer_norm_encoder(encoder_input)
            encoder_input = self.positional_embedding(encoder_input)
            encoder_feature = self.encoder(encoder_input)
            valid_output = self.layer_norm_dep(encoder_feature)

        elif self.pos_embedding is not None:
            pos_embedding = self.pos_embedding(pos_ids)
            valid_output = self.layer_norm_encoder(valid_output)
            pos_embedding = self.pos_norm(pos_embedding)
            valid_output = pos_embedding + valid_output

        if self.use_biaffine:
            valid_output = self.dropout(valid_output)

            arc_h = self.mlp_arc_h(valid_output)
            arc_d = self.mlp_arc_d(valid_output)
            rel_h = self.mlp_rel_h(valid_output)
            rel_d = self.mlp_rel_d(valid_output)

            # get arc and rel scores from the bilinear attention
            # [batch_size, seq_len, seq_len]
            s_arc = self.arc_attn(arc_d, arc_h)
            # [batch_size, seq_len, seq_len, n_rels]
            s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
            # set the scores that exceed the length of each sentence to -inf
            s_arc.masked_fill_(~attention_mask_label.unsqueeze(1), float('-inf'))
        else:
            tmp_arc = self.linear_arc(valid_output).permute(0, 2, 1)
            s_arc = torch.bmm(valid_output, tmp_arc)

            # [batch_size, seq_len, seq_len, n_rels]
            rel_1 = self.rel_classifier_1(valid_output)
            rel_2 = self.rel_classifier_2(valid_output)
            rel_1 = torch.stack([rel_1] * max_len, dim=1)
            rel_2 = torch.stack([rel_2] * max_len, dim=2)
            s_rel = rel_1 + rel_2 + self.bias
            # set the scores that exceed the length of each sentence to -inf
            s_arc.masked_fill_(~attention_mask_label.unsqueeze(1), float('-inf'))

        return s_arc, s_rel

    @staticmethod
    def init_hyper_parameters(args):
        hyper_parameters = DEFAULT_HPARA.copy()
        hyper_parameters['max_seq_length'] = args.max_seq_length
        hyper_parameters['use_bert'] = args.use_bert
        hyper_parameters['use_xlnet'] = args.use_xlnet
        hyper_parameters['use_zen'] = args.use_zen
        hyper_parameters['do_lower_case'] = args.do_lower_case
        hyper_parameters['mlp_dropout'] = args.mlp_dropout
        hyper_parameters['n_mlp_arc'] = args.n_mlp_arc
        hyper_parameters['n_mlp_rel'] = args.n_mlp_rel
        hyper_parameters['use_biaffine'] = args.use_biaffine

        hyper_parameters['use_pos'] = args.use_pos
        hyper_parameters['use_encoder'] = args.use_encoder
        hyper_parameters['num_layers'] = args.num_layers

        return hyper_parameters

    @property
    def model(self):
        return self.state_dict()

    def save_model(self, output_dir, vocab_dir):

        output_model_path = os.path.join(output_dir, 'pytorch_model.bin')
        torch.save(self.state_dict(), output_model_path)

        output_tag_file = os.path.join(output_dir, 'labelset.json')
        save_json(output_tag_file, self.labelmap)

        output_hpara_file = os.path.join(output_dir, 'hpara.json')
        save_json(output_hpara_file, self.hpara)

        if self.pos2id is not None:
            output_hpara_file = os.path.join(output_dir, 'pos2id.json')
            save_json(output_hpara_file, self.pos2id)

        if self.word2id is not None:
            output_hpara_file = os.path.join(output_dir, 'word2id.json')
            save_json(output_hpara_file, self.word2id)

        output_config_file = os.path.join(output_dir, 'config.json')
        with open(output_config_file, "w", encoding='utf-8') as writer:
            if self.bert:
                writer.write(self.bert.config.to_json_string())
            elif self.xlnet:
                writer.write(self.xlnet.config.to_json_string())
            elif self.zen:
                writer.write(self.zen.config.to_json_string())
        output_bert_config_file = os.path.join(output_dir, 'bert_config.json')
        command = 'cp ' + str(output_config_file) + ' ' + str(output_bert_config_file)
        subprocess.run(command, shell=True)

        if self.bert:
            vocab_name = 'vocab.txt'
        elif self.xlnet:
            vocab_name = 'spiece.model'
        elif self.zen:
            vocab_name = 'vocab.txt'
        else:
            raise ValueError()
        vocab_path = os.path.join(vocab_dir, vocab_name)
        command = 'cp ' + str(vocab_path) + ' ' + str(os.path.join(output_dir, vocab_name))
        subprocess.run(command, shell=True)

        if self.zen:
            ngram_name = 'ngram.txt'
            ngram_path = os.path.join(vocab_dir, ngram_name)
            command = 'cp ' + str(ngram_path) + ' ' + str(os.path.join(output_dir, ngram_name))
            subprocess.run(command, shell=True)

    @classmethod
    def load_model(cls, model_path, device):
        tag_file = os.path.join(model_path, 'labelset.json')
        labelmap = load_json(tag_file)

        pos_file = os.path.join(model_path, 'pos2id.json')
        if os.path.exists(pos_file):
            pos2id = load_json(pos_file)
        else:
            pos2id = None

        word_file = os.path.join(model_path, 'word2id.json')
        if os.path.exists(word_file):
            word2id = load_json(word_file)
        else:
            word2id = None

        hpara_file = os.path.join(model_path, 'hpara.json')
        hpara = load_json(hpara_file)
        DEFAULT_HPARA.update(hpara)

        res = cls(labelmap=labelmap, hpara=DEFAULT_HPARA, model_path=model_path, word2id=word2id, pos2id=pos2id)
        res.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location=device))
        return res

    @staticmethod
    def set_not_grad(module):
        for para in module.parameters():
            para.requires_grad = False

    def load_data(self, data_path, do_predict=False):
        if not do_predict:
            flag = data_path[data_path.rfind('/')+1: data_path.rfind('.')]
        else:
            flag = 'predict'

        lines = readfile(data_path, flag)

        examples = self.process_data(lines, flag)

        return examples

    @staticmethod
    def process_data(lines, flag):
        data = []
        for sentence, head, label, pos in lines:
            data.append((sentence, head, label, pos))
        examples = []
        for i, (sentence, head, label, pos) in enumerate(data):
            guid = "%s-%s" % (flag, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, head=head,
                                         label=label, pos=pos))
        return examples

    def get_loss(self, arc_scores, rel_scores, arcs, rels, mask):
        arc_scores, arcs = arc_scores[mask], arcs[mask]
        rel_scores, rels = rel_scores[mask], rels[mask]
        rel_scores = rel_scores[torch.arange(len(arcs)), arcs]
        arc_loss = self.arc_criterion(arc_scores, arcs)
        rel_loss = self.rel_criterion(rel_scores, rels)
        loss = arc_loss + rel_loss

        return loss

    @staticmethod
    def decode(arc_scores, rel_scores, mask):
        arc_preds = eisner(arc_scores, mask)
        rel_preds = rel_scores.argmax(-1)
        rel_preds = rel_preds.gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)

        return arc_preds, rel_preds

    def convert_examples_to_features(self, examples):

        features = []

        length_list = []
        tokens_list = []
        head_idx_list = []
        labels_list = []
        valid_list = []
        label_mask_list = []
        punctuation_idx_list = []

        pos_list = []

        for (ex_index, example) in enumerate(examples):
            textlist = example.text_a.split(' ')
            labellist = example.label
            head_list = example.head
            tokens = []
            head_idx = []
            labels = []
            valid = []
            label_mask = []

            punctuation_idx = []

            poslist = example.pos

            if len(textlist) > self.max_seq_length - 2:
                textlist = textlist[:self.max_seq_length - 2]
                labellist = labellist[:self.max_seq_length - 2]
                head_list = head_list[:self.max_seq_length - 2]
                poslist = poslist[:self.max_seq_length - 2]

            for i, word in enumerate(textlist):
                if ispunct(word):
                    punctuation_idx.append(i+1)
                token = self.tokenizer.tokenize(word)
                tokens.extend(token)
                label_1 = labellist[i]
                for m in range(len(token)):
                    if m == 0:
                        valid.append(1)
                        head_idx.append(head_list[i])
                        labels.append(label_1)
                        label_mask.append(1)
                    else:
                        valid.append(0)
            length_list.append(len(tokens))
            tokens_list.append(tokens)
            head_idx_list.append(head_idx)
            labels_list.append(labels)
            valid_list.append(valid)
            label_mask_list.append(label_mask)
            punctuation_idx_list.append(punctuation_idx)

            pos_list.append(poslist)

        label_len_list = [len(label) for label in labels_list]
        seq_pad_length = max(length_list) + 2
        label_pad_length = max(label_len_list) + 1

        for indx, (example, tokens, head_idxs, labels, valid, label_mask, punctuation_idx, pos) in \
                enumerate(zip(examples, tokens_list, head_idx_list,
                              labels_list, valid_list, label_mask_list, punctuation_idx_list, pos_list)):

            ntokens = []
            segment_ids = []
            label_ids = []
            head_idx = []

            ntokens.append("[CLS]")
            segment_ids.append(0)

            valid.insert(0, 1)
            label_mask.insert(0, 1)
            head_idx.append(-1)
            label_ids.append(self.labelmap["[CLS]"])
            for i, token in enumerate(tokens):
                ntokens.append(token)
                segment_ids.append(0)
            for i in range(len(labels)):
                if labels[i] in self.labelmap:
                    label_ids.append(self.labelmap[labels[i]])
                else:
                    label_ids.append(self.labelmap['<UNK>'])
                head_idx.append(head_idxs[i])
            ntokens.append("[SEP]")

            segment_ids.append(0)
            valid.append(1)

            input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
            input_mask = [1] * len(input_ids)
            while len(input_ids) < seq_pad_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                valid.append(1)
            while len(label_ids) < label_pad_length:
                head_idx.append(-1)
                label_ids.append(0)
                label_mask.append(0)

            eval_mask = copy.deepcopy(label_mask)
            eval_mask[0] = 0
            # ignore all punctuation if not specified
            for idx in punctuation_idx:
                if idx < label_pad_length:
                    eval_mask[idx] = 0

            if self.pos_embedding is not None:
                pos_ids = [self.pos2id['[CLS]']]
                for i in range(len(pos)):
                    if pos[i] in self.pos2id:
                        pos_ids.append(self.pos2id[pos[i]])
                    else:
                        pos_ids.append(self.pos2id['<UNK>'])
                while len(pos_ids) < label_pad_length:
                    pos_ids.append(0)
                assert len(pos_ids) == label_pad_length
            else:
                pos_ids = None

            word_ids = None

            assert len(input_ids) == seq_pad_length
            assert len(input_mask) == seq_pad_length
            assert len(segment_ids) == seq_pad_length
            assert len(valid) == seq_pad_length

            assert len(label_ids) == label_pad_length
            assert len(head_idx) == label_pad_length
            assert len(label_mask) == label_pad_length
            assert len(eval_mask) == label_pad_length

            if self.zen_ngram_dict is not None:
                ngram_matches = []
                #  Filter the ngram segment from 2 to 7 to check whether there is a ngram
                max_gram_n = self.zen_ngram_dict.max_ngram_len

                for p in range(2, max_gram_n):
                    for q in range(0, len(tokens) - p + 1):
                        character_segment = tokens[q:q + p]
                        # j is the starting position of the ngram
                        # i is the length of the current ngram
                        character_segment = tuple(character_segment)
                        if character_segment in self.zen_ngram_dict.ngram_to_id_dict:
                            ngram_index = self.zen_ngram_dict.ngram_to_id_dict[character_segment]
                            ngram_matches.append([ngram_index, q, p, character_segment,
                                                  self.zen_ngram_dict.ngram_to_freq_dict[character_segment]])

                ngram_matches = sorted(ngram_matches, key=lambda s: s[-1], reverse=True)

                max_ngram_in_seq_proportion = math.ceil((len(tokens) / self.max_seq_length) * self.zen_ngram_dict.max_ngram_in_seq)
                if len(ngram_matches) > max_ngram_in_seq_proportion:
                    ngram_matches = ngram_matches[:max_ngram_in_seq_proportion]

                ngram_ids = [ngram[0] for ngram in ngram_matches]
                ngram_positions = [ngram[1] for ngram in ngram_matches]
                ngram_lengths = [ngram[2] for ngram in ngram_matches]
                ngram_tuples = [ngram[3] for ngram in ngram_matches]
                ngram_seg_ids = [0 if position < (len(tokens) + 2) else 1 for position in ngram_positions]

                ngram_mask_array = np.zeros(self.zen_ngram_dict.max_ngram_in_seq, dtype=np.bool)
                ngram_mask_array[:len(ngram_ids)] = 1

                # record the masked positions
                ngram_positions_matrix = np.zeros(shape=(seq_pad_length, self.zen_ngram_dict.max_ngram_in_seq), dtype=np.int32)
                for i in range(len(ngram_ids)):
                    ngram_positions_matrix[ngram_positions[i]:ngram_positions[i] + ngram_lengths[i], i] = 1.0

                # Zero-pad up to the max ngram in seq length.
                padding = [0] * (self.zen_ngram_dict.max_ngram_in_seq - len(ngram_ids))
                ngram_ids += padding
                ngram_lengths += padding
                ngram_seg_ids += padding
            else:
                ngram_ids = None
                ngram_positions_matrix = None
                ngram_lengths = None
                ngram_tuples = None
                ngram_seg_ids = None
                ngram_mask_array = None

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              head_idx=head_idx,
                              label_id=label_ids,
                              valid_ids=valid,
                              label_mask=label_mask,
                              eval_mask=eval_mask,
                              word_ids=word_ids,
                              pos_ids=pos_ids,
                              ngram_ids=ngram_ids,
                              ngram_positions=ngram_positions_matrix,
                              ngram_lengths=ngram_lengths,
                              ngram_tuples=ngram_tuples,
                              ngram_seg_ids=ngram_seg_ids,
                              ngram_masks=ngram_mask_array,
                              ))
        return features

    def feature2input(self, device, feature):
        all_input_ids = torch.tensor([f.input_ids for f in feature], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in feature], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in feature], dtype=torch.long)
        all_head_idx = torch.tensor([f.head_idx for f in feature], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in feature], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in feature], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in feature], dtype=torch.bool)
        all_eval_mask_ids = torch.tensor([f.eval_mask for f in feature], dtype=torch.bool)
        input_ids = all_input_ids.to(device)
        input_mask = all_input_mask.to(device)
        segment_ids = all_segment_ids.to(device)
        head_idx = all_head_idx.to(device)
        label_ids = all_label_ids.to(device)
        valid_ids = all_valid_ids.to(device)
        l_mask = all_lmask_ids.to(device)
        eval_mask = all_eval_mask_ids.to(device)

        if self.zen is not None:
            all_ngram_ids = torch.tensor([f.ngram_ids for f in feature], dtype=torch.long)
            all_ngram_positions = torch.tensor([f.ngram_positions for f in feature], dtype=torch.long)
            # all_ngram_lengths = torch.tensor([f.ngram_lengths for f in train_features], dtype=torch.long)
            # all_ngram_seg_ids = torch.tensor([f.ngram_seg_ids for f in train_features], dtype=torch.long)
            # all_ngram_masks = torch.tensor([f.ngram_masks for f in train_features], dtype=torch.long)

            ngram_ids = all_ngram_ids.to(device)
            ngram_positions = all_ngram_positions.to(device)
        else:
            ngram_ids = None
            ngram_positions = None

        if self.pos_embedding is not None:
            all_pos_ids = torch.tensor([f.pos_ids for f in feature], dtype=torch.long)
            pos_ids = all_pos_ids.to(device)
        else:
            pos_ids = None

        word_ids = None

        return input_ids, input_mask, l_mask, eval_mask, head_idx, label_ids, \
               word_ids, pos_ids, \
               ngram_ids, ngram_positions, segment_ids, valid_ids


class InputExample(object):

    def __init__(self, guid, text_a, text_b=None, head=None, label=None, pos=None):

        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.head = head
        self.label = label
        self.pos = pos


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, head_idx, label_id, valid_ids=None,
                 label_mask=None, eval_mask=None,
                 word_ids=None, pos_ids=None,
                 ngram_ids=None, ngram_positions=None, ngram_lengths=None,
                 ngram_tuples=None, ngram_seg_ids=None, ngram_masks=None,
                 ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.head_idx = head_idx
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        self.eval_mask = eval_mask

        self.word_ids = word_ids
        self.pos_ids = pos_ids

        self.ngram_ids = ngram_ids
        self.ngram_positions = ngram_positions
        self.ngram_lengths = ngram_lengths
        self.ngram_tuples = ngram_tuples
        self.ngram_seg_ids = ngram_seg_ids
        self.ngram_masks = ngram_masks



def readfile(filename, flag):
    data = []
    sentence = []
    head = []
    label = []
    pos = []

    with open(filename, 'r', encoding='utf8') as f:
        lines = f.readlines()
        if not flag == 'predict':
            for line in lines:
                line = line.strip()
                if line == '':
                    if len(sentence) > 0:
                        data.append((sentence, head, label, pos))
                        sentence = []
                        head = []
                        label = []
                        pos = []
                    continue
                splits = line.split('\t')
                sentence.append(splits[1])
                pos.append(splits[3])
                head.append(int(splits[6]))
                label.append(splits[7])
            if len(sentence) > 0:
                data.append((sentence, head, label, pos))
        else:
            raise ValueError()
            # for line in lines:
            #     line = line.strip()
            #     if line == '':
            #         continue
            #     label_list = ['NN' for _ in range(len(line))]
            #     data.append((line, label_list))
    return data
