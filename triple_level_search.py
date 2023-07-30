import gc
import os
import re
import sys
import copy
import json
import random
import numpy as np

import torch
import torch.nn.functional as F

import warnings
from transformers import logging
logger = logging.get_logger(__name__)
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList

from prompter import Prompter



def init_toolkit(
    prompt_template='alpaca', # The prompt template to use, will default to alpaca.)
    temperature=1.0,
    top_p=1.0,
    top_k=50,
    num_beams=1,
    max_new_tokens=640,
    do_sample=False,
    **kwargs,
):
    tools = {}
    tools['prompt'] = Prompter(prompt_template)
    tools['gen_config'] = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True,
        bos_token_id=1, # begin token
        eos_token_id=2, # end token
        **kwargs,
        )
    return tools


token2id_dict = {
    '\n': 13,
    'Input': 10567,
    'Response': 13291,
    '),(': 21336,
    ',': 29892,
    ')': 29897,
    '(': 29898,
}


triple_start_id = [token2id_dict[ele] for ele in ['),(', '(']]
triple_end_id = [token2id_dict[ele] for ele in ['),(', ')']]
ent_rel_start_id = [token2id_dict[ele] for ele in ['),(', ',', '(']]

threshold1 = 0.6
threshold2 = 0.7
threshold3 = 0.5


def gen_step(
    self, tokenizer, input_ids,
    raw_input, relation_set,
    generation_config=None,
    logits_processor=None,
    stopping_criteria=None,
    prefix_allowed_tokens_fn=None,
    synced_gpus=None,
    assistant_model=None,
    streamer=None,
    prob_threshold=0.1,
    select_method='heuristic', # ['max', 'heuristic']
    triple_type='head-first',  # ['head-first', 'rel-first', 'tail-first']
    **kwargs,
):
    batch_size, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]

    if generation_config is None:
        generation_config = self.generation_config
    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)

    bos_token_id, eos_token_id = generation_config.bos_token_id, generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]

    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    if has_default_max_length and generation_config.max_new_tokens is None:
        warnings.warn(
            f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
            "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
            " recommend using `max_new_tokens` to control the maximum length of the generation.",
            UserWarning,
        )
    elif generation_config.max_new_tokens is not None:
        generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
        if not has_default_max_length:
            logger.warn(
                f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                "Please refer to the documentation for more information. "
                "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)",
                UserWarning,
            )

    if input_ids_seq_length >= generation_config.max_length:
        input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        logger.warning(
            f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
            f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
            " increasing `max_new_tokens`."
        )

    # 2. Set generation parameters if not already defined
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    logits_processor = self._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=input_ids,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
    )

    stopping_criteria = self._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria
    )
    logits_warper = self._get_logits_warper(generation_config)

    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    # scores = None

    if input_ids[0,-1].item() == token2id_dict['(']:
        triple_start = input_ids.size(1)
        init_prob_list = []
    else:
        bias = input_ids[0].tolist()[::-1].index(token2id_dict['('])
        triple_start = input_ids.size(1) - bias
        init_prob_list = [1.] * bias

    input_ids, prob_list, _, __, ___ = generate(
        self, tokenizer, input_ids, init_prob_list,
        raw_input, relation_set, triple_start,
        generation_config, model_kwargs,
        logits_processor, logits_warper,
        stopping_criteria, unfinished_sequences,
        prob_threshold=prob_threshold,
        select_method=select_method, triple_type=triple_type,
        gen_to_end=True, **kwargs,
    )

    return input_ids, prob_list


def locate_pos(h, r, t, raw_input, relation_set, MAX_NUM):
    pos_h = raw_input.find(h) if h in raw_input else MAX_NUM-1
    pos_r = relation_set.index(r) if r in relation_set else len(relation_set)
    pos_t = raw_input.find(t) if t in raw_input else MAX_NUM-1
    return pos_h, pos_r, pos_t


def generate(
    self, tokenizer, input_ids, prob_list,
    raw_input, relation_set, triple_start,
    generation_config, model_kwargs,
    logits_processor, logits_warper,
    stopping_criteria, unfinished_sequences,
    prob_threshold=0.1,
    select_method='heuristic', triple_type='head-first',
    gen_to_end=False,
    gen_to_hr_change=None,
    relation_prob={},
    **kwargs,
):
    test_end = False
    bos_token_id, eos_token_id = generation_config.bos_token_id, generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]

    # prob_list = []
    init_len = input_ids.size(1)
    while input_ids.size(1) - init_len < generation_config.max_new_tokens:

        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # forward pass to get next token
        with torch.no_grad():
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )

        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # 一些临时问题的解决方案
        next_token_scores[:, token2id_dict['(']] = torch.HalfTensor([-10000.])[0]
        if '）' not in raw_input:
            next_token_scores[:, 30409] = torch.HalfTensor([-10000.])[0]

        # sample
        probs = F.softmax(next_token_scores, dim=-1)

        candidate_tokens = torch.where(probs[0] > max(prob_threshold, 0.5 * probs.max().item()))[0]
        candidate_first_probs = probs[0, candidate_tokens].cpu().numpy()

        # 当前的可选择项为 结束 ')' 或继续生成下一个 triple '),('
        if len(candidate_tokens) > 1 and candidate_tokens.cpu().tolist() == triple_end_id: # ['),(', ')']
            if gen_to_hr_change is not None:
                assert triple_start-1 == torch.where((input_ids[0,:-1] == token2id_dict['),(']) | (input_ids[0,:-1] == token2id_dict['(']))[0][-1], \
                    (triple_start, torch.where((input_ids[0] == token2id_dict['),(']) | (input_ids[0] == token2id_dict['(']))[0])
                triple_len = input_ids.size(1) - triple_start
                triple_prob = np.prod(prob_list[-triple_len:])
                triple = tokenizer.decode(input_ids[0, -triple_len:], skip_special_tokens=True)

                if triple[:len(gen_to_hr_change)] != gen_to_hr_change: # 删除最后一个 triple 并返回
                    tmp = input_ids[0, :-1].tolist()[::-1] # 不考虑最后一个 token '),('
                    pos = 1 + min(tmp.index(token2id_dict['(']), tmp.index(token2id_dict['),('])) # +1 是因为上一行没有计算最后一个 token
                    input_ids = input_ids[:, :-pos]
                    prob_list = prob_list[:-pos]
                    break

            just_end = False
            if test_end: # 之前已经经历过 结束 或者 继续生成下一个 的选择，当前生成的 triple 是重复 triple，所以直接结束
                tmp = tokenizer.decode(input_ids[0, triple_start:-1], skip_special_tokens=True)
                complete_result = tokenizer.decode(input_ids[0, init_len:triple_start-1], skip_special_tokens=True)
                if tmp in complete_result:
                    just_end = True
                    next_tokens = candidate_tokens[1:] # 结束
                else:
                    next_tokens = candidate_tokens[:1] # 不结束
            else:
                next_tokens = candidate_tokens[:1] # 第一次出现这种选择，默认不结束

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            unfinished_sequences = unfinished_sequences.mul((sum(next_tokens != i for i in eos_token_id)).long())
            prob_list.append( probs[0, next_tokens[0]].item() )

            if not gen_to_end or just_end:
                break
            else:
                triple_start = input_ids.size(1)
                test_end = True # 在生成下一个 triple 后，如果是重复 triple，则强制结束
                continue        # 跳过下面内容

        if len(candidate_tokens) > 1 and token2id_dict['),('] not in candidate_tokens:
            exist_triples = tokenizer.decode(input_ids[0, init_len:], skip_special_tokens=True)
            print(f'  Start branch {input_ids.size(1)}', exist_triples)
            print('  candidate_tokens', [(a, b) for a, b in zip(
                tokenizer.decode(candidate_tokens), candidate_first_probs)])

            candidate_outs, candidate_probs, candidate_rets = [], [], []
            candidate_triples = []
            for __ in range(len(candidate_tokens)):
                next_tokens = candidate_tokens[__:__+1]
                next_token_prob = probs[0, next_tokens[0]].item()
                new_input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                new_model_kwargs = self._update_model_kwargs_for_generation(
                    outputs, copy.deepcopy(model_kwargs), is_encoder_decoder=self.config.is_encoder_decoder
                )
                new_unfinished_sequences = unfinished_sequences.mul((sum(next_tokens != i for i in eos_token_id)).long())

                tmp = input_ids[0].cpu().tolist()[::-1]
                if 1: # (token2id_dict['),('] in tmp and tmp.index(token2id_dict['),(']) < tmp.index(token2id_dict[','])) or tmp.index(token2id_dict['(']) < tmp.index(token2id_dict[',']): # 向前回溯时，先找到'),(' '('而不是','
                    ttt = copy.deepcopy(prob_list) + [probs[0, candidate_tokens].sum().item()]
                else:
                    ttt = copy.deepcopy(prob_list) + [next_token_prob]

                new_input_ids, new_prob_list, new_model_kwargs, new_unfinished_sequences, new_relation_prob = generate(
                    self, tokenizer, new_input_ids, ttt, # copy.deepcopy(prob_list) + [next_token_prob],
                    raw_input, relation_set, triple_start,
                    generation_config, new_model_kwargs,
                    logits_processor, logits_warper,
                    stopping_criteria, new_unfinished_sequences,
                    prob_threshold=prob_threshold,
                    gen_to_end=False, relation_prob=relation_prob,
                    **kwargs,
                )

                assert triple_start-1 == torch.where((new_input_ids[0,:-1] == token2id_dict['),(']) | (new_input_ids[0,:-1] == token2id_dict['(']))[0][-1]
                triple_len = new_input_ids.size(1) - triple_start
                branch_score = np.prod(new_prob_list[-triple_len:])
                branch_triple = tokenizer.decode(new_input_ids[0, -triple_len:-1], skip_special_tokens=True)
                print(f'  branch {input_ids.size(1)}', branch_score, branch_triple, probs[0, candidate_tokens].sum().item()) # , input_ids[0])

                if branch_triple in exist_triples:
                    print(f'WARNING: triple {branch_triple} exists in {exist_triples}')
                    branch_score = 0.

                candidate_outs.append(new_input_ids)
                candidate_probs.append(branch_score)
                candidate_rets.append((new_prob_list, new_model_kwargs, new_unfinished_sequences, new_relation_prob))

                tmp = branch_triple
                if len(re.findall(',', tmp)) == 2:
                    candidate_triples.append(tmp.split(','))
                else:
                    new_triple = None
                    for rel in relation_set:
                        if rel in tmp:
                            if triple_type == 'head-first':
                                h, t = tmp.split(f',{rel},')
                                new_triple = [h, rel, t]
                            elif triple_type == 'tail-first':
                                t, h = tmp.split(f',{rel},')
                                new_triple = [t, rel, h]
                            elif triple_type == 'rel-first':
                                assert tmp[:len(rel)+1] == f'{rel},'
                                tmp = tmp[len(rel)+1:]
                                parts = tmp.split(',')
                                split_point = 1
                                while not (','.join(parts[:split_point]) in raw_input
                                       and ','.join(parts[split_point:]) in raw_input):
                                    split_point += 1
                                h = ','.join(parts[:split_point])
                                t = ','.join(parts[split_point:])
                                new_triple = [rel, h, t]
                            elif triple_type == 'rel-last':
                                assert tmp[-len(rel)-1:] == f',{rel}'
                                tmp = tmp[:-len(rel)-1]
                                parts = tmp.split(',')
                                split_point = 1
                                while not (','.join(parts[:split_point]) in raw_input
                                       and ','.join(parts[split_point:]) in raw_input):
                                    split_point += 1
                                h = ','.join(parts[:split_point])
                                t = ','.join(parts[split_point:])
                                new_triple = [h, t, rel]
                            else:
                                assert 0
                            break
                    if new_triple:
                        candidate_triples.append(new_triple)
                    else:
                        print(f'Warning, rel {branch_triple} not given, set prob to 0')
                        candidate_probs[-1] = 0.
                        candidate_triples.append(branch_triple.split(',')[:3])

            # 1. 分数过低（小于最高值的某一阈值50%），舍弃
            # 2. 因为生成顺序总是和 relation_set 顺序一致，所以优先选择同种 relation（TODO 这里也要加一个舍弃阈值）
            # 3. 如果剩下的 relation 都一致了，那么优先选择 head 的出现位置在句子中比较早的（TODO 这里也要加一个舍弃阈值，这些阈值以triple_search的超参数形式存在）
            # 4. 多个 tail 根据分数
            if select_method == 'heuristic':
                candidate_pos = []
                threshold1 = 0.8 # 在加入了 tail-first-model 之后，这里应该鼓励其生成尽可能多的关系
                prob_threshold = threshold1 * max(candidate_probs)
                MAX_NUM = len(raw_input) # generation_config.max_new_tokens

                for triple, prob in zip(candidate_triples, candidate_probs):
                    # 1. 分数过低（小于最高值的某一阈值50%），舍弃
                    if prob < prob_threshold:
                        # candidate_pos.append((len(relation_set), MAX_NUM-1, 0))
                        candidate_pos.append((len(relation_set), MAX_NUM-1, MAX_NUM-1, 0))
                    # 根据 rel 的给定顺序和 h/t 在句子中的出现次序，选择靠前的（‘位于‘除外，直接选择 max）；对于三元组中的最后一项，如果是h或t的话，不再根据位置，而是根据 prob
                    else:
                        if triple_type == 'head-first':
                            h, rel, t = triple
                            pos_h, pos_r, pos_t = locate_pos(h, rel, t, raw_input, relation_set, MAX_NUM)
                            # candidate_pos.append((pos_r, pos_h, prob))
                            # candidate_pos.append((pos_r, pos_h+len(h), prob))
                            candidate_pos.append((pos_r, pos_h, pos_t, prob))
                        elif triple_type == 'tail-first':
                            t, rel, h = triple
                            pos_h, pos_r, pos_t = locate_pos(h, rel, t, raw_input, relation_set, MAX_NUM)
                            # candidate_pos.append((pos_r, pos_t, prob))
                            candidate_pos.append((pos_r, pos_t, pos_h, prob))
                        elif triple_type == 'rel-first':
                            rel, h, t = triple
                            pos_h, pos_r, pos_t = locate_pos(h, rel, t, raw_input, relation_set, MAX_NUM)
                            # candidate_pos.append((pos_r, pos_h, prob))
                            candidate_pos.append((pos_r, pos_h, pos_t, prob))
                        elif triple_type == 'rel-last':
                            h, t, rel = triple
                            pos_h, pos_r, pos_t = locate_pos(h, rel, t, raw_input, relation_set, MAX_NUM)
                            # candidate_pos.append((pos_r, pos_h, prob))
                            candidate_pos.append((pos_r, pos_h, pos_t, prob))
                        else:
                            assert 0
                        if rel == '位于':
                            if triple_type == 'tail-first': # '位于' 直接选择 prob 最高的
                                candidate_pos[-1] = (pos_r, 0, 0, prob)
                            else: # '位于' 选择 位置靠后的
                                rel_pos, h_pos, t_pos, prob = candidate_pos[-1]
                                candidate_pos[-1] = (rel_pos, MAX_NUM-h_pos, MAX_NUM-t_pos, prob)
                candidate_pos = np.array(candidate_pos)
                # 因为下一行是 argmin，所以用了 1-prob 而不是 prob
                # candidate_pos = candidate_pos[:,0] * MAX_NUM + candidate_pos[:,1] + 1-candidate_pos[:,2]
                candidate_pos = (candidate_pos[:,0] * MAX_NUM + candidate_pos[:,1]) * MAX_NUM + candidate_pos[:,2] + 1-candidate_pos[:,3]
                select_idx = candidate_pos.argmin()
            elif select_method == 'max':
                select_idx = np.array(candidate_probs).argmax()
            else:
                assert 0

            input_ids = candidate_outs[select_idx]
            prob_list, model_kwargs, unfinished_sequences, relation_prob = candidate_rets[select_idx]

            del new_input_ids, new_model_kwargs, new_unfinished_sequences, new_prob_list
            del candidate_outs, candidate_probs, candidate_rets
            gc.collect()
            torch.cuda.empty_cache()

            # triple_len = input_ids.size(1) - triple_start
            # print('\tSelect', tokenizer.decode(input_ids[0, triple_start:], skip_special_tokens=True), input_ids[0, triple_start:], triple_len, prob_list[-triple_len:])

            assert triple_start-1 == torch.where((input_ids[0,:-1] == token2id_dict['),(']) | (input_ids[0,:-1] == token2id_dict['(']))[0][-1], \
                (triple_start, torch.where((input_ids[0] == token2id_dict['),(']) | (input_ids[0] == token2id_dict['(']))[0])
            triple_len = input_ids.size(1) - triple_start
            triple_prob = np.prod(prob_list[-triple_len:])
            triple = tokenizer.decode(input_ids[0, -triple_len:], skip_special_tokens=True)
            if gen_to_end:
                print(triple_prob, 'SELECT', triple, input_ids[0, -triple_len:], triple_len, prob_list[-triple_len:])

            if gen_to_hr_change is not None:
                if triple[:len(gen_to_hr_change)] != gen_to_hr_change: # 删除最后一个 triple 并返回
                    tmp = input_ids[0, :-1].tolist()[::-1] # 不考虑最后一个 token '),('
                    pos = 1 + min(tmp.index(token2id_dict['(']), tmp.index(token2id_dict['),('])) # +1 是因为上一行没有计算最后一个 token
                    input_ids = input_ids[:, :-pos]
                    prob_list = prob_list[:-pos]
                    break

            if len(re.findall(',', triple)) == 2:
                rel = triple.split(',')[1]
            else:
                for rel in relation_set:
                    if rel in triple:
                        break
            relation_prob[rel] = triple_prob

            triple_start = input_ids.size(1)
            test_end = False # 当生成了一个正常triple后，重置 test_end

            # stop when each sentence is finished, or if we exceed the maximum length
            # if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, None):
                break

            assert input_ids[0, -1].item() in [token2id_dict[ele] for ele in ['),(', ')']]
            if not gen_to_end: # 嵌套情况
                break
        else:
            if generation_config.do_sample:
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(probs, dim=-1)

            if input_ids[0, -1].item() in ent_rel_start_id: # ['),(', ',', '(']
                current_character_prob = probs[0, next_tokens.item()].item()

            # update generated ids, model inputs, and length for next step

            # 一些特殊符号的临时解决措施
            if next_tokens.item() == 9774: # replace `.),` with `.`
                next_tokens[:1] = 29889
            elif next_tokens.item() == 1159: # replace `")` with `"`
                next_tokens[:1] = 29908
            elif next_tokens.item() == 1846: # replace `.)` with `.`
                next_tokens[:1] = 29889
            elif next_tokens.item() == 28135: # replace `+)` with `+`
                next_tokens[:1] = 29974
            elif next_tokens.item() == 1723: # replace ` )` with `)`
                next_tokens[:1] = token2id_dict[')']
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            unfinished_sequences = unfinished_sequences.mul((sum(next_tokens != i for i in eos_token_id)).long())

            if gen_to_hr_change is not None and next_tokens[0].item() == token2id_dict[','] \
                and input_ids[0, triple_start:].tolist().count(token2id_dict[',']) > 2:
                triple = tokenizer.decode(input_ids[0, triple_start:], skip_special_tokens=True)

                for rel in relation_set:
                    if rel in triple:
                        break

                if rel in triple:
                    entity0, entity1 = triple.split(f',{rel},')
                    if not (entity1 == '' or entity1 in raw_input):
                        print(f'Warning: generate {triple} and force to stop')
                        input_ids[0, -1] = token2id_dict['),(']
                        prob_list.append(1.)

                        triple_len = input_ids.size(1) - triple_start
                        triple_prob = np.prod(prob_list[-triple_len:])
                        triple = tokenizer.decode(input_ids[0, -triple_len:], skip_special_tokens=True)
                        if triple[:len(gen_to_hr_change)] != gen_to_hr_change:
                            tmp = input_ids[0, :-1].tolist()[::-1] # 不考虑最后一个 token '),('
                            pos = 1 + min(tmp.index(token2id_dict['(']), tmp.index(token2id_dict['),('])) # +1 是因为上一行没有计算最后一个 token
                            input_ids = input_ids[:, :-pos]
                            prob_list = prob_list[:-pos]

                        break

            prob_list.append( probs[0, next_tokens[0]].item() )
            # print(tokenizer.decode(input_ids[0, -1:], skip_special_tokens=True), input_ids[0, -1], prob_list[-1])

            # stop when each sentence is finished, or if we exceed the maximum length
            # if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, None):
                break

            # stop when finishing generating a triple
            if not gen_to_end and next_tokens.item() in triple_end_id:
                break

            if next_tokens.item() in triple_end_id:
                assert triple_start-1 == torch.where((input_ids[0,:-1] == token2id_dict['),(']) | (input_ids[0,:-1] == token2id_dict['(']))[0][-1], \
                    (triple_start, torch.where((input_ids[0] == token2id_dict['),(']) | (input_ids[0] == token2id_dict['(']))[0], input_ids[0])
                triple_len = input_ids.size(1) - triple_start
                triple_prob = np.prod(prob_list[-triple_len:])
                triple = tokenizer.decode(input_ids[0, -triple_len:], skip_special_tokens=True)

                if gen_to_hr_change is not None:
                    if triple[:len(gen_to_hr_change)] != gen_to_hr_change: # 删除最后一个 triple 并返回
                        tmp = input_ids[0, :-1].tolist()[::-1] # 不考虑最后一个 token '),('
                        pos = 1 + min(tmp.index(token2id_dict['(']), tmp.index(token2id_dict['),('])) # +1 是因为上一行没有计算最后一个 token
                        input_ids = input_ids[:, :-pos]
                        prob_list = prob_list[:-pos]
                        break

                if len(re.findall(',', triple)) == 2:
                    rel = triple.split(',')[1]
                else:
                    for rel in relation_set:
                        if rel in triple:
                            break

                print(triple_prob, triple, input_ids[0, triple_start:], triple_len, prob_list[-triple_len:])

                relation_prob[rel] = triple_prob

                triple_start = input_ids.size(1)
                test_end = False # 当生成了一个正常triple后，重置 test_end

    return input_ids, prob_list, model_kwargs, unfinished_sequences, relation_prob


def generate_given_triples(
    self, tokenizer, input_ids,
    raw_input, relation_set, h_rel_pairs,
    generation_config=None,
    logits_processor=None,
    stopping_criteria=None,
    prefix_allowed_tokens_fn=None,
    synced_gpus=None,
    assistant_model=None,
    streamer=None,
    prob_threshold=0.1,
    triple_type='head-first',
    select_method='max',
    **kwargs,
):
    batch_size, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]

    if generation_config is None:
        generation_config = self.generation_config
    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)

    bos_token_id, eos_token_id = generation_config.bos_token_id, generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]

    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    if has_default_max_length and generation_config.max_new_tokens is None:
        warnings.warn(
            f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
            "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
            " recommend using `max_new_tokens` to control the maximum length of the generation.",
            UserWarning,
        )
    elif generation_config.max_new_tokens is not None:
        generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
        if not has_default_max_length:
            logger.warn(
                f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                "Please refer to the documentation for more information. "
                "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)",
                UserWarning,
            )

    if input_ids_seq_length >= generation_config.max_length:
        input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        logger.warning(
            f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
            f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
            " increasing `max_new_tokens`."
        )

    # 2. Set generation parameters if not already defined
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    logits_processor = self._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=input_ids,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
    )

    stopping_criteria = self._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria
    )
    logits_warper = self._get_logits_warper(generation_config)

    # unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    # scores = None

    triple_start = input_ids.size(1)

    result = []
    for h, r, ts in h_rel_pairs:
        hr = f'{h},{r},'
        add_ids = tokenizer(hr)['input_ids']
        assert add_ids[0] == 1
        add_ids = add_ids[1:]
        if add_ids[0] == 29871: # 空格
            add_ids = add_ids[1:] # 第一个字是中文会有空格，否则没有

        add_ids = torch.LongTensor(add_ids).unsqueeze(0).to(device=input_ids.device)
        print('\n\nStart with', hr)

        new_input_ids = torch.cat([input_ids, add_ids], dim=-1)
        unfinished_sequences = new_input_ids.new(new_input_ids.shape[0]).fill_(1)

        init_prob_list = [1.] * add_ids.size(1)

        out_input_ids, prob_list, _, __, ___ = generate(
            self, tokenizer, new_input_ids, init_prob_list,
            raw_input, relation_set, triple_start,
            generation_config, model_kwargs,
            logits_processor, logits_warper,
            stopping_criteria, unfinished_sequences,
            prob_threshold=prob_threshold,
            triple_type=triple_type,
            select_method=select_method,
            gen_to_end=True, gen_to_hr_change=hr[:-1], **kwargs,
        )

        if out_input_ids[0,-1].item() == 2:
            assert out_input_ids[0,-2].item() == token2id_dict[')']
            a, b = out_input_ids[0, new_input_ids.size(1):-1], prob_list[add_ids.size(1):-1]
        elif out_input_ids[0,-1].item() == token2id_dict['),(']:
            a, b = out_input_ids[0, new_input_ids.size(1):], prob_list[add_ids.size(1):]
        else: # 未中止的生成序列
            tmp = out_input_ids[0].tolist()[::-1]
            pos = min(tmp.index(token2id_dict['(']), tmp.index(token2id_dict['),(']))
            a, b = out_input_ids[0, new_input_ids.size(1):-pos], prob_list[add_ids.size(1):-pos]

        print(hr+tokenizer.decode(a, skip_special_tokens=True), np.prod(b))
        print(b)
        if r == '别名' and len(b) == 0:
            print('Trick 0')
            result.append((0., ','.join([h, r, ts])+'),('))
        else:
            result.append((np.prod(b), hr+tokenizer.decode(a, skip_special_tokens=True)))

    return result


def generate_given_triples2(
    self, tokenizer, input_ids, tools,
    raw_input, relation_set, h_rel_pairs,
    generation_config=None,
    logits_processor=None,
    stopping_criteria=None,
    prefix_allowed_tokens_fn=None,
    synced_gpus=None,
    assistant_model=None,
    streamer=None,
    prob_threshold=0.1,
    triple_type='head-first',
    select_method='max',
    **kwargs,
):
    batch_size, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]
    device = input_ids.device

    if generation_config is None:
        generation_config = self.generation_config
    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)

    bos_token_id, eos_token_id = generation_config.bos_token_id, generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]

    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    if has_default_max_length and generation_config.max_new_tokens is None:
        warnings.warn(
            f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
            "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
            " recommend using `max_new_tokens` to control the maximum length of the generation.",
            UserWarning,
        )
    elif generation_config.max_new_tokens is not None:
        generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
        if not has_default_max_length:
            logger.warn(
                f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                "Please refer to the documentation for more information. "
                "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)",
                UserWarning,
            )

    if input_ids_seq_length >= generation_config.max_length:
        input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        logger.warning(
            f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
            f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
            " increasing `max_new_tokens`."
        )

    # 2. Set generation parameters if not already defined
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    logits_processor = self._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=input_ids,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
    )

    stopping_criteria = self._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria
    )
    logits_warper = self._get_logits_warper(generation_config)

    # unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    # scores = None

    triple_start = input_ids.size(1)

    result = []
    for h, r, ts in h_rel_pairs:
        new_relation_set = [rel for rel in relation_set if rel != r]
        random.shuffle(new_relation_set)
        new_relation_set = [r] + new_relation_set

        assert triple_type in ['head-first', 'tail-first']
        if triple_type == 'head-first':
            # triple_format = '"(头实体,关系,尾实体)"'
            triple_format = '(Subject,Relation,Object)'
        elif triple_type == 'tail-first':
            triple_format = '(Object,Relation,Subject)'
        else:
            assert 0
        instruction = f'已知候选的关系列表：{new_relation_set}，请你根据关系列表，' \
                    + f'从以下输入中抽取出可能存在的头实体(Subject)与尾实体(Object)，并给出对应的关系三元组。' \
                    + f'请按照 {triple_format} 的格式回答。'

        prompt = tools['prompt'].generate_prompt(instruction, raw_input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        tmp = '输入中包含的关系三元组是：\n('
        tmp = tokenizer(tmp, return_tensors="pt")
        tmp = tmp['input_ids']

        assert tmp[0, :2].tolist() == [1, 29871] # 表示开始的 token，表示空格的 token
        input_ids = torch.cat((input_ids, tmp[:, 2:].to(device=input_ids.device)), dim=1)

        hr = f'{h},{r},'
        add_ids = tokenizer(hr)['input_ids']
        assert add_ids[0] == 1
        add_ids = add_ids[1:]
        if add_ids[0] == 29871: # 空格
            add_ids = add_ids[1:] # 第一个字是中文会有空格，否则没有

        add_ids = torch.LongTensor(add_ids).unsqueeze(0).to(device=input_ids.device)
        print('\n\nStart with', hr)

        new_input_ids = torch.cat([input_ids, add_ids], dim=-1)
        unfinished_sequences = new_input_ids.new(new_input_ids.shape[0]).fill_(1)

        init_prob_list = [1.] * add_ids.size(1)

        out_input_ids, prob_list, _, __, ___ = generate(
            self, tokenizer, new_input_ids, init_prob_list,
            raw_input, relation_set, triple_start,
            generation_config, model_kwargs,
            logits_processor, logits_warper,
            stopping_criteria, unfinished_sequences,
            prob_threshold=prob_threshold,
            triple_type=triple_type,
            select_method=select_method,
            gen_to_end=True, gen_to_hr_change=hr[:-1], **kwargs,
        )

        if out_input_ids[0,-1].item() == 2:
            assert out_input_ids[0,-2].item() == token2id_dict[')']
            a, b = out_input_ids[0, new_input_ids.size(1):-1], prob_list[add_ids.size(1):-1]
        elif out_input_ids[0,-1].item() == token2id_dict['),(']:
            a, b = out_input_ids[0, new_input_ids.size(1):], prob_list[add_ids.size(1):]
        else: # 未中止的生成序列
            tmp = out_input_ids[0].tolist()[::-1]
            pos = min(tmp.index(token2id_dict['(']), tmp.index(token2id_dict['),(']))
            a, b = out_input_ids[0, new_input_ids.size(1):-pos], prob_list[add_ids.size(1):-pos]

        print(hr+tokenizer.decode(a, skip_special_tokens=True), np.prod(b))
        print(b)
        if r == '别名' and len(b) == 0:
            print('Trick 0')
            result.append((0., ','.join([h, r, ts])+'),('))
        else:
            result.append((np.prod(b), hr+tokenizer.decode(a, skip_special_tokens=True)))

    return result