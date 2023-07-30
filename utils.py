import json
import random

import torch
import numpy as np


from itertools import groupby
def split_list(lst, number):
    return [sublist for sublist in [list(group) for key, group in groupby(lst, lambda x: x == number) if not key] if sublist]


token2id_dict = {
    '\n': 13,
    'Input': 10567,
    'Response': 13291,
    '),(': 21336,
    ',': 29892,
    ')': 29897,
    '(': 29898,
}


def load_data(input_file='data/valid1.json'):
    records = []
    with open(input_file, "r") as reader:
        for line in reader:
            data = json.loads(line)
            records.append(data)
    return records


def gen_inputs(prompt_generator, tokenizer, input, relation_set, mode='head-first', prefix=''):
    assert mode in ['head-first', 'tail-first']
    if mode == 'head-first':
        # triple_format = '"(头实体,关系,尾实体)"'
        triple_format = '(Subject,Relation,Object)'
    elif mode == 'tail-first':
        triple_format = '(Object,Relation,Subject)'
    else:
        assert 0
    instruction = f'已知候选的关系列表：{relation_set}，请你根据关系列表，' \
                + f'从以下输入中抽取出可能存在的头实体(Subject)与尾实体(Object)，并给出对应的关系三元组。' \
                + f'请按照 {triple_format} 的格式回答。'

    prompt = prompt_generator.generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    tmp = '输入中包含的关系三元组是：\n(' + prefix
    tmp = tokenizer(tmp, return_tensors="pt")
    tmp = tmp['input_ids']

    assert tmp[0, :2].tolist() == [1, 29871] # 表示开始的 token，表示空格的 token
    input_ids = torch.cat((input_ids, tmp[:, 2:]), dim=1)
    return input_ids


def split_triple_probs(tokenizer, input, relation_set, input_ids, output_ids, output_prob_list, pooling='prod'):
    assert pooling in ['mean', 'prod']

    if input_ids[-1] == token2id_dict['(']:
        triple_start, prefix_len = len(input_ids), 0
    else: # gen_inputs 时包含 prefix的情况
        prefix_len = input_ids.tolist()[::-1].index(token2id_dict['('])
        triple_start = len(input_ids) - prefix_len
    raw_output_ids = output_ids[triple_start:]
    assert np.all(output_prob_list[:prefix_len] == 1.)
    output_ids = raw_output_ids[prefix_len:]
    output_prob_list = output_prob_list[prefix_len:]

    # 处理意外终止的 output_ids（比如，超过最大生成长度，或者 triple 并不符合 (ent,rel,ent) 形式）
    # 1. 正常结束，则删除最后的 中止token 2
    if output_ids[-1] == 2:
        assert output_ids[-2] == token2id_dict[')']
        output_ids = output_ids[:-1]
        output_prob_list = output_prob_list[:-1]
    # 2. 用于 generate_given_triples ，因为限定了 (head,rel)，当生成其他 head_rel 对时自动中止，所以会以 ),( 结束
    elif output_ids[-1] == token2id_dict['),(']:
        pass
    elif token2id_dict['),('] not in output_ids:
        return None
    else: # 未中止的生成序列，那么截断到最后出现的 ),(
        tmp = output_ids.tolist()[::-1]
        pos = tmp.index(token2id_dict['),('])
        output_ids, prob_list = output_ids[:-pos], prob_list[:-pos]

    # output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    # prefix_text = tokenizer.decode(raw_output_ids[:prefix_len], skip_special_tokens=True)

    assert output_ids[-1].item() in [token2id_dict['),('], token2id_dict[')']]

    split_points = np.where(output_ids == token2id_dict['),('])[0] + 1
    out = []
    for chunk_idx, (id_chunk, prob_chunk) in enumerate(zip(np.split(output_ids, split_points), np.split(output_prob_list, split_points))):
        if chunk_idx == 0:
            id_chunk = np.concatenate((raw_output_ids[:prefix_len], id_chunk))
        triple = tokenizer.decode(id_chunk[:-1], skip_special_tokens=True)

        if triple.count(',') == 2:
            h, rel, t = triple.split(',')
            if rel not in relation_set:
                print(f'WARNING: triple {triple} has no matched relations')
                continue
            if rel == '位于' and f'{h}{t}' in input:
                print(f'INFO: ({h},位于,{t}) -> ({t},位于,{h})')
                h, t = t, h
        else:
            out_triple = None
            for rel in relation_set:
                if f',{rel},' in triple:
                    h, t = triple.split(f',{rel},')
                    if h == t:
                        continue
                    if rel == '位于' and f'{h}{t}' in input:
                        print(f'INFO: ({h},位于,{t}) -> ({t},位于,{h})')
                        h, t = t, h
                    out_triple = (h, rel, t)
                    break
            if out_triple is None:
                continue
            h, rel, t = out_triple
        out_triple = (h, rel, t)

        if pooling == 'mean':
            score = np.mean(prob_chunk)
        elif pooling == 'prod':
            score = np.prod(prob_chunk)
        else:
            assert 0
        print(out_triple, score)
        out.append((score, out_triple))

    return out


def kg2output(kg):
    return  '(' + '),('.join([','.join(ele) for ele in kg]) + ')'


def triple2rel(triple, relation_set, mode):
    if triple.count(',') == 2:
        h, rel, t = triple.split(',')
    else:
        rel = None
        for candidate in relation_set:
            if mode == 'rel-first' and triple[:len(candidate)+1] == f'{candidate},':
                rel = candidate
                break
            elif mode == 'rel-last' and triple[-len(candidate)-1:] == f',{candidate}':
                rel = candidate
                break
            elif f',{candidate},' in triple:
                rel = candidate
                break
    return rel


def exist_head_result(head_result, triple):
    h, rel, t = triple
    return h in head_result and rel in head_result[h] and t in head_result[h][rel]


def exist_tail_result(tail_result, triple):
    h, rel, t = triple
    return t in tail_result and rel in tail_result[t] and h in tail_result[t][rel]


def exist_rel_result(rel_result, triple):
    h, rel, t = triple
    return rel in rel_result and h in rel_result[rel] and t in rel_result[rel][h]


def update_result(triple, head_result, tail_result, rel_result, rel_last_result=None):
    h, rel, t, prob = triple

    if h not in head_result:
        head_result[h] = {'rel_count': 0, 't_count': 0, 't_visit': 0}
    if rel not in head_result[h]:
        head_result[h][rel] = {}
        head_result[h]['rel_count'] += 1
    if t not in head_result[h][rel]: # 因为 new_result 是按从大到小排序，所以如果某个 (h,rel,t) 在 new_result 中多次出现，最开始的一定是 prob 最高的
        head_result[h][rel][t] = []
        head_result[h]['t_count'] += 1
    head_result[h][rel][t].append(prob)
    head_result[h]['t_visit'] += 1

    if rel not in rel_result:
        rel_result[rel] = {}
    if h not in rel_result[rel]:
        rel_result[rel][h] = {'t_count': 0, 't_visit': 0}
    if t not in rel_result[rel][h]:
        rel_result[rel][h][t] = []
        rel_result[rel][h]['t_count'] += 1
    rel_result[rel][h][t].append(prob)
    rel_result[rel][h]['t_visit'] += 1

    if t not in tail_result:
        tail_result[t] = {'rel_count': 0, 'h_count': 0, 'h_visit': 0}
    if rel not in tail_result[t]:
        tail_result[t][rel] = {}
        tail_result[t]['rel_count'] += 1
    if h not in tail_result[t][rel]:
        tail_result[t][rel][h] = []
        tail_result[t]['h_count'] += 1
    tail_result[t][rel][h].append(prob)
    tail_result[t]['h_visit'] += 1

    if rel_last_result is not None:
        if h not in rel_last_result:
            rel_last_result[h] = {'t_count': 0, 'rel_count': 0, 'rel_visit': 0}
        if t not in rel_last_result[h]:
            rel_last_result[h][t] = {}
            rel_last_result[h]['t_count'] += 1
        if rel not in rel_last_result[h][t]:
            rel_last_result[h][t][rel] = []
            rel_last_result[h]['rel_count'] += 1
        rel_last_result[h][t][rel].append(prob)
        rel_last_result[h]['rel_visit'] += 1


def remove_result(triple, head_result, tail_result, rel_result, rel_last_result=None):
    h, rel, t = triple

    head_result[h]['t_count'] -= 1
    head_result[h]['t_visit'] -= len(head_result[h][rel][t])
    del head_result[h][rel][t]
    if len(head_result[h][rel]) == 0:
        head_result[h]['rel_count'] -= 1
        del head_result[h][rel]
        if head_result[h]['rel_count'] == 0:
            assert head_result[h]['t_count'] == 0
            assert head_result[h]['t_visit'] == 0
            assert len(head_result[h]) == 3
            del head_result[h]

    tail_result[t]['h_count'] -= 1
    tail_result[t]['h_visit'] -= len(tail_result[t][rel][h])
    del tail_result[t][rel][h]
    # 如果此时 tail_result[t][rel] 为空，则删除
    if len(tail_result[t][rel]) == 0:
        tail_result[t]['rel_count'] -= 1
        del tail_result[t][rel]
        # 如果此时 tail_result[t] 为空，则删除
        if tail_result[t]['rel_count'] == 0:
            assert tail_result[t]['h_count'] == 0
            assert tail_result[t]['h_visit'] == 0
            assert len(tail_result[t]) == 3
            del tail_result[t]

    rel_result[rel][h]['t_visit'] -= len(rel_result[rel][h][t])
    rel_result[rel][h]['t_count'] -= 1
    del rel_result[rel][h][t]
    if rel_result[rel][h]['t_count'] == 0:
        assert rel_result[rel][h]['t_visit'] == 0
        assert len(rel_result[rel][h]) == 2
        del rel_result[rel][h]
        if len(rel_result[rel]) == 0:
            del rel_result[rel]

    if rel_last_result is not None:
        rel_last_result[h]['rel_visit'] -= len(rel_last_result[h][t][rel])
        rel_last_result[h]['rel_count'] -= 1
        del rel_last_result[h][t][rel]
        if len(rel_last_result[h][t]) == 0:
            del rel_last_result[h][t]
            rel_last_result[h]['t_count'] -= 1
            if rel_last_result[h]['t_count'] == 0:
                assert rel_last_result[h]['rel_count'] == 0
                assert rel_last_result[h]['rel_visit'] == 0
                assert len(rel_last_result[h]) == 3
                del rel_last_result[h]


def sort_dict(result_dict):
    for h, h_dict in result_dict.items():
        if 'count' in h or 'visit' in h or 'value' in h: continue
        for rel, rel_dict in h_dict.items():
            if 'count' in rel or 'visit' in rel or 'value' in rel: continue
            result_dict[h][rel] = {k: v for k, v in sorted([(_k, _v) for _k, _v in rel_dict.items() if 'count' not in _k and 'visit' not in _k and 'value' not in _k], key=lambda x: -max(x[1]))}
            for k in rel_dict.keys():
                if k not in result_dict[h][rel]:
                    result_dict[h][rel][k] = rel_dict[k]
            result_dict[h][rel]['max_value'] = max(result_dict[h][rel][list(result_dict[h][rel].keys())[0]])
        result_dict[h] = {k: v for k, v in sorted([(_k, _v) for _k, _v in result_dict[h].items() if 'count' not in _k and 'visit' not in _k and 'value' not in _k], key=lambda x: -x[1]['max_value'])}
        for k in h_dict.keys():
            if k not in result_dict[h]:
                result_dict[h][k] = h_dict[k]
        for rel, rel_dict in result_dict[h].items():
            if 'count' in rel or 'visit' in rel or 'value' in rel: continue
            del result_dict[h][rel]['max_value']


def construct_result_from_hfm(idx, new_data, start, end, input, relation_set, drop_short_entity=True): # hfm: head_first_model
    head_result_from_hfm, rel_result_from_hfm, tail_result_from_hfm, local_merge_map = {}, {}, {}, {}
    removed_triples = []

    max_score = -1
    max_data = None # head_first_model 进行推断时，通过 shuffle 得到了多个结果。这里 max_data 存放其中得分最高的一个结果
    for i in range(start, end):
        assert new_data[i][0] == idx
        _, relations, output_text, prob_list, output_pairs = new_data[i]

        tmp = []
        new_output_pairs = []
        for prob, triple in output_pairs:
            if triple not in tmp:
                tmp.append(triple)
                new_output_pairs.append((prob, triple))
        output_pairs = new_output_pairs

        # sentence_score = np.prod([ele[0] for ele in output_pairs]) ** (1. / len(output_pairs))
        sentence_score = np.mean([ele[0] for ele in output_pairs])
        if sentence_score > max_score:
            max_score = sentence_score
            max_data = [ele[1] for ele in output_pairs]

        for prob, triple in output_pairs:
            h, rel, t = triple
            if h[0] == '“' and h[-1] == '”':
                h = h[1:-1]
            if t[0] == '“' and t[-1] == '”':
                t = t[1:-1]

            if h not in input:
                print(f'WARNING: drop {(h, rel, t)} as head {h} not exists in raw input')
                continue
            if h == t:
                print(f'WARNING: drop {(h, rel, t)} as the same head and tail')
                continue

            # for 对称 relation
            if rel in ['临近', '兄弟姊妹', '配偶', '接壤']:
                if t not in input:
                    print(f'WARNING: drop {(h, rel, t)} as head {t} not exists in raw input for relation {rel}')
                    removed_triples.append((h, rel, t))
                    continue
                if input.index(h) == input.index(t): # 不接受类似 (北京市动物园,位于,北京市)
                    continue
                if input.index(h) > input.index(t):
                    h, t = t, h
            if rel in ['位于', '所在行政领土'] and f'{h}{t}' in input:
                h, t = t, h

            # 自包含的不可能是别名
            if rel in ['别名', '学名']:
                if input.startswith(t):
                    print(f'WARNING: drop {triple} as raw input start with {t}')
                    removed_triples.append((h, rel, t))
                    continue
                if (h not in input) or (t not in input) or (input.count(h) == input.count(t) and ((h in t) or (t in h))):
                    print(f'WARNING: drop {triple} as this word only occurs once in input and cannot be 别名')
                    removed_triples.append((h, rel, t))
                    continue

            if '时间' not in rel and t not in input: # '时间'的话可能会有'同年7月'这样的词
                print(f'WARNING: drop {triple} as the tail {t} not in input')
                removed_triples.append((h, rel, t))
                continue

            if rel in ['临近', '接壤']:
                if f'{h}{t}' in input and (exist_head_result(head_result_from_hfm, (t, '位于', h)) or exist_head_result(head_result_from_hfm, (t, '所在行政领土', h))):
                    print(f'WARNING: drop {triple} as {(t, "位于", h)} exists')
                    removed_triples.append((h, rel, t))
                    continue
                if f'{t}{h}' in input and (exist_head_result(head_result_from_hfm, (h, '位于', t)) or exist_head_result(head_result_from_hfm, (h, '所在行政领土', t))):
                    print(f'WARNING: drop {triple} as {(h, "位于", t)} exists')
                    removed_triples.append((h, rel, t))
                    continue
            if rel in ['位于', '所在行政领土'] and f'{t}{h}' in input:
                if exist_head_result(head_result_from_hfm, (h, '临近', t)):
                    print(f'WARNING: drop {(h, "临近", t)} as {triple} exists')
                    remove_result((h, '临近', t), head_result_from_hfm, tail_result_from_hfm, rel_result_from_hfm)
                    removed_triples.append((h, '临近', t))
                if exist_head_result(head_result_from_hfm, (h, '接壤', t)):
                    print(f'WARNING: drop {(h, "接壤", t)} as {triple} exists')
                    remove_result((h, '接壤', t), head_result_from_hfm, tail_result_from_hfm, rel_result_from_hfm)
                    removed_triples.append((h, '接壤', t))

                if exist_head_result(head_result_from_hfm, (t, '临近', h)):
                    print(f'WARNING: drop {(t, "临近", h)} as {triple} exists')
                    remove_result((t, '临近', h), head_result_from_hfm, tail_result_from_hfm, rel_result_from_hfm)
                    removed_triples.append((t, '临近', h))
                if  exist_head_result(head_result_from_hfm, (t, '接壤', h)):
                    print(f'WARNING: drop {(t, "接壤", h)} as {triple} exists')
                    remove_result((t, '接壤', h), head_result_from_hfm, tail_result_from_hfm, rel_result_from_hfm)
                    removed_triples.append((t, '接壤', h))

            # '途径' 高于 '位于'
            if rel == '位于' and exist_head_result(head_result_from_hfm, (t, '途径', h)):
                print(f'WARNING: remove {h, rel, t} as {(t, "途径", h)} is found')
                removed_triples.append((h, rel, t))
                continue
            if rel == '途径' and exist_head_result(head_result_from_hfm, (t, '位于', h)):
                print(f'WARNING: remove {(t, "位于", h)} as {h, rel, t} is found')
                remove_result((t, '位于', h), head_result_from_hfm, tail_result_from_hfm, rel_result_from_hfm)
                removed_triples.append((t, '位于', h))

            # '包含' 低于 '位于' 和 '临近'
            if rel == '包含':
                if exist_head_result(head_result_from_hfm, (t, '位于', h)):
                    print(f'WARNING: remove {h, rel, t} as {(t, "位于", h)} is found')
                    removed_triples.append((h, rel, t))
                    continue
                if exist_head_result(head_result_from_hfm, (h, '临近', t)):
                    print(f'WARNING: remove {h, rel, t} as {(h, "临近", t)} is found')
                    removed_triples.append((h, rel, t))
                    continue
                if exist_head_result(head_result_from_hfm, (t, '临近', h)):
                    print(f'WARNING: remove {h, rel, t} as {(t, "临近", h)} is found')
                    removed_triples.append((h, rel, t))
                    continue
            if rel == '位于' and exist_head_result(head_result_from_hfm, (t, '包含', h)):
                print(f'WARNING: remove {(t, "包含", h)} as {h, rel, t} is found')
                remove_result((t, '包含', h), head_result_from_hfm, tail_result_from_hfm, rel_result_from_hfm)
                removed_triples.append((t, '包含', h))
            if rel == '临近':
                if exist_head_result(head_result_from_hfm, (h, '包含', t)):
                    print(f'WARNING: remove {(h, "包含", t)} as {h, rel, t} is found')
                    remove_result((h, '包含', t), head_result_from_hfm, tail_result_from_hfm, rel_result_from_hfm)
                    removed_triples.append((h, '包含', t))
                if exist_head_result(head_result_from_hfm, (t, '包含', h)):
                    print(f'WARNING: remove {(t, "包含", h)} as {h, rel, t} is found')
                    remove_result((t, '包含', h), head_result_from_hfm, tail_result_from_hfm, rel_result_from_hfm)
                    removed_triples.append((t, '包含', h))

            # 所有其他 rel 优先于 '成就'
            if rel == '成就':
                found_flag = False
                for other_rel in relation_set:
                    if other_rel == rel: continue
                    if exist_head_result(head_result_from_hfm, (h, other_rel, t)):
                        print(f'WARNING: remove {h, rel, t} as {(h, other_rel, t)} is found')
                        removed_triples.append((h, rel, t))
                        found_flag = True
                        break
                if found_flag:
                    continue
            if rel != '成就' and exist_head_result(head_result_from_hfm, (h, '成就', t)):
                print(f'WARNING: remove {(h, "成就", t)} as {h, rel, t} is found')
                remove_result((h, '成就', t), head_result_from_hfm, tail_result_from_hfm, rel_result_from_hfm)
                removed_triples.append((h, '成就', t))

            if exist_head_result(head_result_from_hfm, (h, rel, t)):
                update_result((h, rel, t, prob), head_result_from_hfm, tail_result_from_hfm, rel_result_from_hfm)
                continue

            if drop_short_entity:
                if t in tail_result_from_hfm and rel in tail_result_from_hfm[t]:
                    continue_flag = False
                    for old_h in list(tail_result_from_hfm[t][rel].keys()):
                        if len(h) < len(old_h) and h in old_h and input.count(h) == 1:
                            print(f'WARNING: drop0 head {h} for ({rel}, {t}) as we find a better (or longer) head {old_h}')
                            removed_triples.append((h, rel, t))
                            continue_flag = True
                            local_merge_map[(h, rel, t)] = (old_h, rel, t)
                            update_result((old_h, rel, t, prob), head_result_from_hfm, tail_result_from_hfm, rel_result_from_hfm)
                            break
                        if len(old_h) < len(h) and old_h in h and input.count(old_h) == 1:
                            print(f'WARNING: drop head {old_h} for ({rel}, {t}) as we find a better (or longer) head {h}')
                            removed_triples.append((old_h, rel, t))
                            local_merge_map[(old_h, rel, t)] = (h, rel, t)
                            remove_result((old_h, rel, t), head_result_from_hfm, tail_result_from_hfm, rel_result_from_hfm)

                    # 如果该 triple 已经确定放弃，那么继续遍历 output_pairs 中的下一个 triple
                    if continue_flag:
                        continue
                    del old_h

                # # 如果有多个出生时间和死亡时间，选择长的那个
                # # if '时间' in rel and len(head_result_from_hfm[h][rel]):
                # # if ('时间' in rel or rel == '表演者') and len(head_result_from_hfm[h][rel]):
                rel_set0 = ['配偶', '父亲', '职务'] # 选择短的
                if h in head_result_from_hfm and rel in head_result_from_hfm[h]:
                    continue_flag = False
                    for old_t in list(head_result_from_hfm[h][rel].keys()):
                        if (rel in rel_set0 and len(t) > len(old_t) and old_t in t) or (rel not in rel_set0 and len(t) < len(old_t) and t in old_t and input.count(t) == 1):
                            print(f'WARNING: drop0 tail {t} for ({h}, {rel}) as we find a better (or longer) tail {old_t}')
                            removed_triples.append((h, rel, t))
                            continue_flag = True
                            local_merge_map[(h, rel, t)] = (h, rel, old_t)
                            update_result((h, rel, old_t, prob), head_result_from_hfm, tail_result_from_hfm, rel_result_from_hfm)
                            break
                        if (rel in rel_set0 and len(old_t) > len(t) and t in old_t) or \
                           (rel not in rel_set0 and len(old_t) < len(t) and old_t in t and input.count(old_t) == 1):
                            print(f'WARNING: drop tail {old_t} for ({h}, {rel}) as we find a better (or longer) tail {t}')
                            removed_triples.append((h, rel, old_t))
                            local_merge_map[(h, rel, old_t)] = (h, rel, t)
                            remove_result((h, rel, old_t), head_result_from_hfm, tail_result_from_hfm, rel_result_from_hfm)

                    if continue_flag:
                        continue
                    del old_t

            update_result((h, rel, t, prob), head_result_from_hfm, tail_result_from_hfm, rel_result_from_hfm)

            # 如果经过上述处理后，仍然有未排除项，那么只能这样了
            if '时间' in rel and len(head_result_from_hfm[h][rel]) > 0:
                print(f'WARNING: multi tail {head_result_from_hfm[h][rel].keys()} and new tail {t} for ({h}, {rel})')

    sort_dict(head_result_from_hfm)
    sort_dict(rel_result_from_hfm)
    sort_dict(tail_result_from_hfm)

    return head_result_from_hfm, rel_result_from_hfm, tail_result_from_hfm, max_data, local_merge_map, removed_triples

