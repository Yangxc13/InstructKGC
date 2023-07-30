import argparse
import copy
import random
import re
import torch
from tqdm import tqdm

import functools
from contextlib import closing
# import multiprocessing as mp
from multiprocessing import Pool

from utils import *


def worker_process(data_in):
    pid, num_process, records, temp_path, cuda_id = data_in
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]=str(cuda_id)
    import torch
    from load_model import gen_zhixi_model
    from triple_level_search import init_toolkit, gen_step, generate_given_triples2

    print(f'Start process {pid} on cuda {cuda_id}')

    end2end_result, h_rel_result, t_rel_result, mainitem_result = torch.load(temp_path)
    indices = [record['id'] % 1000000 for record in records]

    if num_process > 1:
        h_rel_result = {k: v for k, v in h_rel_result.items() if k in indices}
        t_rel_result = {k: v for k, v in t_rel_result.items() if k in indices}
        mainitem_result = {k: v for k, v in mainitem_result.items() if k in indices}

    head_mode = 'head-first'
    head_model, tokenizer = gen_zhixi_model(
        base_model='model_hub/zhixi-13b-with-lora-merged',
        lora_weights='')
    head_self = head_model._orig_mod
    device = head_self.device

    tail_mode = 'tail-first'
    tail_model, _ = gen_zhixi_model(
        base_model='model_hub/zhixi-13b-with-lora-merged',
        lora_weights='model_hub/zhixi-tail-model-lora')
    tail_self = tail_model._orig_mod

    tools = init_toolkit()

    output = []
    for record in records:
        try:
            idx = record['id'] % 1000000
            input = record['input']
            raw_relation_set = relation_set = record['instruction'].split(']')[0][12:-1].split("', '")
            print(idx, input)

            if raw_relation_set == ['']:
                record['output'] = ''
                record['kg'] = []
                output.append(record)
                continue

            # 中文括号替换英文括号，在本 for 循环末尾会替换回来
            has_bracket = '(' in input or ')' in input
            if has_bracket:
                input = copy.deepcopy(input).replace('(', '（').replace(')', '）').replace('[', '<').replace(']', '>')

            # 替换 relation 关键词， 在本 for 循环末尾会替换回来
            if record['cate'] == '医学':
                change_flag = False
                for rel_idx, rel in enumerate(raw_relation_set):
                    if rel == '包含':
                        raw_relation_set[rel_idx] = '属于'
                        change_flag = True
                if change_flag:
                    print(f'WARNING: change relation_set to {raw_relation_set}')
            if record['cate'] == '自然科学':
                change_flag = False
                for rel_idx, rel in enumerate(raw_relation_set):
                    if rel == '组成':
                        raw_relation_set[rel_idx] = '组成成分'
                        change_flag = True
                    if rel == '性质':
                        raw_relation_set[rel_idx] = '特性'
                        change_flag = True
                    if rel == '用途':
                        raw_relation_set[rel_idx] = '应用场景'
                        change_flag = True
                if change_flag:
                    print(f'WARNING: change relation_set to {raw_relation_set}')

            if idx == 545 and raw_relation_set == ['2020年8月7日']:
                raw_relation_set = ['类型', '上映时间']

            # 从 end2end_result 中选取哪些行对应当前的 idx（在旧版本中，会从中选择得分最高的一组作为最终结果）
            start = end = -1
            for i in range(len(end2end_result)):
                if start < 0 and end2end_result[i][0] == idx:
                    start = i
                if start >= 0 and end < 0 and end2end_result[i][0] != idx:
                    end = i
                    break
            if start >= 0 and end < 0:
                end = len(end2end_result)

            if start < 0:
                print('WARNING: Jump record', idx)
                continue

            merge_map2 = {}
            head_result_from_hfm, rel_result_from_hfm, tail_result_from_hfm, max_data, merge_map2, removed_triples = \
                construct_result_from_hfm(idx, end2end_result, start, end, input, raw_relation_set, drop_short_entity=False)

            # 尝试从上面结果中找到最长的句首实体作为 main_item
            main_item = None
            heads = list(head_result_from_hfm.keys()) + list(tail_result_from_hfm.keys())
            if idx in t_rel_result:
                tmp = []
                for prob, triple_list in t_rel_result[idx].values():
                    if triple_list[-1] == ')':
                        triple_list = triple_list[:-1]
                    elif triple_list[-3:] == '),(':
                        triple_list = triple_list[:-3]

                    for triple in triple_list.split('),('):
                        for rel in raw_relation_set:
                            if f',{rel},' in triple:
                                t = triple.split(f',{rel},')[1]
                                tmp.append(t)
                heads = heads + tmp

            heads = list(set(heads))
            heads = [ele for ele in heads if input.startswith(ele)]
            if len(heads) > 1:
                heads = [max(heads, key=len)]
            if len(heads) == 1:
                main_item = heads[0]
                if any([main_item.endswith(ele) for ele in ['世纪', '年', '月', '日']]):
                    main_item = None

            # merge main_item
            merge_map0 = {}
            # merge_map0_except_bieming = {}
            if '别名' in raw_relation_set and '别名' in rel_result_from_hfm:
                for h, h_dict in rel_result_from_hfm['别名'].items():
                    for t, prob_list in h_dict.items():
                        if f'{h}{t}' in input: # and (f'{h}{t}' in head_result_from_hfm.keys() or f'{h}{t}' in tail_result_from_hfm.keys()):
                            assert f'{h}{t}' not in merge_map0
                            merge_map0[f'{h}{t}'] = h
                            merge_map0[t] = h
                            # merge_map0_except_bieming[t] = h
                            print(f'WARNING: Trigger merge main_item rule 0 before generate_given_triples(). Will map {f"{h}{t}"} to {h}')

            if len(merge_map0):
                # 采用最简单的方法，就是把上面的 head_result_from_hfm, rel_result_from_hfm, tail_result_from_hfm 构建过程重新执行一遍
                new_data_copied = copy.deepcopy(end2end_result[start:end])

                for line_idx in range(len(new_data_copied)):
                    _0, _1, _2, _3, _pairs = new_data_copied[line_idx]
                    for triple_idx, (prob, (h, rel, t)) in enumerate(_pairs):
                        if rel == '别名' and t in merge_map0 and merge_map0[t] == h:
                            continue
                        for old, new in merge_map0.items():
                            change_flag = False
                            if h == old:
                                change_flag = True
                                h = new
                            if t == old:
                                change_flag = True
                                t = new
                            if change_flag:
                                print('WARNING', _pairs[triple_idx], '->', (prob, (h, rel, t)))
                        _pairs[triple_idx] = (prob, (h, rel, t))
                    new_data_copied[line_idx] = (_0, _1, _2, _3, _pairs)
            else:
                new_data_copied = end2end_result[start:end]

            head_result_from_hfm, rel_result_from_hfm, tail_result_from_hfm, max_data, merge_map2, removed_triples = \
                construct_result_from_hfm(idx, new_data_copied, 0, len(new_data_copied), input, raw_relation_set)

            # 利用 h_rel_result 调整候选三元组集合。这个主要是针对 <h, rel> 对应多个 t 的情况
            # 如果某个候选的 t 不能在以 <h, rel> 为开头的情况下生成，那么移除该 t
            # 这部分代码是在复赛中最后3天被加入，由于评测次数有限，并不确定是否真的有用以及对最后结果的提高幅度
            if 1:
                h_rel_pairs = []
                for h, h_dict in head_result_from_hfm.items():
                    for rel, rel_dict in h_dict.items():
                        if rel in ['rel_count', 't_count', 't_visit']: continue
                        if len(rel_dict) == 1: continue # 如果 <h, rel> 只对应单个 t，则跳过
                        h_rel_pairs.append((h, rel, list(rel_dict.keys())))

                if idx not in h_rel_result:
                    h_rel_result[idx] = {}

                # 更新 h_rel_result
                if len(h_rel_pairs):
                    try:
                        new_h_rel_pairs = [(h, rel, ts) for h, rel, ts in h_rel_pairs if (h, rel) not in h_rel_result[idx]]

                        if len(new_h_rel_pairs):
                            print('Re-generate h_rel_pairs', new_h_rel_pairs)
                            # 首先，以 cc 中的 <tail, rel> 对为引子，使用 tail_model 生成可能的 head
                            input_ids = gen_inputs(tools['prompt'], tokenizer, input, raw_relation_set, mode=head_mode, prefix='')
                            # init_result = generate_given_triples(
                            #     head_self, tokenizer, input_ids,
                            #     input, relation_set, new_h_rel_pairs,
                            #     generation_config=tools['gen_config'],
                            #     triple_type=head_mode, select_method='heuristic'
                            # )
                            init_result = generate_given_triples2(
                                head_self, tokenizer, input_ids.to(device), tools,
                                input, relation_set, new_h_rel_pairs,
                                generation_config=tools['gen_config'],
                                triple_type=head_mode
                            )

                            # result_dict[idx] = copy.deepcopy(init_result)
                            for (prob, triple_list), (h, rel, ts) in zip(init_result, new_h_rel_pairs):
                                if triple_list[-1] == ')':
                                    triple_list = triple_list[:-1]
                                elif triple_list[-3:] == '),(':
                                    triple_list = triple_list[:-3]

                                tmp = []
                                for triple in triple_list.split('),('):
                                    assert triple.startswith(f'{h},{rel},')
                                    t = triple.split(f',{rel},')[1]
                                    tmp.append(t)
                                h_rel_result[idx][(h, rel)] = (prob, tmp)
                    except Exception as e:
                        print(f'Error: when update h_rel_result for idx {idx}', e)

                # h_rel_result 是比 end2end_result 还要宽松的生成。移除不在 h_rel_result 中的结果
                tmp_del_triples = []
                for h, h_dict in head_result_from_hfm.items():
                    for rel, rel_dict in h_dict.items():
                        if rel in ['rel_count', 't_count', 't_visit']: continue
                        if len(rel_dict) == 1: continue # 如果 <h, rel> 只对应单个 t，则跳过
                        ts = list(rel_dict.keys())
                        new_ts = [t for t in ts if t in h_rel_result[idx][(h, rel)][1]]
                        for t in ts:
                            if rel == '位于' and f'{t}{h}' in input: continue
                            if t not in new_ts:
                                print(f'WARNING: remove {(h, rel, t)} as not in {h_rel_result[idx][(h, rel)]}')
                                tmp_del_triples.append((h, rel, t))
                                # for other_t in h_rel_result[idx][(h, rel)][1]:
                                #     if other_t in head_result_from_hfm[h][rel] and max(head_result_from_hfm[h][rel][t]) < 0.8 *max(head_result_from_hfm[h][rel][other_t]):
                                #         print(f'WARNING: remove {(h, rel, t)} as not in {h_rel_result[idx][(h, rel)]}')
                                #         tmp_del_triples.append((h, rel, t))
                                #         break

                for triple in tmp_del_triples:
                    remove_result(triple, head_result_from_hfm, tail_result_from_hfm, rel_result_from_hfm)

            # 利用 mainitem_result 调整候选三元组集合。这部分也是在复赛中最后3天被加入，消融实验证明有所提升 0.6791 (f1=0.5656, rouge2=0.7927) -> 0.6869 (f1=0.5719, rouge2=0.8020)
            if main_item is not None:
                if main_item not in head_result_from_hfm:
                    miss_relations = raw_relation_set
                else:
                    rel_for_main_item = [rel for rel in head_result_from_hfm[main_item].keys() if rel not in ['rel_count', 't_count', 't_visit']]
                    # 这时候 tail_result 还没有构建
                    # rel_for_main_item += [rel for rel in tail_result_from_hfm[main_item].keys() if rel not in ['rel_count', 'h_count', 'h_visit']]
                    miss_relations = sorted(list(set(relation_set) - set(rel_for_main_item)))
                if len(miss_relations):
                    # tmp.append(idx)
                    print(f'\nIdx {idx} main_item {main_item} misses relation {miss_relations}')
                    print(input)

                    if not (idx in mainitem_result and mainitem_result[idx]['miss_relations'] == miss_relations):
                        mainitem_result[idx] = {'miss_relations': miss_relations}
                        h_rel_pairs = [(main_item, rel, None) for rel in miss_relations]

                        print(f'\tRe-generate mainitem_rel_pairs {h_rel_pairs}')
                        input_ids = gen_inputs(tools['prompt'], tokenizer, input, raw_relation_set, mode=head_mode, prefix='')
                        # init_result = generate_given_triples(
                        #     head_self, tokenizer, input_ids,
                        #     input, relation_set, h_rel_pairs,
                        #     generation_config=tools['gen_config'],
                        #     triple_type=head_mode
                        # )
                        init_result = generate_given_triples2(
                            head_self, tokenizer, input_ids.to(device), tools,
                            input, relation_set, h_rel_pairs,
                            generation_config=tools['gen_config'],
                            triple_type=head_mode
                        )
                        mainitem_result[idx]['init_result'] = init_result
                    else:
                        init_result = mainitem_result[idx]['init_result']

                    for prob, triple_list in init_result:
                        if triple_list[-1] == ')':
                            triple_list = triple_list[:-1]
                        elif triple_list[-3:] == '),(':
                            triple_list = triple_list[:-3]
                        for triple in triple_list.split('),('):
                            rel = triple2rel(triple, raw_relation_set, 'head-first')
                            h, t = triple.split(f',{rel},')
                            print(f'WARNING: add {(h, rel, t)} for miss relation {rel}')
                            if (h, rel, t) in removed_triples:
                                print(f'\tThis action is abondaned as {(h, rel, t)} in removed_triples')
                            else:
                                update_result((h, rel, t, prob), head_result_from_hfm, tail_result_from_hfm, rel_result_from_hfm)

                # 如果 类型 score 小于 main_item 的类型 score 的 75%
                if main_item in head_result_from_hfm and '类型' in head_result_from_hfm[main_item]:
                    max_score = max([max(probs) for t, probs in rel_result_from_hfm['类型'][main_item].items() if t not in ['t_count', 't_visit']])
                    tmp_del_triples = []
                    for h, h_dict in rel_result_from_hfm['类型'].items():
                        if h == main_item: continue
                        for t in list(h_dict.keys()):
                            if t in ['t_count', 't_visit']: continue
                            if max(h_dict[t]) < 0.75 * max_score:
                                print(f'WARNING: remove {(h, "类型", t)} for low score than main_item {max(h_dict[t])}')
                                tmp_del_triples.append((h, '类型', t))
                    for triple in tmp_del_triples:
                        remove_result(triple, head_result_from_hfm, tail_result_from_hfm, rel_result_from_hfm)

            # 更新 t_rel_result 结果
            if 1:
                if idx not in t_rel_result:
                    t_rel_result[idx] = {}

                t_rel_pairs = []
                for t, t_dict in tail_result_from_hfm.items():
                    for rel, rel_dict in t_dict.items():
                        if rel in ['h_count', 'h_visit', 'rel_count', 't_count', 't_visit']:
                            continue
                        if (t, rel, None) not in t_rel_pairs:
                            t_rel_pairs.append((t, rel, None))
                        if rel == '位于':
                            for h in rel_dict.keys():
                                if (h, rel, None) not in t_rel_pairs:
                                    t_rel_pairs.append((h, rel, None))

                # 为 “绝对星等” 加上等
                for t_rel_idx, (t, rel, hs) in enumerate(t_rel_pairs):
                    if rel == '绝对星等' and t[-1] != '等' and f'{t}等' in input:
                        t_rel_pairs[t_rel_idx] = (f'{t}等', rel, hs)
                    # if rel == '人口' and t[-1] != '人' and f'{t}人' in input:
                    #     t_rel_pairs[t_rel_idx] = (f'{t}人', rel, hs)

                new_t_rel_pairs = [(t, rel, hs) for t, rel, hs in t_rel_pairs if (t, rel) not in t_rel_result[idx]]

                if len(new_t_rel_pairs):
                    print('Re-generate t_rel_pairs', new_t_rel_pairs)
                    # 首先，以 cc 中的 <tail, rel> 对为引子，使用 tail_model 生成可能的 head
                    input_ids = gen_inputs(tools['prompt'], tokenizer, input, raw_relation_set, mode=tail_mode, prefix='')
                    # init_result = generate_given_triples(
                    #     tail_self, tokenizer, input_ids,
                    #     input, relation_set, new_t_rel_pairs,
                    #     generation_config=tools['gen_config'],
                    #     triple_type=tail_mode
                    # )
                    init_result = generate_given_triples2(
                        tail_self, tokenizer, input_ids.to(device), tools,
                        input, relation_set, new_t_rel_pairs,
                        generation_config=tools['gen_config'],
                        triple_type=tail_mode
                    )

                    # result_dict[idx] = copy.deepcopy(init_result)
                    for prob, triple_list in init_result:
                        assert any([f',{rel},' in triple_list for rel in raw_relation_set])
                        for rel in raw_relation_set:
                            if f',{rel},' in triple_list:
                                t = triple_list[:triple_list.index(f',{rel},')]
                                assert (t, rel) not in t_rel_result[idx]
                                t_rel_result[idx][(t, rel)] = (prob, triple_list)
                                break

            init_result = copy.deepcopy([
                t_rel_result[idx][(t, rel)] for t, rel, hs in t_rel_pairs])

            if len(merge_map0):
                # 补充新的 merge_map0
                change_flag = False
                added_triple = []
                for prob, triple_list in init_result:
                    if '别名' in triple_list:
                        if triple_list.endswith('),('):
                            triple_list = triple_list[:-3]
                        elif triple_list.endswith(')'):
                            triple_list = triple_list[:-1]
                        for triple in triple_list.split('),('):
                            t, h = triple.split(',别名,')
                            if f'{h}{t}' in input and f'{h}{t}' not in merge_map0:
                                if t in merge_map0.values():
                                    print(f'WARNING: want to map {h} to {t} but there are already mapped item(s) {[k for k, v in merge_map0.items() if v == t]}')
                                    continue
                                merge_map0[f'{h}{t}'] = h
                                merge_map0[t] = h
                                # merge_map0_except_bieming[t] = h
                                print(f'WARNING: Trigger merge main_item rule 0 after generate_given_triples(). Will map {f"{h}{t}"} to {h}')
                                change_flag = True
                                added_triple.append((h, '别名', t))

                        if t in merge_map0 and merge_map0[t] == h:
                            continue

                if change_flag:
                    print('INFO: Trigger merge main_item rule 0 after generate_given_triples. Now re-generate results_from_hfm')
                    new_data_copied = copy.deepcopy(end2end_result[start:end])

                    for line_idx in range(len(new_data_copied)):
                        _0, _1, _2, _3, _pairs = new_data_copied[line_idx]
                        for triple_idx, (prob, (h, rel, t)) in enumerate(_pairs):
                            if rel == '别名' and t in merge_map0 and merge_map0[t] == h:
                                continue
                            for old, new in merge_map0.items():
                                change_flag = False
                                if h == old:
                                    change_flag = True
                                    h = new
                                if t == old:
                                    change_flag = True
                                    t = new
                                if change_flag:
                                    print('WARNING', _pairs[triple_idx], '->', (prob, (h, rel, t)))
                            _pairs[triple_idx] = (prob, (h, rel, t))
                        new_data_copied[line_idx] = (_0, _1, _2, _3, _pairs)

                    for h, rel, t in added_triple:
                        _pairs.append((1., (h, rel, t)))
                        new_data_copied[line_idx] = (_0, _1, _2, _3, _pairs)

                    head_result_from_hfm, rel_result_from_hfm, tail_result_from_hfm, max_data, merge_map2 = \
                        construct_result_from_hfm(idx, new_data_copied, 0, len(new_data_copied), input, raw_relation_set)

                # 替换 init_result 中结果
                for result_idx, (prob, triple_list) in enumerate(init_result):
                    change_flag = False
                    if '别名' in triple_list:
                        for map_idx, (old_h, new_h) in enumerate(merge_map0.items()):
                            if f'{old_h},别名,{new_h}' in triple_list:
                                triple_list = triple_list.replace(f'{old_h},别名,{new_h}', f'placeholder{map_idx}')

                    for old_h, new_h in merge_map0.items():
                        if old_h in triple_list:
                            change_flag = True
                            triple_list = triple_list.replace(f',{old_h})', f',{new_h})')
                            if triple_list.startswith(f'{old_h},'):
                                triple_list = f'{new_h},' + triple_list[len(old_h)+1:]

                    if '别名' in triple_list and 'placeholder' in triple_list:
                        for map_idx, (old_h, new_h) in enumerate(merge_map0.items()):
                            if f'placeholder{map_idx}' in triple_list:
                                triple_list = triple_list.replace(f'placeholder{map_idx}', f'{old_h},别名,{new_h}')

                    if change_flag:
                        print('WARNING:', init_result[result_idx], '->', (prob, triple_list))
                        init_result[result_idx] = (prob, triple_list)

            raw_result = copy.deepcopy(init_result)
            # 更新 raw_result （特定 relation 的每个 <tail, rel> 只对应唯一的 head）
            for result_idx, (prob, text) in enumerate(raw_result):
                triples_list = text.strip('(').strip(',').strip(')').split('),(')
                rel = triple2rel(triples_list[0], relation_set, tail_mode)
                assert all([rel in item for item in triples_list])
                # 对于以下 relation 只选 top1 （即 tail 只对应唯一的 head） TODO 需要用`寻找是否只对应唯一`的这部分代码确认一下，是否确实是唯一
                if len(triples_list) > 1 and rel in ['别名', '创建时间', '宽度', '长度', '高度', '面积', '出生时间', '出生地点',
                    '死亡时间', '墓地', '字', '号', '绝对星等', '直径', '公转周期', '质量', '死亡人数', '受伤人数', '出版时间', # 删除‘作品’ see valid 274
                    '上映时间', '发行时间', '票房', '生产时间', '学名', '重量', '宽度', '高度', '成立时间', '解散时间', '人口', '面积']:
                    print('Drop tail_for_one_head', triples_list[1:])
                    triples_list = triples_list[:1]

                triples_kg = []
                for triple in triples_list:
                    t, h = triple.split(f',{rel},')
                    if rel == '位于' and f'{h}{t}' in input:
                        h, t = t, h
                    triples_kg.append((h, rel, t))

                raw_result[result_idx] = (prob, triples_kg)

            # merge_map2
            for result_idx, (prob, triples_kg) in enumerate(raw_result):
                change_flag = False
                for triple_idx, (h, rel, t) in enumerate(triples_kg):
                    if (h, rel, t) in merge_map2:
                        print(f'WARNING: based on merge_map2, project raw_result {(h, rel, t)} into {merge_map2[(h, rel, t)]}')
                        triples_kg[triple_idx] = merge_map2[(h, rel, t)]
                        change_flag = True
                if change_flag:
                    raw_result[result_idx] = (prob, triples_kg)

            head_result_from_tfm, rel_result_from_tfm, tail_result_from_tfm = {}, {}, {}
            rel_last_result_from_tfm = {}
            for prob, triple_list in raw_result:
                for h, rel, t in triple_list:
                    if h not in input:
                        print(f'WARNING: drop {(h, rel, t)} as head {h} not exists in raw input')
                        continue
                    if h == t:
                        print(f'WARNING: drop {(h, rel, t)} as the same head and tail')
                        continue

                    # for 对称 relation
                    if rel in ['临近', '兄弟姊妹', '配偶']:
                        if input.index(h) == input.index(t): # 不接受类似 (北京市动物园,位于,北京市)
                            continue
                        if input.index(h) > input.index(t):
                            h, t = t, h
                    if rel in ['位于', '所在行政领土'] and f'{h}{t}' in input:
                        h, t = t, h

                    # 自包含的不可能是别名
                    if rel == '别名':
                        if (h not in input) or (t not in input) or (input.count(h) == input.count(t) and ((h in t) or (t in h))):
                            continue

                    # if h not in hfm_entities:
                    #     print(f'WARNING: drop {(h, rel, t)} as head {h} not exists in hfm results')
                    #     continue

                    update_result((h, rel, t, prob), head_result_from_tfm, tail_result_from_tfm, rel_result_from_tfm, rel_last_result_from_tfm)

            sort_dict(head_result_from_tfm)
            sort_dict(rel_result_from_tfm)
            sort_dict(tail_result_from_tfm)
            sort_dict(rel_last_result_from_tfm)

            # ======
            # 通过比较 head_result_from_tfm 和 head_result_from_hfm 确定最终结果
            bidirectional_result = []

            raw_head_result_from_hfm = head_result_from_hfm
            raw_rel_result_from_hfm  = rel_result_from_hfm
            raw_tail_result_from_hfm = tail_result_from_hfm

            raw_head_result_from_tfm = head_result_from_tfm
            raw_rel_result_from_tfm  = rel_result_from_tfm
            raw_tail_result_from_tfm = tail_result_from_tfm
            raw_rel_last_result_from_tfm = rel_last_result_from_tfm

            head_result_from_hfm = copy.deepcopy(raw_head_result_from_hfm)
            rel_result_from_hfm  = copy.deepcopy(raw_rel_result_from_hfm)
            tail_result_from_hfm = copy.deepcopy(raw_tail_result_from_hfm)

            head_result_from_tfm = copy.deepcopy(raw_head_result_from_tfm)
            rel_result_from_tfm  = copy.deepcopy(raw_rel_result_from_tfm)
            tail_result_from_tfm = copy.deepcopy(raw_tail_result_from_tfm)
            rel_last_result_from_tfm = copy.deepcopy(raw_rel_last_result_from_tfm)

            common_head_result, common_rel_result, common_tail_result = {}, {}, {}

            # 对于 '位于' 关系，只要 f'{t}{h}' 存在，则必然选择
            h_has_locate = []
            for h in list(head_result_from_hfm.keys()):
                for rel in ['位于', '所在行政领土']:
                    if h not in head_result_from_hfm or rel not in head_result_from_hfm[h]: continue
                    for t in list(head_result_from_hfm[h][rel].keys()):
                        assert h != t
                        if t not in head_result_from_hfm[h][rel]: continue
                        if f'{t}{h}' in input and (h, rel, t) not in bidirectional_result:
                            bidirectional_result.append((h, rel, t))
                            if h not in h_has_locate: h_has_locate.append(h)
                            update_result((h, rel, t, max(head_result_from_hfm[h][rel][t])), common_head_result, common_tail_result, common_rel_result)
                            remove_result((h, rel, t), head_result_from_hfm, tail_result_from_hfm, rel_result_from_hfm)
                            if exist_head_result(head_result_from_tfm, (h, rel, t)):
                                remove_result((h, rel, t), head_result_from_tfm, tail_result_from_tfm, rel_result_from_tfm, rel_last_result_from_tfm)

            for h in list(head_result_from_tfm.keys()):
                for rel in ['位于', '所在行政领土']:
                    if h not in head_result_from_tfm or rel not in head_result_from_tfm[h]: continue
                    for t in list(head_result_from_tfm[h][rel].keys()):
                        assert h != t
                        if t not in head_result_from_tfm[h][rel]: continue
                        if f'{t}{h}' in input and (h, rel, t) not in bidirectional_result:
                            bidirectional_result.append((h, rel, t))
                            if h not in h_has_locate: h_has_locate.append(h)
                            update_result((h, rel, t, max(head_result_from_tfm[h][rel][t])), common_head_result, common_tail_result, common_rel_result)
                            remove_result((h, rel, t), head_result_from_tfm, tail_result_from_tfm, rel_result_from_tfm, rel_last_result_from_tfm)
                            if exist_head_result(head_result_from_hfm, (h, rel, t)):
                                remove_result((h, rel, t), head_result_from_hfm, tail_result_from_hfm, rel_result_from_hfm)

            # collect common triples
            for h in list(head_result_from_tfm.keys()):
                if h not in head_result_from_tfm: continue
                for rel in list(head_result_from_tfm[h].keys()):
                    if rel in ['rel_count', 't_count', 't_visit']: continue
                    # if rel not in head_result_from_hfm[h]:
                    #     continue
                    chosen_num = 0 # 考虑 head 只对应唯一的 tail
                    for t in list(head_result_from_tfm[h][rel].keys()):
                        if rel == '位于' and not exist_head_result(head_result_from_hfm, (h, rel, t)) and \
                            exist_head_result(head_result_from_hfm, (t, rel, h)) and not exist_head_result(head_result_from_tfm, (t, rel, h)):
                            bidirectional_result.append((t, rel, h))
                            update_result((t, rel, h, max(head_result_from_tfm[h][rel][t])), common_head_result, common_tail_result, common_rel_result)
                            print(f'WARNING: add {(t, rel, h)} for 位于 as {(t, rel, h)} in hfm but only {(h, rel, t)} in tfm')

                        if exist_head_result(head_result_from_hfm, (h, rel, t)):
                            bidirectional_result.append((h, rel, t))
                            update_result((h, rel, t, max(head_result_from_tfm[h][rel][t])), common_head_result, common_tail_result, common_rel_result)
                            remove_result((h, rel, t), head_result_from_hfm, tail_result_from_hfm, rel_result_from_hfm)
                            remove_result((h, rel, t), head_result_from_tfm, tail_result_from_tfm, rel_result_from_tfm, rel_last_result_from_tfm)
                            if rel in ['位于', '所在行政领土'] and h not in h_has_locate:
                                h_has_locate.append(h)

                            flag1 = exist_head_result(head_result_from_hfm, (t, rel, h))
                            flag2 = exist_head_result(head_result_from_tfm, (t, rel, h))

                            if flag1 and flag2:
                                bidirectional_result.append((t, rel, h))
                                print(f'WARNING: both {(h, rel, t)} and {(t, rel, h)} exist in hfm & tfm')
                                update_result((t, rel, h, max(head_result_from_hfm[t][rel][h])), common_head_result, common_tail_result, common_rel_result)
                                if rel in ['位于', '所在行政领土'] and t not in h_has_locate:
                                    h_has_locate.append(t)
                            if flag1:
                                if not flag2: print(f'INFO: remove {(t, rel, h)} for only existing in hfm but {(h, rel, t)} exists in both')
                                remove_result((t, rel, h), head_result_from_hfm, tail_result_from_hfm, rel_result_from_hfm)
                            if flag2:
                                if not flag1: print(f'INFO: remove {(t, rel, h)} for only existing in tfm but {(h, rel, t)} exists in both')
                                remove_result((t, rel, h), head_result_from_tfm, tail_result_from_tfm, rel_result_from_tfm, rel_last_result_from_tfm)

            # deal with (h, rel, t) and (t, rel, h)
            for h in list(common_head_result.keys()):
                if h not in common_head_result: continue
                for rel in list(common_head_result[h].keys()):
                    if h not in common_head_result or rel not in common_head_result[h]: continue # dict 在执行过程中可能发生变化
                    if rel in ['rel_count', 't_count', 't_visit']: continue
                    for t in list(common_head_result[h][rel].keys()):
                        if h not in common_head_result or rel not in common_head_result[h] or t not in common_head_result[h][rel]: continue
                        if exist_head_result(common_head_result, (t, rel, h)):
                            if main_item is not None and h is main_item:
                                bidirectional_result.remove((t, rel, h))
                                remove_result((t, rel, h), common_head_result, common_tail_result, common_rel_result)
                            elif main_item is not None and t is main_item:
                                bidirectional_result.remove((h, rel, t))
                                remove_result((h, rel, t), common_head_result, common_tail_result, common_rel_result)
                            elif len(raw_head_result_from_hfm[h][rel][t]) >= len(raw_head_result_from_hfm[t][rel][h]) \
                                and max(raw_head_result_from_hfm[h][rel][t]) > 1.2 * max(raw_head_result_from_hfm[t][rel][h]):
                                bidirectional_result.remove((t, rel, h))
                                remove_result((t, rel, h), common_head_result, common_tail_result, common_rel_result)
                            elif len(raw_head_result_from_hfm[h][rel][t]) <= len(raw_head_result_from_hfm[t][rel][h]) \
                                and 1.2 * max(raw_head_result_from_hfm[h][rel][t]) < max(raw_head_result_from_hfm[t][rel][h]):
                                bidirectional_result.remove((h, rel, t))
                                remove_result((h, rel, t), common_head_result, common_tail_result, common_rel_result)
                            elif len(raw_head_result_from_hfm[h][rel][t]) > len(raw_relation_set) // 2 and len(raw_head_result_from_hfm[t][rel][h]) < len(raw_relation_set) // 2:
                                bidirectional_result.remove((t, rel, h))
                                remove_result((t, rel, h), common_head_result, common_tail_result, common_rel_result)
                            elif len(raw_head_result_from_hfm[h][rel][t]) < len(raw_relation_set) // 2 and len(raw_head_result_from_hfm[t][rel][h]) > len(raw_relation_set) // 2:
                                bidirectional_result.remove((h, rel, t))
                                remove_result((h, rel, t), common_head_result, common_tail_result, common_rel_result)
                            else: # 如果不能分辨出 (h, rel, t) 和 (t, rel, h) 那个正确，则全部移除。这个会提高一些 f1-score
                                bidirectional_result.remove((t, rel, h))
                                remove_result((t, rel, h), common_head_result, common_tail_result, common_rel_result)
                                bidirectional_result.remove((h, rel, t))
                                remove_result((h, rel, t), common_head_result, common_tail_result, common_rel_result)
                            # else:
                            #     flag1 = len(raw_head_result_from_hfm[h][rel][t]) > len(raw_head_result_from_hfm[t][rel][h])
                            #     flag2 = len(raw_head_result_from_hfm[h][rel][t]) < len(raw_head_result_from_hfm[t][rel][h])
                            #     if not flag1 and not flag2:
                            #         flag1 = max(raw_head_result_from_hfm[h][rel][t]) > max(raw_head_result_from_hfm[t][rel][h])
                            #         flag2 = max(raw_head_result_from_hfm[h][rel][t]) < max(raw_head_result_from_hfm[t][rel][h])
                            #     if not flag1 and not flag2 and main_item is not None:
                            #         flag1 = h == main_item
                            #         flag2 = t == main_item
                            #     if not flag1 and not flag2:
                            #         flag1 = True

                            #     if flag1:
                            #         bidirectional_result.remove((t, rel, h))
                            #         remove_result((t, rel, h), common_head_result, common_tail_result, common_rel_result)
                            #     else:
                            #         bidirectional_result.remove((h, rel, t))
                            #         remove_result((h, rel, t), common_head_result, common_tail_result, common_rel_result)

            # 其次，处理两个实体间有多种关系的一种情况 (h,rel0,t),(h,rel1,t),(h,rel1,t1) 那么排除 (h,rel1,t) 选择 (h,rel0,t),(h,rel1,t1)
            del_idx = []
            for triple_idx0, (h, rel0, t) in enumerate(bidirectional_result):
                if triple_idx0 in del_idx:
                    continue

                if h not in raw_rel_last_result_from_tfm or t not in raw_rel_last_result_from_tfm[h]:
                    print(f'WARNING: ({h}, {t}) not in raw_rel_last_result_from_tfm, it may be added by hand')
                    continue

                if len(raw_rel_last_result_from_tfm[h][t]) > 1:
                    other_rels = []
                    for rel1 in raw_rel_last_result_from_tfm[h][t].keys():
                        if rel1 == rel0: continue
                        if (h, rel1, t) not in bidirectional_result: continue
                        triple_idx1 = bidirectional_result.index((h, rel1, t))
                        if triple_idx1 in del_idx:
                            continue
                        other_rels.append(rel1)

                    if len(other_rels) > 1:
                        print(f'WARNING: h, t ({h}, {t}) have {len(raw_rel_last_result_from_tfm[h][t])} relations, {raw_rel_last_result_from_tfm[h][t].keys()}, useful {other_rels}, do not know how to tackle')
                    elif len(other_rels) == 1:
                        rel1 = other_rels[0]
                        triple_idx1 = bidirectional_result.index((h, rel1, t))

                        if rel1 in ['父母']: # 允许多个 tail 存在，即 (h,rel1,t),(h,rel1,t1) 同时存在
                            # 那么就根据分数判断 (h,rel0,t),(h,rel1,t) 中应该删除哪一个
                            # 暂暂时根据 hfm 分数
                            if np.mean(raw_head_result_from_hfm[h][rel1][t]) > np.mean(raw_head_result_from_hfm[h][rel0][t]):
                                print(f'WARNING: remove {(h, rel0, t)} (score={raw_head_result_from_hfm[h][rel0][t]} {raw_head_result_from_tfm[h][rel0][t]}),'
                                    + f' as {(h, rel1, t)} is better (score={raw_head_result_from_hfm[h][rel1][t]} {raw_head_result_from_tfm[h][rel1][t]})')
                                del_idx.append(triple_idx0)
                                solved_flag = True
                            else:
                                print(f'WARNING: remove {(h, rel1, t)} (score={raw_head_result_from_hfm[h][rel1][t]} {raw_head_result_from_tfm[h][rel1][t]}),'
                                    + f' as {(h, rel0, t)} is better (score={raw_head_result_from_hfm[h][rel0][t]} {raw_head_result_from_tfm[h][rel0][t]})')
                                del_idx.append(triple_idx1)
                                solved_flag = True
                        else:
                            # (h,rel0) 是否有其他的 t
                            h_rel0_t1 = []
                            if len(raw_head_result_from_tfm[h][rel0]) > 1:
                                for t1 in raw_head_result_from_tfm[h][rel0].keys():
                                    if t1 != t and (h, rel0, t1) in bidirectional_result and bidirectional_result.index((h, rel0, t1)) not in del_idx:
                                        h_rel0_t1.append(t1)
                            # (h,rel1) 是否有其他的 t
                            h_rel1_t1 = []
                            if len(raw_head_result_from_tfm[h][rel1]) > 1:
                                for t1 in raw_head_result_from_tfm[h][rel1].keys():
                                    if t1 != t and (h, rel1, t1) in bidirectional_result and bidirectional_result.index((h, rel1, t1)) not in del_idx:
                                        h_rel1_t1.append(t1)
                            # (t,rel0) 是否有其他的 h1
                            t_rel0_h1 = []
                            if len(raw_tail_result_from_tfm[t][rel0]) > 1:
                                for h1 in raw_tail_result_from_tfm[t][rel0].keys():
                                    if h1 != h and (h1, rel0, t) in bidirectional_result and bidirectional_result.index((h1, rel0, t)) not in del_idx:
                                        t_rel0_h1.append(h1)
                            # (t,rel1) 是否有其他的 h1
                            t_rel1_h1 = []
                            if len(raw_tail_result_from_tfm[t][rel1]) > 1:
                                for h1 in raw_tail_result_from_tfm[t][rel1].keys():
                                    if h1 != h and (h1, rel1, t) in bidirectional_result and bidirectional_result.index((h1, rel1, t)) not in del_idx:
                                        t_rel1_h1.append(h1)

                            solved_flag = False

                            if not solved_flag and len(h_rel0_t1) >= 1 and len(h_rel1_t1) == 0:
                                del_idx.append(triple_idx0)
                                print(f'WARNING: 0716 select {(h, rel1, t)} and {(h, rel0, h_rel0_t1)}, remove {(h, rel0, t)}')
                                solved_flag = True
                            if not solved_flag and len(h_rel0_t1) == 0 and len(h_rel1_t1) >= 1:
                                del_idx.append(triple_idx1)
                                print(f'WARNING: 0716 select {(h, rel0, t)} and {(h, rel1, h_rel1_t1)}, remove {(h, rel1, t)}')
                                solved_flag = True
                            if not solved_flag and len(t_rel0_h1) >= 1 and len(t_rel1_h1) == 0:
                                del_idx.append(triple_idx0)
                                print(f'WARNING: 0716 select {(h, rel1, t)} and {(t_rel0_h1, rel0, t)}, remove {(h, rel0, t)}')
                                solved_flag = True
                            if not solved_flag and len(t_rel0_h1) == 0 and len(t_rel1_h1) >= 1:
                                del_idx.append(triple_idx1)
                                print(f'WARNING: 0716 select {(h, rel0, t)} and {(t_rel1_h1, rel1, t)}, remove {(h, rel1, t)}')
                                solved_flag = True
                            if not solved_flag:
                                print(f'WARNING: do not know how to tackle ({h}, <{rel0}/{rel1}, {t}>), {h_rel0_t1}, {h_rel1_t1}, {t_rel0_h1}, {t_rel1_h1}')

            if len(del_idx):
                print('Remove', [bidirectional_result[triple_idx] for triple_idx in del_idx])
                bidirectional_result = [triple for triple_idx, triple in enumerate(bidirectional_result) if triple_idx not in del_idx]

            del_idx = []
            for triple_idx0, (h, rel0, t) in enumerate(bidirectional_result):
                if triple_idx0 in del_idx:
                    continue

                if h not in raw_rel_last_result_from_tfm or t not in raw_rel_last_result_from_tfm[h]:
                    print(f'WARNING: (in turn 2) ({h}, {t}) not in raw_rel_last_result_from_tfm, it may be added by hand')
                    continue

                if len(raw_rel_last_result_from_tfm[h][t]) > 1:
                    other_rels = []
                    for rel1 in raw_rel_last_result_from_tfm[h][t].keys():
                        if rel1 == rel0: continue
                        if (h, rel1, t) not in bidirectional_result: continue
                        triple_idx1 = bidirectional_result.index((h, rel1, t))
                        if triple_idx1 in del_idx:
                            continue
                        other_rels.append(rel1)

                    if len(other_rels) > 1:
                        print(f'WARNING: (in turn 2) h, t ({h}, {t}) have {len(raw_rel_last_result_from_tfm[h][t])} relations, {raw_rel_last_result_from_tfm[h][t].keys()}, useful {other_rels}, do not know how to tackle')
                    elif len(other_rels) == 1:
                        rel1 = other_rels[0]
                        triple_idx1 = bidirectional_result.index((h, rel1, t))

                        if 1:
                            # (h,rel0) 是否有其他的 t
                            h_rel0_t1 = []
                            if len(raw_head_result_from_tfm[h][rel0]) > 1:
                                for t1 in raw_head_result_from_tfm[h][rel0].keys():
                                    if t1 != t and (h, rel0, t1) in bidirectional_result and bidirectional_result.index((h, rel0, t1)) not in del_idx:
                                        h_rel0_t1.append(t1)
                            # (h,rel1) 是否有其他的 t
                            h_rel1_t1 = []
                            if len(raw_head_result_from_tfm[h][rel1]) > 1:
                                for t1 in raw_head_result_from_tfm[h][rel1].keys():
                                    if t1 != t and (h, rel1, t1) in bidirectional_result and bidirectional_result.index((h, rel1, t1)) not in del_idx:
                                        h_rel1_t1.append(t1)
                            # (t,rel0) 是否有其他的 h1
                            t_rel0_h1 = []
                            if len(raw_tail_result_from_tfm[t][rel0]) > 1:
                                for h1 in raw_tail_result_from_tfm[t][rel0].keys():
                                    if h1 != h and (h1, rel0, t) in bidirectional_result and bidirectional_result.index((h1, rel0, t)) not in del_idx:
                                        t_rel0_h1.append(h1)
                            # (t,rel1) 是否有其他的 h1
                            t_rel1_h1 = []
                            if len(raw_tail_result_from_tfm[t][rel1]) > 1:
                                for h1 in raw_tail_result_from_tfm[t][rel1].keys():
                                    if h1 != h and (h1, rel1, t) in bidirectional_result and bidirectional_result.index((h1, rel1, t)) not in del_idx:
                                        t_rel1_h1.append(h1)

                            solved_flag = False
                            if not solved_flag and len(h_rel0_t1) == 0 and len(h_rel1_t1) == 0 and len(t_rel0_h1) == 0 and len(t_rel1_h1) == 0:
                                print(f'WARNING: head_tail {[h, t]} has two rels {[rel0, rel1]}, and no other involved entities')

                                if len(raw_head_result_from_hfm[h][rel0][t]) > len(raw_head_result_from_hfm[h][rel1][t]):
                                    select_rel = rel0
                                elif len(raw_head_result_from_hfm[h][rel0][t]) < len(raw_head_result_from_hfm[h][rel1][t]):
                                    select_rel = rel1
                                elif np.mean(raw_head_result_from_hfm[h][rel0][t]) > np.mean(raw_head_result_from_hfm[h][rel1][t]):
                                    select_rel = rel0
                                else:
                                    select_rel = rel1
                                print(f'\tselect {(h, select_rel, t)}')
                                if select_rel == rel0:
                                    del_idx.append(bidirectional_result.index((h, rel1, t)))
                                else:
                                    del_idx.append(bidirectional_result.index((h, rel0, t)))

                            if not solved_flag and len(h_rel0_t1) > 0 and len(h_rel1_t1) > 0: # 如果 h 和 t 之间有大于两个 rel 的话，按照目前的写法会按顺序进行比较
                                ts_set = [t] + list(set(h_rel0_t1) & set(h_rel1_t1))
                                alloc2rel0, alloc2rel1 = [], []
                                for _t in ts_set:
                                    if len(raw_head_result_from_hfm[h][rel0][_t]) > len(raw_head_result_from_hfm[h][rel1][_t]):
                                        alloc2rel0.append(_t)
                                    elif len(raw_head_result_from_hfm[h][rel0][_t]) < len(raw_head_result_from_hfm[h][rel1][_t]):
                                        alloc2rel1.append(_t)
                                    elif np.mean(raw_head_result_from_hfm[h][rel0][_t]) > np.mean(raw_head_result_from_hfm[h][rel1][_t]):
                                        alloc2rel0.append(_t)
                                    else:
                                        alloc2rel1.append(_t)
                                print(f'WARNING: checked head {h} has two overlapped rels {[rel0, rel1]}, allocate tails {alloc2rel0} to {rel0} and {alloc2rel1} to {rel1}')
                                for _t in alloc2rel0:
                                    del_idx.append(bidirectional_result.index((h, rel1, _t)))
                                for _t in alloc2rel1:
                                    del_idx.append(bidirectional_result.index((h, rel0, _t)))
                                solved_flag = True

                            if not solved_flag and len(t_rel0_h1) > 0 and len(t_rel1_h1) > 0:
                                hs_set = [h] + list(set(t_rel0_h1) & set(t_rel1_h1))
                                alloc2rel0, alloc2rel1 = [], []
                                for _h in hs_set:
                                    if len(raw_tail_result_from_hfm[t][rel0][_h]) > len(raw_tail_result_from_hfm[t][rel1][_h]):
                                        alloc2rel0.append(_h)
                                    elif len(raw_tail_result_from_hfm[t][rel0][_h]) < len(raw_tail_result_from_hfm[t][rel1][_h]):
                                        alloc2rel1.append(_h)
                                    elif np.mean(raw_tail_result_from_hfm[t][rel0][_h]) > np.mean(raw_tail_result_from_hfm[t][rel1][_h]):
                                        alloc2rel0.append(_h)
                                    else:
                                        alloc2rel1.append(_h)
                                print(f'WARNING: checked tail {t} has two overlapped rels {[rel0, rel1]}, allocate heads {alloc2rel0} to {rel0} and {alloc2rel1} to {rel1}')
                                for _h in alloc2rel0:
                                    del_idx.append(bidirectional_result.index((_h, rel1, t)))
                                for _h in alloc2rel1:
                                    del_idx.append(bidirectional_result.index((_h, rel0, t)))
                                solved_flag = True
                            if not solved_flag:
                                print(f'WARNING: (in turn 2) do not know how to tackle ({h}, <{rel0}/{rel1}, {t}>), {h_rel0_t1}, {h_rel1_t1}, {t_rel0_h1}, {t_rel1_h1}')

            if len(del_idx):
                print('Remove (in turn 2)', [bidirectional_result[triple_idx] for triple_idx in del_idx])
                bidirectional_result = [triple for triple_idx, triple in enumerate(bidirectional_result) if triple_idx not in del_idx]

            # # Rule 2. 如果 head 只对应唯一的 tail，搜索并取 hfm scores 中的 top1
            # 这个要放到上面 (h,rel0,t),(h,rel1,t),(h,rel1,t1) 那么排除 (h,rel1,t) 这一步骤结束之后。因为有相当一部分上面就可以解决了
            rel_with_single_tail = ['出生地点', '出生时间', '死亡时间', '墓地', '发现时间', '改编自', '人口']
            while 1:
                del_idx = []
                for h, rel, t in bidirectional_result:
                    if rel in rel_with_single_tail and h in raw_head_result_from_hfm and rel in raw_head_result_from_hfm[h] and t in raw_head_result_from_hfm[h][rel]:
                        prob = max(raw_head_result_from_hfm[h][rel][t])
                        for other_t, other_prob_list in raw_head_result_from_hfm[h][rel].items():
                            if other_t == t:
                                continue
                            if (h, rel, other_t) not in bidirectional_result:
                                continue
                            if max(other_prob_list) > prob:
                                del_idx.append(bidirectional_result.index((h, rel, t)))
                            else:
                                del_idx.append(bidirectional_result.index((h, rel, other_t)))
                            break

                    if len(del_idx):
                        break
                if len(del_idx):
                    print('Remove', [bidirectional_result[triple_idx] for triple_idx in del_idx])
                    bidirectional_result = [triple for triple_idx, triple in enumerate(bidirectional_result) if triple_idx not in del_idx]
                else:
                    break

            # Rule 1. 如果某个 head 被双向认定是其他实体的别名
            merge_map1 = {}
            for result_idx, (h, rel, t) in enumerate(bidirectional_result):
                if rel == '别名':
                    # 修改 bidirectional_result 中的结果
                    for _idx, (_h, _rel, _t) in enumerate(bidirectional_result):
                        if _h == t:
                            print(f'WARNING: change {bidirectional_result[_idx]} to {(h, _rel, _t)}, as we believe {(h, "别名", _h)}')
                            bidirectional_result[_idx] = (h, rel, _t)
                    merge_map1[t] = h
            # 尝试从 raw_head_result_from_tfm & raw_head_result_from_hfm 中获取新的结果
            # 暂时只考虑一种情况，即 raw_head_result_from_tfm[t]存在的情况。暂时不考虑 raw_head_result_from_hfm[t] 存在的情况，
            #   以及 t 在 raw_head_result_from_tfm & raw_head_result_from_hfm 作为 tail 存在的情况。这个是基于 idx = 280 考虑的
            for old_h, new_h in merge_map1.items():
                if old_h in raw_head_result_from_tfm:
                    for rel, rel_dict in raw_head_result_from_tfm[old_h].items():
                        if rel in ['rel_count', 't_count', 't_visit']:
                            continue
                        # head 只对应唯一的 tail
                        if new_h not in raw_head_result_from_hfm or new_h not in raw_head_result_from_tfm:
                            continue
                        if rel in raw_head_result_from_tfm[new_h] and rel in rel_with_single_tail:
                            continue
                        if rel == '别名': continue
                        if rel not in raw_head_result_from_hfm[new_h]:
                            continue
                        chosen_flag = None # 是否 head 只对应唯一的 tail
                        for t, probs in rel_dict.items():
                            if t in raw_head_result_from_hfm[new_h][rel] and (new_h, rel, t) not in bidirectional_result:
                                if chosen_flag is None:
                                    bidirectional_result.append((new_h, rel, t))
                                else:
                                    print(f'WARNING, h_rel ({h},{rel}) has select one tail {chosen_flag}, drop {t}')
                                # 对于以下 relation 只选 top1 （即 head 只对应唯一的 tail)
                                # if '时间' in rel or rel in ['改编自']:
                                if rel in rel_with_single_tail:
                                    chosen_flag = t

            # 如果某relation在结果中不存在，以 raw_rel_result_from_tfm 分数最高的补齐
            pred_relation_set = [ele[1] for ele in bidirectional_result]
            miss_relations = list(set(raw_relation_set) - set(pred_relation_set))
            if len(miss_relations):
                print(f'WARNING: idx {idx} miss relations {miss_relations}')
                for rel in miss_relations:
                    if rel in raw_rel_result_from_tfm and len(raw_rel_result_from_tfm[rel]):
                        for h, h_dict in raw_rel_result_from_tfm[rel].items():
                            for t, t_probs in h_dict.items():
                                if t in ['t_count', 't_visit']: continue
                                print(f'Add {(h, rel, t)} for miss relation {rel}')
                                print(f'\tWARNING: STOPPED')
                                # bidirectional_result.append((h, rel, t)) # 暂时只选 top1
                                break
                            break
                    else:
                        print(f'Error: miss relation {rel} not found in newdata')

            print('Final result:')
            print(idx, input)
            # 双向选择的结果

            if has_bracket:
                for triple_idx, (h, rel, t) in enumerate(bidirectional_result):
                    change_flag = False
                    if '（' in h or '）' in h:
                        assert h in input
                        h_pos = input.index(h)
                        new_h = record['input'][h_pos:h_pos+len(h)]
                        print(f'WARNING: input has brackets, transform {h} into {new_h}')
                        h = new_h
                        del new_h
                        change_flag = True
                    if '（' in t or '）' in t:
                        assert t in input
                        t_pos = input.index(t)
                        new_t = record['input'][t_pos:t_pos+len(t)]
                        print(f'WARNING: input has brackets, transform {t} into {new_t}')
                        t = new_t
                        del new_t
                        change_flag = True
                    if change_flag:
                        bidirectional_result[triple_idx] = (h, rel, t)

            if record['cate'] == '医学':
                for triple_idx, (h, rel, t) in enumerate(bidirectional_result):
                    if rel in ['属于']:
                        bidirectional_result[triple_idx] = (t, '包含', h)
            if record['cate'] == '自然科学':
                for triple_idx, (h, rel, t) in enumerate(bidirectional_result):
                    if rel in ['组成成分']:
                        bidirectional_result[triple_idx] = (h, '组成', t)
                    if rel in ['特性']:
                        bidirectional_result[triple_idx] = (h, '性质', t)
                    if rel in ['应用场景']:
                        bidirectional_result[triple_idx] = (h, '用途', t)

            same_h_t = [triple for triple in bidirectional_result if triple[0] == triple[-1]]
            if len(same_h_t):
                print('WARNING: remove same head-tail', same_h_t)
                bidirectional_result = [triple for triple in bidirectional_result if triple[0] != triple[-1]]

            del_idx = []
            for triple_idx, (h, rel, t) in enumerate(bidirectional_result):
                if '时间' in rel:
                    figures = re.findall('([0-9]*)', t)
                    for ti in figures:
                        if ti not in input:
                            print(f'WARNING: del time {(h, rel, t)} in {input}')
                            del_idx.append(triple_idx)
            bidirectional_result = [triple for triple_idx, triple in enumerate(bidirectional_result) if triple_idx not in del_idx]

            if record['cate'] == '地理地区':
                for triple_idx, (h, rel, t) in enumerate(bidirectional_result):
                    if rel == '行政中心' and (h, '所在行政领土', t) in bidirectional_result:
                        bidirectional_result[triple_idx] = (t, rel, h)

            # 别名 优先于其他
            if '别名' in raw_relation_set:
                del_idx = []
                for triple_idx, (h, rel, t) in enumerate(bidirectional_result):
                    if rel != '别名' and (h, '别名', t) in bidirectional_result:
                        print(f'WARNING: remove {(h, rel, t)} as {(h, "别名", t)} exists in bidirectional_result')
                        del_idx.append(triple_idx)
                    if rel != '别名' and (t, '别名', h) in bidirectional_result:
                        print(f'WARNING: remove {(h, rel, t)} as {(t, "别名", h)} exists in bidirectional_result')
                        del_idx.append(triple_idx)
                bidirectional_result = [triple for triple_idx, triple in enumerate(bidirectional_result) if triple_idx not in del_idx]

            print('BIDIRECT', kg2output(bidirectional_result))

            # 通过选取分数最高序列 top1 办法得到的结果
            if max_data is not None:
                print('MAX_DATA', kg2output(max_data))
            print('~~~~~~\n\n')

            record['output'] = kg2output(bidirectional_result)
            record['kg'] = bidirectional_result

            output.append(record)

        except Exception as e:
            print('Error', idx, e)

    return output, h_rel_result, t_rel_result, mainitem_result, pid


def get_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/test.json')
    parser.add_argument('--temp_path', type=str, default='temp.pt')
    parser.add_argument('--output_path', type=str, default='output_test.json')
    parser.add_argument('--new_temp_path', type=str, default='new_temp.pt')
    parser.add_argument('--num_process', type=int, default=1)
    parser.add_argument('--gpu', nargs='+', type=int, default=[0])
    return parser.parse_args(args)


if __name__ == '__main__':
    args = get_args()

    if args.num_process == 1 and len(args.gpu) > 0:
        print(f'Will launch on {len(args.gpu)} GPU(s) {args.gpu}')
    elif args.num_process > 1 and args.num_process == len(args.gpu):
        print(f'Will launch {args.num_process} tasks on {args.num_process} GPUs {args.gpu}')
    else:
        print(
'''Running arguments error. For example, you can use:

python main.py --num_process 4 --gpu 0,1,2,3

This command launches 4 process, each on one gpu. It requires more than 50 GB memory for each GPU.

python main.py --num_process 1 --gpu 0,1,2,3

This command launches a single process on 4 gpus. It requires more than 15 GB memory for each GPU.
''')
        exit()

    records = load_data(args.data_path)
    if args.num_process == 1:
        output, out_h_rel_result, out_t_rel_result, out_mainitem_result, _ = \
            worker_process((0, 1, records, args.temp_path, ','.join([str(ele) for ele in args.gpu])))
    else:
        tasks = [(i, args.num_process, records[i::args.num_process], args.temp_path, args.gpu[i]) for i in range(args.num_process)]
        output = []
        out_h_rel_result, out_t_rel_result, out_mainitem_result = {}, {}, {}
        with closing(Pool(processes=args.num_process)) as pool:
            for ret_output, ret_h, ret_t, ret_m, ret_id in pool.imap_unordered(functools.partial(worker_process), tasks):
                print(ret_id, 'finished')
                output += ret_output
                out_h_rel_result.update(ret_h)
                out_t_rel_result.update(ret_t)
                out_mainitem_result.update(ret_m)
        output = sorted(output, key=lambda x: x['id'])

    with open(args.output_path, 'w', encoding='utf-8') as f:
        for record in output:
            json.dump(record, f, ensure_ascii=False)
            f.write('\n')

    import torch
    end2end_result, _, __, ___ = torch.load(args.temp_path)
    torch.save((end2end_result, out_h_rel_result, out_t_rel_result, out_mainitem_result), args.new_temp_path)
