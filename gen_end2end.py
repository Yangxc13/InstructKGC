import argparse
import copy
import numpy as np
import random
import re
from tqdm import tqdm

import functools
from contextlib import closing
# import multiprocessing as mp
from multiprocessing import Pool

from utils import *


def worker_process(data_in):
    pid, records, cuda_id = data_in
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]=str(cuda_id)
    import torch
    from load_model import gen_zhixi_model
    from triple_level_search import init_toolkit, gen_step, generate_given_triples2

    print(f'Start process {pid} on {cuda_id}')

    head_mode = 'head-first'
    head_model, tokenizer = gen_zhixi_model(
        base_model='model_hub/zhixi-13b-with-lora-merged',
        lora_weights='')
    head_self = head_model._orig_mod
    device = head_self.device

    tools = init_toolkit()

    end2end_result = []
    for record in records:
        try:
            idx = record['id'] % 1000000
            input = record['input']
            instruction = record['instruction']
            raw_relation_set = record['instruction'].split(']')[0][12:-1].split("', '")
            print(idx, input)

            if raw_relation_set == ['']:
                print(f'WARNING: Empty relation set for idx {idx}')
                print('~~~~~~\n')
                continue

            # 中文括号替换英文括号，在 main.py 中会替换回来
            has_bracket = '(' in input or ')' in input
            if has_bracket:
                input = copy.deepcopy(input).replace('(', '（').replace(')', '）').replace('[', '<').replace(']', '>')

            # 替换部分 relation 以达到更好效果，在 main.py 中会替换回来
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

            for rel_idx in range(len(raw_relation_set)):
                try:
                    if idx == 539 and rel_idx == 0: continue # 这一条会不断生成相同的三元组结果直到达到最大生成长度，可以手动直接跳过

                    # 对于有 n 个 relation 的句子，我们会将 relation_set 打乱顺序 n 次
                    # 对于第 i 次，将原始 raw_relation_set 中的第 i 个 relation 置于最前，其他的打乱顺序排列
                    relation_set = copy.deepcopy(raw_relation_set)
                    relation_set.remove(raw_relation_set[rel_idx])
                    random.shuffle(relation_set)
                    relation_set = [raw_relation_set[rel_idx]] + relation_set
                    print(relation_set)

                    prefix = ''
                    input_ids = gen_inputs(tools['prompt'], tokenizer, input, relation_set, mode=head_mode, prefix=prefix)
                    output_ids, prob_list = gen_step(
                        head_self, tokenizer, input_ids.to(device),
                        input, relation_set,
                        generation_config=tools['gen_config'],
                        triple_type=head_mode
                    )

                    output_text = prefix + tokenizer.decode(output_ids[0, input_ids.size(1):], skip_special_tokens=True)
                    output_pairs = split_triple_probs(tokenizer, input, relation_set, input_ids[0].cpu().numpy(), output_ids[0].cpu().numpy(), np.array(prob_list))
                    end2end_result.append((idx, relation_set, output_text, np.array(prob_list), output_pairs))
                except Exception as e:
                    print(f'Error: when generate end2end results for {idx} with relation set {relation_set}', e)

            print('~~~~~~\n')

        except Exception as e:
            print('Error', idx, e)

    return end2end_result, pid


def get_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/test.json')
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

python gen_end2end.py --num_process 4 --gpu 0,1,2,3

This command launches 4 process, each on one gpu. It requires more than 50 GB memory for each GPU.

python gen_end2end.py --num_process 1 --gpu 0,1,2,3

This command launches a single process on 4 gpus. It requires more than 15 GB memory for each GPU.
''')
        exit()

    records = load_data(args.data_path)
    if args.num_process == 1:
        end2end_result, _ = worker_process((
            0, records, ','.join([str(ele) for ele in args.gpu])))
    else:
        tasks = [(i, records[i::args.num_process], args.gpu[i]) for i in range(args.num_process)]
        end2end_result = []
        with closing(Pool(processes=args.num_process)) as pool:
            for ret, ret_id in pool.imap_unordered(functools.partial(worker_process), tasks):
                print(ret_id, 'finished')
                end2end_result += ret

    import torch
    h_rel_result, t_rel_result, mainitem_result = {}, {}, {}
    torch.save((end2end_result, h_rel_result, t_rel_result, mainitem_result), args.new_temp_path)
