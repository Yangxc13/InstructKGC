import copy
import json
import random
import re
from tqdm import tqdm


def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as fp:
        for ele in data:
            json.dump(ele, fp, ensure_ascii=False)
            fp.write('\n')


raw_data = open('data/train.json').read().split('\n')
if raw_data[-1] == '':
    raw_data = raw_data[:-1]


selected_idx = open('data/train_selected_idx.txt').read().split('\n')
if selected_idx[-1] == '':
    selected_idx = selected_idx[:-1]
selected_idx = [int(ele) for ele in selected_idx]


# tail-first
output = []
for line in tqdm(raw_data):
    out = json.loads(line)
    if out['id'] not in selected_idx: continue
    input = out['input']
    init_relation_set = out['instruction'].split(']')[0][12:-1].split("', '")

    # 重新排列三元组顺序：首先按照 relation 在 relation_set 中的次序，其次按照尾实体在文本中的位置，再次按照头实体在文本中的位置
    # 同种 relation 的三元组保存在一起，在后面组合的时候也保证同种 relation 的三元组相邻
    results_pre_relation = {}
    for relation in init_relation_set:
        results_pre_relation[relation] = sorted([[ele[2], ele[1], ele[0]] for ele in out['kg'] if ele[1] == relation], key=lambda x: input.index(x[0]) * len(input) + input.index(x[2]))

    for first_relation in init_relation_set:
    	# 数据增强，打乱 relation_set 顺序，保证每种 relation 都处在开头位置一次
        other_relation = [ele for ele in init_relation_set if ele != first_relation]
        random.shuffle(other_relation)
        relation_set = [first_relation] + other_relation

        out['kg'] = []
        for relation in relation_set:
            out['kg'] += results_pre_relation[relation]

        out['instruction'] = f'已知候选的关系列表：{relation_set}，请你根据关系列表，从以下输入中抽取出可能存在的头实体(Subject)与尾实体(Object)，并给出对应的关系三元组。请按照 (Object,Relation,Subject) 的格式回答。'

        out['output'] = ','.join([str(tuple(ele)) for ele in out['kg']])
        out['output'] = out['output'].replace("', '", ',').replace("'", '')

        output.append(copy.deepcopy(out))

save_json(output, 'data/train_tail_first_shuffle_rels.json')
