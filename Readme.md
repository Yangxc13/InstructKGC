## InstructKGC

### 一、简介
本仓库是天池比赛“[CCKS2023 指令驱动的自适应知识图谱构建](https://tianchi.aliyun.com/competition/entrance/532080/introduction)”的参赛代码。此份代码主要的关注点在于，在已有的知识图谱构建大模型 [Zhixi](https://github.com/zjunlp/KnowLM) 的基础上，如何最大化提升单一大模型在知识图谱构建上的效果。此份代码的主要创新点在于：

1. 生成句子时的 `triple-level-search` 方法。区别于 `transformers` 库默认提供的 `greedy-search` 和 `beam-search` 方法，`triple-level-search` 方法会保留每种分支的可能，直到当前三元组生成完成，因此在面对不同长度的三元组时具备更好的自适应能力。
2. `Tail-first-model` 的训练。原版的 Zhixi 大模型在生成三元组时是按照 (head, relation, tail) 的顺序，我们将其称作 `head-first-model`。在原版模型的基础上，我们使用 LoRA 方法进行模型微调，在比赛提供的较少的训练数据上进行训练，得到了按照 (tail, relation, head) 的顺序生成三元组的 `tail-first-model` 模型。在实际推断过程中，当 `head-first-model` 和 `tail-first-model` 生成了等同的三元组时，我们有更大的置信度认为这个三元组是一个正确的结果。
3. 通过打乱 relation_set 的顺序，以及预提供 `<head, rel>` 对或者 `<tail, rel>` 对的形式，来干涉知识图谱构建大模型生成结果的过程，从而得到更多的候选三元组。这也可以看作是另一种 prompt 的形式。这是因为大模型生成句子是一个逐 token、遵循条件概率的过程，relation 的生成顺序和已有的三元组都会影响大模型后续的生成。
4. 从候选三元组的集合中选取最终结果的算法流程。这里主要是一套手动编写的规则，其核心在于，优先选取 `head-first-model` 和 `tail-first-model` 共有的结果，以及对特定 relation 手动制定的处理规则等。

### 二、数据和环境准备

请于[比赛界面](https://tianchi.aliyun.com/competition/entrance/532080/information)下载三个数据文件，放置在 `data/` 文件夹下。

请遵循 [KnowLM](https://github.com/zjunlp/KnowLM#2-quick-start) 中的说明进行大模型运行前的环境准备。

可以从 HuggingFace 中下载基准知识图谱大模型的参数（下载地址：[llama-13b](https://huggingface.co/decapoda-research/llama-13b-hf)、[zhixi-13b-diff](https://huggingface.co/zjunlp/zhixi-13b-diff)、[zhixi-13b-lora](https://huggingface.co/zjunlp/zhixi-13b-lora)）。请从 [GoogleDrive](https://drive.google.com/file/d/1sPBVqlKtlnf1JXYa2nTRR4LhReGsSfQW/view?usp=sharing) （约25M） 中下载我们训练的 `tail-first-model` 的LoRA模型的参数，如果您想自己训练该模型，请参考本文件的 3.2 部分。请将上述模型下载后放置在 `model_hub/` 文件夹下，使该文件夹的组织结构如下：

```shell
	|-- model_hub/
	|	|-- llama-13b-hf/
	|	|-- zhixi-13b-diff/
	|	|-- zhixi-13b-lora/
	|	|-- zhixi-tail-model-lora/
```

复原基准知识图谱大模型 Zhixi （融合 zhixi-13b 和 zhixi-13b-lora 需要python库 peft >= 0.3.0），这里我们将复原后的大模型参数命名为 `zhixi-13b-with-lora-merged`。

```shell
python weight_diff.py recover --path_raw model_hub/llama-13b-hf --path_diff model_hub/zhixi-13b-diff --path_tuned model_hub/zhixi-13b
python weight_diff.py merge --path_zhixi model_hub/zhixi-13b --path_lora model_hub/zhixi-13b-lora --path_sfted model_hub/zhixi-13b-with-lora-merged
```

### 三、运行说明

#### 3.1 快速复现比赛结果

本代码运行的主要速度瓶颈来源于大模型生成句子的过程。在参赛过程中，为避免对同样的输入输出的重复处理，我们把大模型在不同 prompt 下的生成结果存储到了临时文件，每次需要调用大模型前优先从临时文件中直接选择历史结果。同时在执行流程中，新的 prompt 和大模型在该 prompt 的下的输出也会不断添加到临时文件中。这极大的加速了我们对算法流程进行调优的过程。

为快速复现比赛结果，您可以从 [GoogleDrive](https://drive.google.com/file/d/1Xyx8ngWvmP0tbuKAYB-J65YeKR8x3Nqb/view?usp=drive_link) （约10M） 中下载我们已经针对 test 生成好的临时文件。临时文件中的结果全部由 `zhixi-13b-with-lora-merged` 和 `zhixi-tail-model-lora` 两个模型生成。临时文件的加载方式和所含内容说明如下：

```shell
import torch
end2end_result, h_rel_result, t_rel_result, mainitem_result = torch.load('temp.pt')
```
* end2end\_result： 在 relation_set 打乱后的不同顺序下，`head-first-model` （即 Zhixi 大模型）的生成结果。
* h\_rel\_result：给定 `<head, rel>` 对的情况下，`head-first-model` 的生成结果。
* t\_rel\_result：给定 `<tail, rel>` 对的情况下，`tail-first-model` 的生成结果。
* mainitem\_rel\_result：给定 `<mainitem, rel>` 对的情况下，`head-first-model` 的生成结果。这个是因为数据集中大部分句子都是对句首实体进行描述，所以我们优先考虑句首实体和不同 relation 结合。

为快速复现比赛结果，请根据您的GPU显存情况选择执行以下命令：

```shell
## 如果单块GPU显存超过 52G (每块GPU单独运行一个大模型)
python main.py --num_process 1 --gpu 0 --data_path data/test.json --temp_path temp.pt --output_path test_output.json --new_temp_path new_temp.pt

## 或者，如果具有不少于4块GPU，每块显存超过 16G (多块GPU合力运行一个大模型)
python main.py --num_process 1 --gpu 0 1 2 3 --data_path data/test.json --temp_path temp.pt --output_path test_output.json --new_temp_path new_temp.pt
```

该指令会读取数据集 `data/test.json` 和预生成的临时文件 `temp.pt`。输出结果存放于 `test_output.json`，更新后的临时文件位于 `new_temp.pt`。该指令会同时加载 `zhixi-13b-with-lora-merged` 和 `zhixi-tail-model-lora` 两个模型，需要占用 50G 以上 GPU 内存空间。

我们于复赛最后一天测试了当前版本的代码效果，单次运行输出的结果文件取得的分数为 0.6869 （其中，f1-score=0.5719，rouge2-score=0.8020）。我们在复赛中取得的最好成绩为 0.6908 （其中，f1-score=0.5671，rouge2-score=0.8145），这是将多次运行的历史结果文件取并集，整合成一个最终结果文件，所取得的结果。这些历史结果文件在执行流程和阈值选择上有些微差别，但对于本代码的整体效果影响不大，故并未罗列。

如果您想从头开始复现我们的比赛结果，请参考本章节 3.2 和 3.3 的内容。

#### 3.2 使用 LoRA 方法训练 tail-first-model

如果在比赛训练集上，以默认的三元组生成顺序（即 `head-first-model`）进一步训练基准大模型的话，我们发现模型的表现总是倾向于变差。这可能与训练的过拟合有关，因为基准大模型使用的知识图谱构建相关的数据要远多于比赛提供的训练数据。此外，比赛的训练数据有很多肉眼可见的标注异常的情况，尤其是"自然科学"类别中与数学有关的句子。因此，我们在比赛中并没有选择对 `head-first-model` 继续强化训练，而是希望通过一些简单的微调训练让大模型中潜藏的能力暴露出来。例如，基准大模型总是先生成头实体和关系，然后在它们的前提上以类似于条件概率的形式生成尾实体。我们打算挖掘基准大模型在给定尾实体和关系的前提下，判断头实体的能力，即 `tail-first-model`。

我们首先对训练数据进行了筛选，我们计算基准大模型的输出结果和训练集 ground-truth 之间的 f1-score ，仅保留 f1-score>0.66 的样本点用于训练。简便起见，我们将以此方法筛选出的训练数据的序号存储在了 `./data/train_selected_idx.txt` 中。请执行以下命令以生成 `tail-first-model` 对应的训练数据集：

```shell
python gen_tail_model_train_dataset.py
```

使用以下命令进行模型训练：(当前参数需求单块GPU显存超过24G，或者4块GPU每块显存超过12G)

```shell
python finetune_llama.py \
    --base_model 'model_hub/zhixi-13b-with-lora-merged' \
    --train_path 'data/train_tail_first_shuffle_rels.json' \
    --output_dir 'lora/zhixi-tail-model-lora-e8-r8' \
    --batch_size 128 \
    --micro_batch_size 4 \
    --num_epochs 8 \
    --learning_rate 1e-4 \
    --cutoff_len 1536 \
    --val_set_size 100 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length
```

训练得到的模型会存储在 `./lora/zhixi-tail-model-lora-e8-r8` 下。

#### 3.3 完整的执行流程

1. 生成 end2end\_result

```shell
## 如果单块GPU显存超过 28G (每块GPU单独运行一个大模型)
python gen_end2end.py --num_process 1 --gpu 0 --data_path data/test.json --new_temp_path temp_step0.pt

## 或者，如果具有不少于4块GPU，每块显存超过 28G (每块GPU单独运行一个大模型，多GPU并行加速)
python gen_end2end.py --num_process 4 --gpu 0 1 2 3 --data_path data/test.json --new_temp_path temp_step0.pt

## 或者，如果具有不少于4块GPU，每块显存超过 8G (多块GPU合力运行一个大模型)
python gen_end2end.py --num_process 1 --gpu 0 1 2 3 --data_path data/test.json --new_temp_path temp_step0.pt
```

2. 生成 h\_rel\_result, t\_rel\_result, mainitem\_result 和最终结果

```shell
## 如果单块GPU显存超过 52G (每块GPU单独运行一个大模型)
python main.py --num_process 1 --gpu 0 --data_path data/test.json --temp_path temp_step0.pt --output_path test_output.json --new_temp_path temp_step1.pt

## （推荐）或者，如果具有不少于4块GPU，每块显存超过 52G (每块GPU单独运行一个大模型，多GPU并行加速)
python main.py --num_process 4 --gpu 0 1 2 3 --data_path data/test.json --temp_path temp_step0.pt --output_path test_output.json --new_temp_path temp_step1.pt

## 或者，如果具有不少于4块GPU，每块显存超过 16G (多块GPU合力运行一个大模型)
python main.py --num_process 1 --gpu 0 1 2 3 --data_path data/test.json --temp_path temp_step0.pt --output_path test_output.json --new_temp_path temp_step1.pt
```

输出结果存放于 `test_output.json`，更新后的临时文件位于 `temp_step1.pt`

### 四、可能存在的问题

1. `triple-level-search` 方法实现的比较初步，目前只支持 batchsize=1 的生成。在参赛过程中，如 3.1 部分所示，我们把大模型在不同 prompt 下的生成结果存储到了临时文件中，这样在调整算法规则时避免了重复处理同样的输入输出。
2. 基准模型在运行过程中，可能会出现持续不断生成同样的三元组直到达到最大生成长度，或者生成预料之外符号的问题（例如 `"` 和 `)` token\_id=[29908, 29897] 应该作为两个 token 逐个生成，但基准模型会直接生成一个 token  `")` token\_id=1159）。这可能会造成 `triple-level-search` 在某些输入下会报错。我们对部分情况手动撰写了规则进行处理，其他情况在报错后会自动跳过该数据点。这些问题并没有很好的解决方法，若要彻底解决可能需要调整基准大模型 Zhixi 的训练过程，这个是我们目前无法接触到的。

### 五、Acknowledge

本仓库受益于 [KnowLM](https://github.com/zjunlp/KnowLM) 和 [DeepKE](https://github.com/zjunlp/DeepKE)，在此对他们的工作表示致敬和感谢。


