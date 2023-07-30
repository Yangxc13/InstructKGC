import os
import sys
import datetime
import fire
from typing import List

import torch
import torch.nn.functional as F

from datasets import Dataset
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

from prompter import Prompter

import logging
logger = logging.getLogger("__main__")


class SimpleTrainer(transformers.Trainer):

    # def training_step(self, model, inputs):
    #     return super().training_step(model, inputs)

    def compute_loss(self, model, inputs, return_outputs=False):
        inputs['labels'] = None
        # outputs = model._orig_mod(**inputs)
        outputs = model(**inputs)

        try:
            probs = F.softmax(outputs.logits, dim=-1)
            choices = outputs.logits.argmax(dim=-1)

            # 找到训练的开始位置点（即文字 `Response`），该位置之前的部分不参与训练
            input_ids = inputs['input_ids']
            pos = torch.stack(torch.where(input_ids == 13291), dim=1).cpu().numpy()
            pos[:, 1] += 18

            mask_next_one = torch.zeros(input_ids.size(), dtype=bool)
            mask = torch.zeros(input_ids.size(), dtype=bool)

            # 从第一次出现目标概率值小于 0.9 开始，到从第一次出现目标概率值小于 0.1 结束
            # 只对该区间内的文字进行训练
            # s_e = []
            for i, j in pos:
                start = end = None
                if j+1 >= input_ids.size(1):
                    continue

                while j+1 < input_ids.size(1):
                    gt = input_ids[i, j+1]
                    pred = probs[i, j, gt]
                    # if pred < 0.9:
                    #     print(i.item(), j.item(), tokenizer.decode([gt.item()]), pred.item())
                    if pred < 0.9:
                        start = j
                        break
                    if gt.item() == 2:
                        break
                    j += 1

                if j+1 >= input_ids.size(1) or start is None:
                    continue

                mask[i, j] = 1
                mask_next_one[i, j+1] = 1
                if pred >= 0.1:
                    j += 1
                    while j+1 < input_ids.size(1):
                        gt = input_ids[i, j+1]
                        pred = probs[i, j, gt]
                        if pred < 0.1:
                            end = j
                            break
                        if pred < 0.9:
                            # print(i.item(), j.item(), tokenizer.decode([gt.item()]), pred.item())
                            mask[i, j] = 1
                            mask_next_one[i, j+1] = 1
                        if gt.item() == 2:
                            break
                        j += 1

                    if j+1 >= input_ids.size(1):
                        # s_e.append((i, start, input_ids.size(1)))
                        continue
                else:
                    end = j + 1

                # s_e.append((i, start, end))

            weights = mask / mask.sum(1, keepdim=True)
            weights = weights[mask]
            weights = weights / weights.sum()
            weights = weights.to(outputs.logits.device)

            shift_labels = input_ids[mask_next_one]
            shift_logits = probs.view(input_ids.numel(), model.config.vocab_size)[mask.flatten()]

            loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')
            loss = torch.sum(weights * loss)

            return (loss, outputs) if return_outputs else loss

        except Exception as e:
            torch.save(inputs, 'error_inputs.pt')
            assert 0, e


def train(
    # model/data params
    base_model: str = "decapoda-research/llama-7b-hf",  
    train_path: str = "data/train.json",
    valid_path: str = None,
    output_dir: str = "lora",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 1024,
    val_set_size: int = 100,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"train_path: {train_path}\n"
            f"valid_path: {valid_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    ## prepare tokenizer and data
    prompter = Prompter(prompt_template_name)

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            '输入中包含的关系三元组是：\n' + data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    if train_path is not None:
        train_data = Dataset.from_json(
            train_path, 
        )
    if valid_path is not None:
        valid_data = Dataset.from_json(
            valid_path, 
        )
    elif valid_path is None and train_path is not None:
        train_val = train_data.train_test_split(test_size=val_set_size, shuffle=True, seed=42)
        # train_data = train_val["train"] 所有数据都是训练数据
        valid_data = train_val["test"]

    train_data = train_data.shuffle().map(
        generate_and_tokenize_prompt, 
        remove_columns=train_data.column_names,
        load_from_cache_file=True,
    )
    valid_data = valid_data.shuffle().map(
        generate_and_tokenize_prompt, 
        remove_columns=valid_data.column_names,
        load_from_cache_file=True,
    )

    ## prepare model
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = tokenizer.bos_token_id = 1
    model.config.eos_token_id = tokenizer.eos_token_id = 2

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = SimpleTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0.06,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="no",
            save_strategy="epoch",
            #eval_steps=100,
            #save_steps=100,
            output_dir=output_dir,
            # save_total_limit=3,
            load_best_model_at_end=False, # True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    logger.info("*** Training ***")
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    metrics = train_result.metrics
    logger.info("***** Train results *****")
    logger.info(f"{metrics}")
    model.save_pretrained(output_dir)
    trainer.save_state()

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)
