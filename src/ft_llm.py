import argparse
import json
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import setup_chat_format, set_seed as trl_seed
from peft import LoraConfig, AutoPeftModelForCausalLM 
from trl import SFTTrainer
from transformers import set_seed as transf_seed
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np 
from torch.utils.data import DataLoader
from transformers.trainer_utils import EvalLoopOutput
import random, glob
from lightning import seed_everything 

from reformat_data_ft_llm import process

def set_random_seed(seed: int):
    """set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(seed=seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    trl_seed(seed)
    transf_seed(seed)
    

def formatting_prompts_func(samples):
    prompt_texts = [tokenizer.apply_chat_template(
             sample[:-1], tokenize=False, add_generation_prompt=True) for sample in samples["messages"]]
    
    print("=="*50)
    print(prompt_texts[-1])
    print("=="*50)
    return prompt_texts

def split_label(sample):
    tokenized_lb = tokenizer.encode(sample['messages'][-1]['content'], padding='max_length',max_length=10 )
    sample['labels'] = tokenized_lb 
    return sample
 
class LLMErcTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.data_process_args = argparse.Namespace(
            packing=False,
            dataset_text_field=None,
            max_seq_length=kwargs.get('max_seq_length', None),
            formatting_func=formatting_prompts_func,
            num_of_sequences=kwargs.get('num_of_sequences', 1024),
            chars_per_token=kwargs.get('chars_per_token', 3.6),
            remove_unused_columns=kwargs.get('args').remove_unused_columns if kwargs.get('args') is not None else True,
            dataset_kwargs=kwargs.get('dataset_kwargs', {})
        )
        self.eval_dataset = self._process_raw_data(kwargs.get('eval_dataset', None))  
        print("len(eval dataset) = ",  len(self.eval_dataset))
    
    def _process_raw_data(self, dataset):
        dataset2 = dataset.map(split_label)
        dataset = self._prepare_dataset(
                dataset=dataset,
                tokenizer=self.tokenizer,
                packing=False,
                dataset_text_field=None,
                max_seq_length=self.data_process_args.max_seq_length,
                formatting_func=self.data_process_args.formatting_func,
                num_of_sequences=self.data_process_args.num_of_sequences,
                chars_per_token=self.data_process_args.chars_per_token,
                remove_unused_columns=self.data_process_args.remove_unused_columns,
                **self.data_process_args.dataset_kwargs, 
            )
        dataset = dataset.add_column('labels', dataset2['labels']) 
        return dataset 
    
    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        if "input_ids" not in eval_dataset.column_names and "labels" not in eval_dataset.column_names:
            # this is raw data which need to preprocess
            eval_dataset = self._process_raw_data(eval_dataset)
            
        return super().get_eval_dataloader(eval_dataset)
    
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only= None,
        ignore_keys = None,
        metric_key_prefix="eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        model = self.model
        model = model.to(dtype=torch.bfloat16)
        
        model.eval()
            
        # losses/preds/labels on CPU (final containers)
        all_preds = []
        all_labels = []
        all_raw_decoded = []
         
        def post_process(str_out):
            try:
                gen_text = str_out.split("assistant\n")[-1].split("<|im_end|>")[0]
            except:
                gen_text = "error"
            return gen_text
        
        # Main evaluation loop
        with torch.no_grad():
            for step, inputs in enumerate(tqdm(dataloader)):
                inputs = self._prepare_inputs(inputs)
                gen_kwargs = {'max_new_tokens': 10, 
                              'do_sample': False, 
                              'eos_token_id': self.tokenizer.eos_token_id, 
                              'pad_token_id': self.tokenizer.pad_token_id,
                              "temperature": 0.1,
                              }
                generated_tokens = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **gen_kwargs,
                )
                labels = inputs.pop("labels")
                str_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
                
                raw_decoded = [e for e in self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)]
                str_decoded = [post_process(e) for e in raw_decoded]
                all_preds += str_decoded
                all_labels += str_labels 
                all_raw_decoded += raw_decoded
        num_samples = len(dataloader)
          
        f1_weighted = f1_score(
            all_labels,
            all_preds,
            average=f"weighted",
        )
        metrics = { f"{metric_key_prefix}_weighted-f1": f1_weighted  }
        
        json.dump({"metrics": metrics, 
                   "detail_pred": list(zip(all_preds, all_labels, all_raw_decoded))}, 
                   open(f"{self.args.output_dir}/result_{metric_key_prefix}_step-{self.state.global_step}.json", "wt"), indent=1)
        
        # free the memory again
        del model
        torch.cuda.empty_cache()
        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
        
    
if __name__=='__main__':
        
    parser = argparse.ArgumentParser(description='Process ...')
    parser.add_argument('--do_train', action="store_true", help='fine tuning a LLM model with LoRA', default=False)
    parser.add_argument('--do_eval_test', action="store_true", help='eval on test set', default=False)
    parser.add_argument('--do_eval_dev', action="store_true", help='eval on dev set', default=False)
    parser.add_argument('--ft_model_path', type=str, default=None, help='fintuned model path') 
    parser.add_argument('--ft_model_id', type=str, default=None, help='fintuned model id for saving after train it')
    parser.add_argument('--prompting_type', type=str, default='spdescV2', help='prompting style in {cot, fewshot, zeroshot}')
    parser.add_argument('--base_model_id', type=str, default='meta-llama/Llama-2-7b-hf', help='base llm model id')
    parser.add_argument('--extract_prompting_llm_id', type=str, default='Llama-2-7b-chat-hf', help='base llm model id')
    parser.add_argument('--epoch', type=int, default=None, help='training epoch')
    parser.add_argument('--max_steps', type=int, default=None, help='training steps')
    parser.add_argument('--lr_scheduler', type=str, default='constant', help='learning rate scheduler')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate value')
    parser.add_argument('--seed', type=int, default=42, help='random seed value')
    parser.add_argument('--kshot', type=int, default=0, help='k shot examples for llm')
    parser.add_argument('--lora_r', type=int, default=32, help='lora rank')
    parser.add_argument('--eval_delay', type=int, default=200, help='eval delay')
    parser.add_argument('--window', type=int, default=5, help='local context window size')
    parser.add_argument('--max_seq_len', type=int, default=None, help='max sequence length for chunking/packing')
    parser.add_argument('--re_gen_data', action="store_true", help='re generate data', default=False)
    parser.add_argument('--data_name', type=str,  help='data name in {iemocap, meld, emorynlp}', default='iemocap')
    parser.add_argument('--data_folder', type=str,  help='path folder save all data', default='./data/')
    parser.add_argument('--output_folder', type=str,  help='path folder save all data', default='./finetuned_llm/')

    args, unknown = parser.parse_known_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.prompting_type == 'zeroshot':
        args.kshot = 0
    print(args)
    
    set_random_seed(args.seed)
    
    all_path_folder_preprocessed_data = [f"{args.data_folder}/{args.data_name}.{d_type}.{args.kshot}shot_w{args.window}_{args.prompting_type}.jsonl" \
        for d_type in [ 'train' , 'valid',  'test']]
    if args.re_gen_data:
        process(all_path_folder_preprocessed_data, args) 
                    
    # Load jsonl data from disk
    dataset = load_dataset("json", data_files=all_path_folder_preprocessed_data[0], split="train", cache_dir=f'{args.output_folder}/{args.ft_model_id}')
    valid_dataset = load_dataset("json", data_files=all_path_folder_preprocessed_data[1], split="train", cache_dir=f'{args.output_folder}/{args.ft_model_id}')
    test_dataset = load_dataset("json", data_files=all_path_folder_preprocessed_data[2], split="train", cache_dir=f'{args.output_folder}/{args.ft_model_id}')
    

    # Load model and tokenizer
    model_id = args.base_model_id # "codellama/CodeLlama-7b-hf" # or `mistralai/Mistral-7B-v0.1`
    if args.do_train:
        tensor_data_type = torch.bfloat16  
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=tensor_data_type
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if args.ft_model_path is not None:
            model = AutoPeftModelForCausalLM.from_pretrained(
                args.ft_model_path,
                device_map="auto",
                torch_dtype=tensor_data_type,
                load_in_8bit=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                attn_implementation="flash_attention_2",
                torch_dtype=tensor_data_type,
                quantization_config=bnb_config
            )
    else:
        tensor_data_type = torch.float32 # for reduce the miss matching of ouputs of batch inference
        ft_model_path = f"{args.output_folder}/{args.ft_model_id}" if args.ft_model_path is None else args.ft_model_path
        tokenizer = AutoTokenizer.from_pretrained(ft_model_path)
        model = AutoPeftModelForCausalLM.from_pretrained(
            ft_model_path,
            device_map="auto",
            torch_dtype=tensor_data_type
        )
        
    
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = 'left' 

    # # set chat template to OAI chatML, remove if you start from a fine-tuned model
    model, tokenizer = setup_chat_format(model, tokenizer)
    
    # training config 
    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    peft_config = LoraConfig(
            lora_alpha=128,
            lora_dropout=0.05,
            r=args.lora_r,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM", 
    )

    training_args = TrainingArguments(
        output_dir=f'{args.output_folder}/{args.ft_model_id}',                  # directory to save and repository id
        num_train_epochs= args.epoch,                     # number of training epochs
        max_steps=args.max_steps,
        per_device_train_batch_size=4,          # batch size per device during training
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,          # number of steps before performing a backward/update pass
        gradient_checkpointing=True,            # use gradient checkpointing to save memory
        save_total_limit=1,
        optim="adamw_torch_fused",              # use fused adamw optimizer
        eval_delay=args.eval_delay,                       # log every 10 steps meld:200
        logging_steps=50,                       # log every 10 steps
        eval_steps=50,
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model='weighted-f1',
        greater_is_better=True,
        evaluation_strategy='steps',
        save_strategy="steps",                  # save checkpoint every epoch
        learning_rate=args.lr,                     # learning rate, based on QLoRA paper
        bf16=True,                              # use bfloat16 precision
        tf32=True,                              # use tf32 precision
        max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
        lr_scheduler_type=args.lr_scheduler,           # use constant learning rate scheduler
        push_to_hub=False,                      # push model to hub ##########################
        group_by_length=True,
        report_to="tensorboard",                # report metrics to tensorboard
    ) 

    trainer = LLMErcTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=valid_dataset,
        neftune_noise_alpha=5,
        peft_config=peft_config,
        max_seq_length=args.max_seq_len,
        tokenizer=tokenizer,
        packing=True,
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False, # No need to add additional separator token
        }
    )
    
    # n_trainable_pr, total_pr = get_peft_model(model, peft_config).get_nb_trainable_parameters()
    # print(f"total params: {n_trainable_pr}, trainable params {total_pr}, percentage={n_trainable_pr/total_pr*100}")

    if args.do_train:
        # start training, the model will be automatically saved to the hub and the output directory
        # trainer.train(ignore_keys_for_eval=[])
        trainer.train(resume_from_checkpoint=True if len(glob.glob(f'{args.output_folder}/{args.ft_model_id}/checkpoint-*')) > 0 else None)

        # save model 
        trainer.save_model()
        

    ft_model_path = f'{args.output_folder}/{args.ft_model_id}' if args.ft_model_path is None else args.ft_model_path
        
    if args.do_eval_test:
        result = trainer.evaluate(test_dataset, metric_key_prefix='test')
        print(f"Test result = {result}") 
        
    if args.do_eval_dev:
        result = trainer.evaluate(valid_dataset, metric_key_prefix='valid')
        print(f"Valid result = {result}") 
