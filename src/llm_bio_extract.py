
import sys
import os
from torch.utils.data import DataLoader
import traceback
from tqdm import tqdm
import json 
from transformers import LlamaTokenizer, AutoModel, AutoTokenizer, LlamaForCausalLM

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, AutoConfig


dataset_name = 'iemocap'
data_folder = './data/'
prompt_type = 'spdescV2'


print("Loading model ...")
# trained with chat and instruction
model_name = 'meta-llama/Llama-2-70b-chat-hf'
# model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'  #  standard model
tensor_data_type = torch.bfloat16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=tensor_data_type
)
model = LlamaForCausalLM.from_pretrained(
    model_name,
    # return_dict=True,
    load_in_8bit=True,
    device_map="auto",
    # low_cpu_mem_usage=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


class BatchPreprocessor(object): 
    def __init__(self, tokenizer, dataset_name=None, window_ct=2) -> None:
        self.tokenizer = tokenizer
        self.separate_token_id = self.tokenizer.convert_tokens_to_ids("</s>")
        self.dataset_name  = dataset_name
        self.window_ct = window_ct
    
    @staticmethod
    def load_raw_data(path_data):
        raw_data = json.load(open(path_data))
        if isinstance(raw_data, dict):
            new_data_list = []
            for k, v in raw_data.items():
                v['s_id'] = k
                new_data_list.append(v)
            return new_data_list
        elif isinstance(raw_data, list):
            return raw_data
            
    
    @staticmethod
    def get_speaker_name(s_id, gender, data_name):
        if data_name == "iemocap":
            # iemocap: label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
            speaker = {
                        "Ses01": {"F": "Mary", "M": "James"},
                        "Ses02": {"F": "Patricia", "M": "John"},
                        "Ses03": {"F": "Jennifer", "M": "Robert"},
                        "Ses04": {"F": "Linda", "M": "Michael"},
                        "Ses05": {"F": "Elizabeth", "M": "William"},
                    }
            s_id_first_part = s_id[:5]
            return speaker[s_id_first_part][gender].upper()
        elif data_name in ['meld', "emorynlp"]:
            # emorynlp: label index mapping =  {'Joyful': 0, 'Mad': 1, 'Peaceful': 2, 'Neutral': 3, 'Sad': 4, 'Powerful': 5, 'Scared': 6}
            # meld: label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
            gender_idx = gender.index(1) 
            return f"SPEAKER_{gender_idx}"
        elif data_name=='dailydialog':
            # dailydialog:  {'no_emotion': 0, 'happiness': 1, 'sadness': 2, 'surprise': 3,  'anger': 4, 'fear': 5, 'disgust':6}
            return f"SPEAKER_{gender}"
        
    def sentence_mixed_by_surrounding(self, sentences, around_window, s_id, genders, data_name):
        new_sentences = []
        for i, cur_sent in enumerate(sentences):
            tmp_s = ""
            for j in range(max(0, i-around_window), min(len(sentences), i+around_window+1)):
                if i == j:
                    tmp_s += " </s>"
                tmp_s +=  f" {self.get_speaker_name(s_id, genders[j], data_name=data_name)}: {sentences[j]}"
                if i == j:
                    tmp_s += " </s>"
            new_sentences.append(tmp_s)
        return new_sentences
    
    def __call__(self, batch):
        raw_sentences = []
        raw_sentences_flatten = []
        labels = []

        # masked tensor  
        lengths = [len(sample['sentences']) for sample in batch]
        max_len_conversation = max(lengths)
        padding_utterance_masked = torch.BoolTensor([[False]*l_i+ [True]*(max_len_conversation - l_i) for l_i in lengths])

        # collect all sentences
        # - intra speaker
        intra_speaker_masekd_all = torch.BoolTensor(len(batch), max_len_conversation,max_len_conversation)
        for i, sample in enumerate(batch):
            sentences_mixed_arround = self.sentence_mixed_by_surrounding(sample['sentences'], 
                                                                        around_window=self.window_ct, 
                                                                        s_id=sample['s_id'], 
                                                                        genders=sample['genders'],
                                                                        data_name=self.dataset_name)
        
            # conversation padding 
            padded_conversation = sentences_mixed_arround + ["<pad_sentence>"]* (max_len_conversation - lengths[i])
            raw_sentences.append(padded_conversation)
            raw_sentences_flatten += padded_conversation

            # label padding 
            labels += [int(label) for label in sample['labels']] + [-1]* (max_len_conversation - lengths[i])

            # speaker
            intra_speaker_masekd= torch.BoolTensor(len(padded_conversation),len(padded_conversation)).fill_(False)
            for j in  range(len( sample['genders'])):
                for k in  range(len( sample['genders'])):
                    gender_j = sample['genders'][j]
                    gender_k = sample['genders'][k]

                    if gender_j == gender_k:
                        intra_speaker_masekd[j][k] = True
                    else:
                        intra_speaker_masekd[j][k] = False

            intra_speaker_masekd_all[i] = intra_speaker_masekd

        if len(labels)!= len(raw_sentences_flatten):
            print('len(labels)!= len(raw_sentences_flatten)')

        # utterance vectorizer
        # v_single_sentences = self._encoding(sample['sentences'])
        contextual_sentences_ids = self.tokenizer(raw_sentences_flatten,  padding='longest', max_length=512, truncation=True, return_tensors='pt')
        sent_indices, word_indices = torch.where(contextual_sentences_ids['input_ids'] == self.separate_token_id)
        gr_sent_indices = [[] for e in range(len(raw_sentences_flatten))]
        for sent_idx, w_idx in zip (sent_indices, word_indices):
            gr_sent_indices[sent_idx].append(w_idx.item())
            
        cur_sentence_indexes_masked = torch.BoolTensor(contextual_sentences_ids['input_ids'].shape).fill_(False)
        for i in range(contextual_sentences_ids['input_ids'].shape[0]):
            if raw_sentences_flatten[i] =='<pad_sentence>':
                cur_sentence_indexes_masked[i][gr_sent_indices[i][0]] = True
                continue
            for j in range(contextual_sentences_ids['input_ids'].shape[1]):
                if  gr_sent_indices[i][0] <= j <= gr_sent_indices[i][1]:
                    cur_sentence_indexes_masked[i][j] = True

        return (contextual_sentences_ids, torch.LongTensor(labels), padding_utterance_masked, intra_speaker_masekd_all, cur_sentence_indexes_masked, raw_sentences) 


class BatchPreprocessorLLM(BatchPreprocessor):
    def __init__(self, tokenizer, dataset_name=None, window_ct=2, emotion_labels=[]) -> None:
        self.tokenizer = tokenizer
        self.separate_token_id = self.tokenizer.convert_tokens_to_ids("</s>")
        self.dataset_name = dataset_name
        self.window_ct = window_ct
        self.emotion_labels = emotion_labels
        self.printted = False

    @staticmethod
    def load_raw_data(path_data):
        raw_data = json.load(open(path_data))
        if isinstance(raw_data, dict):
            new_data_list = []
            for k, v in raw_data.items():
                v['s_id'] = k
                new_data_list.append(v)
            return new_data_list
        elif isinstance(raw_data, list):
            return raw_data

    @staticmethod
    def get_speaker_name(s_id, gender, data_name):
        if data_name == "iemocap":
            # iemocap: label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
            speaker = {
                "Ses01": {"F": "Mary", "M": "James"},
                "Ses02": {"F": "Patricia", "M": "John"},
                "Ses03": {"F": "Jennifer", "M": "Robert"},
                "Ses04": {"F": "Linda", "M": "Michael"},
                "Ses05": {"F": "Elizabeth", "M": "William"},
            }
            s_id_first_part = s_id[:5]
            return speaker[s_id_first_part][gender].upper()
        elif data_name in ['meld', "emorynlp"]:
            # emorynlp: label index mapping =  {'Joyful': 0, 'Mad': 1, 'Peaceful': 2, 'Neutral': 3, 'Sad': 4, 'Powerful': 5, 'Scared': 6}
            # meld: label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
            gender_idx = gender.index(1)
            return f"SPEAKER_{gender_idx}"
        elif data_name == 'dailydialog':
            # dailydialog:  {'no_emotion': 0, 'happiness': 1, 'sadness': 2, 'surprise': 3,  'anger': 4, 'fear': 5, 'disgust':6}
            return f"SPEAKER_{gender}"

    def sentence_mixed_by_surrounding(self, sentences, around_window, s_id, genders, data_name):
        new_conversations = []
        align_sents = []
        for i, cur_sent in enumerate(sentences):
            tmp_s = ""
            for j in range(max(0, i-around_window), min(len(sentences), i+around_window+1)):
                u_j = f"{self.get_speaker_name(s_id, genders[j], data_name=data_name)}: {sentences[j]}"
                if i == j:
                    align_sents.append(u_j)
                tmp_s += f"\n{u_j}"
            new_conversations.append(tmp_s)
        return new_conversations, align_sents

    def __call__(self, batch):
        raw_sentences = []
        raw_sentences_flatten = []
        labels = []
        speaker_info = []
        listener_info = []

        # masked tensor
        lengths = [len(sample['sentences']) for sample in batch]
        max_len_conversation = max(lengths)
        padding_utterance_masked = torch.BoolTensor(
            [[False]*l_i + [True]*(max_len_conversation - l_i) for l_i in lengths])

        # collect all sentences
        # - intra speaker
        flatten_data = []
        intra_speaker_masekd_all = torch.BoolTensor(
            len(batch), max_len_conversation, max_len_conversation)
        for i, sample in enumerate(batch):
            new_conversations, align_sents = self.sentence_mixed_by_surrounding(sample['sentences'],
                                                                                around_window=self.window_ct,
                                                                                s_id=sample['s_id'],
                                                                                genders=sample['genders'],
                                                                                data_name=self.dataset_name)
            few_shot_example = """\n=======
Context: Given predefined emotional label set [happy, sad, neutral, angry, excited, frustrated], and bellow conversation: 
"
PATRICIA: You know, it's lovely here, the air is sweet.
PATRICIA: No, not sorry.  But, um. But I'm not gonna stay.
JOHN: The trouble is, I planned on sort of sneaking up on you on a period of a week or so.  But they take it for granted that we're all set.
PATRICIA: I knew they would, your mother anyway.
PATRICIA: Well, from her point of view, why else would I come?
PATRICIA: I guess this is why I came.
JOHN: I'm embarrassing you and I didn't want to tell it to you here.  I wanted some place we'd never been before.  A place where we'd be brand new to each other.
PATRICIA: Well, you started to write me
JOHN: You felt something that far back?
PATRICIA: Every day since.
JOHN: Ann, why didn't you let me know?
JOHN: Let's drive someplace.  I want to be alone with you.
JOHN: No.  Nothing like that.
"

Question: What is the emotion of the speaker at the utterance "PATRICIA: Well, from her point of view, why else would I come?"?
Answer: neutral

Question: What is the emotion of the speaker at the utterance "PATRICIA: I guess this is why I came."?
Answer: happy

Question: What is the emotion of the speaker at the utterance "JOHN: I'm embarrassing you and I didn't want to tell it to you here.  I wanted some place we'd never been before.  A place where we'd be brand new to each other."?
Answer: excited
"""
            for i_u, (conv, utterance) in enumerate(zip(new_conversations, align_sents)):
                prompt_extract_context_vect = few_shot_example + \
                    f"\n=======\nContext: Given predefined emotional label set [{', '.join(self.emotion_labels)}], and bellow conversation:\n\"{conv}\n\"\n\nQuestion: What is the emotion of the speaker at the utterance \"{utterance}\"?\nAnswer:"
                if not self.printted:
                    print(prompt_extract_context_vect)
                    self.printted = True

                inputs = self.tokenizer(
                    prompt_extract_context_vect, return_tensors="pt")
                input_ids = inputs["input_ids"]
                flatten_data.append({
                    "s_id": sample['s_id'],
                    "u_idx": i_u,
                    "prompt_content": prompt_extract_context_vect,
                    "input_ids": input_ids,
                }
                )

        return flatten_data


class BatchPreprocessorLLMSpeakerDescription(BatchPreprocessor):
    def __init__(self, tokenizer, dataset_name=None, window_ct=2, emotion_labels=[]) -> None:
        self.tokenizer = tokenizer
        self.separate_token_id = self.tokenizer.convert_tokens_to_ids("</s>")
        self.dataset_name = dataset_name
        self.window_ct = window_ct
        self.emotion_labels = emotion_labels

    @staticmethod
    def load_raw_data(path_data):
        raw_data = json.load(open(path_data))
        if isinstance(raw_data, dict):
            new_data_list = []
            for k, v in raw_data.items():
                v['s_id'] = k
                new_data_list.append(v)
            return new_data_list
        elif isinstance(raw_data, list):
            return raw_data

    @staticmethod
    def get_speaker_name(s_id, gender, data_name):
        if data_name == "iemocap":
            # iemocap: label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
            speaker = {
                "Ses01": {"F": "Mary", "M": "James"},
                "Ses02": {"F": "Patricia", "M": "John"},
                "Ses03": {"F": "Jennifer", "M": "Robert"},
                "Ses04": {"F": "Linda", "M": "Michael"},
                "Ses05": {"F": "Elizabeth", "M": "William"},
            }
            s_id_first_part = s_id[:5]
            return speaker[s_id_first_part][gender].upper()
        elif data_name in ['meld', "emorynlp"]:
            # emorynlp: label index mapping =  {'Joyful': 0, 'Mad': 1, 'Peaceful': 2, 'Neutral': 3, 'Sad': 4, 'Powerful': 5, 'Scared': 6}
            # meld: label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
            gender_idx = gender.index(1)
            return f"SPEAKER_{gender_idx}"
        elif data_name == 'dailydialog':
            # dailydialog:  {'no_emotion': 0, 'happiness': 1, 'sadness': 2, 'surprise': 3,  'anger': 4, 'fear': 5, 'disgust':6}
            return f"SPEAKER_{gender}"

    def preprocess(self, all_conversations):

        new_data = {}
        gr_by_len = {}
        for i, sample in enumerate(all_conversations):

            all_utterances = []
            all_speaker_names = []
            for i_u, u in enumerate(sample['sentences']):
                speaker_name = self.get_speaker_name(
                    sample['s_id'], sample['genders'][i_u], self.dataset_name)
                u_full_name = f'{speaker_name}: {u}'
                all_utterances.append(u_full_name)
                all_speaker_names.append(speaker_name)

            full_conversation = "\n".join(all_utterances)
            prompts_speaker_description_word_ids = {}
            prompting_input = {}
            for speaker_name in set(all_speaker_names):
                prompting = "\nGiven this conversation between speakers: \n\"\n" + full_conversation + \
                    "\n\"\nIn overall of above conversation, what do you think about the characteristics speaker {}? (Note: provide an answer within 250 words)".format(speaker_name)

                prompts_speaker_description_word_ids[speaker_name] = self.tokenizer(
                    prompting, return_tensors="pt")["input_ids"]
                prompting_input[speaker_name] = prompting

                # group by len for batch decode by llm
                if prompts_speaker_description_word_ids[speaker_name].shape[-1] not in gr_by_len:
                    gr_by_len[prompts_speaker_description_word_ids[speaker_name].shape[-1]] = []
                gr_by_len[prompts_speaker_description_word_ids[speaker_name].shape[-1]].append({
                    'w_ids': prompts_speaker_description_word_ids[speaker_name],
                    'conv_id': sample['s_id'],
                    'type_data': sample['type_data'],
                    "prompting_input": prompting,
                    'speaker_name': speaker_name,
                    'all_speaker_names': all_speaker_names
                })

        return gr_by_len


raw_data = []
for type_data in ['valid', 'test', 'train']:
    data_name_pattern = f'{dataset_name}.{type_data}'
    path_processed_data = f'{data_folder}/{data_name_pattern}_{prompt_type}_{model_name.split("/")[-1]}.json'

    org_raw_data = BatchPreprocessorLLMSpeakerDescription.load_raw_data(
        f"{data_folder}/{data_name_pattern}.json")

    if os.path.exists(path_processed_data):
        processed_data = json.load(open(path_processed_data, 'rt'))
        print(
            f'- sucessful processed {len(processed_data)}/{len(org_raw_data)} conversations in data-type ={type_data}')
        json.dump(processed_data, open(
            path_processed_data+"_backup.json", 'wt'), indent=2)
        org_raw_data = [e for e in org_raw_data if e['s_id']
                        not in processed_data]

    print(
        f'- Continue process {len(org_raw_data)} conversations in data-type ={type_data}')
    for e in org_raw_data:
        e['type_data'] = type_data
    raw_data = raw_data + org_raw_data

data_preprocessor = BatchPreprocessorLLMSpeakerDescription(tokenizer, dataset_name=dataset_name, window_ct=4,
                                                           emotion_labels=['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated'])

gr_by_len = data_preprocessor.preprocess(raw_data)
all_data = {}
print_one_time = True
for len_promting, speaker_promts in tqdm(gr_by_len.items()):
    for batch_size in [8, 5, 2, 1]:
        try:
            all_promtings_texts = [e['prompting_input']
                                   for e in speaker_promts]
            data_loader = DataLoader(all_promtings_texts,
                                     batch_size=batch_size,
                                     shuffle=False)
            output_sp_desc = []
            with torch.no_grad():
                for i, speaker_promts_in_batch in enumerate(data_loader):
                    # batch decoded by llm
                    inputs = tokenizer(speaker_promts_in_batch,
                                       return_tensors="pt", padding=False)
                    input_ids = inputs["input_ids"].to("cuda")
                    with torch.no_grad():
                        outputs = model.generate(input_ids, max_new_tokens=300)
                    output_text = tokenizer.batch_decode(
                        outputs, skip_special_tokens=True)

                    for j, e in enumerate(output_text):
                        output_sp_desc.append(
                            e.replace(all_promtings_texts[j], ""))

                    if print_one_time:
                        print(output_text)
                        print(output_sp_desc)
                        print_one_time = False

                for i, out in enumerate(output_sp_desc):
                    speaker_promts[i]['sp_desc'] = out
            break

        except Exception as e:
            traceback.print_exc()
            print(e)
            if batch_size == 1:
                print(["Errr "]*10)

for type_data in ['valid', 'test', 'train']:
    data_name_pattern = f'{dataset_name}.{type_data}'
    path_processed_data = f'{data_folder}/{data_name_pattern}_{prompt_type}_{model_name.split("/")[-1]}.json'

    processed_data = {}
    if os.path.exists(path_processed_data):
        processed_data = json.load(open(path_processed_data, 'rt'))
        print(
            f'- load processed [old] {len(processed_data)} conversations in data-type ={type_data}')

    all_data = {}
    for len_promting, speaker_promts in gr_by_len.items():
        for description in speaker_promts:
            if type_data != description['type_data']:
                continue

            if description['conv_id'] not in all_data:
                all_data[description['conv_id']] = {
                    'all_speaker_names': description['all_speaker_names'],
                    'vocab_sp2desc':  {}
                }
            all_data[description['conv_id']
                     ]['vocab_sp2desc'][description['speaker_name']] = description['sp_desc']

    print(
        f'- sucessful processed [new] {len(all_data)} conversations in data-type ={type_data}')
    # json.dump(all_data, open(f'{path_data}_new.json', 'wt'), indent=2)

    all_data_new = {}
    for k, v in all_data.items():
        all_data_new[k] = []
        for sp_name in v['all_speaker_names']:
            all_data_new[k].append(v['vocab_sp2desc'][sp_name])

    print(
        f'- update processed [new] {len(all_data_new)} + [old] {len(processed_data)} conversations in data-type ={type_data}')
    all_data_new.update(processed_data)
    json.dump(all_data_new, open(f'{path_processed_data}', 'wt'), indent=2)
