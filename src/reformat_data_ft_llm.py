import json
import re

data_name_pattern = 'train'

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


def flatten_conversation_mixed_by_surrounding(conv, around_window, s_id, genders, data_name):
    new_data = []
    for i, cur_sent in enumerate(conv):
        tmp_window = []
        for j in range(max(0, i-around_window), min(len(conv), i+around_window+1)):
            tmp_window.append(f" {get_speaker_name(s_id, genders[j], data_name=data_name)}: {conv[j]}")

        new_data.append(tmp_window)
    return new_data

def get_label_map(data_name):
    all_data_label_map = {
        "iemocap":   {0:'happy',1:'sad',2:'neutral',3:'angry',4:'excited',5:'frustrated'},
        "emorynlp":  ['Joyful', 'Mad', 'Peaceful', 'Neutral', 'Sad', 'Powerful', 'Scared'],
        "meld":  ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger'],
        "dailydialog":  ['no_emotion', 'happiness', 'sadness', 'surprise',  'anger', 'fear', 'disgust']
    }
    return all_data_label_map[data_name]

def preprocess_desc_speaker(str_in):
    str_in = str_in.split("</s>")[0].replace("<s>", "").replace("\n", " ")
    str_out = re.sub(r" {2,}", " ",  str_in)
    return str_out
 
def gen_default_prompting_messages(data_name, conv, around_window, s_id, desc_speaker_data=None):
    new_conv = []
    samples = []
    for i,sent in enumerate(conv['sentences']):
        new_sent_gender = conv['genders'][i]
        sent_name = get_speaker_name(s_id,new_sent_gender, data_name)
        new_sent =  f'{sent_name}: {sent}'
        new_conv.append(new_sent)
    conv_str = "\n".join(new_conv)
    
    flatten_conv = flatten_conversation_mixed_by_surrounding(conv['sentences'], around_window, s_id, conv['genders'], data_name)
    
    for i, sent in enumerate(new_conv):
        system_msg = f'### You are an expert at analyzing the emotion of utterances among speakers in a conversation.'
        conv_str = "\n".join(flatten_conv[i])
        local_context_msg = f"\n### Given the following conversation as a context \n{conv_str}"
        speaker_name = get_speaker_name(s_id, conv["genders"][i], data_name)
        q_msg =  f'Based on above conversation, which emotional label of {speaker_name} in the utterance \"{conv["sentences"][i]}\".'
            
        label_msg =  get_label_map(data_name)[conv['labels'][i]]

        samples.append({
            "messages":  [
                {'role': "system", 'content': system_msg + local_context_msg},
                {'role': "user", 'content': q_msg},
                {'role': "assistant", 'content': label_msg},
            ]
        })
    return samples
     
def gen_spdescV2_prompting_messages(data_name, conv, around_window, s_id, desc_speaker_data):
    new_conv = []
    for i,sent in enumerate(conv['sentences']):
        new_sent_gender = conv['genders'][i]
        sent_name = get_speaker_name(s_id,new_sent_gender, data_name)
        new_sent =  f'{sent_name}: {sent}'
        new_conv.append(new_sent)
    conv_str = "\n".join(new_conv)
    
    flatten_conv = flatten_conversation_mixed_by_surrounding(conv['sentences'], around_window, s_id, conv['genders'], data_name)
    
    samples = []
    for i, sent in enumerate(new_conv):
        system_msg = f'### You are an expert at analyzing the emotion of utterances among speakers in a conversation.'
        speaker_name = get_speaker_name(s_id, conv["genders"][i], data_name)
        
        desc_str = desc_speaker_data[s_id][i].replace("\n", " ")



        desc_msg = f'\n### Given the characteristic of this speaker, {speaker_name}: \n{desc_str}'
        
        conv_str = "\n".join(flatten_conv[i])
        local_context_msg = f"\n### Given the following conversation as a context \n{conv_str}"
        
        q_msg =  f'Based on above conversation and characteristic of the speakers, which emotional label of {speaker_name} in the utterance \"{conv["sentences"][i]}\".'
        label_msg = get_label_map(data_name)[conv['labels'][i]]
        
        samples.append({
            "messages":  [
                {'role': "system", 'content': system_msg + desc_msg + local_context_msg},
                {'role': "user", 'content': q_msg},
                {'role': "assistant", 'content': label_msg},
            ]
        })
        
    return samples 
      
def process(paths_folder_preprocessed_data, args):
    
    process_kwargs = {}
    for path_folder_preprocessed_data in paths_folder_preprocessed_data:
        
        d_type = 'train' if '.train.' in path_folder_preprocessed_data else \
                'valid' if '.valid.' in path_folder_preprocessed_data else \
                'test' if '.test.' in path_folder_preprocessed_data else None  
        
        folder_data = args.data_folder
        around_window = args.window
        data_name = args.data_name
        path_data_out = path_folder_preprocessed_data
        prompting_type = args.prompting_type
        extract_prompting_llm_id = args.extract_prompting_llm_id 
        
        raw_data = f'{folder_data}/{data_name}.{d_type}.json'
        org_data = json.load(open(raw_data)) # ; org_data = dict([(k,v) for k,v in org_data.items()][:10])
        
        new_format = []
        
        # if use speaker description -> load raw data and preprocess
        if prompting_type not in ["default" ]:
            desc_speaker_data = json.load(open(f'{folder_data}/{data_name}.{d_type}_{prompting_type}_{extract_prompting_llm_id}.json'))
            processed_desc_speaker_data = {}
            if desc_speaker_data is not None and "spdesc" in prompting_type:
                for s_id, desc_all_conv in desc_speaker_data.items():
                    processed_desc_speaker_data[s_id] = [preprocess_desc_speaker(spdesc) for spdesc in desc_all_conv]
                desc_speaker_data = processed_desc_speaker_data   
        else:
            desc_speaker_data = None
            
        # path data out 
        path_processed_data = raw_data.replace(".json", f".0shot_w{around_window}_{prompting_type}.jsonl") if path_data_out is None else path_data_out
        
        # prompting process function 
        process_function_map = {
            "spdescV2": gen_spdescV2_prompting_messages,
            "default": gen_default_prompting_messages,
        }
        
        process_func = process_function_map.get(prompting_type, process_function_map['default'])
        print(f"- process prompting by {process_func.__name__}")
        
        for s_id, conv in org_data.items(): 
            process_args = [data_name, conv, around_window, s_id, desc_speaker_data]
            samples = process_func(*process_args, **process_kwargs)
            new_format = new_format + samples
            
        with open(f'{path_processed_data}', 'wt') as f:
            new_format = [json.dumps(e) for e in new_format]
            f.write("\n".join(new_format))


# if __name__=="__main__":
#     process('train', around_window=5, use_spdesc=True)
#     process('test', around_window=5, use_spdesc=True)
#     process('valid', around_window=5, use_spdesc=True)

