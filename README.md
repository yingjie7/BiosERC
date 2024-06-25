## BiosERC: Integrating Biography Speakers Supported by LLMs for ERC Tasks
In the Emotion Recognition in Conversation task, recent investigations have utilized attention mechanisms exploring relationships among utterances from intra- and inter-speakers for modeling emotional interaction between them. However, attributes such as speaker personality traits remain unexplored and present challenges in terms of their applicability to other tasks or compatibility with diverse model architectures. Therefore, this work introduces a novel framework named BiosERC, which investigates speaker characteristics in a conversation. By employing Large Language Models (LLMs), we extract the ``biographical information'' of the speaker within a conversation as supplementary knowledge injected into the model to classify emotional labels for each utterance. Our proposed method achieved state-of-the-art (SOTA) results on three famous benchmark datasets: IEMOCAP, MELD, and EmoryNLP, demonstrating the effectiveness and generalization of our model and showcasing its potential for adaptation to various conversation analysis tasks.

## Results 
Performance comparison between our proposed method and previous works on the test sets.
|                                                |       |             |              |           |
| :--------------------------------------------- | :---: | :---------: | :----------: | :-------: |
| **Methods**                                    |       | **IEMOCAP** | **EmoryNLP** | **MELD**  |
| HiTrans                                        |       |    64.50    |    36.75     |   61.94   |
| DAG                                            |       |    68.03    |    39.02     |   63.65   |
| DialogXL                                       |       |    65.94    |    34.73     |   62.14   |
| DialogueEIN                                    |       |    68.93    |    38.92     |   65.37   |
| SGED + DAG-ERC                                 |       |    68.53    |    40.24     |   65.46   |
| S+PAGE                                         |       |    68.93    |    40.05     |   64.67   |
| InstructERC   _+(ft LLM)_                      |       |  **71.39**  |    41.39     |   69.15   |
|                                                |       |             |              |           |
| Intra/inter ERC (baseline)   ${[AccWR]}_{MLP}$ |       |    67.65    |    39.33     |   64.58   |
| _BiosERC_ $_{  BERT-based}$                    |       |    67.79    |    39.89     |   65.51   |
| _BiosERC_  +ft LLM $_{Llama-2-7b}$             |       |    69.02    |    41.44     |   68.72   |
| _BiosERC_   +ft LLM $_{Llama-2-13b}$           |       |    71.19    |  **41.68**   | **69.83** |
|                                                |       |             |              |           |

##  Data  
unzip the file `data.zip` to extract data.
- IEMOCAP
    Data structure examples: 
    ```json
    {
        # this is first conversation 
        "Ses05M_impro03": { 
            "labels": [
            4,
            2,
            4,
            4 
            ],
            "sentences": [
            "Guess what?",
            "what?",
            "I did it, I asked her to marry me.",
            "Yes, I did it."
            ], 
            "genders": [
            "M",
            "F",
            "M",
            "M",
            "F", 
            ]
        },

        # this is second conversation 
        "Ses05M_impro03": { 
            "labels": [
            4,
            2,
            ],
            "sentences": [
            "Guess what?",
            "what?", 
            ], 
            "genders": [
            "M",
            "F",  
            ]
        }
    }
    ```

##  Python ENV 
Init python environment 
```cmd
    conda create --prefix=./env_py38  python=3.9
    conda activate ./env_py38 
    pip install -r requirements.txt
```

## Run 
1. Init environment follow the above step.
2. Data peprocessing. 
   1. Put all the raw data to the folder `data/`.
    The overview of data structure:
        ```
            .
            ├── data/
            │   ├── llm_vectors/  # save speaker description data
            │   │   ├── meld.valid_spdescV2_Llama-2-70b-chat-hf.json            
            │   │   ├── meld.train_spdescV2_Llama-2-70b-chat-hf.json 
            │   │   ├── meld.test_spdescV2_Llama-2-70b-chat-hf.json
            │   │   └── ...
            │   ├── meld.test.json
            │   ├── meld.train.json
            │   ├── meld.valid.json
            │   ├── ...
            │   ├── iemocap.test.json
            │   ├── iemocap.train.json
            │   └── iemocap.valid.json
            ├── src/
            ├── finetuned_llm/
            └── ...
        ```
3. Train  
    Run following command to train a new model. 
    ```bash 
    python src/llm_bio_extract.py # to extract speaker bio
    bash scrips/train_llm.sh # to train a llm model
    ```
    > **Note**: Please check this scripts to check the setting and choose which data you want to run. 

## Citation 
   
    ```bibtex
    @InProceedings{bioserc,
    author="Xue, Jieying
    and Nguyen, Minh-Phuong
    and Matheny, Blake
    and Nguyen, Le-Minh",
    editor="[updating]",
    title="BiosERC: Integrating Biography Speakers Supported by LLMs for ERC Tasks",
    booktitle="Artificial Neural Networks and Machine Learning -- ICANN 2024",
    year="2024",
    publisher="Springer Nature Switzerland",
    address="Cham",
    pages="[updating]", 
    isbn="[updating]"
    }

    ```
