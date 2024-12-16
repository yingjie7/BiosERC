## BiosERC: Integrating Biography Speakers Supported by LLMs for ERC Tasks
In the Emotion Recognition in Conversation task, recent investigations have utilized attention mechanisms exploring relationships among utterances from intra- and inter-speakers for modeling emotional interaction between them. However, attributes such as speaker personality traits remain unexplored and present challenges in terms of their applicability to other tasks or compatibility with diverse model architectures. Therefore, this work introduces a novel framework named BiosERC, which investigates speaker characteristics in a conversation. By employing Large Language Models (LLMs), we extract the ``biographical information'' of the speaker within a conversation as supplementary knowledge injected into the model to classify emotional labels for each utterance. Our proposed method achieved state-of-the-art (SOTA) results on three famous benchmark datasets: IEMOCAP, MELD, and EmoryNLP, demonstrating the effectiveness and generalization of our model and showcasing its potential for adaptation to various conversation analysis tasks.

Full paper here: [https://link.springer.com/chapter/10.1007/978-3-031-72344-5_19](https://link.springer.com/chapter/10.1007/978-3-031-72344-5_19)

## Results 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bioserc-integrating-biography-speakers/emotion-recognition-in-conversation-on-meld)](https://paperswithcode.com/sota/emotion-recognition-in-conversation-on-meld?p=bioserc-integrating-biography-speakers)<br/>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bioserc-integrating-biography-speakers/emotion-recognition-in-conversation-on-4)](https://paperswithcode.com/sota/emotion-recognition-in-conversation-on-4?p=bioserc-integrating-biography-speakers)<br/>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bioserc-integrating-biography-speakers/emotion-recognition-in-conversation-on)](https://paperswithcode.com/sota/emotion-recognition-in-conversation-on?p=bioserc-integrating-biography-speakers)

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
            │   ├── meld.valid_spdescV2_Llama-2-70b-chat-hf.json # speaker biography will be generated by run `python src/llm_bio_extract.py`
            │   ├── meld.train_spdescV2_Llama-2-70b-chat-hf.json # speaker biography will be generated by run `python src/llm_bio_extract.py`
            │   ├── meld.test_spdescV2_Llama-2-70b-chat-hf.json  # speaker biography will be generated by run `python src/llm_bio_extract.py`
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
    
4. Infer trained model from huggingface
    Download model and data from [https://huggingface.co/phuongnm94/BiosERC](https://huggingface.co/phuongnm94/BiosERC) 
    ```bash 
    bash scrips/infer.sh # to infer a llm-based BiosERC: Llama-13b
    ```
    or run 
    ```bash 
    infer_bioserc_bertbased.ipynb # to infer BiosERC bert based model 
    ```
    > **Note**: Please check all the path of data and models related.
## Citation 
   
```bibtex
@InProceedings{10.1007/978-3-031-72344-5_19,
    author="Xue, Jieying
    and Nguyen, Minh-Phuong
    and Matheny, Blake
    and Nguyen, Le-Minh",
    editor="Wand, Michael
    and Malinovsk{\'a}, Krist{\'i}na
    and Schmidhuber, J{\"u}rgen
    and Tetko, Igor V.",
    title="BiosERC: Integrating Biography Speakers Supported by LLMs for ERC Tasks",
    booktitle="Artificial Neural Networks and Machine Learning -- ICANN 2024",
    year="2024",
    publisher="Springer Nature Switzerland",
    address="Cham",
    pages="277--292",
    abstract="In the Emotion Recognition in Conversation task, recent investigations have utilized attention mechanisms exploring relationships among utterances from intra- and inter-speakers for modeling emotional interaction between them. However, attributes such as speaker personality traits remain unexplored and present challenges in terms of their applicability to other tasks or compatibility with diverse model architectures. Therefore, this work introduces a novel framework named BiosERC, which investigates speaker characteristics in a conversation. By employing Large Language Models (LLMs), we extract the ``biographical information'' of the speaker within a conversation as supplementary knowledge injected into the model to classify emotional labels for each utterance. Our proposed method achieved state-of-the-art (SOTA) results on three famous benchmark datasets: IEMOCAP, MELD, and EmoryNLP, demonstrating the effectiveness and generalization of our model and showcasing its potential for adaptation to various conversation analysis tasks. Our source code is available at https://github.com/yingjie7/BiosERC.",
    isbn="978-3-031-72344-5"
}

```

## Licensing Information
- **MELD**: Licensed under [GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).
- **EMORYNLP**: Licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).
- **IEMOCAP**: Licensed under a non-commercial research license. Refer to the official [IEMOCAP website](https://sail.usc.edu/iemocap/) for terms of use.