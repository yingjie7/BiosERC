## AccumWR: Accumulating Word Representations in Multi-level Context Integration for ERC Task 

Emotion Recognition in Conversations (ERC) has attracted increasing attention recently because of its high applicability, which is to predict the sentiment label of each utterance given a conversation as context.    
In order to identify the emotion of a focal sentence, it is crucial to model its meaning fused with contextual information. Many recent studies have focused on capturing different types of context as supporting information and integrated it in various ways: local and global contexts or at the speaker level through intra-speaker and inter-speaker integration. However, the importance of word representations after context integration has not been investigated completely, while word information is also essential to reflect the speaker's emotions in the conversation.
Therefore, in this paper, we investigate the effect of accumulating word vector representations on sentence modeling fused with multi-level contextual integration.  To this end, we propose an effective method for sentence modeling in ERC tasks and achieve competitive state-of-the-art results on four well-known benchmark datasets: Iemocap, MELD, EmoryNLP, and DailyDialog.  
## Results 
Performance comparison between our proposed method and previous works on the test sets.
|                                                 |    |         |              |           |
|:------------------------------------------------|:-----------:|:-----------:|:------------:|:---------:|
| **Methods**                                     | | **IEMOCAP** | **EmoryNLP** | **MELD**  | 
 HiTrans                      |                          | 64.50                | 36.75                 | 61.94             
 DAG                         |                          | 68.03                | 39.02                 | 63.65             
 DialogXL                   |                          | 65.94                | 34.73                 | 62.14             
 DialogueEIN             |                          | 68.93                | 38.92                 | 65.37             
 SGED + DAG-ERC                      |                          | 68.53                | 40.24                 | 65.46             
 S+PAGE                         |                          | 68.93                | 40.05                 | 64.67             
 InstructERC   _+(ft LLM)_ |                          | $\textbf{71.39}$       | 41.39                 | 69.15      
|       
 Intra/inter ERC (baseline)   $\texttt{[AccWR]}_{MLP}$       |     | 67.65                | 39.33                 | 64.58             
 $\texttt{BiosERC}_\texttt{ BERT-based}$           |      | 67.79                | 39.89      | 65.51 
  $\texttt{BiosERC}$  +ft LLM $_\texttt{Llama-2-7b}$  |     | 69.02              | 41.44            | 68.72          
  $\texttt{BiosERC}$  +ft LLM $_\texttt{Llama-2-13b}$ |       | 71.19              | $\textbf{41.68}$        | $\textbf{69.83}$    
|                                                 |    |         |              |           |

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
            ├── data
            │   ├── meld.test.json
            │   ├── meld.train.json
            │   ├── meld.valid.json
            │   ├── iemocap.test.json
            │   ├── iemocap.train.json
            │   └── iemocap.valid.json
            └── ...
        ```
3. Train  
    Run following command to train a new model. 
    ```bash 
    bash scrips/train_llm.sh
    ```
    > **Note**: Please check this scripts to check the setting and choose which data you want to run. 

## Citation 
   
    [updating]