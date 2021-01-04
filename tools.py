import os
import torch
from torch.utils.data import TensorDataset,DataLoader



def make_dataset(input_id, input_segment, input_attention, input_maskLM,input_pos):
    all_input_id = torch.tensor(
        [input_id for input_id in input_id], dtype=torch.long)
    all_input_segment = torch.tensor(
        [input_segment for input_segment in input_segment], dtype=torch.long)
    all_input_attention = torch.tensor(
        [input_attention for input_attention in input_attention], dtype=torch.long)
    all_input_maskLM = torch.tensor(
        [input_maskLM for input_maskLM in input_maskLM], dtype=torch.long)
    all_input_pos = torch.tensor(
        [input_pos for input_pos in input_pos], dtype=torch.long)
    full_dataset = TensorDataset(
        all_input_id, all_input_segment, all_input_attention, all_input_maskLM,all_input_pos)

    return full_dataset

def get_trained_model_path(folder_name):
    model_folder_list=os.listdir("trained_model/")
    model_folder_list.sort()
    new_trained_model_path=""
    folder = os.path.exists("trained_model/"+folder_name)
    if not folder:
        os.makedirs('trained_model/'+folder_name)
    new_trained_model_path='trained_model/'+folder_name

    assert new_trained_model_path!=""

    return new_trained_model_path

def write_message(file_path,message):
    with open(file_path,'a',encoding='utf-8') as f:
        f.write(message)

# 用來做原始BERT 需要的Embedding
class OriBertEmbedding:
    def __init__(self,input_sentence):
       self.input_sentence=input_sentence


    def segment_embedding(self):
        # segment_embedding
        SEP_flag=False
        sep_token="[SEP]"
        input_segment=[]
        for token in self.input_sentence:  
            if (SEP_flag):
                input_segment.append(1)
            else:
                if token != sep_token:
                    input_segment.append(0) 
                elif token==sep_token :
                    SEP_flag=True
                    input_segment.append(0) 
        return input_segment

    def attention_embedding(self):
        # attention_embedding
        input_attention=[]

        input_attention.extend([1]*len(self.input_sentence))

        return input_attention