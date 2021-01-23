import os
import torch
import spacy
import argparse
import torch.nn.functional as F

from tqdm import tqdm
from transformers import BertTokenizer,BertConfig,BertForMaskedLM
from tools import get_accepted_pos_list,get_dataset_list
from data_sequence_preprocess import get_all_syntactic_keyword_list,insert_sep_token

## parser settings
parser = argparse.ArgumentParser()
parser.add_argument('--file_path', '-fp', type=str)
parser.add_argument('--model_path', '-mdp', type=str)
parser.add_argument('--output_file_name', '-oflnm', type=str)
parser.add_argument('--gpu_num', '-gpun', type=int)
args = parser.parse_args()

##　預設值
tokenizer = BertTokenizer(vocab_file='bert-base-uncased-vocab.txt')
# nlp = spacy.load("model/spacy/en_core_web_md-2.3.1/en_core_web_md/en_core_web_md-2.3.1")
accepted_pos_list=get_accepted_pos_list()

## for debug
args.file_path='dataset/train_10.txt'
args.model_path='test_squencemodel/0/pytorch_model.bin'
args.output_file_name="sequence.txt"
args.gpu_num=0
##
def main():

    # GPU setting
    if args.gpu_num > 1 :
        device = torch.device('cuda')
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    else:
        device = torch.device('cpu')
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    # # load model
    config = BertConfig.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained(args.model_path, config=config)
    model.to(device)
    model.eval()

    # read data
    semantic_list,syntactic_list=get_dataset_list(args.file_path)
    all_syntactic_keyword_list=get_all_syntactic_keyword_list(syntactic_list)
    all_syntactic_keyword_with_sep_list=insert_sep_token(all_syntactic_keyword_list)
    
    for index,semantic_sentence in enumerate(semantic_list) :
        predict_sentence=["[MASK]"]
    
        while("[MASK]" in predict_sentence):
            input_ids_list,input_segment_list,input_attention_list=data_preprocess(semantic_sentence,all_syntactic_keyword_with_sep_list[index],predict_sentence)
            input_id_tensor,input_segment_tensor,input_attention_tensor=convert_to_tensor(input_ids_list,input_segment_list,input_attention_list)
            
            outputs=model(return_dict=True,input_ids=input_id_tensor,token_type_ids=input_segment_tensor,
                    attention_mask=input_attention_tensor)
            logits=outputs[0]
            maskpos=input_ids_list.index(103)
            predicted_index = torch.argmax(logits[0, maskpos]).item()
            predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]

            
            predict_sentence.remove("[MASK]")
            
            if predicted_token!="[SEP]":
                predict_sentence.append(predicted_token)
                predict_sentence.append("[MASK]")
            else:
                predict_sentence.append(predicted_token)
                print(predict_sentence)
                break
            
    return 0

def data_preprocess(semantic_sentence,syntactic_keyword_with_sep,predict_sentence):

    # 處理 input_ids
    input_token_list=[]
    input_token_list.append("[CLS]")
    input_token_list.extend(tokenizer.tokenize(semantic_sentence))
    input_token_list.append("[SEP]")
    input_token_list.extend(syntactic_keyword_with_sep)
    input_token_list.extend(predict_sentence)
    
    input_ids_list=tokenizer.convert_tokens_to_ids(input_token_list)
    
    # 處理 segment embedding (token type)
    semantic_part=[]
    semantic_part.append("[CLS]")
    semantic_part.extend(tokenizer.tokenize(semantic_sentence))
    semantic_part.append("[SEP]")
    # input分三個部分
    semantic_part_len=len(semantic_part)
    keyword_length=len(syntactic_keyword_with_sep)
    sequence_length=len(predict_sentence)


    input_segment_list=[]
    input_segment_list.extend([0]*semantic_part_len)
    input_segment_list.extend([1]*keyword_length)
    input_segment_list.extend([0]*sequence_length)

    # 處理 attention embedding
    input_attention_list=[]
    input_attention_list.extend([1]*len(input_token_list))

    padding_len=512-len(input_token_list)
    input_ids_list.extend([0]*padding_len)
    input_segment_list.extend([0]*padding_len)
    input_attention_list.extend([0]*padding_len)
    assert len(input_ids_list)==len(input_segment_list)
    assert len(input_ids_list)==len(input_attention_list)
    
    return input_ids_list,input_segment_list,input_attention_list

def convert_to_tensor(input_ids_list,input_segment_list,input_attention_list):
    input_id_tensor=torch.tensor([input_ids_list])
    input_segment_tensor=torch.tensor([input_segment_list])
    input_attention_tensor=torch.tensor([input_attention_list])

    return input_id_tensor,input_segment_tensor,input_attention_tensor
if __name__ == "__main__":
    main()