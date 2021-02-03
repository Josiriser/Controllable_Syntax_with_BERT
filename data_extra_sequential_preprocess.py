import torch
import spacy
import pickle
import argparse

from tqdm import tqdm
from torch.utils.data import TensorDataset
from transformers import BertConfig, BertTokenizer
from tools import get_dataset_list,get_token_list_position
from tools import get_accepted_pos_list,get_sentence_tag_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file_path', '-trfp', type=str)
    parser.add_argument('--test_file_path', '-tefp', type=str)
    args = parser.parse_args()

    ##　for debug
    if args.train_file_path==None:
        args.train_file_path="dataset/mingda_train_10.txt"
    if args.test_file_path==None:
        args.test_file_path="dataset/mingda_test_5.txt"
    ##

    ## pre-load
    tokenizer=BertTokenizer(vocab_file='bert-base-uncased-vocab.txt')
    nlp = spacy.load("model/spacy/en_core_web_md-2.3.1/en_core_web_md/en_core_web_md-2.3.1")
    accepted_pos_list=get_accepted_pos_list()
    ##
    data_path_dict={
    "train":args.train_file_path,
    "test":args.test_file_path
    }
    for key,data_path in data_path_dict.items():
        print("get {0} data ...".format(key))
        semantics_list,syntactic_list=get_dataset_list(data_path)
        print("get {0} all syntactic keyword list ...".format(key))
        extrapolate_sequential_syntactic(syntactic_list,accepted_pos_list,tokenizer,nlp)
        print("")

def extrapolate_sequential_syntactic(syntactic_list,accepted_pos_list,tokenizer,nlp):
    syntactic_list_in_dict={}
    part_maskLM_embedding_list_in_dict={}
    for index,syntactic_sentence in enumerate(tqdm(syntactic_list)):
        tag_dict=get_sentence_tag_dict(nlp,syntactic_sentence)
        token_list=tokenizer.tokenize(syntactic_sentence)
        syntactic_sentence_token_position_dict=get_token_list_position(token_list)
        syntactic_token_position_dict,syntactic_token_position_with_empty_dict=replace_unassigned_token_with_empty(token_list,accepted_pos_list,tag_dict)
        
        syntactic_list_in_dict[index],part_maskLM_embedding_list_in_dict[index]=extrapolate(syntactic_sentence,syntactic_sentence_token_position_dict,syntactic_token_position_dict,syntactic_token_position_with_empty_dict,tokenizer)
        
        input()


def replace_unassigned_token_with_empty(token_list,accepted_pos_list,tag_dict):
    # 用 [E] 取代那些沒有被選到的詞性的token
    syntactic_token_position_dict={}
    syntactic_token_position_with_empty_dict={}
    for i,token in enumerate(token_list):
        if token in tag_dict:
            if tag_dict[token] in accepted_pos_list:
                syntactic_token_position_dict[i]=token
                syntactic_token_position_with_empty_dict[i]=token
            else:
                syntactic_token_position_with_empty_dict[i]="[E]"
        else:
            syntactic_token_position_with_empty_dict[i]="[E]"
    return syntactic_token_position_dict,syntactic_token_position_with_empty_dict

def extrapolate(syntactic_sentence,syntactic_sentence_token_position_dict,syntactic_token_position_dict,syntactic_token_position_with_empty_dict,tokenizer):
    all_sequential_extrapolate_list=[]
    sequential_first_extrapolate_list,sequential_first_extrapolate_pair_list=sequential_extrapolate(syntactic_sentence_token_position_dict,syntactic_token_position_dict)
   

    # recall_extrapolate_pair_list=sequential_first_extrapolate_pair_list.copy()
    # all_sequential_extrapolate_list.append(sequential_first_extrapolate_list.copy())
    # part_labels_embeddings_list=[]
    # for index,position_token_pair in enumerate(recall_extrapolate_pair_list):
    #     if position_token_pair[1]=="[MASK]":
    #         border=recall_extrapolate_pair_list[index+1][0]
    #         labels=[]
    #         length=len(recall_extrapolate_pair_list)-(index+1)
    #         for j in range(border):
    #             # 找 label 
    #             labels.append(tokenizer.convert_tokens_to_ids(syntactic_sentence_token_position_dict[j]))
    #             labels.extend([-100]*length)
    #             part_labels_embeddings_list.append(labels.copy())

    #     print()
    return 0,1

def sequential_extrapolate(syntactic_sentence_token_position_dict,syntactic_token_position_dict):
    all_sequential_extrapolate_list=[]
    sequential_extrapolate_list=[]
    sequential_extrapolate_pair_list=[]
    input_sentence=[]
    input_sentence_dict={}
    temp_input_sentence=[]
    single_sequence_sentence_list=[]
    # 每個 token 前後都插入 [X]
    temp_pos=-999
    temp_input_sentence.append((temp_pos,"[X]"))
    temp_pos+=1
    for position in syntactic_token_position_dict:
        temp_input_sentence.append((position,syntactic_token_position_dict[position]))
        temp_input_sentence.append((temp_pos,"[X]"))
        temp_pos+=1

    # 取得 [X] 的位置 (取第一個看到的)
    for position_token_pair in temp_input_sentence:
        if position_token_pair[1]=="[X]":
            continue

    # 把[X]後半段先存下來
    # reapted_sentence=[]
    # for i in range(x_pos+1,len(temp_input_sentence)):
    #     reapted_sentence.append(temp_input_sentence[i])

    # 第一個是[M]
    # sequence_sentence_list=[]
    # sequence_sentence_list.append("[MASK]")
    # sequence_sentence_list.extend(reapted_sentence)
    # single_sequence_sentence_list.append(sequence_sentence_list.copy())
   



    return sequential_extrapolate_list,sequential_extrapolate_pair_list

if __name__=="__main__":
    main()

