
import spacy
import argparse
import random
from tqdm import tqdm
from transformers import BertTokenizer
from tools import get_dataset_list,get_accepted_pos_list,padding,convert_embedding_to_feature




def main():
    ## 外部參數設定
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file_path', '-trfp', type=str)
    parser.add_argument('--test_file_path', '-tefp', type=str)
    parser.add_argument('--valid_file_path', '-vafp', type=str)
    parser.add_argument('--output_file_path', '-outfp', type=str)
    args = parser.parse_args()

    ## 預設值

    tokenizer = BertTokenizer(vocab_file='bert-base-uncased-vocab.txt')
    nlp = spacy.load("model/spacy/en_core_web_md-2.3.1/en_core_web_md/en_core_web_md-2.3.1")
    accepted_pos_list=get_accepted_pos_list()
    ##

    ## for debug
    # args.train_file_path="/user_data/Project/Controllable_Syntax_with_BERT/dataset/train_10.txt"
    # args.test_file_path="/user_data/Project/Controllable_Syntax_with_BERT/dataset/test_5.txt"
    # args.valid_file_path="/user_data/Project/Controllable_Syntax_with_BERT/dataset/validation_5.txt"
    # args.output_file_path="/user_data/Project/Controllable_Syntax_with_BERT/sequential_dataset"
    ## 


    data_path_dict={
    "train":args.train_file_path,
    "test":args.test_file_path,
    "validation":args.valid_file_path
    }
    for key,data_path in data_path_dict.items():
        print("get {0} data ...".format(key))
        semantic_list,syntactic_list=get_dataset_list(data_path)

        print(" get {0} all syntactic keyword list ...".format(key))
        all_syntactic_keyword_list=get_all_syntactic_keyword_list(syntactic_list,accepted_pos_list,tokenizer,nlp)

        print(" insert sep to {0} all syntactic keyword list ...".format(key))
        all_syntactic_keyword_with_sep_list=insert_sep_token(all_syntactic_keyword_list)

        print(" get  {0} all sequence sentence list ...".format(key))
        all_sequence_sentence_list=get_all_sequence_sentence_list(syntactic_list,tokenizer)

        print(" get {0} embeddings ...".format(key))
        token_embedding_id_list,segment_embedding_list,attention_embedding_list,maskLM_embedding_list=get_embeddings(semantic_list,syntactic_list,all_syntactic_keyword_with_sep_list,all_sequence_sentence_list,tokenizer)

        print(" convert to feature {0} embeddings ...".format(key))
        convert_embedding_to_feature(args.output_file_path,key,token_embedding_id_list,
        segment_embedding_list,attention_embedding_list,maskLM_embedding_list)
        
        print(" {0} data finished".format(key))

    return 0

def get_all_syntactic_keyword_list(syntactic_list,accepted_pos_list,tokenizer,nlp):
    all_syntactic_keyword_list=[]
    for syntactic_sentence in tqdm(syntactic_list):
        syntactic_keyword_list=get_syntactic_keyword(syntactic_sentence,accepted_pos_list,tokenizer,nlp)
        all_syntactic_keyword_list.append(syntactic_keyword_list)

    return all_syntactic_keyword_list

def get_syntactic_keyword(syntactic_sentence,accepted_pos_list,tokenizer,nlp):
    syntactic_keyword_list=[]
    doc = nlp(syntactic_sentence)
    token_list=tokenizer.tokenize(syntactic_sentence)
    for token in (doc):
        # 檢查 keyword 也要在 token_list 裡的字，避免 BERT 預測是 UNK
        if token.tag_ in accepted_pos_list and token.text in token_list:
            syntactic_keyword_list.append(token.text)
    # if len(syntactic_keyword_list)==0:
    #     token_len=len(token_list)
    #     rand_pos=random.randint(0,token_len-2)
    #     syntactic_keyword_list.append(token_list[rand_pos])
    return syntactic_keyword_list

def insert_sep_token(all_syntactic_keyword_list):
    all_syntactic_keyword_with_sep_list=[]
    for i in tqdm(range(len(all_syntactic_keyword_list))):
        syntactic_keyword_with_sep_list=[]
        for token in all_syntactic_keyword_list[i]:
            syntactic_keyword_with_sep_list.append(token)
            syntactic_keyword_with_sep_list.append("[SEP]")
        all_syntactic_keyword_with_sep_list.append(syntactic_keyword_with_sep_list)
    return all_syntactic_keyword_with_sep_list


def get_all_sequence_sentence_list(syntactic_list,tokenizer):
    all_sequence_sentence_list=[]
    
    for syntactic_sentence in tqdm(syntactic_list):
        single_sequence_sentence_list=[]
        # 紀錄已出現過的token
        appeared_token_list=[]

        # 第一個是[M]
        sequence_sentence_list=[]
        sequence_sentence_list.append("[MASK]")
        single_sequence_sentence_list.append(sequence_sentence_list.copy())
        sequence_sentence_list.clear()

        token_list=tokenizer.tokenize(syntactic_sentence)
        for token in (token_list):
            # 先存現在的token
            appeared_token_list.append(token)

            # 把出現過的token 加到sequence_sentence
            sequence_sentence_list.extend(appeared_token_list)
            sequence_sentence_list.append("[MASK]")
            single_sequence_sentence_list.append(sequence_sentence_list.copy())
            sequence_sentence_list.clear()
        all_sequence_sentence_list.append(single_sequence_sentence_list.copy())

            
    return all_sequence_sentence_list


def get_embeddings(semantic_list,syntactic_list,all_syntactic_keyword_with_sep_list,all_sequence_sentence_list,tokenizer):

    token_embedding_id_list=[]
    segment_embedding_list = []
    attention_embedding_list = []
    maskLM_embedding_list = []
    for index,semantic_sentence in enumerate(tqdm(semantic_list)):
        count =0
        syntactic_token_list=tokenizer.tokenize(syntactic_list[index])
        for sequence_sentence in (all_sequence_sentence_list[index]):
            
            # 處理 input_ids
            input_token_list=[]
            input_token_list.append("[CLS]")
            input_token_list.extend(tokenizer.tokenize(semantic_sentence))
            input_token_list.append("[SEP]")
            input_token_list.extend(all_syntactic_keyword_with_sep_list[index])
            input_token_list.extend(sequence_sentence)
            token_embedding_id_list.append(tokenizer.convert_tokens_to_ids(input_token_list.copy()))
            
            # 處理 segment embedding (token type)
            semantic_part=[]
            semantic_part.append("[CLS]")
            semantic_part.extend(tokenizer.tokenize(semantic_sentence))
            semantic_part.append("[SEP]")
            # input分三個部分
            semantic_part_len=len(semantic_part)
            keyword_length=len(all_syntactic_keyword_with_sep_list[index])
            sequence_len=len(sequence_sentence)

            input_segment_list=[]
            input_segment_list.extend([0]*semantic_part_len)
            input_segment_list.extend([1]*keyword_length)
            input_segment_list.extend([0]*sequence_len)
            segment_embedding_list.append(input_segment_list.copy())

            # 處理 attention embedding
            input_attention_list=[]
            input_attention_list.extend([1]*len(input_token_list))
            attention_embedding_list.append(input_attention_list.copy())

            # 處理 maskLM 
            input_maskLM_list=[]
            input_maskLM_list.extend([-100]*semantic_part_len)
            input_maskLM_list.extend([-100]*keyword_length)
            input_maskLM_list.extend([-100]*count)
            if count < len(syntactic_token_list):
                input_maskLM_list.extend(tokenizer.convert_tokens_to_ids([syntactic_token_list[count]]))
                
            else:
                input_maskLM_list.extend(tokenizer.convert_tokens_to_ids(["[SEP]"]))
            count+=1
            maskLM_embedding_list.append(input_maskLM_list.copy())

            assert len(input_token_list)==len(input_segment_list)
            assert len(input_token_list)==len(input_attention_list)
            assert len(input_token_list)==len(input_maskLM_list)
    print("padding...")
    token_embedding_id_list, segment_embedding_list, attention_embedding_list, maskLM_embedding_list = padding(
        token_embedding_id_list, segment_embedding_list, attention_embedding_list, maskLM_embedding_list)
    assert len(token_embedding_id_list) == len(segment_embedding_list)
    assert len(token_embedding_id_list) == len(attention_embedding_list)
    assert len(token_embedding_id_list) == len(maskLM_embedding_list)

    return token_embedding_id_list,segment_embedding_list,attention_embedding_list,maskLM_embedding_list


if __name__ == "__main__":
    main()

