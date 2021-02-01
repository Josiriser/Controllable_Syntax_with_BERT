import os
import torch
import spacy
import random
import argparse
import torch.nn.functional as F
from tqdm import tqdm
from data_extra_preprocess import extrapolate_syntactic
from data_extra_preprocess_pos import get_pos_embedding,convert_tuple_to_dict,get_sentence_tag_dict
from data_extra_preprocess_pos import pos_match
from transformers import BertConfig, BertForMaskedLM, BertTokenizer
from transformers import AutoModelForMaskedLM,AutoConfig
from train_pos_copy import POS_Model
from tools import segment_embedding,attention_embedding,output_evaluate,embedding_padding

# parser settings
parser = argparse.ArgumentParser()
parser.add_argument('--file_path', '-fp', type=str)
parser.add_argument('--model_path', '-mdp', type=str)
parser.add_argument('--output_file_name', '-oflnm', type=str)
args = parser.parse_args()

## 
# python predict.py -fp  mingda_chen_dataset/test_input.txt -mdp trained_model/baseline_small_model/4/pytorch_model.bin -oflnm small_model_pk.txt
##

## for debug
args.file_path='mingda_chen_dataset/test_input.txt'
args.model_path='trained_model/baseline_small_model/4/pytorch_model.bin'
args.output_file_name="baseline_small_test_SEP_model2_DataParallel.txt"
##

# pre-load file,information
tokenizer =BertTokenizer(vocab_file='bert-base-uncased-vocab.txt')
nlp =spacy.load("model/spacy/en_core_web_md-2.3.1/en_core_web_md/en_core_web_md-2.3.1")
pos_encoder_dict=convert_tuple_to_dict(nlp.get_pipe("tagger").labels)

def read_test_data():
    semantic_list=[]
    syntactic_list=[]
    with open(args.file_path,'r',encoding='utf-8') as f:
        for line in f:
            line_split_list=line.split('\t')
            semantic_list.append(line_split_list[0])
            syntactic_list.append(line_split_list[1].strip("\n"))
    
    return semantic_list,syntactic_list

def data_preprocess(sentence):
    ignored_pos_list=['NN','NNS','NNP','NNPS',"IN","VB"]
    accepted_pos_list=["WRB","WP","WP$","WDT","IN","TO","MD"]
    doc = nlp(sentence)
    syntactic_sentence=[]
    syntactic_sentence.append("[MASK]")
    for i,token in enumerate(doc): 
        # if token.tag_ not in ignored_pos_list:
        #     syntactic_sentence.append(token.text)
        #     syntactic_sentence.append("[MASK]")
        if token.tag_  in accepted_pos_list:
            syntactic_sentence.append(token.text)
            syntactic_sentence.append("[MASK]")
    # print(syntactic_sentence)
    return syntactic_sentence

def delete_sep(output_sentence):
    clean_sentence=[]
    for token in output_sentence:
        if token!="[SEP]":
            clean_sentence.append(token)
    return clean_sentence

def extract_sentence_from_list(sentence_list):
    string=""

    for token in sentence_list:
        string=string+token+" "
    if string=="":
        return "#"
    return string

def check_special_token(sentence_list):
    if '[MASK]' in sentence_list:
        return True
    return False

def select_model(model_num = 0):
    if model_num ==0:
        bert_config, bert_class = (BertConfig, BertForMaskedLM)
        # config = bert_config.from_pretrained('trained_model/baseline_small_model/4/config.json')
        config = AutoConfig.from_pretrained('bert-base-uncased')
        model = bert_class.from_pretrained(args.model_path, config=config)
    elif model_num ==1:
        model = torch.load(args.model_path)
        
    return model

# def segment_embedding(input_sentence):
#     #segment_embedding
#     SEP_flag=False
#     sep_token="[SEP]"
#     input_segment=[]
#     for token in input_sentence:  
#         if (SEP_flag):
#             input_segment.append(1)
#         else:
#             if token != sep_token:
#                 input_segment.append(0) 
#             elif token==sep_token :
#                 SEP_flag=True
#                 input_segment.append(0) 
#     return input_segment

# def attention_embedding(input_sentence):
#     input_attention=[]

#     input_attention.extend([1]*len(input_sentence))

#     return input_attention

def pos_embedding(ori_semantic_sentence,ori_syntactic_sentence,input_syntactic):
    input_pos = []
    semantics_tag_dict=get_sentence_tag_dict(nlp,ori_semantic_sentence)
    syntactic_tag_dict=get_sentence_tag_dict(nlp,ori_syntactic_sentence)

    semantics_token=[]
    semantics_token.append("[CLS]")
    semantics_token.extend(tokenizer.tokenize(ori_semantic_sentence))
    semantics_token.append("[SEP]")

    semantics_pos=pos_match(semantics_tag_dict,semantics_token,pos_encoder_dict)
    syntactic_pos=pos_match(syntactic_tag_dict,input_syntactic,pos_encoder_dict)
    input_pos.extend(semantics_pos)
    input_pos.extend(syntactic_pos)
    return input_pos

def maskLM_embedding(input_sentence):
    input_maskLM=[]
    SEP_flag=False
    for token in input_sentence:
        if (SEP_flag):
            # if token in sepcial_token_list:
            #     input_maskLM.append(-1)
            # else:
            #     input_maskLM.append(tokenizer.convert_tokens_to_ids(token))
            input_maskLM.extend(tokenizer.convert_tokens_to_ids(token))
        else:
            if token!="[SEP]":
                input_maskLM.append(-100)
            else:
                input_maskLM.append(-100)
                SEP_flag=True
    return input_maskLM

def extroplate_mask(syntactic_sentence_list):
    process_syntactic_sentence_list=[]
    process_syntactic_sentence_list.append("[MASK]")
    for token in syntactic_sentence_list:
        process_syntactic_sentence_list.append(token)
        process_syntactic_sentence_list.append("[MASK]")
    
    return process_syntactic_sentence_list

def main():

    # model select
    # 0 : base
    # 1 : pos
    model_num = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda")

    ori_semantic_list,ori_syntactic_list=read_test_data()
    model = select_model(model_num)
    model.to(device)
    model.eval()

    output_txt=""
    eval_dict={}
    semantic=[]
    syntactic=[]
    ref=[]
    count=0
    for index,ori_semantic_sentence in enumerate(tqdm(ori_semantic_list)):
        input_sentence=[]
        input_semantic=[]
        input_syntactic=[]
        input_segment=[]
        input_attention=[]
        input_maskLM=[]
        input_pos=[]
        
        tokenized_text=tokenizer.tokenize(ori_semantic_sentence)
        
        input_semantic.append("[CLS]")
        input_semantic.extend(tokenized_text)
        input_semantic.append("[SEP]")
       
        # extroplation
        input_syntactic=data_preprocess(ori_syntactic_list[index])

        syntactic_iteration=input_syntactic.copy()
        # # 太短的句子不要
        # if len(syntactic_iteration) < 9 or len(tokenized_text) < 9 :
        #     continue
        while check_special_token(syntactic_iteration):
            # input_sentence
            input_sentence.extend(input_semantic)
            input_sentence.extend(syntactic_iteration.copy())
            if len(input_sentence)>512:
                break
            syntactic_iteration.clear()
            
            # input_segment
            input_segment=segment_embedding(input_sentence)
            # input_attention
            input_attention=attention_embedding(input_sentence)
            # input_maskLM
            # input_maskLM=maskLM_embedding(input_sentence)
            assert len(input_sentence)==len(input_segment)
            assert len(input_sentence)==len(input_attention)
            input_sentence_ids=tokenizer.convert_tokens_to_ids(input_sentence)
            # padding
            input_sentence_ids=embedding_padding(512,input_sentence_ids)
            input_segment=embedding_padding(512,input_segment)
            input_attention=embedding_padding(512,input_attention)

            input_id_tensor=torch.tensor([input_sentence_ids]).to(device)
            input_segment_tensor=torch.tensor([input_segment]).to(device)
            input_attention_tensor=torch.tensor([input_attention]).to(device)

            if model_num==0:
                outputs=model(input_ids=input_id_tensor,token_type_ids=input_segment_tensor,
                attention_mask=input_attention_tensor)
            elif model_num==1:
                input_pos=pos_embedding(ori_semantic_sentence,ori_syntactic_list[index],input_syntactic)
                input_pos=embedding_padding(512,input_pos)
                # assert len(input_sentence)==len(input_pos)
                input_pos_tensor=torch.tensor([input_pos]).to(device)
                outputs=model(input_ids=input_id_tensor,token_type_ids=None,
                attention_mask=input_attention_tensor,pos_ids=input_pos_tensor)

            predictions=outputs[0]
            # predictions=outputs[1]
            output_sentence=[]
            syn_flag=False
            for word in input_sentence: 
                if (syn_flag) and word !="[MASK]":
                    # output_sentence.append(word)
                    syntactic_iteration.append(word)
                if word =='[MASK]':
                    maskpos=input_sentence.index('[MASK]')
                    logit_prob = F.softmax(predictions[0, maskpos]).data.tolist()
                    predicted_index = torch.argmax(predictions[0, maskpos]).item()
                    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
                    # print(predicted_token)
                    # exit()
                    # print(predicted_token,logit_prob[predicted_index])
                    # output_sentence.append(predicted_token)
                    input_sentence[maskpos]=predicted_token
                    if predicted_token!='[SEP]' and predicted_token!='[UNK]':
                        syntactic_iteration.append('[MASK]')
                        syntactic_iteration.append(predicted_token)
                        syntactic_iteration.append('[MASK]')
                if (not syn_flag) and (word=="[SEP]"):
                    syn_flag=True
            # print(syntactic_iteration)
            input_sentence.clear()
        output_sentence=syntactic_iteration
        # print(input_semantic)
        # print(input_syntactic)
        # print(ori_syntactic_list[index])
        # print(output_sentence)
        count+=1
        input_semantic_clean=extract_sentence_from_list(input_semantic)
        input_syntactic_clean=extract_sentence_from_list(input_syntactic)
        output_sentence_clean=extract_sentence_from_list(output_sentence)
        output_txt=output_txt+"input_semantic  : "+input_semantic_clean+"\n"
        output_txt=output_txt+"ori_syntactic   : "+ori_syntactic_list[index]+"\n"
        output_txt=output_txt+"input_syntactic : "+input_syntactic_clean+"\n"
        output_txt=output_txt+"output_sentence : "+output_sentence_clean+"\n"
        output_txt=output_txt+"\n\n"

        semantic.append(ori_semantic_sentence)
        syntactic.append(ori_syntactic_list[index])
        ref.append(output_sentence_clean)
        

    # output_file_path=os.path.join("result",args.output_file_name)
    # with open(output_file_path,'a',encoding='utf-8') as f:
    #     f.write(output_txt)
    # output_txt=""
    #　產出evaluate file
    assert len(semantic)==len(syntactic)
    assert len(semantic)==len(ref)
    output_evaluate(ref=ref,filename="extra")



   

    
            
       
        
    print("finished")

def test():

    # model select
    # 0 : base
    # 1 : pos
    model_num = 0
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # device = torch.device("cuda")
    device = torch.device("cpu")
    model = select_model(model_num)
    model.eval()


    input_sentence=[]
    input_semantic=[]
    input_syntactic=[]
    input_segment=[]
    input_attention=[]
    ori_semantic_sentence=" so we all need to deal with it . "
    syntactic_sentence_list=["to","with","of"]
    tokenized_text=tokenizer.tokenize(ori_semantic_sentence)

    input_semantic.append("[CLS]")
    input_semantic.extend(tokenized_text)
    input_semantic.append("[SEP]")
    
    # extroplation
    input_syntactic=extroplate_mask(syntactic_sentence_list)

    

    syntactic_iteration=input_syntactic.copy()
    while check_special_token(syntactic_iteration):
        input_sentence.extend(input_semantic)
        input_sentence.extend(syntactic_iteration.copy())

        # print(input_sentence)
        if len(input_sentence)>512:
                break
        syntactic_iteration.clear()

        # input_segment
        input_segment=segment_embedding(input_sentence)
        # input_attention
        input_attention=attention_embedding(input_sentence)

        assert len(input_sentence)==len(input_segment)
        assert len(input_sentence)==len(input_attention)
        
        input_sentence_ids=tokenizer.convert_tokens_to_ids(input_sentence)

        input_id_tensor=torch.tensor([input_sentence_ids])
        input_segment_tensor=torch.tensor([input_segment])
        input_attention_tensor=torch.tensor([input_attention])
        if model_num==0:
            outputs=model(input_ids=input_id_tensor,token_type_ids=input_segment_tensor,
            attention_mask=input_attention_tensor)
        predictions=outputs[0]
        output_sentence=[]
        syn_flag=False
        for word in input_sentence: 
            if (syn_flag) and word !="[MASK]":
                # output_sentence.append(word)
                syntactic_iteration.append(word)
            if word =='[MASK]':
                maskpos=input_sentence.index('[MASK]')
                logit_prob = F.softmax(predictions[0, maskpos]).data.tolist()
                predicted_index = torch.argmax(predictions[0, maskpos]).item()
                predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
                # print(predicted_token,logit_prob[predicted_index])
                # output_sentence.append(predicted_token)
                input_sentence[maskpos]=predicted_token
                if predicted_token!='[SEP]' and predicted_token!='[UNK]':
                    syntactic_iteration.append('[MASK]')
                    syntactic_iteration.append(predicted_token)
                    syntactic_iteration.append('[MASK]')
            if (not syn_flag) and (word=="[SEP]"):
                syn_flag=True
        # print(syntactic_iteration)
        input_sentence.clear()
    output_sentence=syntactic_iteration
    # print(input_semantic)
    # print(input_syntactic)
    # print(ori_syntactic_list[index])
    print(output_sentence)
    # semantic_sentence="can i see in your binoculars ? "
    # syntactic_sentence=["can","I","binoculars"]
    
    # tokenized_text=tokenizer.tokenize(semantic_sentence)
    # process_sentence=[]
    # process_sentence.append("[CLS]")
    # for word in tokenized_text:
    #     process_sentence.append(word)
    # process_sentence.append("[SEP]")

    # process_sentence.append("[MASK]")

    # for syntactic in syntactic_sentence:
    #     process_sentence.append(syntactic)
    #     process_sentence.append("[MASK]")
    
    # aa=tokenizer.convert_tokens_to_ids(process_sentence)
    # tokens_tensor = torch.tensor([aa])
    
    # outputs=model(tokens_tensor)
    # predictions=outputs[0]
    # SEP_flag =False
    # output_sentence=""
    # for token in process_sentence:
    #     if token =='[SEP]':
    #         SEP_flag=True
    #     elif SEP_flag and token=='[MASK]':
    #         maskpos=process_sentence.index('[MASK]')

    #         logit_prob = F.softmax(predictions[0, maskpos]).data.tolist()
    #         predicted_index = torch.argmax(predictions[0, maskpos]).item()
    #         predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    #         print(predicted_token,logit_prob[predicted_index])
    #         process_sentence[maskpos]=predicted_token
    #     else:
    #         output_sentence=output_sentence+" "+token
    # print(output_sentence)

if __name__ == "__main__":

    # test()
    main()



    




    


