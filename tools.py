import os
import torch
import pickle
from tqdm import tqdm
from torch.utils.data import TensorDataset,DataLoader
from transformers import BertTokenizer

tokenizer =BertTokenizer(vocab_file='bert-base-uncased-vocab.txt')

def make_dataset_with_pos(input_id, input_segment, input_attention, input_maskLM,input_pos):
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

# 用來產生BERT預設需要的Embedding
def segment_embedding(input_sentence):
    # segment_embedding
    SEP_flag=False
    sep_token="[SEP]"
    input_segment=[]
    for token in input_sentence:  
        if (SEP_flag):
            input_segment.append(1)
        else:
            if token != sep_token:
                input_segment.append(0) 
            elif token==sep_token :
                SEP_flag=True
                input_segment.append(0) 
    return input_segment

def attention_embedding(input_sentence):
    # attention_embedding
    input_attention=[]

    input_attention.extend([1]*len(input_sentence))

    return input_attention

def embedding_padding(padding_length,embedding_list):
    padding_list=embedding_list.copy()
    while len(padding_list) < padding_length:
        padding_list.append(0)
    return padding_list
    
def output_evaluate(ref,filename):
    test_ref=""
    for i in range(len(ref)):
        test_ref=test_ref+ref[i]+"\n"
    test_ref_path=os.path.join("result","for_evaluate",filename+"_ref.txt")
    with open(test_ref_path,'w',encoding='utf-8') as f:
        f.write(test_ref)

def predict_test_in_train_mode(input_ids,labels,logits):
   
    input_ids = input_ids.cpu().numpy().tolist()[0]
    labels= labels.cpu().numpy().tolist()[0]
    resume_token(input_ids,labels)
    maskpos_list=find_maskpos(input_ids)
    labels_list=[]
    predict_list=[]
    for maskpos in maskpos_list:
        predicted_index = torch.argmax(logits[0, maskpos]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
        labels_list.append(tokenizer.convert_ids_to_tokens([labels[maskpos]][0]))
        predict_list.append(predicted_token)
        # if predicted_token!="[SEP]":
            # print('label:',tokenizer.convert_ids_to_tokens([labels[maskpos]][0]))
            # print('predict:',predicted_token)
    print('label:',labels_list)
    print('predict:',predict_list)

    return 0

def resume_token(input_ids,labels):
    text = ""
    stop_index = 0
    for index,input_token_id in enumerate(input_ids):
        if input_token_id != 0:
            text = text+tokenizer.convert_ids_to_tokens([input_token_id])[0] + " "
        else:
            stop_index=index
            break
    # print("input_ids",text)
    text = ""
    for index,label_token_id in enumerate(labels):
        if index<stop_index:
            text = text+tokenizer.convert_ids_to_tokens([label_token_id])[0] + " "
    # print("label",text)

    return 0

def find_maskpos(input_ids):
    find = 103
    return [i for i,v in enumerate(input_ids) if v==find]

def get_sentence_tag_dict(nlp,sentence):
    tag_dict={}
    doc=nlp(sentence)
    for token in doc:
        if token.text in tag_dict.keys():
            continue
        tag_dict[token.text.lower()]=token.tag_
    
    return tag_dict

def SEP_token_change(maskLM_embedding_list):
    maskLM_changed_embedding_list=maskLM_embedding_list.copy()

    for i in range(len(maskLM_embedding_list)):
        for j,label in enumerate(maskLM_embedding_list[i]):
            if label == 102:
                # 13044 = garbage
                maskLM_changed_embedding_list[i][j]=13044

    return maskLM_changed_embedding_list

def convert_tuple_to_dict(tuple_obj):
    dict_obj={}
    for i in range(len(tuple_obj)):
        dict_obj[i]=tuple_obj[i]
    return dict_obj

def get_dataset_list(data_path):

    semantics_list=[]
    syntactic_list=[]
    with open(data_path,'r',encoding='utf-8') as f:

        for line in f:
            line_processed_list=line.strip("\n").split("\t")
            semantics_list.append(line_processed_list[0])
            syntactic_list.append(line_processed_list[1])
      
    return semantics_list,syntactic_list

def get_accepted_pos_list():
    accepted_pos_list=["WRB","WP","WP$","WDT","IN","TO","MD"]
    return accepted_pos_list

# 標準 BERT padding
def padding(token_embedding_id_list, segment_embedding_list, attention_embedding_list, maskLM_embedding_list):

    max_len=512
    for i in tqdm(range(len(token_embedding_id_list))):
        while len(token_embedding_id_list[i]) < max_len:
            token_embedding_id_list[i].append(0)
            segment_embedding_list[i].append(0)
            attention_embedding_list[i].append(0)
            maskLM_embedding_list[i].append(-100)
    return token_embedding_id_list, segment_embedding_list, attention_embedding_list, maskLM_embedding_list

def convert_embedding_to_feature(data_path,data_type,token_embedding_id_list, segment_embedding_list, attention_embedding_list, maskLM_embedding_list):

    data_feature = {"input_id": token_embedding_id_list,
                    "input_segment": segment_embedding_list,
                    "input_attention": attention_embedding_list,
                    "input_maskLM": maskLM_embedding_list
                    }
    if not os.path.isdir(data_path):
            os.mkdir(data_path)
    output = open("{0}/{1}_data_feature.pkl".format(data_path,data_type), "wb")
    pickle.dump(data_feature, output)

def get_data_set(path):
    data = open(path, "rb")
    data_feature = pickle.load(data)
    input_id = data_feature["input_id"]
    input_segment = data_feature["input_segment"]
    input_attention = data_feature["input_attention"]
    input_maskLM = data_feature["input_maskLM"]
    dataset = make_dataset(input_id, input_segment, input_attention, input_maskLM)
    return dataset

def make_dataset(input_id, input_segment, input_attention, input_maskLM):
    all_input_id = torch.tensor(
        [input_id for input_id in input_id], dtype=torch.long)
    all_input_segment = torch.tensor(
        [input_segment for input_segment in input_segment], dtype=torch.long)
    all_input_attention = torch.tensor(
        [input_attention for input_attention in input_attention], dtype=torch.long)
    all_input_maskLM = torch.tensor(
        [input_maskLM for input_maskLM in input_maskLM], dtype=torch.long)
    full_dataset = TensorDataset(
        all_input_id, all_input_segment, all_input_attention, all_input_maskLM)

    return full_dataset


def make_dataset_add_pos(input_id, input_segment, input_attention, input_maskLM,input_pos):
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

def extract_sentence_from_list(sentence_list):
    string=""

    for token in sentence_list:
        string=string+token+" "
    if string=="":
        return "#"
    return string

def produce_analysis_file(syntactic_keyword_list,ref_list,filnm):

    # 讀取原作者的test_input
    mingda_test_input="mingda_chen_dataset/test_input.txt"
    with open(mingda_test_input,'r',encoding='utf-8') as mingda:
        test_input=mingda.readlines()

    assert len(syntactic_keyword_list)==len(test_input)
    assert len(syntactic_keyword_list)==len(ref_list)
    output_txt=""
    for i in range(len(syntactic_keyword_list)):
        output_txt=output_txt+"semantic   : "+test_input[i].split("\t")[0]+"\n"
        output_txt=output_txt+"syntactic  : "+test_input[i].split("\t")[1]
        output_txt=output_txt+"keyword    : "+extract_sentence_from_list(syntactic_keyword_list[i])+"\n"
        output_txt=output_txt+"ref        : "+ref_list[i]+"\n"
        output_txt=output_txt+"\n\n"

    output_file_path=os.path.join("result",filnm+".txt")
    with open(output_file_path,'a',encoding='utf-8') as f:
        f.write(output_txt)

# if __name__ == "__main__":
#     produce_compared_analysis_file(1,1,1)
