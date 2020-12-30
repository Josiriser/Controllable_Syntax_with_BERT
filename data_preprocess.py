import torch
import spacy
import pickle
from tqdm import tqdm
from transformers import BertConfig
from transformers import BertTokenizer
from torch.utils.data import TensorDataset

mask_token="[MASK]"
mask_token_pair=(-1,mask_token)
sep_token="[SEP]"
sep_token_pair=(-1,sep_token)

sepcial_token_list=["[CLS]","[SEP]","[MASK]"]

def get_data_list(data_path):

    
    semantics_list=[]
    unporcessed_syntactic_list=[]
    count=0
    with open(data_path,'r',encoding='utf-8') as f:
        for line in f:
            line_processed_list=line.strip("\n").split("\t")
            semantics_list.append(line_processed_list[0])
            unporcessed_syntactic_list.append(line_processed_list[1])
            count+=1
            
    return semantics_list,unporcessed_syntactic_list

def extrapolate_syntactic(ori_syntactic_list,nlp):
    # 外插 [MASK] ， 產出所有可能
    # tokenizer =BertTokenizer(vocab_file='bert-base-uncased-vocab.txt')
    ignored_pos_list=['NN','NNS','NNP','NNPS',"IN","VB"]
    accepted_pos_list=["WRB","WP","WP$","WDT","IN","TO","MD"]
    syntactic_list_in_dict={}
    part_maskLM_embedding_list_in_dict={}
    count=0
    for index,ori_syntactic_sentence in enumerate(tqdm(ori_syntactic_list)):
        
        doc = nlp(ori_syntactic_sentence)
        if len(doc)>=512:
            print(ori_syntactic_sentence)
        # 取得ori_syntactic_sentence 每個token 的 位置  
        ori_syntactic_sentence_token_position_dict=get_position(doc)
    
        # 先取出spacy 的 token，以及留下來的位置
        # syntactic_token_position_dict,syntactic_token_position_with_empty_dict=spacy_tokened(doc,ignored_pos_list)
        # 改成留下不重要的字，留下句型結構
        syntactic_token_position_dict,syntactic_token_position_with_empty_dict=spacy_tokened(doc,accepted_pos_list)
        # 檢查是否全都是Empty
        # if (check_syntactic_token_position_with_empty_dict(syntactic_token_position_with_empty_dict)):
        #     continue
        # 再來外插
        # if len(doc)!=len(syntactic_token_position_dict):
        #     count+=1
        #     if count==228:
        #     # print(count)
        #         syntactic_list_in_dict[index]=extrapolate(ori_syntactic_sentence,ori_syntactic_sentence_token_position_dict,syntactic_token_position_dict,syntactic_token_position_with_empty_dict)
        syntactic_list_in_dict[index],part_maskLM_embedding_list_in_dict[index]=extrapolate(ori_syntactic_sentence,ori_syntactic_sentence_token_position_dict,syntactic_token_position_dict,syntactic_token_position_with_empty_dict)
        count+=1
        # if count==200:
        #     break
        
    return syntactic_list_in_dict,part_maskLM_embedding_list_in_dict

def get_position(doc):
    position_dict={}
    for i,token in enumerate(doc):
        position_dict[i]=token.text
    
    return position_dict

def spacy_tokened(doc,accepted_pos_list):
    # syntactic_token_list=[]
    syntactic_token_position_dict={}
    syntactic_token_position_with_empty_dict={}
    for i,token in enumerate(doc): 
        # if token.tag_ =="IN":
        #     print(token.text)
        # if token.tag_ not in ignored_pos_list:
        if token.tag_ in accepted_pos_list:
            # syntactic_token_list.append(token.text)
            syntactic_token_position_dict[i]=token.text
            syntactic_token_position_with_empty_dict[i]=token.text
        else:
            syntactic_token_position_with_empty_dict[i]="[E]"
    return syntactic_token_position_dict,syntactic_token_position_with_empty_dict

def extrapolate(ori_syntactic_sentence,ori_syntactic_sentence_token_position_dict,syntactic_token_position_dict,syntactic_token_position_with_empty_dict):
    tokenizer =BertTokenizer(vocab_file='bert-base-uncased-vocab.txt')
    all_extrapolate_list=[]
    first_extrapolate_list,first_extrapolate_pair_list=first_extrapolate(syntactic_token_position_dict)
    all_extrapolate_list.append(first_extrapolate_list.copy())
    # print(first_extrapolate_list)
    # input_maskLM
    
    part_maskLM_embedding_list = []
    recall_extrapolate_pair_list=first_extrapolate_pair_list.copy()
    while(check_special_token_in_(recall_extrapolate_pair_list)) or (check_empty_to_fillout(syntactic_token_position_with_empty_dict)):
        temp_extrapolate_pair_list=[]
        input_maskLM=[]
        last_token_position = -2
        for index,position_token_pair in enumerate(recall_extrapolate_pair_list):
            if position_token_pair[1]==mask_token:
                # 處理 [M]
                if ((recall_extrapolate_pair_list[index-1][0]==len(ori_syntactic_sentence_token_position_dict)-1) and (index!=0)):
                    # 如果最後一個 [M] token 已經超過ori_syntactic_sentence ，也就是[M]的前一個token 的位置是答案的最後一個 
                    # 直接把 [M] 換成 [SEP]
                    temp_extrapolate_pair_list.append(sep_token_pair)
                    input_maskLM.append(tokenizer.convert_tokens_to_ids(sep_token))
                elif(index==0 and recall_extrapolate_pair_list[index+1][0]==0):
                     # 或者是 第一個 [M] 已經沒有可以預測的字
                    temp_extrapolate_pair_list.append(sep_token_pair)
                    input_maskLM.append(tokenizer.convert_tokens_to_ids(sep_token))
                elif (index+1 < len(recall_extrapolate_pair_list) and last_token_position+1==recall_extrapolate_pair_list[index+1][0]):
                    # 下一個token position 剛好 等於下一個 token position
                    temp_extrapolate_pair_list.append(sep_token_pair)
                    input_maskLM.append(tokenizer.convert_tokens_to_ids(sep_token))
                else:
                    last_unmasked_token_position=recall_extrapolate_pair_list[index-1][0]
                    # 
                    if last_unmasked_token_position+1==len(ori_syntactic_sentence_token_position_dict)-1:
                        # 判斷此mask  是不是最後一個字  
                        k=last_unmasked_token_position+1
                        temp_extrapolate_pair_list.append(mask_token_pair)
                        temp_extrapolate_pair_list.append((k,ori_syntactic_sentence_token_position_dict[k]))
                        input_maskLM.append(tokenizer.convert_tokens_to_ids(ori_syntactic_sentence_token_position_dict[k]))
                        temp_extrapolate_pair_list.append(mask_token_pair)
                        syntactic_token_position_with_empty_dict[k]=ori_syntactic_sentence_token_position_dict[k]
                    else:    
                        next_token_position=recall_extrapolate_pair_list[index-1][0]+2
                        for k,v in syntactic_token_position_with_empty_dict.items():
                            if k < next_token_position  and k > last_unmasked_token_position and v=="[E]":
                                temp_extrapolate_pair_list.append(mask_token_pair)
                                temp_extrapolate_pair_list.append((k,ori_syntactic_sentence_token_position_dict[k]))
                                input_maskLM.append(tokenizer.convert_tokens_to_ids(ori_syntactic_sentence_token_position_dict[k]))
                                temp_extrapolate_pair_list.append(mask_token_pair)
                                syntactic_token_position_with_empty_dict[k]=ori_syntactic_sentence_token_position_dict[k]
                                break
            else:
                temp_extrapolate_pair_list.append((position_token_pair[0],position_token_pair[1]))
                # 紀錄上一個token 的index
                last_token_position=position_token_pair[0]
                input_maskLM.append(-100)
        recall_extrapolate_pair_list=temp_extrapolate_pair_list.copy()
        recall_extrapolate_pair_list=delete_sep_token(recall_extrapolate_pair_list)
         # result_extrapolate_list=convert_token_specialtoken_pair_to_extrapolate_list(temp_extrapolate_pair_list)
        # print(result_extrapolate_list)
        
        if (check_special_token_in_(recall_extrapolate_pair_list)):
            # 如果還有special token 再加進去
            result_extrapolate_list=convert_token_specialtoken_pair_to_extrapolate_list(recall_extrapolate_pair_list)
       
        
            all_extrapolate_list.append(result_extrapolate_list.copy())
        part_maskLM_embedding_list.append(input_maskLM.copy())
        # recall_extrapolate_pair_list=delete_sep_token(recall_extrapolate_pair_list)
    # print(ori_syntactic_sentence)
    return all_extrapolate_list,part_maskLM_embedding_list

def first_extrapolate(syntactic_token_position_dict):
    # 第一次，因為沒有[MASK]，所以如果有少字，都加上 [MASK]
    first_extrapolate_list=[]
    first_extrapolate_pair_list=[]
    first_extrapolate_list.append(mask_token)
    first_extrapolate_pair_list.append((-1,mask_token))
    for position,token in syntactic_token_position_dict.items():
        first_extrapolate_list.append(token)
        first_extrapolate_list.append(mask_token)
        first_extrapolate_pair_list.append((position,token))
        first_extrapolate_pair_list.append((-1,mask_token))

    return first_extrapolate_list,first_extrapolate_pair_list

def convert_token_specialtoken_pair_to_extrapolate_list(temp_extrapolate_pair_list):
    extrapolate_list=[]
    for position_token_pair in temp_extrapolate_pair_list:
        extrapolate_list.append(position_token_pair[1])
    return extrapolate_list

def check_special_token_in_(recall_extrapolate_pair_list):
    for position_token_pair in recall_extrapolate_pair_list:
        if (position_token_pair[1]==mask_token):
            return True
    return False

def check_empty_to_fillout(syntactic_token_position_with_empty_dict):
    for key in syntactic_token_position_with_empty_dict:
        if syntactic_token_position_with_empty_dict[key]=="[E]":
            return True
    return False

def delete_sep_token(recall_extrapolate_pair_list):
    recall_extrapolate_pair_list_copy=recall_extrapolate_pair_list.copy()
    for position_token_pair in recall_extrapolate_pair_list_copy:
        if position_token_pair[1]==sep_token:
            recall_extrapolate_pair_list.remove(position_token_pair)
    return recall_extrapolate_pair_list

def check_syntactic_token_position_with_empty_dict(syntactic_token_position_with_empty_dict):
    count=0
    for key in syntactic_token_position_with_empty_dict:
        if syntactic_token_position_with_empty_dict[key]=="[E]":
            count+=1
    if count==len(syntactic_token_position_with_empty_dict):
        return True
    return False

def padding(token_embedding_id_list, segment_embedding_list, attention_embedding_list, maskLM_embedding_list):
    # for i in range(len(token_embedding_id_list)):
    #     value_list.append(len(token_embedding_id_list[i]))
    # max_len = max(value_list)
    max_len=512
    print("max_len:",max_len)

    for i in tqdm(range(len(token_embedding_id_list))):
        while len(token_embedding_id_list[i]) < max_len:
            token_embedding_id_list[i].append(0)
            segment_embedding_list[i].append(0)
            attention_embedding_list[i].append(0)
            maskLM_embedding_list[i].append(-100)
    return token_embedding_id_list, segment_embedding_list, attention_embedding_list, maskLM_embedding_list

def get_embedding(semantics_list,ori_syntactic_list,syntactic_list_in_dict,part_maskLM_embedding_list_in_dict):
    token_embedding_id_list = []
    segment_embedding_list = []
    attention_embedding_list = []
    maskLM_embedding_list = []
    tokenizer =BertTokenizer(vocab_file='bert-base-uncased-vocab.txt')
    
    for i in tqdm(range(len(ori_syntactic_list))):
        for j,syntactice_sentence in enumerate(syntactic_list_in_dict[i]):
            input_token = []
            input_segment = []
            input_attention = []
            input_maskLM = []
            # [CLS] + semantics_sentence(X2) + [SEP] + syntactic_sentence(X1) 
            input_token.append("[CLS]")
            input_token.extend(tokenizer.tokenize(semantics_list[i]))
            input_token.append("[SEP]")
            input_token.extend(syntactice_sentence)
            token_embedding_id_list.append(tokenizer.convert_tokens_to_ids(input_token.copy()))
            # print(input_token)

            #segment_embedding
            SEP_flag=False
            for token in input_token:  
                if (SEP_flag):
                    input_segment.append(1)
                else:
                    if token != sep_token:
                        input_segment.append(0) 
                    elif token==sep_token :
                        SEP_flag=True
                        input_segment.append(0) 
            segment_embedding_list.append(input_segment.copy())
            # print(input_segment)
            
            #attention_embedding
            # for k in range(len(input_token)):
            #     input_attention.append(1)  # [CLS]
            input_attention.extend([1]*len(input_token))
            attention_embedding_list.append(input_attention.copy())
            # print(input_attention)

            SEP_flag=False
            for token in input_token:
                if (SEP_flag):
                    # if token in sepcial_token_list:
                    #     input_maskLM.append(-1)
                    # else:
                    #     input_maskLM.append(tokenizer.convert_tokens_to_ids(token))
                    input_maskLM.extend(part_maskLM_embedding_list_in_dict[i][j])
                    break
                else:
                    if token!=sep_token:
                        input_maskLM.append(-100)
                    else:
                        input_maskLM.append(-100)
                        SEP_flag=True
            maskLM_embedding_list.append(input_maskLM.copy())
            # print(input_maskLM)
            assert len(input_token) == len(input_segment)
            assert len(input_token) == len(input_attention)
            assert len(input_token) == len(input_maskLM)
    print("資料筆數:",len(token_embedding_id_list))
    assert len(token_embedding_id_list) == len(segment_embedding_list)
    assert len(token_embedding_id_list) == len(attention_embedding_list)
    assert len(token_embedding_id_list) == len(maskLM_embedding_list)
    
    print("padding...")
    token_embedding_id_list, segment_embedding_list, attention_embedding_list, maskLM_embedding_list = padding(
        token_embedding_id_list, segment_embedding_list, attention_embedding_list, maskLM_embedding_list)
    assert len(token_embedding_id_list) == len(segment_embedding_list)
    assert len(token_embedding_id_list) == len(attention_embedding_list)
    assert len(token_embedding_id_list) == len(maskLM_embedding_list)
    return token_embedding_id_list, segment_embedding_list, attention_embedding_list, maskLM_embedding_list

def convert_embedding_to_feature(data_type,token_embedding_id_list, segment_embedding_list, attention_embedding_list, maskLM_embedding_list):

    data_feature = {"input_id": token_embedding_id_list,
                    "input_segment": segment_embedding_list,
                    "input_attention": attention_embedding_list,
                    "input_maskLM": maskLM_embedding_list
                    }
  
    output = open("data_feature/{0}_data_feature.pkl".format(data_type), "wb")
    pickle.dump(data_feature, output)

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
def main():
    tokenizer =BertTokenizer(vocab_file='bert-base-uncased-vocab.txt')
    nlp =spacy.load("model/spacy/en_core_web_md-2.3.1/en_core_web_md/en_core_web_md-2.3.1")
    nlp.get_pipe("tagger").labels
   
    data_path_dict={
    "train":"/user_data/Project/Controllable_Syntax_with_BERT/dataset/train_6000.txt",
    "test":"/user_data/Project/Controllable_Syntax_with_BERT/dataset/test_3000.txt",
    "validation":"/user_data/Project/Controllable_Syntax_with_BERT/dataset/validation_3000.txt"
    }
    for key,data_path in data_path_dict.items():
        print("get {0} data ...".format(key))
        semantics_list,ori_syntactic_list=get_data_list(data_path)
        print("get {0} data extrapolate syntactic ...".format(key))
        syntactic_list_in_dict,part_maskLM_embedding_list_in_dict=extrapolate_syntactic(ori_syntactic_list,nlp)
        print("{0} syntactic data's length:".format(key),len(syntactic_list_in_dict))

        print("get {0} data embedding...".format(key))
        token_embedding_id_list, segment_embedding_list, attention_embedding_list, maskLM_embedding_list=get_embedding(semantics_list,ori_syntactic_list,syntactic_list_in_dict,part_maskLM_embedding_list_in_dict)
        print("convert {0} data embedding to feature...".format(key))
        convert_embedding_to_feature(key,token_embedding_id_list, segment_embedding_list, attention_embedding_list, maskLM_embedding_list)
        
    print("all finished")
if __name__ == "__main__":
   
    main()