import spacy
import pickle
import data_preprocess
from tqdm import tqdm
from transformers import BertTokenizer

def convert_tuple_to_dict(tuple_obj):
    dict_obj={}
    for i in range(len(tuple_obj)):
        dict_obj[i]=tuple_obj[i]
    return dict_obj
def get_sentence_tag_dict(nlp,sentence):
    tag_dict={}
    doc=nlp(sentence)
    for token in doc:
        if token.text in tag_dict.keys():
            continue
        tag_dict[token.text.lower()]=token.tag_
    
    return tag_dict

def pos_match(tag_dict,token_list,pos_encoder_dict):
    # 尋找詞性
    pos_list=[]
    pos_embedding=[]
    for index,token in enumerate(token_list):
        if '#' in token:
            ori_word = (token_list[index-1]+token).replace('#','')
            if ori_word in tag_dict.keys():
                pos_list.append(tag_dict[ori_word])
                pos_list[index-1]=tag_dict[ori_word]
            else:
                pos_list.append('NONE')
                pos_list[index-1]='NONE'
        else:
            if token.lower() in tag_dict.keys():
                pos_list.append(tag_dict[token.lower()])
            else:
                pos_list.append('NONE')

    for pos in pos_list:
        if pos=='NONE':
            pos_embedding.append(0)
        else:
            pos_embedding.extend(get_key(pos_encoder_dict,pos))
       
    return pos_embedding

def get_key (pos_encoder_dict, value):
    return [k for k, v in pos_encoder_dict.items() if v == value]

def get_pos_embedding(semantics_list,ori_syntactic_list,syntactic_list_in_dict,nlp):
    pos_embedding_list = []
    pos_encoder_dict_temp=convert_tuple_to_dict(nlp.get_pipe("tagger").labels)
    pos_encoder_dict={}
    for key_num,pos_tag in pos_encoder_dict_temp.items():
        pos_encoder_dict[key_num+1]=pos_tag
    # print(pos_encoder_dict)
    tokenizer =BertTokenizer(vocab_file='bert-base-uncased-vocab.txt')
    for i in tqdm(range(len(ori_syntactic_list))):
        # 先產生兩句話詞性的dict
        semantics_tag_dict=get_sentence_tag_dict(nlp,semantics_list[i])
        syntactic_tag_dict=get_sentence_tag_dict(nlp,ori_syntactic_list[i])

        semantics_token=[]
        semantics_token.append("[CLS]")
        semantics_token.extend(tokenizer.tokenize(semantics_list[i]))
        semantics_token.append("[SEP]")
        semantics_pos=pos_match(semantics_tag_dict,semantics_token,pos_encoder_dict)
        for j,syntactic_sentence in enumerate(syntactic_list_in_dict[i]):
            input_pos=[]
            input_token = []
            syntactic_token=[]
            # [CLS] + semantics_sentence(X2) + [SEP] + syntactic_sentence(X1) 
            input_token=semantics_token.copy()
            syntactic_token=syntactic_sentence
            input_token.extend(syntactic_sentence)
            
            
            syntactic_pos=pos_match(syntactic_tag_dict,syntactic_token,pos_encoder_dict)
            
            input_pos.extend(semantics_pos)
            input_pos.extend(syntactic_pos)
            assert len(input_token)==len(input_pos)
            # print("input_token",input_token)
            # print("input_pos",input_pos)
            pos_embedding_list.append(input_pos.copy())

    return pos_embedding_list

def padding(pos_embedding_list):
    # for i in range(len(token_embedding_id_list)):
    #     value_list.append(len(token_embedding_id_list[i]))
    # max_len = max(value_list)
    max_len=512
    print("max_len:",max_len)
    for i in range(len(pos_embedding_list)):
        while len(pos_embedding_list[i]) < max_len:
            pos_embedding_list[i].append(0)
    return pos_embedding_list

def convert_embedding_to_feature(type_,token_embedding_id_list, segment_embedding_list, attention_embedding_list, maskLM_embedding_list,pos_embedding_list):

    data_feature = {"input_id": token_embedding_id_list,
                    "input_segment": segment_embedding_list,
                    "input_attention": attention_embedding_list,
                    "input_maskLM": maskLM_embedding_list,
                    "input_pos":pos_embedding_list
                    }
   
    output = open("data_feature/"+type_+"_data_feature_pos.pkl", "wb")
   

    pickle.dump(data_feature, output)
    
def main():
    nlp =spacy.load("model/spacy/en_core_web_md-2.3.1/en_core_web_md/en_core_web_md-2.3.1")

    data_path_dict={
        "train":"/user_data/Project/Controllable_Syntax_with_BERT/dataset/train_1000.txt",
        "test":"/user_data/Project/Controllable_Syntax_with_BERT/dataset/test_500.txt",
        "validation":"/user_data/Project/Controllable_Syntax_with_BERT/dataset/validation_500.txt"
    }
    for key,data_path in data_path_dict.items():

        print("get_data_list...")
        semantics_list,ori_syntactic_list=data_preprocess.get_data_list(data_path)
        print("get extrapolate_syntactic...")
        syntactic_list_in_dict,part_maskLM_embedding_list_in_dict=data_preprocess.extrapolate_syntactic(ori_syntactic_list,nlp)
        print("get embedding...")
        # 新增詞性的embedding
        pos_embedding_list=get_pos_embedding(semantics_list,ori_syntactic_list,syntactic_list_in_dict,nlp)
        print("pos padding...")
        pos_embedding_list=padding(pos_embedding_list)

        token_embedding_id_list, segment_embedding_list, attention_embedding_list, maskLM_embedding_list=data_preprocess.get_embedding(semantics_list,ori_syntactic_list,syntactic_list_in_dict,part_maskLM_embedding_list_in_dict)
        assert len(pos_embedding_list)==len(token_embedding_id_list)
        convert_embedding_to_feature(key,token_embedding_id_list, segment_embedding_list, attention_embedding_list, maskLM_embedding_list,pos_embedding_list)
        print(key+"_data finished")




if __name__ == "__main__":
    main()