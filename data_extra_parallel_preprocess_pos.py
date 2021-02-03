import spacy
import pickle
import data_extra_preprocess
from tqdm import tqdm
from transformers import BertTokenizer
from tools import get_sentence_tag_dict,convert_tuple_to_dict,get_dataset_list



def pos_match(tag_dict,token_list,pos_encoder_dict):
    pos_encoder_dict={
        1:"IN",
        2:"JJ",
        3:"NNS",
        4:"NN",
        5:"PRP",
        6:"VBD",
        7:".",
        8:"VBZ",
        9:"DT"
    }
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
            elif token == '[CLS]':
                pos_list.append('[CLS]')
            elif token == '[SEP]':
                pos_list.append('[SEP]')
            elif token == '[MASK]':
                pos_list.append('[MASK]')
            else:
                pos_list.append('NONE')

    # for pos in pos_list:
    #     if pos=='NONE':
    #         pos_embedding.append(0)
    #     elif pos == '[CLS]':
    #         pos_embedding.append(51)
    #     elif pos == '[SEP]':
    #         pos_embedding.append(52)
    #     elif pos == '[MASK]':
    #         pos_embedding.append(53)
    #     else:
    #         pos_embedding.extend(get_key(pos_encoder_dict,pos))
    for pos in pos_list:
        if pos=='NONE':
            pos_embedding.append(0)
        elif pos == '[CLS]':
            pos_embedding.append(10)
        elif pos == '[SEP]':
            pos_embedding.append(11)
        elif pos == '[MASK]':
            pos_embedding.append(12)
        else:
            pos_embedding.extend(get_key(pos_encoder_dict,pos))   
    return pos_embedding

def get_reduced_pos_encoder_dict():
    SYM_list=['$',"SYM"]
    PUNCT_list=["``","''",",","-LRB-","-RRB-",".",":","HYPH","NFP"]
    X_list=["ADD","FW","GW",'LS',"NIL","XX","NONE","[CLS]"]
    ADJ_list=["AFX","JJ","JJR","JJS"]
    PART_list=["POS","TO"]
    ADV_list=["RB","RBR","RBS","WRB"]
    VERB_list=["MD","VB","VBD","VBG","VBN","VBP","VBZ"]
    CCONJ_list=["CC"]
    ADP_list=["IN","RP"]
    NUM_list=["CD"]
    DET_list=["DT","PDT","PRP$","WDT","WP$"]
    PRON_list=["EX","NNP","NNPS","PRP","WP"]
    NOUN_list=["NN","NNS"]
    SPACE_list=["SP","_SP"]
    INTJ_list=["UH"]
    SEP_list=["[SEP]"]
    MASK_list=["[MASK]"]
    all_kind_list=[]
    all_kind_list.append(SYM_list)
    all_kind_list.append(PUNCT_list)
    all_kind_list.append(X_list)
    all_kind_list.append(ADJ_list)
    all_kind_list.append(PART_list)
    all_kind_list.append(ADV_list)
    all_kind_list.append(VERB_list)
    all_kind_list.append(CCONJ_list)
    all_kind_list.append(ADP_list)
    all_kind_list.append(NUM_list)
    all_kind_list.append(DET_list)
    all_kind_list.append(PRON_list)
    all_kind_list.append(NOUN_list)
    all_kind_list.append(SPACE_list)
    all_kind_list.append(INTJ_list)
    all_kind_list.append(SEP_list)
    all_kind_list.append(MASK_list)
    reduced_pos_encoder_dict={}
    for index,poses_list in enumerate(all_kind_list):
        reduced_pos_encoder_dict[index+1]=poses_list 
    return reduced_pos_encoder_dict

def reduced_pos_match(tag_dict,token_list,reduced_pos_encoder_dict):
    
    pos_list=[]
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
            elif token == '[CLS]':
                pos_list.append('[CLS]')
            elif token == '[SEP]':
                pos_list.append('[SEP]')
            elif token == '[MASK]':
                pos_list.append('[MASK]')
            else:
                pos_list.append('NONE')

    pos_embedding=[]
    for pos in pos_list:
        for key,set_pos_list in reduced_pos_encoder_dict.items():
            if pos in set_pos_list: 
                pos_embedding.append(key)
                break
    return pos_embedding

def get_key (pos_encoder_dict, value):
    return [k for k, v in pos_encoder_dict.items() if v == value]

def get_pos_embedding(semantics_list,ori_syntactic_list,syntactic_list_in_dict,nlp):
    pos_embedding_list = []
    # pos_encoder_dict_temp=convert_tuple_to_dict(nlp.get_pipe("tagger").labels)

    # pos_encoder_dict={}
    # for key_num,pos_tag in pos_encoder_dict_temp.items():
    #     pos_encoder_dict[key_num+1]=pos_tag
    reduced_pos_encoder_dict=get_reduced_pos_encoder_dict()
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
        # semantics_pos=pos_match(semantics_tag_dict,semantics_token,pos_encoder_dict)
        semantics_pos=reduced_pos_match(semantics_tag_dict,semantics_token,reduced_pos_encoder_dict)
        for j,syntactic_sentence in enumerate(syntactic_list_in_dict[i]):
            input_pos=[]
            input_token = []
            syntactic_token=[]
            # [CLS] + semantics_sentence(X2) + [SEP] + syntactic_sentence(X1) 
            input_token=semantics_token.copy()
            syntactic_token=syntactic_sentence
            input_token.extend(syntactic_sentence)
            
            
            # syntactic_pos=pos_match(syntactic_tag_dict,syntactic_token,pos_encoder_dict)
            syntactic_pos=reduced_pos_match(syntactic_tag_dict,syntactic_token,reduced_pos_encoder_dict)
            
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
        "train":"/user_data/Project/Controllable_Syntax_with_BERT/dataset/mingda_train_1000.txt",
        "test":"/user_data/Project/Controllable_Syntax_with_BERT/dataset/mingda_test_500.txt",
        "validation":"/user_data/Project/Controllable_Syntax_with_BERT/dataset/mingda_validation_500.txt"
    }
    for key,data_path in data_path_dict.items():

        print("get_data_list...")
        semantics_list,ori_syntactic_list=get_dataset_list(data_path)
        print("get extrapolate_syntactic...")
        syntactic_list_in_dict,part_maskLM_embedding_list_in_dict=data_extra_preprocess.extrapolate_syntactic(ori_syntactic_list,nlp)
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