import spacy
from tools import convert_tuple_to_dict,get_sentence_tag_dict
from data_preprocess import get_data_list
from transformers import BertTokenizer
from tqdm import tqdm
nlp=spacy.load("model/spacy/en_core_web_md-2.3.1/en_core_web_md/en_core_web_md-2.3.1")
tokenizer =BertTokenizer(vocab_file='bert-base-uncased-vocab.txt')


ori_pos_encoder_dict=convert_tuple_to_dict(nlp.get_pipe("tagger").labels)

print(ori_pos_encoder_dict)

ori_syntactic_list=["he had no idea about such terms ."]



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
for index,pos_list in enumerate(all_kind_list):
    reduced_pos_encoder_dict[index+1]=pos_list
# print(reduced_pos_encoder_dict)

tag_dict=get_sentence_tag_dict(nlp,ori_syntactic_list[0])
token_list= tokenizer.tokenize(ori_syntactic_list[0])

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

# print(pos_list)
pos_embedding=[]
for pos in pos_list:
    for key,set_pos_list in reduced_pos_encoder_dict.items():
        if pos in set_pos_list: 
            pos_embedding.append(key)
        else :
            pos_embedding.append(0)
for index,pos_tag in ori_pos_encoder_dict.items():
    print()
# data_path='mingda_chen_dataset/train.txt'
# semantics_list,ori_syntactic_list=get_data_list(data_path)




# tag_set=[]
# for i in tqdm(range(len(semantics_list))):
#     doc=nlp(semantics_list[i])
#     for token in doc:
#         if token.tag_ not in tag_set:
#             tag_set.append(token.tag_)
#     doc2 =nlp(ori_syntactic_list[i])      
#     for token in doc2:
#          if token.tag_ not in tag_set:
#             tag_set.append(token.tag_)

# print(tag_set)