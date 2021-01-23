import random
import spacy
from transformers import BertTokenizer
nlp =spacy.load("model/spacy/en_core_web_md-2.3.1/en_core_web_md/en_core_web_md-2.3.1")
# ignored_pos_list=['NN','NNS','NNP','NNPS',"IN","VB"]
accepted_pos_list=["WRB","WP","WP$","WDT","IN","TO","MD"]
tokenizer =BertTokenizer(vocab_file='bert-base-uncased-vocab.txt')

train_data_count=6000
test_data_count=int(train_data_count/2)
validation_data_count=int(train_data_count/2)
# 考慮相似分數
def data_preprocess(path):
    count=0
    sentence_train_list=[]
    sentence_test_list=[]
    sentence_validation_list=[]
    with open(path, "r", encoding='utf-8') as f:     
        for line in f:
            line_split_list=line.split('\t')
            if len(tokenizer.tokenize(line_split_list[0])) >=200 or len(tokenizer.tokenize(line_split_list[1])) >=200:
                continue
            if random.random() > 0.5:
                simlarlity_socore= float(line_split_list[2])
                # 把符合相似分數區間的句子找出來
                if simlarlity_socore < 0.9 and simlarlity_socore > 0.4 and  check_syntactic_sentence(line_split_list[1],accepted_pos_list):
                    count+=1
                    if count <= train_data_count:
                        sentence_train_list.append(line_split_list[0]+"\t"+line_split_list[1])
                    elif count > train_data_count and count <= train_data_count+test_data_count:
                        sentence_test_list.append(line_split_list[0]+"\t"+line_split_list[1])
                    else :
                        sentence_validation_list.append(line_split_list[0]+"\t"+line_split_list[1])
                    # print(count)
                    if count % 1000 ==0:
                        print(count)
                    if count ==test_data_count+train_data_count+validation_data_count:
                        break
                
    return sentence_train_list,sentence_test_list,sentence_validation_list


# 使用作者的training data
def data_preprocess_from_mingda_trining_data(path):
    count=0
    sentence_train_list=[]
    sentence_test_list=[]
    sentence_validation_list=[]
    with open(path, "r", encoding='utf-8') as f:     
        for line in f:
            line_split_list=line.split('\t')
            if len(tokenizer.tokenize(line_split_list[0])) >=200 or len(tokenizer.tokenize(line_split_list[1])) >=200:
                continue
            # 把符合相似分數區間的句子找出來
            if check_syntactic_sentence(line_split_list[1].strip("\n"),accepted_pos_list):
                count+=1
                if count <= train_data_count:
                    sentence_train_list.append(line_split_list[0]+"\t"+line_split_list[1].strip("\n"))
                elif count > train_data_count and count <= train_data_count+test_data_count:
                    sentence_test_list.append(line_split_list[0]+"\t"+line_split_list[1].strip("\n"))
                else :
                    sentence_validation_list.append(line_split_list[0]+"\t"+line_split_list[1].strip("\n"))
                # print(count)
                if count % 1000 ==0:
                    print(count)
                if count ==test_data_count+train_data_count+validation_data_count:
                    break
                
    return sentence_train_list,sentence_test_list,sentence_validation_list


def save_files(sentence_train_list,sentence_test_list,sentence_validation_list):
    # 儲存檔案
    save_path="/user_data/Project/Controllable_Syntax_with_BERT/dataset/"
    with open(save_path+"mingda_train_"+str(train_data_count)+".txt",'w',encoding='utf-8') as f:
        for sentence in sentence_train_list:
            f.write(sentence+"\n")
    with open(save_path+"mingdatest_"+str(test_data_count)+".txt",'w',encoding='utf-8') as f:
        for sentence in sentence_test_list:
            f.write(sentence+"\n")
    with open(save_path+"mingdavalidation_"+str(validation_data_count)+".txt",'w',encoding='utf-8') as f:
        for sentence in sentence_validation_list:
            f.write(sentence+"\n")

def check_syntactic_sentence(ori_syntactic_sentence,accepted_pos_list):
    doc=nlp(ori_syntactic_sentence)
    syntactic_token_list=spacy_tokened_(doc,accepted_pos_list)
    if len(doc)!=len(syntactic_token_list) and len(syntactic_token_list)>=3:
        return True
    return False

def spacy_tokened_(doc,accepted_pos_list):
    syntactic_token_list=[]
    token_list=tokenizer.tokenize(doc.text)
    for i,token in enumerate(doc): 
        if token.tag_ in accepted_pos_list and token.text in token_list:
            syntactic_token_list.append(token.text)
    return syntactic_token_list

if __name__ == "__main__":

    # dataset_path="/user_data/Project/Controllable_Syntax_with_BERT/dataset/para-nmt-50m.txt"

    # sentence_train_list,sentence_test_list,sentence_validation_list=data_preprocess(dataset_path)

    mingda_dataset_path='/user_data/Project/Controllable_Syntax_with_BERT/mingda_chen_dataset/train.txt'
    sentence_train_list,sentence_test_list,sentence_validation_list=data_preprocess_from_mingda_trining_data(mingda_dataset_path)
    save_files(sentence_train_list,sentence_test_list,sentence_validation_list)
