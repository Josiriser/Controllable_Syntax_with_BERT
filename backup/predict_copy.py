import os
import torch
import spacy
import torch.nn.functional as F
from data_preprocess import extrapolate_syntactic
from transformers import BertConfig, BertForMaskedLM, BertTokenizer

tokenizer =BertTokenizer(vocab_file='bert-base-uncased-vocab.txt')
nlp =spacy.load("model/spacy/en_core_web_md-2.3.1/en_core_web_md/en_core_web_md-2.3.1")
def read_test_data():
    data_path="/user_data/Project/Controllable_Syntax_with_BERT/dataset/test.txt"
    semantic_list=[]
    syntactic_list=[]
    with open(data_path,'r',encoding='utf-8') as f:
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
    return string

def write_files(input_semantic,input_syntactic,ori_syntactic,output_sentence):
    file_path='result/output.txt'
    with open(file_path,'a',encoding='utf-8') as f:

        f.write("input_semantic : "+extract_sentence_from_list(input_semantic) +"\n")
        f.write("input_syntactic : "+extract_sentence_from_list(input_syntactic)+"\n")
        f.write("ori_syntactic : "+ori_syntactic+"\n")
        f.write("output_sentence : "+extract_sentence_from_list(output_sentence)+"\n")
        f.write("\n\n")

def check_special_token(sentence_list):
    if '[MASK]' in sentence_list:
        return True
    return False
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda")

    ori_semantic_list,ori_syntactic_list=read_test_data()
    bert_config, bert_class = (BertConfig, BertForMaskedLM)

    config = bert_config.from_pretrained('trained_model/2/4/config.json')
    model = bert_class.from_pretrained('trained_model/2/4/pytorch_model.bin', config=config)
    model.eval()

   

    output_txt=""
    count=0
    for index,ori_semantic_sentence in enumerate(ori_semantic_list):
        input_sentence=[]
        input_semantic=[]
        input_syntactic=[]

        tokenized_text=tokenizer.tokenize(ori_semantic_sentence)

        input_semantic.append("[CLS]")
        input_semantic.extend(tokenized_text)
        input_semantic.append("[SEP]")

        
        # extroplation
        input_syntactic=data_preprocess(ori_syntactic_list[index])

        syntactic_iteration=input_syntactic.copy()
        while check_special_token(syntactic_iteration):

            input_sentence.extend(input_semantic)
            input_sentence.extend(syntactic_iteration.copy())
            if len(input_sentence)>512:
                break
            syntactic_iteration.clear()
            
            input_sentence_ids=tokenizer.convert_tokens_to_ids(input_sentence)
            tokens_tensor=torch.tensor([input_sentence_ids])
            outputs=model(tokens_tensor)
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
        # print(output_sentence)
        
        count+=1
        output_txt=output_txt+"input_semantic  : "+extract_sentence_from_list(input_semantic) +"\n"
        output_txt=output_txt+"input_syntactic : "+extract_sentence_from_list(input_syntactic)+"\n"
        output_txt=output_txt+"output_sentence : "+extract_sentence_from_list(output_sentence)+"\n"
        output_txt=output_txt+"ori_syntactic   : "+ori_syntactic_list[index]+"\n"
        output_txt=output_txt+"\n\n"
        print(count)
        if count % 50 ==0:
            file_path='result/output2_4.txt'
            with open(file_path,'a',encoding='utf-8') as f:
                f.write(output_txt)
            output_txt=""
            
       
        
    print("finished")




    # semantic_sentence="My name is olympia."
    # syntactic_sentence=["you","can","me"]

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

    # for word in process_sentence:
    #     if word !='[MASK]':
    #         print(word)
    #     else:
    #         maskpos=process_sentence.index('[MASK]')

    #         logit_prob = F.softmax(predictions[0, maskpos]).data.tolist()
    #         predicted_index = torch.argmax(predictions[0, maskpos]).item()
    #         predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    #         print(predicted_token,logit_prob[predicted_index])
    #         process_sentence[maskpos]=predicted_token


