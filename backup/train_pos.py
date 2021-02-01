import os
import json
import torch
import spacy
import torch.nn as nn
from transformers import BertConfig,BertForMaskedLM,AdamW
import argparse
import train_pos_copy
import data_preprocess
import data_preprocess_pos
from torch.utils.data import DataLoader,TensorDataset
from tqdm import tqdm
from train import get_trained_model_path
from tools import predict_test_in_train_mode
from Model_holder import ModelHolder

def make_dataset(input_id, input_pos, input_attention, input_maskLM):
    all_input_id = torch.tensor(
        [input_id for input_id in input_id], dtype=torch.long)
    all_input_pos = torch.tensor(
        [input_pos for input_pos in input_pos], dtype=torch.long)
    all_input_attention = torch.tensor(
        [input_attention for input_attention in input_attention], dtype=torch.long)
    all_input_maskLM = torch.tensor(
        [input_maskLM for input_maskLM in input_maskLM], dtype=torch.long)
   
    full_dataset = TensorDataset(
        all_input_id, all_input_pos,all_input_attention, all_input_maskLM)
    return full_dataset
# load model
token_type_size=13
config=BertConfig.from_pretrained('bert-base-uncased')
model=BertForMaskedLM.from_pretrained('bert-base-uncased')
model.bert.embeddings.token_type_embeddings=nn.Embedding(token_type_size,config.hidden_size)
nlp =spacy.load("model/spacy/en_core_web_md-2.3.1/en_core_web_md/en_core_web_md-2.3.1")

semantics_list=["about such concepts as absurdity it knew nothing .",]
ori_syntactic_list=["he had no idea about such terms ."]
syntactic_list_in_dict,part_maskLM_embedding_list_in_dict=data_preprocess.extrapolate_syntactic(ori_syntactic_list,nlp)
token_embedding_id_list, segment_embedding_list, attention_embedding_list, maskLM_embedding_list=data_preprocess.get_embedding(semantics_list,ori_syntactic_list,syntactic_list_in_dict,part_maskLM_embedding_list_in_dict)
pos_embedding_list=data_preprocess_pos.get_pos_embedding(semantics_list,ori_syntactic_list,syntactic_list_in_dict,nlp)
pos_embedding_list=data_preprocess_pos.padding(pos_embedding_list)




train_dataset = make_dataset(token_embedding_id_list, pos_embedding_list,attention_embedding_list, maskLM_embedding_list)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

print("get path")
trained_model_path=get_trained_model_path("new_POS_train")


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")
model.to(device)

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {"params": [p for n, p in model.named_parameters() if not any(
        nd in n for nd in no_decay)], "weight_decay": 0.1, },
    {"params": [p for n, p in model.named_parameters() if any(
        nd in n for nd in no_decay)], "weight_decay": 0.0},
]
optimizer = AdamW(optimizer_grouped_parameters, lr=5e-6, eps=1e-8)
epoch_count=200
for epoch in range(epoch_count):
    # trian model
    sum_train_loss=0.0
    count=0
    print("training")
    model.train()
    for batch_index, batch in enumerate(tqdm(train_dataloader)):       
        count+=1      
        batch = tuple(t.to(device) for t in batch)
        
        output = model(input_ids=batch[0], token_type_ids=batch[1],
                        attention_mask=batch[2], masked_lm_labels=batch[3])

        loss,logits = output[:2]
        sum_train_loss+=loss.item()
        loss.sum().backward()
        optimizer.step()

        model.zero_grad()


    average_trian_loss=round(sum_train_loss/count,3)

    # test model
    sum_test_loss=0.0
    count=0
    print("testing...")
    model.eval()
    for batch_index, batch in enumerate(tqdm(train_dataloader)):  
        count+=1      
        batch = tuple(t.to(device) for t in batch)
        
        output = model(input_ids=batch[0], token_type_ids=batch[1],
                        attention_mask=batch[2], masked_lm_labels=batch[3])

        loss,logits = output[:2]
        sum_test_loss+=loss.item()
        predict_test_in_train_mode(input_ids=batch[0],labels=batch[3],logits=logits)
        
    average_test_loss=round(sum_test_loss/count,3)

    message='第' + str(epoch+1) + '次' + '訓練模式，loss為:' + str(average_trian_loss) + '，測試模式，loss為:' + str(average_test_loss)
    print(message)
    
    # if epoch+1 ==epoch_count:
    #     model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    #     model_to_save.save_pretrained(trained_model_path+'/'+str(epoch))