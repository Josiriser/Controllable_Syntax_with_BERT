import pickle
import torch
import os
from tqdm import tqdm 
import datetime
from tools import write_message
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertConfig, BertForMaskedLM, AdamW,AutoTokenizer
from data_extra_preprocess import make_dataset

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def get_trained_model_path(folder_name):
    model_folder_list=os.listdir("trained_model/")
    model_folder_list.sort()
    new_trained_model_path=""
    # for each_folder in model_folder_list:
    #     folder = os.path.exists('trained_model/'+str(int(each_folder)+1))
    #     if not folder:
    #         os.makedirs('trained_model/'+str(int(each_folder)+1))
    #         new_trained_model_path='trained_model/'+str(int(each_folder)+1)
    #         break
    folder = os.path.exists("trained_model/"+folder_name)
    if not folder:
        os.makedirs('trained_model/'+folder_name)
    new_trained_model_path='trained_model/'+folder_name

    assert new_trained_model_path!=""

    return new_trained_model_path



if __name__ == "__main__":
    print("start")
    pre_trained_model_name="bert-base-uncased"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # device = torch.device("cuda")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("start to read test data feature")
    test_data = open("data_feature/test_data_feature_5.pkl", "rb")
    test_data_feature = pickle.load(test_data)
    test_input_id = test_data_feature["input_id"]
    test_input_segment = test_data_feature["input_segment"]
    test_input_attention = test_data_feature["input_attention"]
    test_input_maskLM = test_data_feature["input_maskLM"]
    test_dataset = make_dataset(
        test_input_id, test_input_segment, test_input_attention, test_input_maskLM)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    print("start to read train data feature")
    train_data = open("data_feature/train_data_feature_10.pkl", "rb")
    train_data_feature = pickle.load(train_data)
    train_input_id = train_data_feature["input_id"]
    train_input_segment = train_data_feature["input_segment"]
    train_input_attention = train_data_feature["input_attention"]
    train_input_maskLM = train_data_feature["input_maskLM"]
    train_dataset = make_dataset(
        train_input_id, train_input_segment, train_input_attention, train_input_maskLM)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    print("finish reading feature")
    config = BertConfig.from_pretrained(pre_trained_model_name, type_vocab_size=2)

    model = BertForMaskedLM.from_pretrained(
        pre_trained_model_name, from_tf=bool('.ckpt' in pre_trained_model_name), config=config)
    #####DataParallel
    model=nn.DataParallel(model)
    #####DataParallel
    model.to(device)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], "weight_decay": 0.1, },
        {"params": [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-6, eps=1e-8)
    # optimizer = AdamW(model.parameters(), lr=5e-3)

    model.zero_grad()
    print("get path")
    folder_name="baseline_small_test_SEP_model_DataParallel"
    trained_model_path=get_trained_model_path(folder_name)
    loss_information_file_path=os.path.join("loss",folder_name+".txt")
    print(loss_information_file_path)
    for epoch in range(10):
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
            # sum_train_loss+=loss.item()
            # loss.sum().backward()
             #####DataParallel
            sum_train_loss+=loss.mean().item()
            loss.mean().backward()
             #####DataParallel
            optimizer.step()

            # _out = torch.argmax(logits,dim=-1).detach().cpu().numpy().tolist()[0]

            # _out = _out[:20]
            # _label = batch[3][0][:20].detach().cpu().numpy().tolist()
            
            # _new_out = []
            # _new_label = []
            # for _o,_l in zip(_out,_label):
            #     if _l != -100:
            #         _new_out.append(_o)
            #         _new_label.append(_l)
            
            # print('out:',tokenizer.decode(_new_out),'label:',tokenizer.decode(_new_label))

            model.zero_grad()


        average_trian_loss=round(sum_train_loss/count,3)

        # test model
        sum_test_loss=0.0
        count=0
        print("testing...")
        model.eval()
        for batch_index, batch in enumerate(tqdm(test_dataloader)):  
            count+=1      
            batch = tuple(t.to(device) for t in batch)
            output = model(input_ids=batch[0], token_type_ids=batch[1],
                           attention_mask=batch[2], masked_lm_labels=batch[3])

            loss,logits = output[:2]
            # sum_test_loss+=loss.item()
             #####DataParallel
            sum_test_loss+=loss.mean().item()
             #####DataParallel
        average_test_loss=round(sum_test_loss/count,3)

        message='第' + str(epoch+1) + '次' + '訓練模式，loss為:' + str(average_trian_loss) + '，測試模式，loss為:' + str(average_test_loss)
        print(message)
        write_message(loss_information_file_path,message+"\n")
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(trained_model_path+'/'+str(epoch))
