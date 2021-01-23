import os
import pickle
import torch
import datetime
import argparse
from tqdm import tqdm 
import torch.nn as nn
from tools import get_data_set,predict_test_in_train_mode,write_message
from torch.utils.data import DataLoader
from transformers import BertConfig,BertForMaskedLM,AdamW
## 外部參數設定
parser = argparse.ArgumentParser()
parser.add_argument('--train_file_path', '-trfp', type=str)
parser.add_argument('--test_file_path', '-tefp', type=str)
parser.add_argument('--valid_file_path', '-vafp', type=str)
parser.add_argument('--trained_model_path', '-modpath', type=str)
parser.add_argument('--batch_size', '-batch', type=int)
parser.add_argument('--gpu_num', '-gpun', type=int)
parser.add_argument('--epoch_num', '-epoch', type=int)
parser.add_argument('--loss_file_name', '-lossnm', type=str)

args = parser.parse_args()

## 預設值
pre_trained_model_name="bert-base-uncased"
file_name=args.loss_file_name
loss_information_file_path=os.path.join("loss",file_name+".txt")




if __name__ == "__main__":
    print("start to trian")

    # GPU setting
    if args.gpu_num > 1 :
        device = torch.device('cuda')
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    else:
        device = torch.device('cpu')
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    train_dataset=get_data_set(args.train_file_path)
    train_dataloader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)

    test_dataset=get_data_set(args.test_file_path)
    test_dataloader=DataLoader(test_dataset,batch_size=args.batch_size,shuffle=True)

    config = BertConfig.from_pretrained(pre_trained_model_name, type_vocab_size=2)

    model = BertForMaskedLM.from_pretrained(
        pre_trained_model_name, from_tf=bool('.ckpt' in pre_trained_model_name), config=config)
    
    if args.gpu_num > 1 :
        model=nn.DataParallel(model)

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
    
    for epoch in range(args.epoch_num):
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
            if args.gpu_num > 1:
                sum_train_loss+=loss.mean().item()
                loss.mean().backward()
            else:   
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
        for batch_index, batch in enumerate(tqdm(test_dataloader)):  
            count+=1      
            batch = tuple(t.to(device) for t in batch)
            output = model(input_ids=batch[0], token_type_ids=batch[1],
                           attention_mask=batch[2], masked_lm_labels=batch[3])

            loss,logits = output[:2]
            # predict_test_in_train_mode(input_ids=batch[0],labels=batch[3],logits=logits)
            if args.gpu_num > 1:
                sum_test_loss+=loss.mean().item()
            else:
                sum_test_loss+=loss.item()
          
        average_test_loss=round(sum_test_loss/count,3)

        message='第' + str(epoch+1) + '次' + '訓練模式，loss為:' + str(average_trian_loss) + '，測試模式，loss為:' + str(average_test_loss)
        print(message)
        write_message(loss_information_file_path,message+"\n")
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        if not os.path.isdir(args.trained_model_path):
            os.mkdir(args.trained_model_path)
        model_to_save.save_pretrained(args.trained_model_path+'/'+str(epoch))
