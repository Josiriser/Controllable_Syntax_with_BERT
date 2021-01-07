import os
import torch
import pickle
import torch.nn as nn
import data_preprocess
from tools import make_dataset,get_trained_model_path,write_message
from Model_holder import ModelHolder
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset,DataLoader
from transformers import AutoModel,AutoConfig,AutoTokenizer,AdamW,BertForMaskedLM,AutoModelForMaskedLM
from transformers.modeling_bert import BertOnlyMLMHead,MaskedLMOutput
import transformers.configuration_bert
##
class Bert_config(transformers.configuration_bert.BertConfig):
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=51,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        gradient_checkpointing=False,
        position_embedding_type="absolute",
        use_cache=True,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache

##
class POS_Model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.config = AutoConfig.from_pretrained('bert-base-uncased')
        self.config = Bert_config()
        self.model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
        self.model.bert.embeddings.token_type_embeddings=nn.Embedding(self.config.type_vocab_size, self.config.hidden_size)
        # self.pos_embs = nn.Embedding(51, self.config.hidden_size)
        self.mask_predict = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.cls = BertOnlyMLMHead(self.config)
        
        

    def forward(self,input_ids=None,token_type_ids=None,attention_mask=None,labels=None,pos_ids=None):
        model_embedding = self.model.bert.embeddings
        # pos_ids 是詞性 不是position

        # inputs_embeds = model_embedding(input_ids=input_ids,token_type_ids=token_type_ids
                                    # ,position_ids=pos_ids)
        inputs_embeds = model_embedding(input_ids=input_ids,token_type_ids=pos_ids)
        
        # return 0
        
        outputs = self.model(output_attentions=True,output_hidden_states=True,
        inputs_embeds=inputs_embeds,labels=labels,attention_mask=attention_mask)
      
        # # loss
        
        if labels !=None:
            hidden_states = outputs[2]
            output_attentions=outputs[3]
        else:
            hidden_states = outputs[1]
            output_attentions=outputs[2]
        
        # #　add_pos
        last_hidden=hidden_states[12]
        
        prediction_scores = self.cls(last_hidden)
        masked_lm_loss = None
        
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=hidden_states,
            attentions=output_attentions
        )

        
def test_model():
    ### fake data
    intput_token=[101,101, 2941, 1010, 5252, 26176, 2032, 1012, 102, 103, 2000, 103, 2044, 103]
    input_segment=[0,0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    input_attention=[1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    input_maskLM=[-100,-100, -100, -100, -100, -100, -100, -100, -100, 100, -100, 5376, -100, 2032]
    intput_pos=[50,50, 30, 2, 23, 38, 28, 5, 50, 50, 35, 50, 15, 50]

    intput_token_tensor=torch.tensor([intput_token],dtype=torch.long)
    input_segment_tensor=torch.tensor([input_segment],dtype=torch.long)
    input_attention_tensor=torch.tensor([input_attention],dtype=torch.long)
    input_maskLM_tensor=torch.tensor([input_maskLM],dtype=torch.long)
    intput_pos_tensor=torch.tensor([intput_pos],dtype=torch.long)

    Pos_model = POS_Model()
    logits=Pos_model(input_ids=intput_token_tensor,token_type_ids=input_segment_tensor,
    attention_mask=input_attention_tensor, labels=input_maskLM_tensor,
    pos_ids=intput_pos_tensor)

def main():
    folder_name="test_parallel"
    print("start")
    pre_trained_model_name="bert-base-uncased"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    device = torch.device("cuda")


    print("start to read test data feature")
    test_data = open("data_feature/test_data_feature_pos_3000.pkl", "rb")
    test_data_feature = pickle.load(test_data)
    test_input_id = test_data_feature["input_id"]
    test_input_segment = test_data_feature["input_segment"]
    test_input_attention = test_data_feature["input_attention"]
    test_input_maskLM = test_data_feature["input_maskLM"]
    test_input_pos = test_data_feature["input_pos"]
    test_dataset = make_dataset(
        test_input_id, test_input_segment, test_input_attention, test_input_maskLM,test_input_pos)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)
    print("start to read train data feature")
    train_data = open("data_feature/train_data_feature_pos_6000.pkl", "rb")
    train_data_feature = pickle.load(train_data)
    train_input_id = train_data_feature["input_id"]
    train_input_segment = train_data_feature["input_segment"]
    train_input_attention = train_data_feature["input_attention"]
    train_input_maskLM = train_data_feature["input_maskLM"]
    train_input_pos = train_data_feature["input_pos"]
    train_dataset = make_dataset(
        train_input_id, train_input_segment, train_input_attention, train_input_maskLM,train_input_pos)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    print("finish reading feature")

    print("get path")
    trained_model_path=get_trained_model_path(folder_name)

    model=POS_Model()
    model.to(device)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], "weight_decay": 0.1, },
        {"params": [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-6, eps=1e-8)

    with ModelHolder(model) as (model_holder,model):
        for epoch in range(5):
            print("trian model")
            # running_loss_val = 0.0
            sum_train_loss=0.0
            count=0
    
            # running_acc = 0.0
            model.train()
            
            for batch_index, batch in enumerate(tqdm(train_dataloader)):       
                count+=1      
                batch = tuple(t.to(device) for t in batch)
                output = model(input_ids=batch[0], token_type_ids=batch[1],
                            attention_mask=batch[2], labels=batch[3],pos_ids=batch[4])

                loss,logits = output[:2]
                sum_train_loss+=loss.item()
                loss.sum().backward()
                optimizer.step()

                model.zero_grad()
               
                
                
            average_trian_loss=round(sum_train_loss/count,3)

            print("model to save")
            ModelHolder.save_checkpoint2(self=None,model=model,save_path=trained_model_path+'/'+str(epoch)+"/")
            print("test model")

            sum_test_loss=0.0
            count=0
            # running_acc = 0.0
            model.eval()
            for batch_index, batch in enumerate(tqdm(test_dataloader)):  
                count+=1      
                batch = tuple(t.to(device) for t in batch)
                output = model(input_ids=batch[0], token_type_ids=batch[1],
                            attention_mask=batch[2], labels=batch[3],pos_ids=batch[4])

                loss,logits = output[:2]
                sum_test_loss+=loss.item()
                
            average_test_loss=round(sum_test_loss/count,3)

            message='第' + str(epoch+1) + '次' + '訓練模式，loss為:' + str(average_trian_loss) + '，測試模式，loss為:' + str(average_test_loss)
            write_message(os.path.join("loss",folder_name+".txt"),message+"\n")
            print(message)

            


if __name__ == "__main__":
    # test_model()
    main()