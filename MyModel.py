import torch
import torch.nn as nn
from transformers import AutoConfig,AutoModelForMaskedLM


class MyEmbeddings(nn.Module):
    def __init__(self,embeddings,config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0).cuda()
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size).cuda()
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size).cuda()
        
        ## 增加 pos_embeddings
        self.pos_embeddings=nn.Embedding(13,config.hidden_size).cuda()

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps).cuda()
        self.dropout = nn.Dropout(config.hidden_dropout_prob).cuda()

    def forward(self,input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None,pos_ids=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        ## 增加 pos_embeddings
        pos_embeddings=self.pos_embeddings(pos_ids)

        embeddings = inputs_embeds  +pos_embeddings
        # embeddings = inputs_embeds + position_embeddings + token_type_embeddings +pos_embeddings
        # embeddings = self.LayerNorm(embeddings)
        # embeddings = self.dropout(embeddings)
        return embeddings

class POS_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model= AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
        self.config=AutoConfig.from_pretrained("bert-base-uncased")
        self.model.bert.embeddings=MyEmbeddings(self.model.bert.embeddings,config=self.config)
        
    def forward(self,input_ids=None,token_type_ids=None,attention_mask=None,labels=None,pos_ids=None):
        embeddings_input = self.model.bert.embeddings(input_ids=input_ids,token_type_ids=token_type_ids,pos_ids=pos_ids)
        outputs = self.model(return_dict=True,inputs_embeds=embeddings_input,labels=labels,attention_mask=attention_mask)
        

        if labels is None:
            loss = None
            logits = outputs['logits']
        else :
            loss= outputs['loss']
            logits = outputs['logits']

        return loss,logits
if __name__ == "__main__":
    semantics_list=["about such concepts as absurdity it knew nothing .",]
    ori_syntactic_list=["he had no idea about such terms ."]



    model= AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    # print(model)
    config=AutoConfig.from_pretrained("bert-base-uncased")
    new_embedding=MyEmbeddings(model.bert.embeddings,config=config)
    model.bert.embeddings=new_embedding()
    model()
    print(new_embedding)