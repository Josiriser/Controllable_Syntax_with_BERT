import os
import argparse
import sacrebleu
from transformers import BertTokenizer

def main():

    ## 外部參數設定
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file_path', '-i', type=str)
    parser.add_argument('--ref_file_path', '-r', type=str)
    args = parser.parse_args()

    if args.input_file_path==None:
        args.input_file_path="mingda_chen_dataset/test_input.txt"
    if args.ref_file_path==None:
        args.ref_file_path="mingda_chen_dataset/test_ref.txt"

    ref,sys=read_data(args.input_file_path,args.ref_file_path)
    eva_bleu(ref,sys)
    

def eva_bleu(ref,sys):
   
    bleu=sacrebleu.corpus_bleu(sys,ref)
    print(bleu.score)
    

def read_data(input_file_path,ref_file_path):
    tokenizer = BertTokenizer(vocab_file='bert-base-uncased-vocab.txt')
    ref1=[]
    ref2=[]
    ref=[]
    with open(input_file_path,"r",encoding='utf-8') as f:
        for line in f:

            ref1.append(token_process(line.split("\t")[0],tokenizer))
            ref2.append(token_process(line.split("\t")[1].strip("\n"),tokenizer))
        ref.append(ref1)
        ref.append(ref2)
    sys=[]
    with open(ref_file_path,"r",encoding='utf-8') as f:
        for line in f:
            sys.append(line.strip("\n"))
    return ref,sys

def token_process(sentence,tokenizer):
    token_list = tokenizer.tokenize(sentence)
    text=""
    for token in token_list:
        text=text+token+" "

    return text


if __name__ == "__main__":
    main()