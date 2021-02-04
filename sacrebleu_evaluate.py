import os
import argparse
import sacrebleu

def main():

    ## 外部參數設定
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file_path', '-i', type=str)
    parser.add_argument('--ref_file_path', '-r', type=str)
    args = parser.parse_args()

    if args.input_file_path==None:
        args.input_file_path="mingda_chen_dataset/test_ref.txt"
    if args.ref_file_path==None:
        args.ref_file_path="mingda_chen_dataset/mingda_test_predict.txt"

    ref,sys=read_data(args.input_file_path,args.ref_file_path)
    eva_bleu(ref,sys)
    

def eva_bleu(ref,sys):
   
    bleu=sacrebleu.corpus_bleu(sys,ref)
    print(bleu.score)
    

def read_data(input_file_path,ref_file_path):

    ref_temp=[]
    ref=[]
    with open(input_file_path,"r",encoding='utf-8') as f:
        for line in f:
            ref_temp.append(line.strip("\n"))
        ref.append(ref_temp)

    sys=[]
    with open(ref_file_path,"r",encoding='utf-8') as f:
        for line in f:
            sys.append(line.strip("\n"))
    return ref,sys

if __name__ == "__main__":
    main()