import os
import sacrebleu


def eva_bleu():
    refs=[]
    sys=[]
    path=os.path.join("result","bleu_text.txt")
    with open(path, "r", encoding='utf-8') as f:     
        for line in f:
            line_split_list=line.split('|||')
            refs.append([line_split_list[0]])
            sys.append(line_split_list[1])
    bleu=sacrebleu.corpus_bleu(sys,refs)
    print(bleu.score)
    
    return 0

def main():
    eva_bleu()



if __name__ == "__main__":
    main()