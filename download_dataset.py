import os

dataset_url = "https://www.dropbox.com/s/prmr8ujytwlfq3a/mingda_dataset.rar"

os.system("wget "+dataset_url)

file_name = "mingda_dataset.rar"
file_Dirname="dataset"

if not os.path.isdir(file_Dirname):
    os.makedirs(file_Dirname)
    
os.system("rar x "+file_name+" "+file_Dirname)
