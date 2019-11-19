python main.py --data ./data/drugbank_100/ --pretrained_emb False


python main.py --data ./data/drugbank_100_new/ --pretrained_emb False --output_folder ./checkpoints/drugbank100_new/


python evaluate.py --data ./data/drugbank_1000/ --pretrained_emb False
