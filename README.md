# Learning Attention-based Embeddings for Relation Prediction in Knowledge Graphs

This program provides the implementation of the attention based model for the knowledge base completion task.

## Usage

### Requirements
- [conda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)

Please download miniconda from above link and create an environment using the following command:

        conda env create -f pytorch35.yml

Activate the environment before executing the program as follows:

        source activate pytorch35

### Training

**Parameters:** 

`--data`: Specify the folder of the dataset. 

`--epochs_gat`: Number of epochs for gat training.

`--epochs_conv`: Number of epochs for convolution training.

`--lr`: Initial learning rate.

`--weight_decay_gat`: L2 reglarization for gat.

`--weight_decay_conv`: L2 reglarization for conv.

`--get_2hop`: Get a pickle object of 2 hop neighbors.

`--use_2hop`: Use 2 hop neighbors for training.  

`--output_folder`: Path of output folder for saving models.
  
`--batch_size_gat`: Batch size for gat model.

`--valid_invalid_ratio_gat`: Ratio of valid to invalid triples for GAT training.
  
`--drop_gat`: Dropout probability for attention layer.

`--alpha`: LeakyRelu alphas for attention layer.

`--nhead_GAT`: Number of heads for multihead attention.

`--margin`: Margin used in hinge loss.

`--batch_size_conv`: Batch size for convolution model.

`--valid_invalid_ratio_conv`: Ratio of valid to invalid triples for conv training.

`--out_channels`: Number of output channels in conv layer.

`--drop_conv`: Dropout probability for conv layer.

### Reproducing results 

To reproduce the results published in the paper:      
                
* **Wordnet**
        
        $ python3 main.py --get_2hop True

* **Freebase**

        $ python3 main.py --data ./data/FB15k-237/ --epochs_gat 3000 --epochs_conv 150 --weight_decay_gat 0.00001 --get_2hop True --partial_2hop True --batch_size_gat 272115 --margin 1 --out_channels 50 --drop_conv 0.3
        
