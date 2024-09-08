#! /usr/bin/env python3
import sys
# append this as the system path
sys.path.append('pytorch_influence_functions')
from pytorch_influence_functions.utils import *
from pytorch_influence_functions.calc_influence_function import *
from pytorch_influence_functions.influence_function import * 
from MT_Dataloader import load_trained_model
from transformers import GPT2Tokenizer, GPT2LMHeadModel

if __name__ == "__main__":
    config = get_default_config()
    # load the pretrained transformer model
    #model = load_trained_model('/home/psen/Machine-Translation/model')
    model = GPT2LMHeadModel.from_pretrained('/home/psen/Machine-Translation/model') 
    #total_params = sum(p.numel() for p in model.parameters()) #numel:Returns the total number of elements in the input tensor.
    # it's the index of test sample will be used in calculation later


    num_test_samples = 50 # suppose there are 1000 test samples
    sample_list = [i for i in range(num_test_samples)] # list

    # get the dataloader along with source text list and target text list.
    #train_loader,test_loader,_,train_texts,train_labels,test_texts,test_labels= load_data(config['SRC_DIR'],config['LABEL_DIR'])

    train_file = config['TRAIN_DIR']
    test_file = config['TEST_DIR']

    # set the directory for log file
    # ptif.init_logging('/users/yangwr/InfluenceFunctions/logfile.log')
    init_logging(config['logdir'])

    # Implement influence function
    influences = calc_img_wise(config, model, train_file, test_file, sample_list)
