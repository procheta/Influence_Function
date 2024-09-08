import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import ast
import numpy as np
import logging

def load_data(file_path, sep=","):
    """加载CSV数据集"""
    return load_dataset("csv", data_files=file_path, sep=sep)["train"]


def tokenize_data(dataset, tokenizer, max_length=128):
    """对数据集进行分词处理"""

    def tokenize_function(examples):
        inputs = tokenizer(examples['Bad_Practices'], padding='max_length', truncation=True, max_length=max_length,
                           return_tensors="pt")
        labels = tokenizer(examples['Good_Practices'], padding='max_length', truncation=True, max_length=max_length,
                           return_tensors="pt")
        return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'],
                'labels': labels['input_ids']}

    return dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

def load_trained_model(model_path, tokenizer_name="gpt2"):

    # Load the trained model
    model = GPT2LMHeadModel.from_pretrained(model_path)

    # Load the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)

    # Set the pad token for the tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_data(train_file, test_file, model_name="gpt2", batch_size=32, max_length=128):
    """主函数：加载和处理数据，返回所需的所有数据对象"""
    # 初始化tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 加载数据集
    train_dataset = load_data(train_file,sep="###")
    test_dataset = load_data(test_file, sep="###")

    # 分词处理
    train_encoded = tokenize_data(train_dataset, tokenizer, max_length)
    test_encoded = tokenize_data(test_dataset, tokenizer, max_length)

    # 创建DataLoader
    train_loader = create_dataloader(train_encoded, batch_size)
    test_loader = create_dataloader(test_encoded, batch_size)

    # 提取原始文本和标签
    train_texts = train_dataset['Bad_Practices']
    train_labels = train_dataset['Good_Practices']
    test_texts = test_dataset['Bad_Practices']
    test_labels = test_dataset['Good_Practices']

    return {
        'train_loader': train_loader,
        'test_loader': test_loader,
        'train_texts': train_texts,
        'train_labels': train_labels,
        'test_texts': test_texts,
        'test_labels': test_labels,
        'tokenizer': tokenizer,
        'train_encoded': train_encoded,
        'test_encoded': test_encoded
    }


def process_test_samples_with_neighbors(test_file, train_file, model_name="gpt2", batch_size=32, max_length=128):
    """处理测试样本及其对应的训练样本最近邻"""
    # 加载测试数据
    test_data = load_dataset("csv", data_files='test_50.csv', sep=",")["train"]

    # 加载和处理训练数据
    data = get_data(train_file, test_file, model_name, batch_size, max_length)
    train_encoded = data['train_encoded']
    for i, row in enumerate(test_data):
        test_sample = row['test_sample']
        try:
            # 使用 numpy 解析标签字符串
            x=row["helpful_sample_id"]
            x=x.replace("[","")
            x=x.replace("]","")
            xx=x.split(" ")
            neighbor_indices=[]
            for x1 in xx:
             try:
              neighbor_indices.append(int(x1))
             except:
                c=0
            #neighbor_indices = np.fromstring('1 2 3 4',dtype=int,  sep=' ')
            #print(neighbor_indices)
            # 检查索引是否有效
            max_index = len(train_encoded)
            valid_indices=[]
            for j in neighbor_indices:
             if j >=0 and j < max_index:
              valid_indices.append(j)
            #valid_indices = neighbor_indices[(neighbor_indices >= 0) & (neighbor_indices < max_index)]
            print("valid", valid_indices)
            if len(valid_indices) != len(neighbor_indices):
                logging.warning(
                    f"Some indices were out of range for sample {i}. Original length: {len(neighbor_indices)}, Valid length: {len(valid_indices)}")
            print("train encoded", train_encoded)
            # 为当前测试样本创建最近邻训练样本的DataLoader
            neighbors_loader = get_nearest_neighbors_loader(i, valid_indices, train_encoded, batch_size)

            yield {
                'test_sample': test_sample,
                'neighbors_loader': neighbors_loader
            }
        except Exception as e:
            logging.error(f"Error processing sample {i}: {str(e)}")
            continue


def get_nearest_neighbors_loader(test_sample_index, neighbor_indices, train_encoded, batch_size=32):
    """为给定的测试样本创建包含其最近邻训练样本的DataLoader"""
    try:
        subset = Subset(train_encoded, neighbor_indices)
        return create_dataloader(subset, batch_size)
    except Exception as e:
        logging.error(f"Error creating loader for sample {test_sample_index}: {str(e)}")
        return None


def create_dataloader(encoded_dataset, batch_size=32):
    """创建PyTorch DataLoader"""
    try:
        input_ids = torch.tensor([item['input_ids'] for item in encoded_dataset])
        attention_mask = torch.tensor([item['attention_mask'] for item in encoded_dataset])
        labels = torch.tensor([item['labels'] for item in encoded_dataset])
        dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    except Exception as e:
        logging.error(f"Error creating DataLoader: {str(e)}")
        return None
