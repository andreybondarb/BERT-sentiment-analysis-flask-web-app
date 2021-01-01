import pickle
import numpy as np
import transformers
from transformers import BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split
import torch
import time

print ("Preparing classifier")
start_time = time.time()

device = torch.device("cpu")
tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased', do_lower_case=True)

# Model load.
model = torch.load('bert_model_Russian_01.pt', map_location=torch.device('cpu'))
print ("Classifier is ready")
print (time.time() - start_time, "seconds")

def return_ids_masks(sentences):
    input_ids = []
    attention_masks = []

    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 300,           # Pad & truncate all sentences.
                            truncation=True,
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                       )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids, attention_masks


def predict(loader, is_probs=False, is_targets=False):
    model.eval()
    predictions = []
    probabilities = [] #
    targets = []
    softmax = torch.nn.Softmax(dim=1)
    for batch in loader:
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            logits = model(batch[0], token_type_ids=None, 
            attention_mask=batch[1])

            predictions.append(torch.argmax(logits[0], dim=1).detach().cpu().numpy())

            if is_probs:
                probs = softmax(logits[0]).detach().cpu().numpy()
                probabilities.append(probs)
                if is_targets:
                    targets.append(batch[1].to('cpu').numpy())

    if is_probs and not is_targets:
        return np.concatenate(probabilities, axis=0)
    
    elif is_probs and is_targets:
        return np.concatenate(probabilities, axis=0), np.concatenate(targets)
    
    else:
        return np.concatenate(predictions)



