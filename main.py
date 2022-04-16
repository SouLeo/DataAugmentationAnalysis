"""
Author: Selma Wanna
Institution: University of Texas at Austin

This project runs experiments on data augmentation for data scarce NLP tasks by ablating
the following models and techniques:

Data Augmentation (DA) neural models à la Kumar et al. 2021  :
- GPT-2 (autoregressive)
- BERT (auto-encoding)
- BART (seq2seq)
- Back-translation (NMT methodology)

Investigating the following behaviors:
- n-gram diversity
- entropy of predictions (on downstream SA task on IMDB dataset)
- noise injections in hidden layer
- noise injections in input sequence

Future Work:
- investigating conditional generation techniques for DA
- unsupervised learning techniques
"""


from transformers import GPT2Tokenizer, GPT2LMHeadModel

from data_input import get_low_resource_imdb_dataset


if __name__ == '__main__':
    # construct data-sparse IMDB data loader sourced from torchnlp.datasets
    imdb_dataloader = get_low_resource_imdb_dataset(ds_size=10)

    # load GPT-2 models from huggingface
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
