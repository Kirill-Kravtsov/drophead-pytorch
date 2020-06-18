# drophead-pytorch
An implementation of drophead regularization for pytorch transformers

## Introduction
This is an Pytorch implementation of [Scheduled DropHead: A Regularization Method for Transformer Models](https://arxiv.org/pdf/2004.13342.pdf), a regularization method for transformers. This implementation was designed to work on top of [transformers](https://github.com/huggingface/transformers). Currently works for Bert, Roberta and XLM-Roberta.

## How to use
There is only one main function - `set_drophead(model, p_drophead)`. As model you can provide any of following:
* transformers.BertModel
* transformers.RobertaModel
* transformers.XLMRobertaModel
* Any downstream model from transformers which uses one of above (e.g. transformers.BertForSequenceClassification)
* Any custom downstream model which which uses first 3 above (has it as attribute). See [example](https://github.com/kirill-kravtsov/drophead-pytorch/blob/master/example.ipynb)

Note:
* Function `set_drophead` works inplace
* If for some reason you are using multiple base model inside one single custom class (e.g. inside you model you average predcitions from Bert and Roberta) than apply function directly to your base models. See 2nd from [here](https://github.com/kirill-kravtsov/drophead-pytorch/blob/master/example.ipynb)
* In this repo only drophead mechanism itself is implemented. If you want scheduled drophead like suggested in paper then simply add call `set_drophead(model, p_drophead)` into your training where p_drophead will be changing according to your schedule.

## Requirements
Was tested with python3, pytorch 1.4.0 and transformers 2.9.0 but probably will work with older versions of last two.
