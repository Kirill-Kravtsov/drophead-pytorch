# Drophead - a pytorch implementation for transformers

## Introduction
This is a Pytorch implementation of [Scheduled DropHead: A Regularization Method for Transformer Models](https://arxiv.org/pdf/2004.13342.pdf), a regularization method for transformers. This implementation was designed to work on top of [transformers](https://github.com/huggingface/transformers) package. Currently it works for Bert, Roberta and XLM-Roberta.

## How to use
You can just copy `drophead.py` to your project.
There is only one main function - `set_drophead(model, p_drophead)`. As `model` you can provide any of the following:
* `transformers.BertModel`
* `transformers.RobertaModel`
* `transformers.XLMRobertaModel`
* Any downstream model from transformers which uses one of the above (e.g. `transformers.BertForSequenceClassification`).
* Any custom downstream model which uses first 3 above (has it as an attribute). See [example](https://github.com/kirill-kravtsov/drophead-pytorch/blob/master/example.ipynb).

Note:
* Function `set_drophead` works inplace.
* `model.train()` and `model.eval()` work the same as for usual dropout.
* If you use multiple base models inside one single custom class (e.g. inside your model you average predictions from Bert and Roberta) then apply function directly to your base models. See 2nd example from [here](https://github.com/kirill-kravtsov/drophead-pytorch/blob/master/example.ipynb).
* In this repo only drophead mechanism itself is implemented. If you want a scheduled drophead like suggested in paper then simply add a call `set_drophead(model, p_drophead)` into your training loop where `p_drophead` will be changing according to your schedule.

## Requirements
The code was tested with python3, pytorch 1.4.0 and transformers 2.9.0 but probably will work with older versions of the last two.
