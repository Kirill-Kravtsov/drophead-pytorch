from transformers import BertModel, RobertaModel, XLMRobertaModel
import torch
import torch.nn.functional as F


VALID_CLS = (BertModel, RobertaModel, XLMRobertaModel)


def drophead_hook(module, input, output):
    if (not module.training) or (module.p_drophead==0):
        return output

    batch_size = output[0].shape[0]
    dist = torch.distributions.Bernoulli(torch.tensor([1-module.p_drophead]))
    mask = dist.sample((batch_size, module.num_attention_heads))
    mask = mask.to(output[0].device).unsqueeze(-1)
    count_ones = mask.sum(dim=1).unsqueeze(-1)

    orig_shape = output[0].shape
    self_att_res = module.transpose_for_scores(output[0])
    self_att_res = self_att_res * mask * module.num_attention_heads / count_ones
    self_att_res = self_att_res.permute(0, 2, 1, 3).view(*orig_shape)
    return (self_att_res,) + output[1:]


def valid_type(obj):
    return isinstance(obj, VALID_CLS)


def get_base_model(model):
    """
    Check model type. If not correct then try to find in attributes,
    """
    if not valid_type(model):
        attrs = [name for name in dir(model) if valid_type(getattr(model, name))]
        if len(attrs) == 0:
            raise ValueError("Please provide valid model")
        model =  getattr(model, attrs[0])
    return model


def set_drophead(model, p=0.1):
    """
    Adds drophead to model. Works inplace.
    Args:
        model: an instance of transformers.BertModel / transformers.RobertaModel /
            transformers.XLMRobertaModel or downstream model (e.g. transformers.BertForSequenceClassification)
            or any custom downstream model
    """
    if (p < 0) or (p > 1):
        raise ValueError("Wrong p argument")

    model = get_base_model(model)

    for bert_layer in model.encoder.layer:
        if not hasattr(bert_layer.attention.self, "p_drophead"):
            bert_layer.attention.self.register_forward_hook(drophead_hook)
        bert_layer.attention.self.p_drophead = p
