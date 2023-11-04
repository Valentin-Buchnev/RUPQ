AVAILABLE_PARAMS_TYPE = ["all", "weight", "quant", "quant_input", "quant_weight"]


def get_params(model, params_type=None):
    model_parameters = []
    quant_input_parameters = []
    quant_weight_parameters = []
    for name, p in model.named_parameters():
        if "input_quantizer" in name:
            quant_input_parameters.append(p)
        elif "weight_quantizer" in name:
            quant_weight_parameters.append(p)
        else:
            model_parameters.append(p)

    if not params_type or params_type == "all":
        return model_parameters + quant_input_parameters + quant_weight_parameters
    elif params_type == "weight":
        return model_parameters
    elif params_type == "quant":
        return quant_input_parameters + quant_weight_parameters
    elif params_type == "quant_input":
        return quant_input_parameters
    elif params_type == "quant_weight":
        return quant_weight_parameters
    else:
        raise Exception("Expected params type one of {}, but got {}".format(AVAILABLE_PARAMS_TYPE, params_type))
