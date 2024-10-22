def function_mse(model_output, gt):
    return {'func_loss': ((model_output['model_out'] - gt['func']) ** 2).mean()}