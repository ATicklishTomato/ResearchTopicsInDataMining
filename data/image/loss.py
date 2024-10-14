def mean_squared_error(model_output, gt):
    return {'img_loss': ((model_output['model_out'] - gt['img']) ** 2).mean()}