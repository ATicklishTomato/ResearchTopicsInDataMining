import skimage.measure

def mean_squared_error(model_output, gt):
    return {'img_loss': ((model_output['model_out'] - gt['img']) ** 2).mean()}

structural_similarity = skimage.metrics.structural_similarity
peak_signal_noise_ratio = skimage.metrics.peak_signal_noise_ratio