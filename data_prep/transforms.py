import torch

class AddGaussianNoise(object):
    def __init__(self, snr_db=30):
        self.snr_db = snr_db

    def __call__(self, tensor):
        speech_power_db = 10 * np.log10(torch.mean(tensor**2))
        noise_avg_db = speech_power_db - self.snr_db
        noise_avg = 10 ** (noise_avg_db / 10)
        noise = torch.randn(tensor.size()) * torch.sqrt(noise_avg)
        return tensor + noise


    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1}, snr_db={2})'.format(self.mean, self.std, self.snr_db)