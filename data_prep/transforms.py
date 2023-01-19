import torch

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., snr_db=30):
        self.std = std
        self.mean = mean
        self.snr_db = snr_db
        self.snr = 10 ** (self.snr_db / 20)

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        speech_rms = tensor.norm(p=2)
        noise_rms = noise.norm(p=2)
        scale = self.snr * noise_rms / speech_rms

        return (tensor + noise/scale) / 2

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1}, snr_db={2})'.format(self.mean, self.std, self.snr_db)