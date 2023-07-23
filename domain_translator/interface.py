import os
import torch
from domain_translator.models import models_wgan, models_wgan_large


class DomainTranslator:
    def __init__(self, translator_dir, translator_name, normalizer_tar):
        """Loading pretrained domain translator for the second step: training classifier
        """
        model_path = os.path.join(translator_dir, translator_name, 'ckpt-translator-best.pth')
        state_dict = torch.load(model_path)

        # load translator meta info
        arch = state_dict['arch']
        image_size = state_dict['image_size']
        num_res_blocks = state_dict['num_res_blocks']

        if arch in ('sngan', 'wgan', 'wgangp'):
            self.translator = models_wgan.create_generator(image_size,
                                                           num_res_blocks)
        elif arch in ('wgan_large', 'wgangp_large'):
            self.translator = models_wgan_large.create_generator(num_res_blocks)
        else:
            raise Exception(f'Unknown translator architecture {arch}')

        self.translator.load_state_dict(state_dict['netG'])
        self.translator.eval()
        self.normalizer_tar = normalizer_tar

    def translate(self, source_input):
        # source_input is a normalized tensor
        translated_output = self.translator(source_input)
        return self.normalizer_tar(translated_output)
