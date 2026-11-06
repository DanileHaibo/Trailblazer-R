"""Explicit external official backend integration; no machine-local fallback paths."""
import json
import sys
from pathlib import Path

def configure(official_repo, libero_repo=None):
    root = Path(official_repo).expanduser().resolve()
    code = root / 'rynnvla-002'
    if not (code / 'model').is_dir() or not (root / 'xllmx').is_dir():
        raise ValueError('Expected pinned RynnVLA-002 checkout with rynnvla-002/ and xllmx/')
    paths = [str(code), str(root)]
    if libero_repo:
        paths.append(str(Path(libero_repo).expanduser().resolve()))
    sys.path[:0] = paths

def make_processor(tokenizer_path, image_tokenizer_path, device='cuda'):
    import numpy as np
    from data.pre_tokenize_action_state import ItemProcessor
    from data.item_processor import generate_crop_size_list, FlexARItemProcessor_Action_State
    from data.convertsation import Conversation
    from xllmx.data.item_processor import MMConvItemProcessor
    import model.chameleon_vae_ori as vae

    class FastProcessor(ItemProcessor):
        def __init__(self):
            MMConvItemProcessor.__init__(self,
                {'<|image|>': self.process_image, '<|action|>': self.process_action, '<|state|>': self.process_state},
                ['<|image|>', '<|action|>', '<|state|>'], str(tokenizer_path), Conversation)
            self.patch_size = 32
            self.crop_size_list = generate_crop_size_list(64, 32)
            image_root = Path(image_tokenizer_path)
            self.chameleon_ori_vocab = vae.VocabInfo(json.loads((image_root/'text_tokenizer.json').read_text())['model']['vocab'])
            self.chameleon_ori_translation = vae.VocabTranslation(self.chameleon_ori_vocab, device=device)
            self.chameleon_ori_image_tokenizer = vae.ImageTokenizer(str(image_root/'vqgan.yaml'), str(image_root/'vqgan.ckpt'), device=device)
            self.n_bins, self.min_action, self.max_action = 256, -1, 1
            self.bins = np.linspace(-1, 1, 256)
            self.bin_centers = (self.bins[:-1]+self.bins[1:])/2
            self.device = device
        def process_item(self, *a, **kw):
            return ItemProcessor.process_item(self, *a, **kw)
        def process_image(self, *a, **kw):
            return FlexARItemProcessor_Action_State.process_image(self, *a, **kw)
    return FastProcessor()
