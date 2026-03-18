import os
import triton
import triton.common.build
from collections import namedtuple
import torch
import esm
from typing import List, Union, Optional
from protein_lm.modeling.scripts.train import compute_esm_embedding, load_ckpt, make_esm_input_ids
from protein_lm.tokenizer.tokenizer import PTMTokenizer
from torch.nn.utils.rnn import pad_sequence

_original_libcuda_dirs = triton.common.build.libcuda_dirs

def patched_libcuda_dirs():
    try:
        dirs = _original_libcuda_dirs()
    except AssertionError:
        dirs = []
    user_dir = os.path.expanduser('~/my_libs')
    if user_dir not in dirs:
        dirs.append(user_dir)
    return dirs

triton.common.build.libcuda_dirs = patched_libcuda_dirs
print("Triton patch applied: libcuda_dirs patched to include ~/my_libs.")

Output = namedtuple("output", ["logits", "hidden_states"])

class PTMMamba:
    def __init__(self, ckpt_path, device='cuda',use_esm=True) -> None:
        self.use_esm = use_esm
        self._tokenizer = PTMTokenizer()
        self._model = load_ckpt(ckpt_path, self.tokenizer, device)
        self._device = device
        self._model.to(device)
        self._model.eval()
        self.esm_model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.esm_model.eval()

    @property
    def model(self) -> torch.nn.Module:
        return self._model


    @property
    def tokenizer(self) -> PTMTokenizer:
        return self._tokenizer
    
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    
    
    def infer(self, seq: str) -> Output:
        input_id = self.tokenizer(seq)
        input_ids = torch.tensor(input_id,device=self.device).unsqueeze(0)
        outputs = self._infer(input_ids)
        return outputs
    
    @torch.no_grad()
    def _infer(self, input_ids):
        if self.use_esm:
            esm_input_ids = make_esm_input_ids(input_ids, self.tokenizer)
            embedding = compute_esm_embedding(
                self.tokenizer, self.esm_model, self.batch_converter, esm_input_ids
            )
        else:
            embedding = None
        outputs = self.model(input_ids, embedding=embedding)
        return outputs
    
    
    def infer_batch(self, seqs: list) -> Output:
        input_ids = self.tokenizer(seqs)
        input_ids = pad_sequence(
            [torch.tensor(x) for x in input_ids],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        input_ids = torch.tensor(input_ids,device=self.device)
        outputs = self._infer(input_ids)
        return outputs
    
    def __call__(self, seq: Union[str, List]) -> Output:
        if isinstance(seq, str):
            return self.infer(seq)
        elif isinstance(seq, list):
            return self.infer_batch(seq)
        else:
            raise ValueError("Input must be a string or a list of strings, got {}".format(type(seq)))
        
        
if __name__ == "__main__":
    ckpt_path = "ckpt/best.ckpt"
    mamba = PTMMamba(ckpt_path,device='cuda:0')
    # seq = '<N-acetylmethionine>EAD<Phosphoserine>PAGPGAPEPLAEGAAAEFS<Phosphoserine>LLRRIKGKLFTWNILKTIALGQMLSLCICGTAITSQYLAERYKVNTPMLQSFINYCLLFLIYTVMLAFRSGSDNLLVILKRKWWKYILLGLADVEANYVIVRAYQYTTLTSVQLLDCFGIPVLMALSWFILHARYRVIHFIAVAVCLLGVGTMVGADILAGREDNSGSDVLIGDILVLLGASLYAISNVCEEYIVKKLSRQEFLGMVGLFGTIISGIQLLIVEYKDIASIHWDWKIALLFVAFALCMFCLYSFMPLVIKVTSATSVNLGILTADLYSLFVGLFLFGYKFSGLYILSFTVIMVGFILYCSTPTRTAEPAESSVPPVTSIGIDNLGLKLEENLQETH<Phosphoserine>AVL'
    # seq = "M<N-acetylalanine>K"
    # output = mamba(seq)
    # print(output.logits.shape)
    # print(output.hidden_states.shape)

    seq= "PEHPSGQSHGPPTPPTTPKTELQSGKADPKRDGRSMGEGGKPHIDFGNVDI"
    output = mamba(seq)
    print(output.logits.shape)
    print(output.hidden_states.shape)

    seq= "GQSHGPPTPPTTPKTELQSGKADPKRDGRSMGEGGKPHID"
    output = mamba(seq)
    print(output.logits.shape)
    print(output.hidden_states.shape)
    
    
    
