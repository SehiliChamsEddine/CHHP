# Pythia_main.py
# Helpers to load and run SparseLLM-style pruning/eval on Pythia / GPT-NeoX style models.
# This file intentionally tries to avoid changing existing OPT/LLaMA flows: it provides
# a loader, an autodiscovery helper, a small adapter SparseGPT_Pythia that delegates
# to the existing SparseGPT_LlaMA implementation in pruning_utils.py, and wrapper
# functions pythia_sparsellm and pythia_eval that mirror the llama_sparsellm / opt_sparsellm
# flow but adapt to common Pythia/GPT-NeoX naming patterns.

import torch
import torch.nn as nn
import math
import copy
import transformers
from pruning_utils import find_layers, SparseGPT_LlaMA

try:
    from transformers import GPTNeoXForCausalLM, AutoModelForCausalLM
except Exception:
    GPTNeoXForCausalLM = None
    from transformers import AutoModelForCausalLM


def get_pythia(args):
    """
    Load a Pythia / GPT-NeoX style model. Tries GPTNeoXForCausalLM first then falls back to AutoModelForCausalLM.
    Sets model.seqlen from config.max_position_embeddings when available.
    """
    def skip(*a, **k):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    try:
        if GPTNeoXForCausalLM is not None:
            model = GPTNeoXForCausalLM.from_pretrained(args.model, torch_dtype="auto")
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")
    except Exception as e:
        raise RuntimeError(f"Failed to load Pythia/GPT-NeoX model '{args.model}': {e}")
    model.seqlen = getattr(model.config, "max_position_embeddings", 2048)
    return model


def pythia_model_info(model):
    """
    Try to detect a Pythia / GPT-NeoX style layout and return a dict with useful items:
      - layers: list/Sequence of layer modules
      - embed_module: embeddings module (or None)
      - final_norm_module: final normalization layer (or None)
      - target_layer_names: list of MLP projection names commonly used for the up/gate/down
      - activation: string hint with activation function name

    The function recognizes common shapes:
      - model.model.gpt_neox.layers (EleutherAI GPT-NeoX style)
      - model.gpt_neox.layers
      - model.model.layers (some wrappers)
    """
    info = {
        "layers": None,
        "embed_module": None,
        "final_norm_module": None,
        "target_layer_names": ["mlp.dense_h_to_4h", "mlp.dense_4h_to_h"],
        "activation": "gelu",
    }

    # pattern 1: model.model.gpt_neox.layers
    if hasattr(model, "model") and hasattr(model.model, "gpt_neox"):
        g = model.model.gpt_neox
        if hasattr(g, "layers"):
            info["layers"] = g.layers
            info["embed_module"] = getattr(g, "embed_in", None) or getattr(g, "word_embeddings", None)
            info["final_norm_module"] = getattr(g, "final_layer_norm", None)
            return info

    # pattern 2: model.gpt_neox.layers
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        g = model.gpt_neox
        info["layers"] = g.layers
        info["embed_module"] = getattr(g, "embed_in", None) or getattr(g, "word_embeddings", None)
        info["final_norm_module"] = getattr(g, "final_layer_norm", None)
        return info

    # pattern 3: model.model.layers (generic)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        info["layers"] = model.model.layers
        info["embed_module"] = getattr(model.model, "embed_in", None) or getattr(model.model, "embed_tokens", None)
        info["final_norm_module"] = getattr(model.model, "final_layer_norm", None) or getattr(model.model, "norm", None)
        return info

    raise ValueError(
        "Could not autodetect Pythia/GPT-NeoX block layout. Inspect `print(model)` to find where layers live."
    )


class SparseGPT_Pythia:
    """
    Adapter class for Pythia layers. It wraps SparseGPT_LlaMA and maps common Pythia MLP names
    to the LLaMA-style names the SparseGPT_LlaMA add_batch expects. This avoids duplicating
    the expensive pruning implementations.

    Behaviour:
      - Delegates fasterprune(), fasterprune_vacuum(), free() to inner SparseGPT_LlaMA
      - Exposes H, batch_inp, batch_out properties that reference the inner one
    """

    def __init__(self, layer):
        # use the LlaMA implementation which already supports mlp.up_proj/mlp.down_proj naming
        self.inner = SparseGPT_LlaMA(layer)
        self.layer = self.inner.layer
        # expose frequently used attributes for compatibility
        self.H = self.inner.H
        self.batch_inp = self.inner.batch_inp
        self.batch_out = self.inner.batch_out
        self.dev = self.inner.dev
        # quantizer passthrough if used by calling code
        if hasattr(self.inner, 'quantizer'):
            self.quantizer = self.inner.quantizer

    def _map_name(self, name):
        # Map common Pythia names to the expected LLaMA names used inside SparseGPT_LlaMA
        # Most Pythia MLPs use: 'mlp.dense_h_to_4h' and 'mlp.dense_4h_to_h'. We map them to
        # mlp.up_proj and mlp.down_proj respectively. If there's a gate projection, try to map it
        if name.endswith("dense_h_to_4h"):
            return 'mlp.up_proj'
        if name.endswith("dense_4h_to_h"):
            return 'mlp.down_proj'
        if "gate" in name or name.endswith("dense_gate"):
            return 'mlp.gate_proj'
        # fallback: return original name
        return name

    def add_batch(self, inp, out, name, blocksize=1024):
        mapped = self._map_name(name)
        # delegate to inner class; inner expects the layer object 'self.inner' and will store
        # batch inputs/outputs when called with names like 'mlp.up_proj' etc.
        return self.inner.add_batch(inp, out, mapped, blocksize=blocksize)

    def fasterprune(self, *args, **kwargs):
        return self.inner.fasterprune(*args, **kwargs)

    def fasterprune_vacuum(self, *args, **kwargs):
        return self.inner.fasterprune_vacuum(*args, **kwargs)

    def free(self):
        return self.inner.free()


@torch.no_grad()
def pythia_sparsellm(model, dataloader, dev, args):
    """
    Pythia pruning flow that mirrors llama_sparsellm but uses the Pythia/GPT-NeoX layout.
    It collects activations with forward hooks, wraps layers with SparseGPT_Pythia and calls
    the same pruning routines (fasterprune / fasterprune_vacuum). The more advanced
    z/p optimization loops from the LLaMA flow are not reimplemented here in full; this
    wrapper focuses on integrating Pythia models with the existing SparseGPT code paths.

    NOTE: The function intentionally follows the structure used in model_utils.py and
    should not modify any OPT/LLaMA code. If you want full parity (including the
    multi-step ADMM-like z/p optimisation), we can copy the exact LLaMA code into
    this function — but that duplicates a lot of code. This version keeps pruning
    identical by delegating to SparseGPT_LlaMA via the adapter above.
    """
    print("Starting Pythia SparseLLM flow...")

    use_cache = model.config.use_cache
    model.config.use_cache = False

    info = pythia_model_info(model)
    layers = info['layers']

    # move embedding modules to device if present
    if info['embed_module'] is not None:
        try:
            info['embed_module'] = info['embed_module'].to(dev)
        except Exception:
            pass

    # bring first layer to device to let catcher collect activations
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs.get('attention_mask', None)
            raise ValueError

    layers[0] = Catcher(layers[0])

    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass

    # restore
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if info['embed_module'] is not None:
        try:
            info['embed_module'] = info['embed_module'].cpu()
        except Exception:
            pass
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    print("Ready.")

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)

        # by default, process all keys sequentially
        sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}

            gpts = {}
            for name in subset:
                # simple layer filter logic based on args (reuse same flags used elsewhere)
                if (not (args.minlayer <= i < args.maxlayer and args.prune_only in name)) == (not args.invert):
                    continue
                # create adapter wrapper
                gpts[name] = SparseGPT_Pythia(subset[name])
                if args.wbits < 16:
                    from quant import Quantizer
                    gpts[name].quantizer = Quantizer()
                    gpts[name].quantizer.configure(args.wbits, perchannel=True, sym=False, mse=False)

            def add_batch(name):
                def tmp(_, inp, out):
                    # replicate same call pattern as other flows
                    gpts[name].add_batch(inp[0].data, out.data, name)
                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

            for h in handles:
                h.remove()

            # detect target layer names in this subset
            detected_targets = [n for n in subset.keys() if any(x in n for x in ["dense_h_to_4h", "dense_4h_to_h", "gate"]) ]
            target_layer_names = detected_targets if len(detected_targets) > 0 else info['target_layer_names']

            # prune non-targets quickly
            for name in list(gpts.keys()):
                if name not in target_layer_names:
                    print(i, name)
                    print('Pruning ...')
                    sparsity = args.sparsity
                    gpts[name].fasterprune(sparsity, prunen=args.prunen, prunem=args.prunem, percdamp=args.percdamp, blocksize=args.blocksize)
                    gpts[name].free()

            # For target layers we keep them for potential advanced optimization; here we also call fasterprune
            for name in target_layer_names:
                if name in gpts:
                    print(i, name)
                    if args.use_vacuum:
                        print('Pruning with VACUUM ...')
                        gpts[name].fasterprune_vacuum(args.sparsity, prunen=args.prunen, prunem=args.prunem, blocksize=args.blocksize, percdamp=args.percdamp, n_vac=getattr(args, 'n_vac', 3), lmbda=getattr(args, 'lmbda_vac', 0), cooking_iters=getattr(args, 'cooking_iters', 0), lr_vac=getattr(args,'lr_vac',0))
                    else:
                        print('Pruning with SparseGPT ...')
                        gpts[name].fasterprune(args.sparsity, prunen=args.prunen, prunem=args.prunem, percdamp=args.percdamp, blocksize=args.blocksize)
                    gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    print("Pythia SparseLLM flow finished.")


@torch.no_grad()
def pythia_eval(model, testenc, dev, args, dataset: str):
    """
    Evaluation wrapper for Pythia models. Mirrors the opt_eval / llama_eval evaluation flows
    but uses autodiscovered embeddings and layer containers.
    """
    print('Evaluating Pythia model ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False

    info = pythia_model_info(model)
    layers = info['layers']

    if info['embed_module'] is not None:
        try:
            info['embed_module'] = info['embed_module'].to(dev)
        except Exception:
            pass

    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs.get('attention_mask', None)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    if info['embed_module'] is not None:
        try:
            info['embed_module'] = info['embed_module'].cpu()
        except Exception:
            pass
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.gmp:
            subset = find_layers(layer)
            for name in subset:
                W = subset[name].weight.data
                thresh = torch.sort(torch.abs(W.flatten()))[0][int(W.numel() * args.sparsity)]
                W.data[torch.abs(W.data) <= thresh] = 0

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    # final norm if any
    if info['final_norm_module'] is not None:
        try:
            info['final_norm_module'] = info['final_norm_module'].to(dev)
        except Exception:
            pass
    # model.lm_head may need to be moved
    try:
        model.lm_head = model.lm_head.to(dev)
    except Exception:
        pass

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if info['final_norm_module'] is not None:
            hidden_states = info['final_norm_module'](hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")

    model.config.use_cache = use_cache
