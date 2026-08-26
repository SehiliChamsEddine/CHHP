import argparse
import copy
import inspect
import math

import torch
import torch.nn as nn
import transformers
from transformers import GPTNeoXForCausalLM

from datautils import get_loaders
from pruning_utils import DEBUG, SparseGPT_OPT, find_layers
from quant import Quantizer


class SparseGPT_Pythia(SparseGPT_OPT):
    def add_batch(self, inp, out, name, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)

        ###### added code
        if name in ["mlp.dense_h_to_4h", "mlp.dense_4h_to_h", "dense_h_to_4h", "dense_4h_to_h"]:
            self.batch_inp.append(inp[0].clone().detach())
            if len(out.shape) == 3:
                out = out.squeeze(0)
            self.batch_out.append(out.clone().detach())
        ######

        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())


def get_pythia(args):
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = GPTNeoXForCausalLM.from_pretrained(args.model, torch_dtype="auto")
    model.seqlen = model.config.max_position_embeddings
    return model


def _match_mlp_name(names, candidates):
    for candidate in candidates:
        if candidate in names:
            return candidate
    return None


def _build_pythia_position_embeddings(model, hidden_states, position_ids):
    rotary_emb = getattr(model.gpt_neox, "rotary_emb", None)
    if rotary_emb is None or position_ids is None:
        return None

    try:
        return rotary_emb(hidden_states, position_ids)
    except TypeError:
        try:
            return rotary_emb(hidden_states, position_ids=position_ids)
        except TypeError:
            return None


def _pythia_layer_forward(layer, hidden_states, model, layer_kwargs):
    layer_signature = getattr(layer, "_sparsellm_forward_signature", None)
    if layer_signature is None:
        layer_signature = inspect.signature(layer.forward).parameters
        layer._sparsellm_forward_signature = layer_signature
    forward_kwargs = {}

    attention_mask = layer_kwargs.get("attention_mask")
    if "attention_mask" in layer_signature and attention_mask is not None:
        forward_kwargs["attention_mask"] = attention_mask

    position_ids = layer_kwargs.get("position_ids")
    cache_position = layer_kwargs.get("cache_position")

    if position_ids is None:
        if torch.is_tensor(cache_position):
            position_ids = cache_position.unsqueeze(0) if cache_position.dim() == 1 else cache_position
        else:
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
    elif torch.is_tensor(position_ids) and position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)

    if "position_ids" in layer_signature and position_ids is not None:
        forward_kwargs["position_ids"] = position_ids
    if "cache_position" in layer_signature and cache_position is not None:
        forward_kwargs["cache_position"] = cache_position

    if "position_embeddings" in layer_signature:
        position_embeddings = layer_kwargs.get("position_embeddings")
        if position_embeddings is None:
            position_embeddings = _build_pythia_position_embeddings(model, hidden_states, position_ids)
        if position_embeddings is not None:
            forward_kwargs["position_embeddings"] = position_embeddings

    return layer(hidden_states, **forward_kwargs)[0]


@torch.no_grad()
def pythia_sparsellm(model, dataloader, dev, args):
    print("Starting ...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.gpt_neox.layers

    model.gpt_neox.embed_in = model.gpt_neox.embed_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "layer_kwargs": {}}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["layer_kwargs"] = {
                "attention_mask": kwargs.get("attention_mask", None),
                "position_ids": kwargs.get("position_ids", None),
                "cache_position": kwargs.get("cache_position", None),
                "position_embeddings": kwargs.get("position_embeddings", None),
            }
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.gpt_neox.embed_in = model.gpt_neox.embed_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    layer_kwargs = cache["layer_kwargs"]

    print("Ready.")

    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            if (not (args.minlayer <= i < args.maxlayer and args.prune_only in name)) == (not args.invert):
                continue
            gpts[name] = SparseGPT_Pythia(subset[name])
            if args.wbits < 16:
                gpts[name].quantizer = Quantizer()
                gpts[name].quantizer.configure(args.wbits, perchannel=True, sym=False, mse=False)

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data, name)

            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = _pythia_layer_forward(layer, inps[j].unsqueeze(0), model, layer_kwargs)
        for h in handles:
            h.remove()

        target_layer_names = ["mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]

        for name in gpts:
            if name not in target_layer_names:
                print(i, name)
                print("Pruning ...")
                print("Pruning with SparseGPT ...")
                gpts[name].fasterprune(
                    args.sparsity,
                    prunen=args.prunen,
                    prunem=args.prunem,
                    percdamp=args.percdamp,
                    blocksize=args.blocksize,
                )
                gpts[name].free()

        fc1_name = _match_mlp_name(gpts.keys(), ["mlp.dense_h_to_4h", "dense_h_to_4h"])
        fc2_name = _match_mlp_name(gpts.keys(), ["mlp.dense_4h_to_h", "dense_4h_to_h"])

        if fc1_name is not None and fc2_name is not None:
            alpha = 0.1
            beta = 0.1
            gamma = 0.1
            opt_epochs = 4

            X_list = gpts[fc1_name].batch_inp
            Y_list = gpts[fc2_name].batch_out
            X = torch.stack(X_list, dim=0)
            Y = torch.stack(Y_list, dim=0)
            X, Y = X.reshape((-1, X.size(-1))).T, Y.reshape((-1, Y.size(-1))).T

            X_list, Y_list = None, None
            gpts[fc1_name].batch_inp.clear()
            gpts[fc2_name].batch_out.clear()

            hidden_z_list = gpts[fc1_name].batch_out
            z = torch.stack(hidden_z_list, dim=0)
            hidden_z_list = None
            gpts[fc1_name].batch_out.clear()
            hidden_p_list = gpts[fc2_name].batch_inp
            p = torch.stack(hidden_p_list, dim=0)
            hidden_p_list = None
            gpts[fc2_name].batch_inp.clear()

            z = z.reshape((-1, z.size(-1))).T.to(dev)
            p = p.reshape((-1, p.size(-1))).T.to(dev)

            torch.cuda.empty_cache()

            Xinv = torch.pinverse(X.to(dtype=torch.float32)).half()

            for opt_step in range(opt_epochs):
                ##############
                # optimize W
                ##############
                if opt_step > 0:
                    fc1_bias = subset[fc1_name].bias
                    if fc1_bias is None:
                        bias = torch.zeros((z.size(0), z.size(-1)), device=z.device, dtype=z.dtype)
                    else:
                        bias = fc1_bias.unsqueeze(1).expand(-1, z.size(-1))
                    weight_matrix_1 = torch.matmul(z - bias, Xinv)
                    gpts[fc1_name].layer.weight.copy_(weight_matrix_1)
                    del bias, weight_matrix_1

                    weight_matrix_2 = copy.deepcopy(gpts[fc2_name].layer.weight).to(dtype=torch.float32).requires_grad_(True)
                    fc2_bias = subset[fc2_name].bias
                    if fc2_bias is None:
                        bias = torch.zeros((Y.size(0), Y.size(-1)), device=Y.device, dtype=Y.dtype)
                    else:
                        bias = fc2_bias.unsqueeze(1).expand(-1, Y.size(-1))
                    learning_rate = 0.01
                    w_epochs = 10
                    for _ in range(w_epochs):
                        with torch.enable_grad():
                            y_pred = torch.matmul(weight_matrix_2, p.to(dtype=torch.float32)) + bias.to(dtype=torch.float32)
                            loss = (y_pred - Y.to(dtype=torch.float32)).pow(2).mean()
                        loss.backward()
                        weight_matrix_2 -= learning_rate * weight_matrix_2.grad
                        weight_matrix_2.grad.zero_()
                    weight_matrix_2 = weight_matrix_2.half()
                    gpts[fc2_name].layer.weight.copy_(weight_matrix_2)

                    del bias, weight_matrix_2, y_pred
                    torch.cuda.empty_cache()

                ##############
                # prune W
                ##############
                if opt_step > 0:
                    tmp_H = torch.zeros_like(gpts[fc2_name].H)
                    tmp_p = p.T.reshape((args.nsamples, -1, p.size(0)))
                    tmp_nsamples = 0
                    for j in range(args.nsamples):
                        tmp_inp = tmp_p[j].unsqueeze(0)
                        tmp = tmp_inp.shape[0]
                        if isinstance(gpts[fc2_name].layer, nn.Linear) or isinstance(gpts[fc2_name].layer, transformers.Conv1D):
                            if len(tmp_inp.shape) == 3:
                                tmp_inp = tmp_inp.reshape((-1, tmp_inp.shape[-1]))
                            tmp_inp = tmp_inp.t()
                        tmp_H *= tmp_nsamples / (tmp_nsamples + tmp)
                        tmp_nsamples += tmp
                        tmp_inp = math.sqrt(2 / tmp_nsamples) * tmp_inp.float()
                        tmp_H += tmp_inp.matmul(tmp_inp.t())
                    gpts[fc2_name].H.copy_(tmp_H)
                    del tmp_H, tmp_p
                    torch.cuda.empty_cache()

                for name in [fc1_name, fc2_name]:
                    print(i, name)
                    if args.use_vacuum:
                        print("Pruning with VACUUM ...")
                        gpts[name].fasterprune_vacuum(
                            args.sparsity,
                            prunen=args.prunen,
                            prunem=args.prunem,
                            blocksize=args.blocksize,
                            percdamp=args.percdamp,
                            n_vac=args.n_vac,
                            lmbda=args.lmbda_vac,
                            cooking_iters=args.cooking_iters,
                            lr_vac=args.lr_vac,
                        )
                    else:
                        print("Pruning with SparseGPT ...")
                        gpts[name].fasterprune(
                            args.sparsity,
                            prunen=args.prunen,
                            prunem=args.prunem,
                            percdamp=args.percdamp,
                            blocksize=args.blocksize,
                        )

                ##############
                # optimize p
                ##############
                next_weight = subset[fc2_name].weight
                m1 = beta * torch.matmul(next_weight.T, next_weight)
                m2 = gamma * torch.eye(m1.shape[0], device=m1.device)
                av = torch.inverse(m1 + m2).to(dtype=torch.float16)

                del m1, m2
                torch.cuda.empty_cache()

                layer_nl_output = nn.functional.gelu(z)

                fc2_bias = subset[fc2_name].bias
                if fc2_bias is None:
                    bias = torch.zeros((Y.size(0), Y.size(-1)), device=Y.device, dtype=Y.dtype)
                else:
                    bias = fc2_bias.unsqueeze(1).expand(-1, Y.size(-1))
                m3 = beta * torch.matmul(next_weight.T, Y - bias)

                del next_weight, bias
                torch.cuda.empty_cache()

                m4 = gamma * layer_nl_output
                af = m3 + m4
                del m3, m4, layer_nl_output

                p = torch.matmul(av, af)

                del av, af
                torch.cuda.empty_cache()

                ##############
                # optimize z
                ##############
                w = subset[fc1_name].weight
                fc1_bias = subset[fc1_name].bias
                if fc1_bias is None:
                    bias = torch.zeros((z.size(0), z.size(-1)), device=z.device, dtype=z.dtype)
                else:
                    bias = fc1_bias.unsqueeze(1).expand(-1, z.size(-1))
                m = torch.matmul(w, X) + bias
                sol1 = (gamma * p + alpha * m) / (gamma + alpha)
                sol2 = m
                del w, bias
                torch.cuda.empty_cache()

                z1 = torch.zeros_like(p)
                z2 = torch.zeros_like(p)

                chunk_size = 500
                for k in range(0, sol1.size(0), chunk_size):
                    chunk = slice(k, k + chunk_size)

                    z1_chunk = z1[chunk]
                    sol1_chunk = sol1[chunk]
                    z1_chunk[sol1_chunk >= 0.0] = sol1_chunk[sol1_chunk >= 0.0]
                    z1[chunk] = z1_chunk

                    z2_chunk = z2[chunk]
                    sol2_chunk = sol2[chunk]
                    z2_chunk[sol2_chunk <= 0.0] = sol2_chunk[sol2_chunk <= 0.0]
                    z2[chunk] = z2_chunk

                del z1_chunk, z2_chunk, sol1_chunk, sol2_chunk, sol1, sol2
                torch.cuda.empty_cache()

                for k in range(0, z1.size(0), chunk_size):
                    chunk = slice(k, k + chunk_size)

                    fz_1_chunk = gamma * torch.square(p[chunk] - nn.functional.gelu(z1[chunk])) + alpha * torch.square(z1[chunk] - m[chunk])
                    fz_2_chunk = gamma * torch.square(p[chunk] - nn.functional.gelu(z2[chunk])) + alpha * torch.square(z2[chunk] - m[chunk])

                    index_z1_chunk = fz_1_chunk <= fz_2_chunk
                    index_z2_chunk = fz_2_chunk < fz_1_chunk

                    z[chunk][index_z1_chunk] = z1[chunk][index_z1_chunk]
                    z[chunk][index_z2_chunk] = z2[chunk][index_z2_chunk]

                del fz_1_chunk, fz_2_chunk, index_z1_chunk, index_z2_chunk, z1, z2, m, chunk
                torch.cuda.empty_cache()

            gpts[fc1_name].free()
            gpts[fc2_name].free()

        for j in range(args.nsamples):
            outs[j] = _pythia_layer_forward(layer, inps[j].unsqueeze(0), model, layer_kwargs)

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache


@torch.no_grad()
def pythia_eval(model, testenc, dev, args, dataset: str):
    print("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.gpt_neox.layers

    model.gpt_neox.embed_in = model.gpt_neox.embed_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "layer_kwargs": {}}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["layer_kwargs"] = {
                "attention_mask": kwargs.get("attention_mask", None),
                "position_ids": kwargs.get("position_ids", None),
                "cache_position": kwargs.get("cache_position", None),
                "position_embeddings": kwargs.get("position_embeddings", None),
            }
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.gpt_neox.embed_in = model.gpt_neox.embed_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    layer_kwargs = cache["layer_kwargs"]

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
            outs[j] = _pythia_layer_forward(layer, inps[j].unsqueeze(0), model, layer_kwargs)
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.gpt_neox.final_layer_norm is not None:
        model.gpt_neox.final_layer_norm = model.gpt_neox.final_layer_norm.to(dev)
    model.embed_out = model.embed_out.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.gpt_neox.final_layer_norm is not None:
            hidden_states = model.gpt_neox.final_layer_norm(hidden_states)
        lm_logits = model.embed_out(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")

    model.config.use_cache = use_cache


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="EleutherAI/pythia-410m", help="Pythia model to load")
    parser.add_argument("--dataset", type=str, choices=["wikitext2", "ptb", "c4"], default="c4", help="Dataset for calibration.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for sampling calibration data.")
    parser.add_argument("--nsamples", type=int, default=64, help="Number of calibration data samples.")
    parser.add_argument("--percdamp", type=float, default=0.01, help="Percent of Hessian diagonal for dampening.")
    parser.add_argument("--sparsity", type=float, default=0.5, help="Target sparsity.")
    parser.add_argument("--prunen", type=int, default=0, help="N for N:M pruning.")
    parser.add_argument("--prunem", type=int, default=0, help="M for N:M pruning.")
    parser.add_argument("--blocksize", type=int, default=128, help="Blocksize for adaptive mask selection.")
    parser.add_argument("--gmp", action="store_true", help="Run GMP baseline.")
    parser.add_argument("--wbits", type=int, default=16, help="Quantization bits.")
    parser.add_argument("--minlayer", type=int, default=-1, help="Prune layers with id >= this.")
    parser.add_argument("--maxlayer", type=int, default=1000, help="Prune layers with id < this.")
    parser.add_argument("--prune_only", type=str, default="", help="Prune only layers containing this text.")
    parser.add_argument("--invert", action="store_true", help="Invert subset.")
    parser.add_argument("--save", type=str, default="", help="Path to save model.")
    parser.add_argument("--true-sequential", action="store_true", help="Run in true sequential mode.")
    parser.add_argument("--log_wandb", action="store_true", help="Log to W&B.")
    parser.add_argument(
        "--use_vacuum",
        action="store_true",
        help="Whether to use the vacuum pruning method for FFN layers.",
    )
    parser.add_argument("--n_vac", type=int, default=3, help="Power of the vacuum function w^(2n+1).")
    parser.add_argument("--n_vac_att", type=int, default=1, help="Power of the vacuum function w^(2n+1).")
    parser.add_argument("--lmbda_vac", type=float, default=0.01, help="Lambda regularization for the vacuum cooking phase.")
    parser.add_argument("--cooking_iters", type=int, default=20, help="Number of optimization steps in the vacuum cooking phase.")
    parser.add_argument("--lr_vac", type=float, default=1e-3, help="Learning rate for the vacuum optimizer.")
    args = parser.parse_args()

    model = get_pythia(args)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
    )

    if (args.sparsity or args.prunen) and not args.gmp:
        pythia_sparsellm(model, dataloader, torch.device("cuda"), args)

    for dataset in ["wikitext2", "c4"]:
        dataloader, testloader = get_loaders(dataset, seed=args.seed, model=args.model, seqlen=model.seqlen)
        pythia_eval(model, testloader, torch.device("cuda"), args, dataset)

    if args.save:
        model.save_pretrained(args.save)


if __name__ == "__main__":
    main()
