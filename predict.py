# Prediction utilities for CLIP prefix captioning (sin Cog, sin Predictor)

import os
import sys
from typing import Tuple, List, Union, Optional

import clip
import numpy as np
import torch
from torch import nn
import torch.nn.functional as nnf
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AdamW,
    get_linear_schedule_with_warmup,
)
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Tipos auxiliares (del código original)
N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]

D = torch.device
CPU = torch.device("cuda")


class MLP(nn.Module):
    """Pequeña MLP usada para proyectar el embedding de CLIP al espacio de GPT-2."""

    def __init__(self, sizes: Tuple[int, ...], bias: bool = True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x: T) -> T:
        return self.model(x)


class ClipCaptionModel(nn.Module):
    """
    Modelo base: GPT-2 condicionado por un prefijo (proyección del embedding de CLIP).
    """

    def __init__(self, prefix_length: int, prefix_size: int = 512):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]

        if prefix_length > 10:  # por memoria
            self.clip_project = nn.Linear(
                prefix_size, self.gpt_embedding_size * prefix_length
            )
        else:
            self.clip_project = MLP(
                (
                    prefix_size,
                    (self.gpt_embedding_size * prefix_length) // 2,
                    self.gpt_embedding_size * prefix_length,
                )
            )

    def get_dummy_token(self, batch_size: int, device: D) -> T:
        return torch.zeros(
            batch_size, self.prefix_length, dtype=torch.int64, device=device
        )

    def forward(
        self,
        tokens: T,
        prefix: T,
        mask: Optional[T] = None,
        labels: Optional[T] = None,
    ):
        """
        tokens: ids de GPT-2 [B, L]
        prefix: embeddings de CLIP [B, D] que serán proyectados y vistos como tokens iniciales
        """
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(
            -1, self.prefix_length, self.gpt_embedding_size
        )
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)

        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)

        out = self.gpt(
            inputs_embeds=embedding_cat,
            labels=labels,
            attention_mask=mask,
        )
        return out


class ClipCaptionPrefix(ClipCaptionModel):
    """
    Variante donde solo se entrena la proyección (prefix), dejando GPT-2 congelado.
    """

    def parameters(self, recurse: bool = True):
        # Solo entrenar la capa de proyección desde CLIP
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        # GPT-2 siempre en eval (congelado)
        self.gpt.eval()
        return self


def generate_beam(
    model: ClipCaptionModel,
    tokenizer: GPT2Tokenizer,
    beam_size: int = 5,
    prompt: Optional[str] = None,
    embed: Optional[T] = None,
    entry_length: int = 67,
    temperature: float = 1.0,
    stop_token: str = ".",
    stop_token_count: int = 1,
):
    """
    Generación con beam search. Devuelve lista de captions ordenados por score.

    `stop_token_count` es cuántas veces tiene que aparecer `stop_token` antes de
    cortar el beam. Con el valor por defecto (1) el comportamiento es el de
    siempre: se corta en el primer punto. Hace falta subirlo a 2 para las captions
    de dos frases del fine-tuning con BBPS ("...descending colon. Bowel preparation
    score BBPS is 7/9."), porque si no la segunda frase -la del BBPS- nunca se
    llega a generar. Ver reportes/finetuning_bbps_igho.md.
    """
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    stop_token_count = max(1, int(stop_token_count))
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    stop_counts = torch.zeros(beam_size, device=device)

    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)

        for _ in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()

            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                    beam_size, -1
                )
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
                stop_counts = stop_counts[next_tokens_source]

            next_token_embed = model.gpt.transformer.wte(
                next_tokens.squeeze()
            ).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            stop_counts = stop_counts + next_tokens.eq(stop_token_index).squeeze().float()
            is_stopped = stop_counts >= stop_token_count
            if is_stopped.all():
                break

    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length)])
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


def _reorder_kv_cache(past_key_values, index: T):
    """Reordena el KV-cache para que cada fila siga al beam del que desciende.

    `past_key_values` es una tupla de 12 capas, cada una con (key, value) de forma
    [filas, cabezas, longitud, dim]. En cada paso del beam search las ramas se
    reordenan (un beam puede descender de otro), y el cache tiene que seguir
    exactamente ese mismo reordenamiento: si no, cada fila continuaria generando
    sobre el historial de atencion equivocado, en silencio y sin error.
    """
    return tuple(
        tuple(tensor.index_select(0, index) for tensor in layer)
        for layer in past_key_values
    )


@torch.no_grad()
def generate_beam_batched(
    model: ClipCaptionModel,
    tokenizer: GPT2Tokenizer,
    embed: T,
    beam_size: int = 5,
    entry_length: int = 67,
    temperature: float = 1.0,
    stop_token: str = ".",
    stop_token_count: int = 1,
) -> List[str]:
    """Beam search sobre un lote de prefijos, reutilizando el KV-cache de GPT-2.

    Equivale a llamar `generate_beam()` una vez por imagen y quedarse con el mejor
    caption (`[0]`), pero procesa las B imagenes a la vez y evita re-procesar la
    secuencia completa en cada paso. Con captions de ~17 tokens eso baja de ~333
    posiciones-token por imagen a ~27, y sobre todo amortiza entre B imagenes el
    costo fijo por paso (leer los pesos de GPT-2 + lanzar los kernels de las 12
    capas), que es lo que realmente domina el tiempo con batch de una sola imagen.

    embed: [B, prefix_length, D], prefijos ya proyectados por `clip_project`.
    Devuelve B captions, el mejor beam de cada imagen, en el orden de entrada.
    """
    model.eval()
    device = embed.device
    batch_size = embed.shape[0]
    stop_token_index = tokenizer.encode(stop_token)[0]
    stop_token_count = max(1, int(stop_token_count))
    rows = batch_size * beam_size
    temperature = temperature if temperature > 0 else 1.0

    # Paso 0: el prefijo se procesa una sola vez por imagen (aun sin beams).
    outputs = model.gpt(inputs_embeds=embed, use_cache=True, return_dict=True)
    logits = (outputs.logits[:, -1, :] / temperature).softmax(-1).log()
    vocab_size = logits.shape[-1]

    scores, next_tokens = logits.topk(beam_size, -1)  # [B, beam]
    tokens = next_tokens.unsqueeze(-1)                # [B, beam, 1]

    # El cache del prefijo es comun a las beam_size ramas de cada imagen: se
    # replica para que las filas queden ordenadas como imagen*beam_size + beam.
    past = _reorder_kv_cache(
        outputs.past_key_values,
        torch.arange(batch_size, device=device).repeat_interleave(beam_size),
    )

    seq_lengths = torch.ones(batch_size, beam_size, device=device)
    stop_counts = next_tokens.eq(stop_token_index).float()
    is_stopped = stop_counts >= stop_token_count
    batch_index = (
        torch.arange(batch_size, device=device).unsqueeze(1).expand(batch_size, beam_size)
    )

    for _ in range(entry_length - 1):
        if bool(is_stopped.all()):
            break

        token_embed = model.gpt.transformer.wte(next_tokens.reshape(rows)).unsqueeze(1)
        outputs = model.gpt(
            inputs_embeds=token_embed,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        logits = (outputs.logits[:, -1, :] / temperature).softmax(-1).log()

        # Un beam ya cerrado no debe seguir acumulando score: se lo obliga a
        # repetir el token 0 con delta 0 (mismo criterio que generate_beam).
        # El enmascarado se hace con los logits todavia en 2-D [filas, vocab]:
        # combinar una mascara 2-D con un indice entero sobre un tensor 3-D es
        # un IndexError en torch, no un broadcast.
        flat_stopped = is_stopped.reshape(rows)
        logits[flat_stopped] = -float(np.inf)
        logits[flat_stopped, 0] = 0

        logits = logits.view(batch_size, beam_size, vocab_size)
        scores_sum = scores.unsqueeze(-1) + logits
        seq_lengths = seq_lengths + (~is_stopped).float()
        scores_sum_average = scores_sum / seq_lengths.unsqueeze(-1)

        scores_sum_average, flat_next = scores_sum_average.view(batch_size, -1).topk(
            beam_size, -1
        )
        beam_source = torch.div(flat_next, vocab_size, rounding_mode="floor")  # [B, beam]
        next_tokens = flat_next % vocab_size                                   # [B, beam]

        seq_lengths = seq_lengths[batch_index, beam_source]
        tokens = tokens[batch_index, beam_source]
        tokens = torch.cat((tokens, next_tokens.unsqueeze(-1)), dim=-1)
        scores = scores_sum_average * seq_lengths
        is_stopped = is_stopped[batch_index, beam_source]
        stop_counts = stop_counts[batch_index, beam_source]
        past = _reorder_kv_cache(
            past, (batch_index * beam_size + beam_source).reshape(rows)
        )

        stop_counts = stop_counts + next_tokens.eq(stop_token_index).float()
        is_stopped = stop_counts >= stop_token_count

    scores = scores / seq_lengths
    best_beam = scores.argmax(dim=-1)  # [B]

    tokens_cpu = tokens.cpu().numpy()
    lengths_cpu = seq_lengths.cpu().numpy()
    best_cpu = best_beam.cpu().numpy()
    return [
        tokenizer.decode(tokens_cpu[i, best][: int(lengths_cpu[i, best])])
        for i, best in enumerate(best_cpu)
    ]


def generate2(
    model: ClipCaptionModel,
    tokenizer: GPT2Tokenizer,
    tokens: Optional[T] = None,
    prompt: Optional[str] = None,
    embed: Optional[T] = None,
    entry_count: int = 1,
    entry_length: int = 67,  # máximo de tokens generados
    top_p: float = 0.8,
    temperature: float = 1.0,
    stop_token: str = ".",
    stop_token_count: int = 1,
) -> str:
    """
    Generación por nucleus sampling (top-p). Devuelve un solo caption (string).

    `stop_token_count`: ver la nota en generate_beam. Con captions de dos frases
    (reporte + BBPS) hay que pasar 2.
    """
    model.eval()
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    stop_token_count = max(1, int(stop_token_count))
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():
        for _ in range(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            cur_tokens = tokens
            seen_stops = 0

            for _ in range(entry_length):
                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    nnf.softmax(sorted_logits, dim=-1), dim=-1
                )

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value

                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)

                cur_tokens = torch.cat((cur_tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)

                if stop_token_index == next_token.item():
                    seen_stops += 1
                    if seen_stops >= stop_token_count:
                        break

            output_list = list(cur_tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    # Devolvemos solo el primer caption
    return generated_list[0]
