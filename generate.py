import torch
import numpy as np
import torch.nn.functional as F



def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens



@torch.no_grad()
def generate_from_partially_masked(
    model,
    prompt,
    masked_indexes,
    attention_mask=None,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.,
    cfg_scale=0.,
    remasking='low_confidence',
    mask_id=126336,
    logits_eos_inf=False,
    confidence_eos_eot_inf=False,
):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (B, L). Tokens at `masked_indexes` will be replaced by [MASK].
        masked_indexes: list[int], 1-indexed positions in `prompt` to be masked and generated.
        attention_mask: Optional tensor of shape (B, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Total number of generated tokens, INCLUDING masked tokens from `prompt`.
                    If gen_length > len(masked_indexes), extra generation slots are appended to
                    the right of `prompt`.
        block_length: Block length over the generatable positions, less than or equal to gen_length.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
        logits_eos_inf: Whether to set the logits of EOS token to -inf.
        confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf.
    Returns:
        A tensor of shape (B, L + max(0, gen_length - len(masked_indexes))).
    '''
    batch_size, prompt_len = prompt.shape
    device = model.device

    if gen_length <= 0:
        raise ValueError(f'gen_length must be positive, got {gen_length}')
    if block_length <= 0:
        raise ValueError(f'block_length must be positive, got {block_length}')
    if gen_length % block_length != 0:
        raise ValueError(f'gen_length ({gen_length}) must be divisible by block_length ({block_length})')
    if steps % (gen_length // block_length) != 0:
        raise ValueError(
            f'steps ({steps}) must be divisible by the number of blocks ({gen_length // block_length})'
        )

    if len(masked_indexes) == 0 and gen_length == 0:
        return prompt.clone()

    # Convert 1-indexed -> 0-indexed, validate, deduplicate, sort.
    masked_pos = sorted(set(int(i) - 1 for i in masked_indexes))
    for pos in masked_pos:
        if pos < 0 or pos >= prompt_len:
            raise ValueError(
                f'All masked_indexes must be in [1, {prompt_len}], got index {pos + 1}'
            )

    num_prompt_masks = len(masked_pos)
    if num_prompt_masks > gen_length:
        raise ValueError(
            f'gen_length ({gen_length}) must be >= len(masked_indexes) ({num_prompt_masks}) '
            'because gen_length includes masked tokens from prompt.'
        )

    appended_gen_length = gen_length - num_prompt_masks
    total_len = prompt_len + appended_gen_length

    x = torch.full(
        (batch_size, total_len),
        mask_id,
        dtype=torch.long,
        device=device,
    )
    x[:, :prompt_len] = prompt.clone()

    # Mask the requested prompt positions.
    if num_prompt_masks > 0:
        prompt_mask_tensor = torch.tensor(masked_pos, dtype=torch.long, device=device)
        x[:, prompt_mask_tensor] = mask_id
    else:
        prompt_mask_tensor = torch.empty(0, dtype=torch.long, device=device)

    if attention_mask is not None:
        if attention_mask.shape != prompt.shape:
            raise ValueError(
                f'attention_mask must have shape {prompt.shape}, got {attention_mask.shape}'
            )
        if appended_gen_length > 0:
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (batch_size, appended_gen_length),
                        dtype=attention_mask.dtype,
                        device=device,
                    ),
                ],
                dim=-1,
            )
        else:
            attention_mask = attention_mask.clone()

    # prompt_index marks positions that must stay fixed for CFG unconditional branch.
    prompt_index = (x != mask_id)

    # Generatable positions = masked prompt positions + appended suffix positions.
    if appended_gen_length > 0:
        appended_positions = torch.arange(
            prompt_len, total_len, device=device, dtype=torch.long
        )
        generation_positions = torch.cat([prompt_mask_tensor, appended_positions], dim=0)
    else:
        generation_positions = prompt_mask_tensor

    if generation_positions.numel() != gen_length:
        raise RuntimeError(
            f'Internal error: expected {gen_length} generatable positions, '
            f'got {generation_positions.numel()}'
        )

    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_start = num_block * block_length
        block_end = (num_block + 1) * block_length
        current_block_positions = generation_positions[block_start:block_end]

        # Among the current and future blocks, which positions in this block are still masked?
        block_mask_index = (x[:, current_block_positions] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = (x == mask_id)

            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)

                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                else:
                    attention_mask_ = None

                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # (B, total_len)

            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = -torch.inf
                logits_with_noise[:, :, 126348] = -torch.inf

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)),
                    -1,
                )  # (B, total_len)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            # Only allow transfers inside blocks up to and including the current block.
            # Future blocks remain frozen at this stage.
            if block_end < gen_length:
                future_positions = generation_positions[block_end:]
                x0_p[:, future_positions] = -np.inf

            # Never transfer onto fixed / already-known tokens.
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                k = int(num_transfer_tokens[j, i].item())
                if k > 0:
                    _, select_index = torch.topk(confidence[j], k=k)
                    transfer_index[j, select_index] = True

            x[transfer_index] = x0[transfer_index]

    return x
