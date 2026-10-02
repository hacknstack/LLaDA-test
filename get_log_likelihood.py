import torch
import torch.nn.functional as F



def forward_process(batch, prompt_index, mask_id):
    b, l = batch.shape

    target_len = (l - prompt_index.sum()).item()
    k = torch.randint(1, target_len + 1, (), device=batch.device)

    x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
    x = ((x - 1) % target_len) + 1
    assert x.min() >= 1 and x.max() <= target_len

    indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
    is_mask = indices < x.unsqueeze(1)
    for i in range(b):
        is_mask[i] = is_mask[i][torch.randperm(target_len)]

    is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)
    noisy_batch = torch.where(is_mask, mask_id, batch)

    # Return the masked batch and the mask ratio
    return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)


def get_logits(model, batch, prompt_index, cfg_scale, mask_id):
    if cfg_scale > 0.:
        assert len(prompt_index) == batch.shape[1]
        prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
        un_batch = batch.clone()
        un_batch[prompt_index] = mask_id
        batch = torch.cat([batch, un_batch])

    input = batch
    logits = model(input).logits

    if cfg_scale > 0.:
        logits, un_logits = torch.chunk(logits, 2, dim=0)
        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
    return logits



@torch.no_grad()
def get_log_likelihood_from_partially_masked(
    model,
    prompt,
    masked_indexes,
    mc_num=128,
    batch_size=16,
    cfg_scale=0.,
    mask_id=126336,
):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (l). This is the full target sequence z.
        masked_indexes: list[int], 1-indexed positions in `prompt` that are treated as unknown /
                        generated positions. The complement positions are treated as observed conditioning.
        mc_num: Monte Carlo estimation times.
        batch_size: Mini batch size.
        cfg_scale: Unsupervised classifier-free guidance scale.
        mask_id: The token id of [MASK] is 126336.

    Returns:
        Monte Carlo estimate of log P(z_masked_positions | z_observed_positions).
    '''
    device = model.device

    if prompt.dim() != 1:
        raise ValueError(f'prompt must have shape (l), got {tuple(prompt.shape)}')

    seq_len = prompt.shape[0]
    if seq_len == 0:
        raise ValueError('prompt must be non-empty')

    if mc_num <= 0:
        raise ValueError(f'mc_num must be positive, got {mc_num}')
    if batch_size <= 0:
        raise ValueError(f'batch_size must be positive, got {batch_size}')
    if mc_num % batch_size != 0:
        raise ValueError(f'mc_num ({mc_num}) must be divisible by batch_size ({batch_size})')

    # Convert 1-indexed -> 0-indexed.
    masked_pos = sorted(set(int(i) - 1 for i in masked_indexes))
    for pos in masked_pos:
        if pos < 0 or pos >= seq_len:
            raise ValueError(
                f'All masked_indexes must be in [1, {seq_len}], got index {pos + 1}'
            )

    if len(masked_pos) == 0:
        return 0.0

    seq = prompt[None, :].repeat((batch_size, 1)).to(device)

    # IMPORTANT: keep this 1D, not batched.
    # True = observed/conditioning token, False = token whose likelihood we evaluate.
    prompt_index = torch.ones(seq_len, dtype=torch.bool, device=device)
    prompt_index[torch.tensor(masked_pos, dtype=torch.long, device=device)] = False

    loss_ = []
    for _ in range(mc_num // batch_size):
        perturbed_seq, p_mask = forward_process(seq, prompt_index, mask_id)
        mask_index = perturbed_seq == mask_id

        logits = get_logits(model, perturbed_seq, prompt_index, cfg_scale, mask_id)

        loss = F.cross_entropy(
            logits[mask_index],
            seq[mask_index],
            reduction='none',
        ) / p_mask[mask_index]

        loss = loss.sum() / batch_size
        loss_.append(loss.item())

    return -sum(loss_) / len(loss_)
