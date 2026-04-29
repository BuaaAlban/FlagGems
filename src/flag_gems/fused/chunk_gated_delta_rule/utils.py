import torch


def eager_chunk_gated_delta_rule(
    q, k, v, g, beta, scale, initial_state=None, output_final_state=False, **kwargs
):
    b, t, h, kdim = q.shape
    vdim = v.shape[-1]
    device = q.device
    dtype = q.dtype

    state = (
        initial_state.clone().to(dtype=dtype, device=device)
        if initial_state is not None
        else torch.zeros(b, h, kdim, vdim, dtype=dtype, device=device)
    )
    out = torch.empty(b, t, h, vdim, dtype=dtype, device=device)

    for i_t in range(t):
        q_t = q[:, i_t, :, :]
        k_t = k[:, i_t, :, :]
        v_t = v[:, i_t, :, :]
        beta_t = beta[:, i_t, :, None, None]
        g_t = g[:, i_t, :, None]

        proj = torch.einsum("bhk,bhkd->bhd", k_t, state)
        v_new = v_t - proj
        state = state * g_t.unsqueeze(-1) + beta_t * torch.einsum(
            "bhk,bhd->bhkd", k_t, v_new
        )
        out[:, i_t, :, :] = torch.einsum("bhk,bhkd->bhd", q_t, state)

    return out, state
