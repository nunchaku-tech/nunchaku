import torch
from torch.nn import RMSNorm

from nunchaku.models.linear import SVDQW4A4Linear

from .gemm import svdq_gemm_w4a4_cuda


def fused_gelu_mlp(x: torch.Tensor, fc1: SVDQW4A4Linear, fc2: SVDQW4A4Linear):
    # a fused operator of fc1 and fc2 with gelu
    batch_size, seq_len, channels = x.shape
    x = x.view(batch_size * seq_len, channels)
    quantized_x, ascales, lora_act = fc1.quantize(x)

    qout_act = torch.empty(batch_size * seq_len, fc1.out_features // 2, dtype=torch.uint8, device=x.device)
    if fc2.precision == "nvfp4":
        qout_ascales = torch.empty(
            fc1.out_features // 16, batch_size * seq_len, dtype=torch.float8_e4m3fn, device=x.device
        )
    else:
        qout_ascales = torch.empty(fc1.out_features // 64, batch_size * seq_len, dtype=x.dtype, device=x.device)
    qout_lora_act = torch.empty(batch_size * seq_len, fc2.proj_down.shape[1], dtype=torch.float32, device=x.device)

    # for int4, we shift the activation after gelu to make it all positive to improve quality.
    is_unsigned = fc2.precision != "nvfp4"

    # if we pass the qout to this kernel, it will do the gelu fusion.
    svdq_gemm_w4a4_cuda(
        act=quantized_x,
        wgt=fc1.qweight,
        qout=qout_act,
        ascales=ascales,
        wscales=fc1.wscales,
        oscales=qout_ascales,
        lora_act_in=lora_act,
        lora_up=fc1.proj_up,
        lora_down=fc2.proj_down,
        lora_act_out=qout_lora_act,
        bias=fc1.bias,
        smooth_factor=fc2.smooth_factor,
        act_unsigned=is_unsigned,
        fp4=fc1.precision == "nvfp4",
        alpha=fc1.wtscale,
        wcscales=fc1.wcscales,
    )

    output = fc2.forward_quant(qout_act, qout_ascales, qout_lora_act)
    output = output.view(batch_size, seq_len, -1)
    return output


def fused_qkv_norm_rottary(
    x: torch.Tensor,
    proj: SVDQW4A4Linear,
    norm_q: RMSNorm,
    norm_k: RMSNorm,
    rotary_emb: torch.Tensor,
    output: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    attn_tokens: int = 0,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert isinstance(norm_q, RMSNorm)
    assert isinstance(norm_k, RMSNorm)

    batch_size, seq_len, channels = x.shape
    x = x.view(batch_size * seq_len, channels)
    quantized_x, ascales, lora_act = proj.quantize(x)

    if output is None:
        output = torch.empty(quantized_x.shape[0], proj.out_features, dtype=x.dtype, device=x.device)

    if isinstance(output, tuple):
        assert len(output) == 3
        output_q, output_k, output_v = output
        svdq_gemm_w4a4_cuda(
            act=quantized_x,
            wgt=proj.qweight,
            ascales=ascales,
            wscales=proj.wscales,
            lora_act_in=lora_act,
            lora_up=proj.proj_up,
            bias=proj.bias,
            fp4=proj.precision == "nvfp4",
            alpha=proj.wtscale,
            wcscales=proj.wcscales,
            norm_q=norm_q.weight,
            norm_k=norm_k.weight,
            rotary_emb=rotary_emb,
            out_q=output_q,
            out_k=output_k,
            out_v=output_v,
            attn_tokens=attn_tokens,
        )
        return output_q, output_k, output_v
    else:
        svdq_gemm_w4a4_cuda(
            act=quantized_x,
            wgt=proj.qweight,
            out=output,
            ascales=ascales,
            wscales=proj.wscales,
            lora_act_in=lora_act,
            lora_up=proj.proj_up,
            bias=proj.bias,
            fp4=proj.precision == "nvfp4",
            alpha=proj.wtscale,
            wcscales=proj.wcscales,
            norm_q=norm_q.weight,
            norm_k=norm_k.weight,
            rotary_emb=rotary_emb,
        )
        output = output.view(batch_size, seq_len, -1)
        return output
