import torch

def quantize_per_group(w, group_size=2, bits=4):
    ''' 
    w(int8) -> per-group quant -> int4 -> 2xint4 pack int8
    '''
    N, K = w.shape
    
    num_groups_per_row = K // group_size
    w_grouped = w.view(N, num_groups_per_row, group_size)

    abs_max_vals = torch.max(torch.abs(w_grouped), dim=2).values # (N, num_groups_per_row)

    scales = abs_max_vals / (2**(bits-1) - 1)
    
    scales_expanded = scales.unsqueeze(2)

    quantized_w = w_grouped / scales_expanded

    i4 = torch.round(quantized_w).clamp(-8, 7).to(torch.int8)
    quantized_2d = i4.view(N, K)
    
    high = quantized_2d[:, 0::2]
    low = quantized_2d[:, 1::2]

    packed_w_i8 = (high << 4) | (low & 0x0F)

    return packed_w_i8, scales
    

if __name__ == "__main__":
    M, K = 6, 8
    w = torch.randint(0, 16, (M, K), dtype=torch.uint8).to("cuda")

    quantized_w, scales = quantize_per_group(w)

    print(w)
    print(quantized_w)
    print(scales)