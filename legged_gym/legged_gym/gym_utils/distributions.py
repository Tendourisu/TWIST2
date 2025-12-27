import torch


def sample_truncated_normal(mu, sigma, low, high, shape, device, out_dtype=torch.float32):
    dtype = torch.float64
    mu = torch.as_tensor(mu, device=device, dtype=dtype)
    sigma = torch.as_tensor(sigma, device=device, dtype=dtype)
    low = torch.as_tensor(low, device=device, dtype=dtype)
    high = torch.as_tensor(high, device=device, dtype=dtype)

    alpha = (low - mu) / sigma
    beta = (high - mu) / sigma
    cdf_lo = torch.special.ndtr(alpha)
    cdf_hi = torch.special.ndtr(beta)
    # guard ndtri from hitting ±inf when bounds land exactly on 0/1
    cdf_lo = torch.nextafter(cdf_lo, torch.ones_like(cdf_lo))
    cdf_hi = torch.nextafter(cdf_hi, torch.zeros_like(cdf_hi))

    u = cdf_lo + (cdf_hi - cdf_lo) * torch.rand(shape + cdf_lo.shape, device=device, dtype=dtype)
    z = torch.special.ndtri(u)
    samples = mu + sigma * z
    samples = torch.clamp(samples, low, high)
    return samples.to(out_dtype)
