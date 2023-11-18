import torch

import build.pte as pte


def main() -> None:
    device: torch.device = torch.device('cuda')
    # g: torch.Generator = torch.Generator(device=device)
    # a: torch.Tensor = torch.rand((3, 2), generator=g, dtype=torch.float32, device=device)
    a: torch.Tensor = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32, device=device)
    print(a)
    b = pte.add_one(a)
    print(b)
    c = pte.cross2d(a, b)
    print(c)

    d = torch.arange(0, 32 * 4).float().to(device)
    print(d)
    pte.print(d)


if __name__ == '__main__':
    main()
