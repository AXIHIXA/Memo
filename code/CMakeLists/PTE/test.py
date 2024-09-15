import torch

import pte


def main() -> None:
    device: torch.device = torch.device('cuda')
    # g: torch.Generator = torch.Generator(device=device)
    # a: torch.Tensor = torch.rand((3, 2), generator=g, dtype=torch.float32, device=device)
    a: torch.Tensor = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32, device=device)
    print(f'a: torch.Tensor = \n{a}\n')
    b: torch.Tensor = pte.add_one(a)
    print(f'b: torch.Tensor = pte.add_one(a) = \n{b}\n')
    c: torch.Tensor = pte.cross2d(a, b)
    print(f'c: torch.Tensor = pte.cross2d(a, b) = \n{c}\n')

    print('Testing print() and pte.print(torch.Tensor):')
    d: torch.Tensor = torch.arange(0, 2 * 4).float().to(device)
    print(d)
    pte.print(d)


if __name__ == '__main__':
    main()
