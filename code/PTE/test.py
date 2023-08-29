import torch

import cmake_build_release.pte as pte


def main() -> None:
    device: torch.device = torch.device('cpu')
    g: torch.Generator = torch.Generator(device=device)
    a: torch.Tensor = torch.rand((11,), generator=g, dtype=torch.float32, device=device)
    print(a)
    print(pte.add_one(a))


if __name__ == '__main__':
    main()
