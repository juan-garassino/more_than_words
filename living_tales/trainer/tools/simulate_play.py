from core.cartridge import CartridgeSpec
from generator.path_sampler import PathSampler


def main() -> None:
    spec = CartridgeSpec.load("cases/amber_cipher/spec.json")
    sampler = PathSampler(spec)
    path = sampler.sample_path()
    if path is None:
        print("No valid path found.")
        return
    print(f"Sampled path with {len(path)} triads:")
    for i, triad in enumerate(path):
        token_ids = ", ".join(t.id for t in triad)
        print(f"  {i+1:02d}: {token_ids}")


if __name__ == "__main__":
    main()
