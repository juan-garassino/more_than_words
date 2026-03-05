import torch


def main():
    print("\n" + "=" * 60)
    print("  THORNFIELD TRAINING PIPELINE")
    print("  Symbolic Hopfield Networks for iOS Cartridges")
    print("=" * 60)

    from trainer.train_mystery import train_mystery_cartridge
    from validator.convergence_proof import ConvergenceProof
    from packager.export_mystery import export_mystery_cartridge, export_tamagotchi_cartridge
    from core.cartridge import CartridgeSpec, TamagotchiSpec
    from trainer.train_tamagotchi import train_tamagotchi_cartridge

    mystery_model, history = train_mystery_cartridge(
        spec_path="cases/amber_cipher/spec.json",
        output_dir="outputs/amber_cipher/",
        n_paths=1000,
        n_epochs=50,
        convergence_rate=0.25,
        min_turns=10,
        max_turns=18,
        device="cpu",
    )

    spec = CartridgeSpec.load("cases/amber_cipher/spec.json")
    proof = ConvergenceProof().run(mystery_model, spec, n_test_paths=500)

    export_mystery_cartridge(
        model=mystery_model,
        spec=spec,
        output_path="outputs/AmberCipher.cartridge",
        proof_report=proof,
    )

    thornling_model = train_tamagotchi_cartridge(
        spec_path="cases/thornling/spec.json",
        output_dir="outputs/thornling/",
        n_trajectories=500,
        trajectory_length=200,
        n_epochs=80,
        device="cpu",
    )

    t_spec = TamagotchiSpec.load("cases/thornling/spec.json")
    export_tamagotchi_cartridge(
        model=thornling_model,
        spec=t_spec,
        output_path="outputs/Thornling.cartridge",
    )

    print("\n" + "=" * 60)
    print("  Both cartridges ready.")
    print("  AmberCipher.cartridge  →  drop into iOS app bundle")
    print("  Thornling.cartridge    →  drop into iOS app bundle")
    print("=" * 60)


if __name__ == "__main__":
    main()
