from __future__ import annotations


class ConvergenceProof:
    def run(
        self,
        model,
        spec,
        n_test_paths: int = 500,
        max_attempts: int | None = None,
        verbose: bool = True,
    ) -> dict:
        from generator.path_sampler import PathSampler
        from trainer.hopfield_analyzer import HopfieldAnalyzer

        analyzer = HopfieldAnalyzer()

        print("\n" + "─" * 50)
        print(f"  CONVERGENCE PROOF: {spec.title}")
        print("─" * 50)

        sampler = PathSampler(
            spec,
            sampling_temperature=1.8,
            enforce_monotone=True,
            min_affinity=0.05,
            allow_partial=False,
        )
        test_paths = sampler.sample_batch(
            n_test_paths,
            verbose=verbose,
            max_attempts=max_attempts,
        )
        print(f"  Sampled {len(test_paths)} held-out paths", flush=True)

        converged = 0
        correct_invariants = 0
        turns_list = []

        invariant_ids = set(t.id for t in spec.invariant_tokens)

        for path in test_paths:
            last_triad_tokens = set(t.id for t in path[-1])
            if last_triad_tokens == invariant_ids:
                converged += 1
                correct_invariants += 1
                turns_list.append(len(path))

        lyapunov = analyzer.lyapunov_check(model, test_paths[:100])
        basin = analyzer.basin_size(model, spec, n_samples=200)
        spurious = analyzer.spurious_attractor_scan(model, spec)

        report = {
            "passed": (
                converged == len(test_paths)
                and correct_invariants == len(test_paths)
                and lyapunov["passed"]
                and basin >= 0.90
                and len(spurious) == 0
            ),
            "convergence_rate": converged / max(len(test_paths), 1),
            "invariant_accuracy": correct_invariants / max(len(test_paths), 1),
            "lyapunov_passed": lyapunov["passed"],
            "lyapunov_monotone_rate": lyapunov["monotone_rate"],
            "basin_coverage": basin,
            "spurious_attractors": len(spurious),
            "avg_turns": sum(turns_list) / max(len(turns_list), 1),
            "min_turns": min(turns_list) if turns_list else 0,
            "max_turns": max(turns_list) if turns_list else 0,
        }

        status = "✓ PASSED" if report["passed"] else "✗ FAILED"
        print(f"\n  Proof: {status}")
        print(f"  Convergence rate:    {report['convergence_rate']:.1%}")
        print(f"  Invariant accuracy:  {report['invariant_accuracy']:.1%}")
        print(
            f"  Lyapunov:            {'✓' if report['lyapunov_passed'] else '✗'} "
            f"({report['lyapunov_monotone_rate']:.1%} monotone)"
        )
        print(f"  Basin coverage:      {report['basin_coverage']:.1%}")
        print(f"  Spurious attractors: {report['spurious_attractors']}")
        print(f"  Turn range:          {report['min_turns']}–{report['max_turns']}")

        if not report["passed"]:
            print("\n  ⚠ Cartridge blocked. Cannot export until proof passes.")
            print("  Suggested fixes:")
            if report["convergence_rate"] < 1.0:
                print("    — Increase w_attractor loss weight")
                print("    — Review attractor weight specification")
            if not report["lyapunov_passed"]:
                print("    — Add LyapunovRegularization to training")
                print("    — Increase training epochs")
            if report["basin_coverage"] < 0.90:
                print("    — Sample more diverse training paths (higher temperature)")
            if report["spurious_attractors"] > 0:
                print("    — Add spurious attractor penalty to loss")

        return report
