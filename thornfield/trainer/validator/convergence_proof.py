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

        print("\n" + "=" * 60)
        print(f"  CONVERGENCE PROOF: {spec.title}")
        print("=" * 60)
        print(f"  n_test_paths={n_test_paths}  max_attempts={max_attempts}", flush=True)

        print("\n[PROOF] Sampling held-out paths (temperature=1.8, monotone=True, partial=False)...", flush=True)
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
        print(f"[PROOF] Sampled {len(test_paths)}/{n_test_paths} held-out paths", flush=True)

        if not test_paths:
            print("\n[PROOF] RESULT: FAILED")
            print("[PROOF] CAUSE:  path sampler produced 0 paths — convergence threshold unreachable")
            print("[PROOF] FIXES:")
            print("  — Raise convergence_rate in pack_case.py (currently too low to reach threshold)")
            print("  — Increase max_turns in spec")
            print("  — Lower convergence_threshold in spec")
            return {
                "passed": False,
                "convergence_rate": 0.0,
                "invariant_accuracy": 0.0,
                "lyapunov_passed": False,
                "lyapunov_monotone_rate": 0.0,
                "basin_coverage": 0.0,
                "spurious_attractors": 0,
                "avg_turns": 0,
                "min_turns": 0,
                "max_turns": 0,
            }

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

        print(f"[PROOF] Checking convergence on {len(test_paths)} paths...", flush=True)

        print("[PROOF] Running Lyapunov check (first 100 paths)...", flush=True)
        lyapunov = analyzer.lyapunov_check(model, test_paths[:100])
        print(
            f"[PROOF] Lyapunov: monotone_rate={lyapunov['monotone_rate']:.1%}  "
            f"violations={len(lyapunov['violations'])}  paths_checked={lyapunov['n_paths_checked']}",
            flush=True,
        )

        print("[PROOF] Estimating basin coverage (200 samples)...", flush=True)
        basin = analyzer.basin_size(model, spec, n_samples=200)
        print(f"[PROOF] Basin coverage: {basin:.1%}", flush=True)

        print("[PROOF] Scanning for spurious attractors...", flush=True)
        spurious = analyzer.spurious_attractor_scan(model, spec)
        print(f"[PROOF] Spurious attractors: {len(spurious)}", flush=True)

        conv_rate = converged / max(len(test_paths), 1)
        inv_acc = correct_invariants / max(len(test_paths), 1)

        # Pass thresholds
        CONV_THRESHOLD = 1.0
        INV_THRESHOLD = 1.0
        LYAPUNOV_THRESHOLD = 0.90
        BASIN_THRESHOLD = 0.90

        conv_ok = conv_rate >= CONV_THRESHOLD
        inv_ok = inv_acc >= INV_THRESHOLD
        lyap_ok = lyapunov["passed"]  # already uses 90% threshold
        basin_ok = basin >= BASIN_THRESHOLD
        spurious_ok = len(spurious) == 0

        report = {
            "passed": conv_ok and inv_ok and lyap_ok and basin_ok and spurious_ok,
            "convergence_rate": conv_rate,
            "invariant_accuracy": inv_acc,
            "lyapunov_passed": lyap_ok,
            "lyapunov_monotone_rate": lyapunov["monotone_rate"],
            "basin_coverage": basin,
            "spurious_attractors": len(spurious),
            "avg_turns": sum(turns_list) / max(len(turns_list), 1),
            "min_turns": min(turns_list) if turns_list else 0,
            "max_turns": max(turns_list) if turns_list else 0,
        }

        def _tick(ok): return "PASS" if ok else "FAIL"

        print("\n[PROOF] RESULTS")
        print("[PROOF] " + "-" * 45)
        print(f"[PROOF] paths_sampled    : {len(test_paths)}/{n_test_paths}")
        print(f"[PROOF] convergence_rate : {conv_rate:.1%}  (need={CONV_THRESHOLD:.0%})  [{_tick(conv_ok)}]")
        print(f"[PROOF] invariant_acc    : {inv_acc:.1%}  (need={INV_THRESHOLD:.0%})  [{_tick(inv_ok)}]")
        print(f"[PROOF] lyapunov_monotone: {lyapunov['monotone_rate']:.1%}  (need≥{LYAPUNOV_THRESHOLD:.0%})  [{_tick(lyap_ok)}]")
        print(f"[PROOF] basin_coverage   : {basin:.1%}  (need≥{BASIN_THRESHOLD:.0%})  [{_tick(basin_ok)}]")
        print(f"[PROOF] spurious         : {len(spurious)}  (need=0)  [{_tick(spurious_ok)}]")
        print(f"[PROOF] turn_range       : {report['min_turns']}–{report['max_turns']}  avg={report['avg_turns']:.1f}")
        print("[PROOF] " + "-" * 45)
        print(f"[PROOF] OVERALL: {'PASSED' if report['passed'] else 'FAILED'}", flush=True)

        if not report["passed"]:
            print("\n[PROOF] FIXES NEEDED:")
            if not conv_ok:
                print("  convergence_rate low  → Increase w_attractor loss weight or review attractor weights")
            if not inv_ok:
                print("  invariant_acc low     → Review graph edges to invariant tokens")
            if not lyap_ok:
                print(f"  lyapunov {lyapunov['monotone_rate']:.1%} < 90%   → Add LyapunovRegularization or increase epochs")
            if not basin_ok:
                print("  basin_coverage low    → Sample more diverse paths (higher temperature)")
            if not spurious_ok:
                print("  spurious attractors   → Add spurious attractor penalty to loss")

        return report
