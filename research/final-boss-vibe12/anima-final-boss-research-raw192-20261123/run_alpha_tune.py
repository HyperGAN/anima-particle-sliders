"""Run the PR command with the admitted Anima native environment."""

from pathlib import Path
import sys

from prepare import ROOT, guarded, pin, read


def main():
    import particle_sliders
    from lumen_studio.core_source import verify_core

    guarded()
    verify_core()
    bridge = ROOT / "prompt-only-alpha-tune"
    staged = bridge / "pr-wheel-visible"
    manifest = read(staged / "module-hashes.json")
    for name, sha256 in manifest.items():
        if pin(staged / "particle_sliders" / name)["sha256"] != sha256:
            raise ValueError("Staged alpha CLI differs from the built PR wheel")
    sys.path.insert(0, str(bridge / "renderer-installed"))
    particle_sliders.__path__.append(str(staged / "particle_sliders"))
    from particle_sliders import alpha_tune, alpha_tune_jev
    for module in (alpha_tune, alpha_tune_jev):
        if Path(module.__file__).parent.resolve() != (staged / "particle_sliders").resolve():
            raise ValueError("The Anima trial did not load the PR wheel command")
    return alpha_tune.main(sys.argv[3:])


if __name__ == "__main__":
    main()
