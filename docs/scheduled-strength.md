# Scheduled strength for Anima

Use the **early** schedule for the measured Anima Turbo Candlelit RAW256
profile. It uses the same RAW checkpoint and ten native denoising steps with
more strength in the first half of the traversed noise range. Across six
matched 512 × 512 cases, Candlelit gain was +0.00609 with early steering,
+0.00379 at constant strength and +0.00186 with late steering.

## Render with the measured schedule

Use the Candlelit RAW256 adapter with SHA-256
`90f831220f630c5a23923da3a7b571b5cc5ab5d99c4150dd054e98b4ddc24ac1`.
The inference command multiplies `--strength` by the supplied gains for each
denoiser call and saves the native sigmas and effective strengths next to the
PNG. Pass the neutral prompt of the case you want to render.

```sh
python -m release_tools.infer \
  --model /path/to/anima-turbo-v1.1 --adapter /path/to/RAW-0256.safetensors \
  --format particles --prompt "$PROMPT" --seed 29001 --width 512 --height 512 \
  --steps 10 --strength 1 \
  --step-gains '[1.5,1.5,1.5,1.5,1.5,1.5,1.5,0.9666667853461673,0.5,0.5]' \
  --output candlelit-early.png
```

For an OFF comparison, replay at `--strength 0`. For the constant ON comparison,
omit `--step-gains` and use `--strength 1`.

## Final Boss RAW192

Use early strength at nominal 2/3 for the tested Final Boss RAW192 checkpoint
on Anima Turbo v1.1 at 512 × 512 with ten native Euler intervals and CFG 1.
The four held-out cases averaged +0.34333 Final Boss intensity over matched
OFF, compared with +0.18833 at equal-dose constant strength and +0.34083
at constant strength 1. Early improved three of four cases over strength 1.
All 16 images passed the integrity check. An independent CUDA render
reproduced all four early PNGs byte for byte.

With the exact RAW192 export, pass `--strength 0.6666666666666666` and the
ten relative gains recorded in the evidence. Match the native sigma grid in
the render sidecar before using the result as a replay.
[Checkpoint, gains and per-case evidence](final-boss-raw192-scheduled-strength.json)

## Apply it to another Anima slider

Use the saved RAW adapter. Resolve the gains from that render's native noise
intervals, then pass the gains in denoiser order. If you change the step count
or scheduler, resolve the gains on the new grid. Compare early, late and
constant renders on matched cases and use the profile with the highest mean
intended-effect gain. Check catastrophic output separately.
[Particle Sliders schedule guide](https://github.com/HyperGAN/particle-sliders/blob/main/docs/strength-schedules.md)
