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

## Apply it to another Anima slider

Use the saved RAW adapter. Resolve the gains from that render's native noise
intervals, then pass the gains in denoiser order. If you change the step count
or scheduler, resolve the gains on the new grid. Compare early, late and
constant renders on matched cases and use the profile with the highest mean
intended-effect gain. Check catastrophic output separately.
[Particle Sliders schedule guide](https://github.com/HyperGAN/particle-sliders/blob/main/docs/strength-schedules.md)
