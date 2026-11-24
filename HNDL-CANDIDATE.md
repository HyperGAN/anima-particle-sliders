# Optional Anima HNDL backend

The current isolated package `0.4.0a1` targets Particle Sliders core
0.15.2 and ParticleGAN 0.11.1 with HNDL 0.10.0 and the pinned HyperGAN
builder. Its backend name is `anima-hndl`.

```sh
python -m pip install 'setuptools>=84.2.1'
python cohorts/hypergan-hndl/build.py --out dist-hndl
python -m pip install 'dist-hndl/anima_particle_sliders-0.4.0a1-py3-none-any.whl[native]'
python -m pip check
```

Use a separate environment from the default Anima distribution. The
builder verifies locked package sources. Inference uses RAW artifacts and
earlier checkpoint cohorts use their original revisions. The installed
package has CPU fixture coverage. Native pretrained and GPU execution need
their own qualification.

## Historical reproduction

The [original candidate record](https://github.com/HyperGAN/anima-particle-sliders/blob/791a80a4f38a9b8cb096d719b6758a79325bd7d2/HNDL-CANDIDATE.md) includes its archived builders and checkpoint tests.
