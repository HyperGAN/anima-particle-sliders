# How the Anima sliders learn

These are nonlinear attention adapters for **Anima Turbo v1.1**. The base diffusion transformer, text encoders, text conditioner, and VAE stay frozen. Bad Intent, Candlelit and Moonlit each have their own adapter, critic, and learned particle cloud. Bad Intent uses the same game with a broader psychological-horror target: expression, pose and framing may change as well as atmosphere. Its internal training identifier remains `uncanny`.

## The particle branch

Every self-attention and cross-attention Q, K, V, and output projection receives an additive branch: 224 projections across 28 transformer blocks. For a projection input $x$, define

$$
u = W_{\mathrm{down}}x,\qquad q=R(u),\qquad a=\operatorname{softmax}\left(\frac{qP^\top}{\sqrt{4}}\right),\qquad z=aP.
$$

Here $W_{\mathrm{down}}$ maps into rank 8, $P\in\mathbb R^{128\times4}$ is the shared particle cloud, and $R$ is a per-projection MLP of width 16. The branch MLP $F$ has width 48. All hidden activations are LeakyReLU with slope 0.2; both MLPs have three hidden linear layers and a final output layer. The projection becomes

$$
y = W_0x + s\,W_{\mathrm{up}}F([u,z]).
$$

The up matrix maps rank 8 back to the projection output width and starts at zero. The same soft routing runs during training and inference. Strength $s=0$ bypasses the branch exactly. Training uses $s=1$; the historical gallery included extrapolation through $s=5$. Calibrated inference additionally multiplies the branch by its stored $\alpha/8$, and the current gallery uses nominal strength 1. No training weights or losses change. For multiple sliders the deltas sum on the same input. Studio normalizes nonnegative mix proportions as $s_j=E m_j/\sum_k m_k$.

An ordinary LoRA has a linear branch $BAx$. The routed particle branch is nonlinear, so its full behavior cannot be merged into a fixed weight matrix. The [distilled LoRAs](DISTILLATION.md) are fitted approximations.

## Paired frozen targets

For each matched neutral/positive prompt pair and seed, generate both frozen trajectories from the same initial noise. At every one of the ten sampling positions on **both** paths, evaluate the neutral and positive frozen velocities at that same latent state. No spatial pooling is used.

Let $v_0(z,t,c_n)$ and $v_0(z,t,c_p)$ be those frozen velocities. Let $v_\theta(z,t,c_n)$ be the adapted neutral prediction. The desired and student edits are

$$
\Delta_T=v_0(z,t,c_p)-v_0(z,t,c_n),\qquad
\Delta_S=v_\theta(z,t,c_n)-v_0(z,t,c_n).
$$

The normalized residual is

$$
e_t=\frac{\Delta_S-\Delta_T}{S_t}
=\frac{v_\theta(z,t,c_n)-v_0(z,t,c_p)}{S_t}.
$$

Each $S_t$ is fitted only on training targets. First compute the per-coordinate sample standard deviation of $\Delta_T$, floored at $10^{-4}$. Multiply that scale by the median per-row RMS of the normalized edits, so their median RMS becomes one. Development data never fits normalization.

## The paired-error game

Use the same Gaussian noise vector for real and fake within a pair:

$$
x_r=\sigma_t(k)\epsilon,\qquad x_f=\sigma_t(k)\epsilon+e_t,\qquad \epsilon\sim\mathcal N(0,I).
$$

The critic is the pinned global-mix network: a projection of the full velocity residual into eight width-48 tokens, one attention layer with four heads, and output score $8\tanh(\cdot/8)$. Separate saved streams choose the D/G rows and noise.

$$
L_D=\mathbb E\,\operatorname{softplus}(D(x_f)-D(x_r))+L_{\mathrm{cap}},\qquad
L_G=\mathbb E\,\operatorname{softplus}(D(x_r)-D(x_f))+L_{\mathrm{VIC}}.
$$

Every fourth update the critic receives the exact autograd gradient cap, with lazy multiplier four:

$$
L_{\mathrm{cap}}=\frac{4}{2}\mathbb E\left[
\max(0,\|\nabla D(x_r)\|_2-1)^2+
\max(0,\|\nabla D(x_f)\|_2-1)^2\right].
$$

Other updates have zero cap loss. VIC acts on a random 64-particle subset: mean $\max(0,1-\sqrt{\operatorname{Var}(P_j)+10^{-4}})$ plus the squared off-diagonal sample covariance sum divided by particle dimension four. Both regularizer coefficients are one. There is no reconstruction or perceptual auxiliary loss in the particle training game.

Noise starts at the training edit RMS divided by 0.28. It decays geometrically toward 0.03 over 1,600 updates, with an absolute hold at 1.0 when the start exceeds that hold. Update $k$ uses schedule index $k-1$. The pilot keeps the same horizon.

## Released recipe and evidence

| Setting | Value |
|---|---|
| Checkpoint | EMA after 1,600 updates, seed 7 |
| Adam learning rates | Generator $2\times10^{-5}$; critic $9\times10^{-4}$; particles $6\times10^{-3}$ |
| Adam betas / decay | $(0,0.999)$ / zero |
| Effective batch / microbatch | 8 / 1 |
| EMA decay | 0.995 |
| Training resolution | 512 × 512 |
| Public samples | 768 × 768, 10 Euler steps, CFG 1, scheduler shift 3 |
| Per slider training data | 24 paired prompt rows × 16 seeds; both trajectories × 10 positions |

The original adult character catalog separates 12 training characters, four development characters, and eight final-test characters. Six definitions train each slider; two extra paraphrases are held out. Candlelit and Moonlit edit lighting; Bad Intent uses expression, pose, framing and atmosphere definitions. The public gallery uses development cases, including a bare prompt, and is not a final-test benchmark.

These are final-budget experimental checkpoints. They were not selected as the best checkpoint by perceptual quality. Strengths above one extrapolate beyond training and can change clothing, pose, setting, or composition. The measured 1,600-update runs did not establish convergence. Training traces, normalization provenance, checkpoint hashes, and development measurements accompany the weights.

## Shared algorithm implementation

The reusable routing, global-mix critic, paired losses, particle regularizer, noise function and ordinary-LoRA solver live in [sliders-conceptmod](https://github.com/mikkel/sliders-conceptmod/tree/beaffeb3640c4554a7315998c04a5909f384b972/packages/concept-slider-core). This release pins that revision in `requirements.txt` and [core.lock.json](https://huggingface.co/ntc-ai/anima-concept-sliders/blob/main/core.lock.json). Anima owns the model integration, target collection, training recipe and sample evidence. The extracted reference implementation is byte-identical to the original; runtime identity hashes the implementation rather than its compatibility import. [Architecture and research](https://github.com/mikkel/sliders-conceptmod/blob/main/docs/shared-core.md).
