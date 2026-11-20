# CADA-on-GenCast implementation

## Boundary

GenCast and its controller run entirely in the `gencast-da` environment.
They do not import the PyTorch `cada` package. The two backends share only
filesystem contracts under `CADA_ROOT`.

## Controller

`CADAController` is a small Haiku network applied to atmospheric temperature.
It receives the current noisy temperature, conditioning temperature,
observation context and mask, previous control, and diffusion noise level.
Its zero-initialized output projection makes a new controller a no-op.

`cada_guided.py` injects `gamma * control` into each DPM-Solver++ denoising
substep while GenCast parameters remain frozen. The productive denoiser call
uses rematerialization. The current implementation uses a warm-start followed
by a short Python-unrolled loop; it does not use `hk.scan`.

## Training

`train_cada_controller.py` supports one or more detached autoregressive
physical steps. For each step it computes:

- cosine-latitude-weighted full-field temperature MSE;
- masked observation MSE accumulated over denoising substeps;
- optional score and control diagnostics;
- controller L2 regularization.

Only controller parameters are differentiated. Checkpoints contain controller
parameters, Optax state, step, controller configuration, diffusion schedule,
and NumPy RNG states. Signal-triggered checkpoints are written atomically.

## Evaluation

`run_cada_gencast.py` advances a two-frame autoregressive memory and reports
full-field, observed-cell, and unobserved-cell RMSE at each 12-hour lead. It
writes native structured outputs when `--output-dir` is supplied by the
GenCast CLI.
