# Adaptive Noise Cancellation

A MATLAB-based project for **adaptive noise cancellation (ANC) of speech signals** using adaptive filtering and second-order optimization ideas. The repository explores and compares **NLMS**, **RLS**, and a **Quasi-Newton / Hessian-preconditioned adaptive filter** under multiple cost/update formulations, and also implements a **partial noise-suppression mode using a second-order notch filter**.

The goal is to estimate unwanted noise from a reference noise signal and subtract that estimate from a noisy speech signal while preserving as much of the desired speech as possible.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Adaptive Noise Cancellation Principle](#adaptive-noise-cancellation-principle)
- [System Architecture](#system-architecture)
- [Algorithms Implemented](#algorithms-implemented)
- [Cost / Update Variants](#cost--update-variants)
- [Main Implementation](#main-implementation)
- [Partial Noise Suppression with a Notch Filter](#partial-noise-suppression-with-a-notch-filter)
- [Dataset and Signals](#dataset-and-signals)
- [Repository Structure](#repository-structure)
- [Important Parameters](#important-parameters)
- [Performance Evaluation](#performance-evaluation)
- [Reported Results](#reported-results)
- [How to Run](#how-to-run)
- [Expected Outputs](#expected-outputs)
- [Advantages and Trade-offs](#advantages-and-trade-offs)
- [Applications](#applications)
- [References](#references)

---

## Project Overview

Adaptive Noise Cancellation is a signal-processing technique in which an adaptive filter learns the relationship between a **reference noise signal** and the noise contained in a **corrupted primary signal**.

This project works with three signals:

- **Clean speech** `s(n)` — the desired speech signal used for evaluation.
- **Noisy speech** — speech contaminated by noise, conceptually `s(n) + v(n)`.
- **External/reference noise** `w(n)` — a reference signal correlated with the unwanted noise.

The adaptive filter processes the external noise and generates an estimate of the noise present in the noisy speech. This estimate is subtracted from the noisy speech to obtain the final output.

The repository focuses on comparing adaptive algorithms and studying the effect of different optimization/update strategies on noise suppression and SNR.

---

## Problem Statement

Speech recordings are often corrupted by background or interfering noise. A fixed filter may not perform well when the characteristics of that noise change over time.

The problem addressed in this project is therefore:

> **Estimate and suppress noise from a noisy speech signal using a reference noise signal and an adaptive filter whose coefficients continuously adjust according to the observed data.**

The project also considers a **partial suppression** case in which a known tonal component can be removed using a notch filter while the adaptive stage handles the remaining noise.

---

## Objectives

The repository is designed to:

1. Implement adaptive noise cancellation for speech signals in MATLAB.
2. Estimate noise using an external/reference noise signal.
3. Recover a cleaner speech signal by subtracting the estimated noise.
4. Compare **NLMS**, **RLS**, and **Quasi-Newton/Hessian-based** approaches.
5. Experiment with **MSE**, **log-cosh**, and **weighted least-squares (WLS)** style updates.
6. Evaluate filtering quality using **Signal-to-Noise Ratio (SNR)**.
7. Study computational and convergence trade-offs among the algorithms.
8. Support partial suppression of known tonal interference using a second-order notch filter.
9. Visualize signals, filter weights, frequency response, and spectral behavior.

---

## Adaptive Noise Cancellation Principle

A simplified ANC structure used by the project is:

```text
                    Primary input
              d(n) = s(n) + v(n)
                       |
                       |------------------------+
                       |                        |
                       |                        v
Reference noise       |                  +-----------+
      x(n) ---------->| Adaptive Filter  | Subtract  |----> e(n)
                      |       w(n)       +-----------+
                      |          |             ^
                      |          v             |
                      |        y(n) -----------+
                      |
                      +--------------------------------

y(n) = estimated noise
e(n) = d(n) - y(n)
```

The adaptive filter output is

```text
y(n) = wᵀ(n)x(n)
```

and the residual/error signal is

```text
e(n) = d(n) - y(n)
```

where:

- `x(n)` is the reference-noise input vector,
- `w(n)` is the adaptive-filter coefficient vector,
- `y(n)` is the estimated noise,
- `d(n)` is the noisy speech,
- `e(n)` is the final filtered output.

If the reference signal is sufficiently correlated with the unwanted component in the primary input, adaptation drives the filter toward a model of that noise path.

---

## System Architecture

The overall processing flow is:

```text
Clean Speech -------------------------------> Used for SNR evaluation
                                                     |
                                                     v
Noisy Speech -------------------------> Adaptive Noise Canceller
                                              |
External Noise --> Optional Notch Filter --> Adaptive Filter
                                              |
                                              v
                                       Estimated Noise
                                              |
                                              v
                                  Noisy Speech - Estimate
                                              |
                                              v
                                       Filtered Speech
                                              |
                         +--------------------+--------------------+
                         |                    |                    |
                         v                    v                    v
                        SNR              Time Plots              FFT
```

In the main `partialfinal.m` implementation, the reference noise can first pass through a notch filter when `mode = 'partial'`.

---

## Algorithms Implemented

### 1. NLMS — Normalized Least Mean Squares

NLMS is an LMS-family adaptive algorithm whose update is normalized by the energy of the current input vector.

The implementation follows the form

```text
w(n+1) = w(n) + [ μ / (xᵀx + ε) ] x(n)e(n)
```

where:

- `μ` is the step size,
- `ε` prevents division by very small values.

Normalization makes the update less sensitive to changes in reference-signal power.

**Repository file:**

```text
RLS-NLMS-QuasiNewton_Comparison/NLMS/nlms.m
```

---

### 2. RLS — Recursive Least Squares

RLS recursively updates the filter using an inverse correlation/covariance matrix. It uses a forgetting factor to give more importance to recent observations.

The implementation calculates a gain vector approximately as

```text
g(n) = P(n-1)x(n) /
       [λ + xᵀ(n)P(n-1)x(n) + ε]
```

and updates the weights using

```text
w(n) = w(n-1) + g(n)e(n)
```

The inverse correlation matrix is then recursively updated.

**Repository file:**

```text
RLS-NLMS-QuasiNewton_Comparison/RLS/Rls.m
```

The project uses values including:

```text
λ = 0.9995
δ = 100
N = 8
```

---

### 3. Quasi-Newton / Hessian-Preconditioned Adaptive Filtering

The main project explores second-order optimization ideas by maintaining an approximation related to the inverse Hessian and using it to precondition the adaptive update.

The motivation comes from Newton's optimization step:

```text
Δx = -H⁻¹(x)∇f(x)
```

and

```text
x(k+1) = x(k) - H⁻¹(x(k))∇f(x(k))
```

The main implementation maintains an inverse-Hessian-related matrix and updates the filter using a step of the form:

```text
w = w + μ H_inv x e
```

This introduces curvature information into the update direction rather than relying only on a first-order gradient.

The final implementation also estimates a maximum-eigenvalue-related quantity and uses:

```text
μ = 2 / λmax
```

with safeguards in the code.

The project slides describe the Hessian inverse update as a **dynamic Broyden-like / Quasi-Newton update**.

**Main file:**

```text
partialfinal.m
```

**Additional Quasi-Newton experiments:**

```text
RLS-NLMS-QuasiNewton_Comparison/Quasi-Newton/
```

---

## Cost / Update Variants

The comparison folders contain experiments with three formulations.

### MSE

Mean-squared-error-based adaptation is the conventional baseline. The project notes that the MSE cost surface is convex in the standard linear adaptive-filter setting, which is useful for stable optimization.

### Log-Cosh

The repository also experiments with a log-cosh-style robust update.

Since

```text
d/de log(cosh(e)) = tanh(e)
```

the implementations use `tanh(e)` in the adaptive update.

Examples:

```text
RLS-NLMS-QuasiNewton_Comparison/NLMS/nlms_log.m
RLS-NLMS-QuasiNewton_Comparison/RLS/rls_log_cosh.m
RLS-NLMS-QuasiNewton_Comparison/Quasi-Newton/lms_log_cosh.m
```

### WLS

Weighted Least Squares variants assign weights to errors or observations so their contributions to adaptation can differ.

Examples:

```text
RLS-NLMS-QuasiNewton_Comparison/NLMS/nlmswls.m
RLS-NLMS-QuasiNewton_Comparison/RLS/rls_wls.m
RLS-NLMS-QuasiNewton_Comparison/Quasi-Newton/lms_wls_.m
```

---

## Main Implementation

The main runnable experiment is:

```text
partialfinal.m
```

It performs the following operations:

1. Loads clean speech, noisy speech, and external noise.
2. Uses an adaptive-filter length of `N = 8`.
3. Uses a sampling frequency of `44.1 kHz`.
4. Selects a processing mode.
5. In partial mode, constructs a second-order notch filter for the specified tonal frequency.
6. Filters the reference noise through the notch filter.
7. Stores recent reference samples in an adaptive-filter input buffer.
8. Estimates the noise using the current adaptive weights.
9. Subtracts the estimated noise from the noisy speech.
10. Updates the Hessian/inverse-Hessian-related approximation.
11. Updates the adaptive filter coefficients.
12. Calculates SNR before and after adaptive cancellation.
13. Optionally applies the notch filter to the final output.
14. Calculates an additional post-notch SNR.
15. Plots time-domain signals and filter weights.
16. Plots the notch-filter magnitude and phase responses.
17. Compares noisy and filtered spectra using the FFT.
18. Plays the filtered signal and writes the post-notch output to a WAV file.

---

## Partial Noise Suppression with a Notch Filter

The main program contains:

```matlab
mode = 'partial';
```

For this mode, the project defines:

```matlab
tonal_freqs = [2000];
r = 0.999;
```

A second-order notch section is constructed for each selected tonal frequency.

For angular frequency

```text
θ = 2πf/fs
```

the numerator is

```text
b = [1, -2cos(θ), 1]
```

and the denominator is

```text
a = [1, -2r cos(θ), r²]
```

The zeros create the notch at the desired frequency while poles with radius `r` control the notch characteristics.

Multiple notch sections can be combined by convolution when multiple tonal frequencies are specified.

The project documentation notes a trade-off: increasing selectivity/filter order can improve suppression but can also increase phase effects and potentially distort the desired signal.

---

## Dataset and Signals

The root of the repository contains:

```text
clean_speech2.txt
noisy_speech_file2_time_varying_amplitude_f0_2kHz_f1_8243Hz.txt
external_noise_file2_time_varying_amplitude_f0_2kHz_f1_8243Hz.txt
```

Each of these files contains **517,377 samples**.

At the sampling frequency used by the MATLAB program:

```text
fs = 44,100 Hz
```

this corresponds to approximately:

```text
517377 / 44100 ≈ 11.73 seconds
```

of signal data.

From the filenames and code:

- `clean_speech2.txt` is the clean reference speech.
- `noisy_speech_file2...txt` is the corrupted speech.
- `external_noise_file2...txt` is the reference noise.
- The supplied experiment names identify time-varying-amplitude components at approximately **2 kHz** and **8.243 kHz**.

---

## Repository Structure

```text
Adaptive-Noise-Cancellation/
│
├── README.md
├── partialfinal.m
│
├── clean_speech2.txt
├── noisy_speech_file2_time_varying_amplitude_f0_2kHz_f1_8243Hz.txt
├── external_noise_file2_time_varying_amplitude_f0_2kHz_f1_8243Hz.txt
│
├── Course_Project___Final_Evaluation.pdf
├── Course_Project___Extra_Slides.pdf
│
└── RLS-NLMS-QuasiNewton_Comparison/
    │
    ├── NLMS/
    │   ├── nlms.m
    │   ├── nlms_log.m
    │   └── nlmswls.m
    │
    ├── RLS/
    │   ├── Rls.m
    │   ├── rls_log_cosh.m
    │   └── rls_wls.m
    │
    └── Quasi-Newton/
        ├── lms_log_cosh.m
        └── lms_wls_.m
```

> **Note:** Some comparison scripts load generic files named `clean_speech.txt`, `noisy_speech.txt`, and `external_noise.txt`. Those exact filenames are not present in the supplied repository snapshot. To run those comparison scripts directly, provide the corresponding files in their working directory or update the `load(...)` paths to the available dataset.

---

## Important Parameters

The final project documentation gives the following design choices.

| Parameter | Project Value | Purpose |
|---|---:|---|
| Filter length `N` | `8` | Number of adaptive-filter coefficients |
| Sampling frequency `fs` | `44100 Hz` | Signal sampling rate |
| Forgetting factor `γ` | `0.91` | Controls memory in the Hessian-related update |
| Regularization `ε` | `1e-6` | Improves numerical conditioning |
| Step size | `2 / λmax` | Dynamically derived from an eigenvalue estimate |
| Notch pole radius `r` | `0.999` | Controls notch characteristics |
| Partial-mode tone | `2000 Hz` | Tonal frequency selected in `partialfinal.m` |

The final-evaluation slides recommend approximately:

| Parameter | Recommended Range |
|---|---|
| Filter length `N` | `5 ≤ N ≤ 32` |
| Forgetting factor `γ` | `0.9 ≤ γ < 1` |
| Step size `μ` | typically `0.01 ≤ μ ≤ 1` |
| Regularization `ε` | `10⁻⁸ ≤ ε ≤ 10⁻³` |

These ranges are project documentation rather than universal optimal values; actual tuning depends on the signal and noise environment.

---

## Performance Evaluation

The main metric used by the repository is **Signal-to-Noise Ratio (SNR)**.

Before cancellation:

```text
SNR_before =
10 log10 [ Σ s²(n) / Σ(noisy(n) - s(n))² ]
```

After cancellation:

```text
SNR_after =
10 log10 [ Σ s²(n) / Σ(output(n) - s(n))² ]
```

The SNR improvement is calculated as:

```text
SNR Gain = SNR_after - SNR_before
```

The main program also calculates SNR after the optional final notch-filtering stage.

---

## Reported Results

The supplied project slides report the following **SNR after filtering** values:

| Cost Function | NLMS (dB) | RLS (dB) | Quasi-Newton (dB) |
|---|---:|---:|---:|
| MSE | 9.593 | 22.3 | 24.86 |
| Log-cosh | 22.3006 | 22.43 | 24.91 |
| WLS | 10.4 | 12.25 | 22.3057 |

These are the values reported in `Course_Project___Extra_Slides.pdf` for the project's experiments. They should be interpreted as results for the tested signals, parameters, and implementation rather than as universal performance guarantees.

The highest value shown in that comparison table is **24.91 dB for the Quasi-Newton log-cosh experiment**.

---

## How to Run

### Requirements

- MATLAB
- Audio output support for `sound` / `audioplayer`
- The supplied `.txt` signal files

The project does not contain a Python dependency file or external package setup. Its implementation is MATLAB-based.

### Run the Main Experiment

1. Clone the repository:

```bash
git clone https://github.com/Deepakreddy1510/Adaptive-Noise-Cancellation.git
```

2. Open MATLAB.

3. Change the MATLAB current folder to the repository root.

4. Make sure these files are present in the same working directory as `partialfinal.m`:

```text
clean_speech2.txt
noisy_speech_file2_time_varying_amplitude_f0_2kHz_f1_8243Hz.txt
external_noise_file2_time_varying_amplitude_f0_2kHz_f1_8243Hz.txt
```

5. Run:

```matlab
partialfinal
```

### Change the Operating Mode

The main script currently contains:

```matlab
mode = 'partial';
```

`partial` enables the reference-noise notch-filter stage and the additional output-notch processing.

The `else` branch of the script bypasses the reference-noise notch filter for other mode strings.

### Change the Tonal Frequency

Edit:

```matlab
tonal_freqs = [2000];
```

For multiple frequencies, the code is structured to accept an array, for example:

```matlab
tonal_freqs = [2000 8243];
```

Each frequency generates a notch section and the sections are combined.

---

## Expected Outputs

Running `partialfinal.m` prints values such as:

```text
Mode of Operation: PARTIAL
SNR Before Noise Cancellation: ... dB
SNR After Adaptive Cancellation: ... dB
SNR Gain After Adaptive Cancellation: ... dB
SNR After Output Notch Filtering: ... dB
```

The script also produces several visualizations.

### Signal Plots

- Clean speech
- Noisy speech
- Estimated noise
- Final output signal
- Adaptive filter weights

### Notch-Filter Plots

- Magnitude response
- Phase response

### Output Comparison

- Clean speech
- Adaptive output before final notch filtering
- Output after notch filtering

### Frequency-Domain Comparison

An FFT magnitude plot compares:

- Noisy speech
- Filtered output

### Audio Output

The main program plays the adaptive-filtered signal and, in partial mode, writes:

```text
filtered_output_notched.wav
```

Several comparison scripts also generate WAV files such as filtered NLMS, RLS, or WLS outputs.

---

## Advantages and Trade-offs

According to the supplied final-evaluation material:

| Criterion | LMS + Hessian | NLMS | RLS |
|---|---|---|---|
| Computational complexity | `O(N²)` | `O(N)` | reported as `O(N³)` in the project slides |
| Memory | Hessian approximation, about `N²` | Low, about `N` | Inverse correlation matrix, about `N²` |
| Reported SNR behavior | High | Moderate | High |

### NLMS

**Advantages**

- Relatively simple.
- Low memory requirements.
- Normalization improves robustness to input-power changes.

**Trade-off**

- Convergence/performance depends on the selected step size and signal characteristics.

### RLS

**Advantages**

- Uses correlation information.
- Typically adapts quickly in many adaptive-filtering problems.

**Trade-off**

- Requires substantially more matrix computation and memory than NLMS.

### Quasi-Newton / Hessian Approach

**Advantages**

- Uses curvature information to improve the update direction.
- Supports experimentation with different loss/cost formulations.
- Produced strong SNR values in the project's reported comparison.

**Trade-off**

- Requires maintaining and updating matrix information.
- More computationally expensive than a basic NLMS implementation.
- Numerical conditioning and parameter selection become important.

---

## References

The project documentation cites:

1. Paulo S. R. Diniz, *Adaptive Filtering: Algorithms and Practical Implementation*.
2. Course material on optimization / logistic regression from EE2802 Machine Learning slides.
3. The Sherman-Morrison formula.

The mathematical implementation also uses standard concepts from adaptive filtering, least-squares estimation, normalized LMS, recursive least squares, Newton/Quasi-Newton optimization, and digital notch filtering.

---

## Repository

Project repository:

```text
https://github.com/Deepakreddy1510/Adaptive-Noise-Cancellation
```

---

## Summary

This project demonstrates adaptive speech-noise cancellation using multiple adaptive optimization strategies. It compares **NLMS**, **RLS**, and **Quasi-Newton/Hessian-based filtering**, investigates **MSE, log-cosh, and WLS** variants, and combines adaptive filtering with a **second-order notch filter** for partial tonal-noise suppression.

The supplied experimental results show that the Quasi-Newton variants achieved strong SNR values on the project's test signals, with the project slides reporting **24.91 dB** after filtering for the Quasi-Newton log-cosh experiment.

The repository is primarily an **experimental/academic MATLAB implementation** intended for studying adaptive-filter behavior, optimization choices, noise suppression, and SNR performance.
