# Project Summary

## Abstract

This project designs a new audio segmentation model based on a two-layer bidirectional LSTM architecture. The input to the model is the output of the Whisper Encoder, while the output is the signal value (alphas) for each frame.

By predicting these alphas, the model is able to estimate the number of tokens within an audio segment, generate timestamps, and guide the segmentation of audio chunks. These predictions not only support the Whisper Decoder during decoding but also enable more efficient streaming speech recognition.

---

## CIF (Continuous Integrate-and-Fire) Method

The training of this model follows the CIF method (Continuous Integrate-and-Fire), which has been successfully applied in ASR models such as Paraformer.

The general process is as follows:

1. Pass the audio through the encoder to obtain the encoder output.
2. Transform each frame’s feature into a scalar and apply a sigmoid function, yielding an alpha value between 0 and 1.
3. Continuously accumulate alpha values until they reach a predefined threshold, then trigger a “fire.”
4. When firing, reset the accumulator to zero. The encoder features of the frames whose alphas sum to 1 are combined through weighted averaging to produce a new frame as the output.
5. The new frame is then sent into the decoder to produce a token.

In previous implementations, CIF typically functioned as part of a larger model, meaning training was applied to the full model rather than CIF itself. In contrast, this project uses Whisper’s encoder and decoder with frozen parameters, and specifically trains a small model for signal prediction.

---

## Role of Signal Values (alphas)

The output of this model is the per-frame signal value (alpha), which serves several purposes:

1. **Token Count Prediction**
   By summing the alphas, we can estimate the token count of the current audio segment. This allows us to design a decoder-side penalty function that guides the decoding process and mitigates hallucinations, such as repeated or erroneous tokens.

2. **Truncation Detection**
   The sum of alphas indicates whether truncation occurs within the current segment. For instance, if the sum equals 2.5, the segment is truncated before the third token. If the decoder produces more than two tokens, it may have generated a truncated and thus erroneous token, which should be discarded.

3. **Timestamp Generation**
   By integrating the alpha values over time, we can compute the timestamp of each token.

4. **Guiding Next-Segment Splitting**
   The timestamp of the last complete token can serve as the starting point for the next audio segment, preventing truncation at the beginning of new segments and reducing decoding errors.

---

## Experiment Process

### Dataset Preprocessing

#### Whisper Model Adjustment

In Whisper’s Audio Encoder module, the input audio is required to be exactly 30 seconds long. Any shorter input raises an error. Since this project focuses on streaming recognition, I set the segment length to 1 second.

Padding a 1-second segment to 30 seconds wastes computation, so I slightly modified Whisper’s Audio Encoder by removing the assert statement enforcing the 30-second input. This allows direct processing of 1-second audio chunks.

#### Timestamp Generation

Both LibriSpeech (English) and AIShell-1 (Chinese) datasets lack timestamp annotations, so I generated them myself.

* For **LibriSpeech**, I used WhisperX.
* For **AIShell-1**, I used the `fa_zh` model from FunASR.

#### Label Generation

Timestamps were converted into label sequences. Given Whisper’s frame length of 0.02s, each timestamp was mapped to a frame index.

For each token, frames between its start and end indices were labeled with signal values, while others were set to 0 (blank). The signal value for each token was defined as:

$$
\alpha = \frac{1}{F_{end} - F_{start}}
$$

where \$F\_{start}\$ and \$F\_{end}\$ are the start and end frame indices of the token.

These alpha labels are hypothetical, serving only as training signals to help the model estimate token counts and timestamps. They are not direct ground truth values.

#### Dataset Construction

The model input consists of Whisper Encoder outputs from 1-second audio segments. Steps:

1. Traverse the timestamp list and extract 1-second segments every 5 tokens, aligned with token boundaries. This reduces overlap and ensures padding at the beginning.
2. Pad shorter segments to 1 second.
3. Encode the segments using Whisper Encoder to obtain feature representations.
4. Extract corresponding labels and pad them if needed.

This process yielded **100,000 audio segments** for training.

---

### Loss Function Design

The loss function is critical to performance. I designed a composite loss consisting of three parts: token count loss, timestamp regression loss, and blank loss:

$$
\mathcal{L} = \lambda_{\mathrm{count}} \ell_{\mathrm{count}} + \lambda_{\mathrm{time}} \ell_{\mathrm{time}} + \lambda_{\mathrm{blank}} \ell_{\mathrm{blank}}
$$

* \$\lambda\_{\mathrm{count}}\$, \$\lambda\_{\mathrm{time}}\$, and \$\lambda\_{\mathrm{blank}}\$ are weighting factors.

#### Token Count Loss

Ensures the model predicts the number of tokens in a segment:

$$
\ell_{\mathrm{count}} = \frac{1}{B} \sum_{i=1}^B \mathrm{Smooth}_{L_1}(C_i, c_i)
$$

where \$C\_i = \sum\_{t=1}^T \alpha\_{i,t}\$ is the predicted token count, and \$c\_i\$ is the ground truth.

#### Timestamp Regression Loss

Trains the model to predict per-token timestamps:

$$
\ell_{\mathrm{time}} = \frac{1}{|\{i:c_i>0\}|} \sum_{i:c_i>0} \mathrm{Smooth}_{L_1}(\hat\tau_{i}, \tau_{i})
$$

where \$\hat\tau\_{i}\$ is the predicted timestamp sequence computed via attention weighting, and \$\tau\_{i}\$ is the ground truth sequence.

#### Blank Loss

Encourages the model to output near-zero alphas for blank frames (e.g., background noise):

$$
\ell_{\mathrm{blank}} = \frac{1}{B} \sum_{i=1}^B \sum_{t=1}^T \alpha_{i,t} \cdot m_{i,t}
$$

where \$m\_{i,t}\$ is a mask set to 1 if the frame is blank, else 0.

---

### Training Results

I trained on both LibriSpeech (English) and AIShell-1 (Chinese) datasets, using Whisper’s base, small, medium, and large-v3 models. Optimizer: AdamW, learning rate: 1e-4, batch size: 8, epochs: 20.

Key results (base model shown as example):

* **Chinese dataset**: Validation loss = 0.7213 (Count Loss: 0.0968, Time Loss: 0.4646, Blank Loss: 0.0632).
* **English dataset**: Validation loss = 2.8246 (sub-losses not recorded).

Performance was notably better on Chinese, likely because it is a monosyllabic language. In English, multi-syllabic words such as *“everyone”* may be misinterpreted as multiple tokens (*“every” + “one”*), an issue absent in Chinese.

---

## Local Demo Test

To further evaluate, I implemented a local demo using my own recordings. Example timestamps:

| Start | End   | Token |
| ----- | ----- | ----- |
| 59.5  | 70.5  | 大     |
| 70.5  | 82.5  | 家     |
| 83.5  | 95.5  | 好     |
| 127.5 | 139.5 | 我     |
| 139.5 | 151.5 | 叫     |
| 151.5 | 162.5 | 小     |
| 162.5 | 174.5 | 明     |
| 201.5 | 213.5 | 很     |
| 213.5 | 224.5 | 高     |
| 224.5 | 232.5 | 兴     |
| 232.5 | 242.5 | 认     |
| 242.5 | 254.5 | 识     |
| 254.5 | 275.7 | 你     |

### Demo Workflow

1. Extract the first 1s of audio and pass it through the Whisper Encoder and the model.
2. Use the predicted token boundary to cut the next segment.
3. Repeat until the full recording is processed.

### Accuracy

Predicted timestamps aligned closely with reference values, with only 2–3 frames of deviation, which is acceptable given timestamp variability.

### Robustness

* On blank segments (background noise >1s), the model output nearly zero alphas, showing robustness to silence.
* On token-to-token gaps, alphas also remained near zero until a token appeared, further confirming robustness.

