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

The loss function plays a crucial role in this model, as it significantly impacts performance. I designed a composite loss consisting of three components: **Token Count Loss**, **Timestamp Regression Loss**, and **Blank Loss**. The overall formula is:

$$
\mathcal{L} = \lambda_{\mathrm{count}} \ell_{\mathrm{count}} + \lambda_{\mathrm{time}} \ell_{\mathrm{time}} + \lambda_{\mathrm{blank}} \ell_{\mathrm{blank}}
$$

where:

* \$\lambda\_{\mathrm{count}}\$, \$\lambda\_{\mathrm{time}}\$, and \$\lambda\_{\mathrm{blank}}\$ are the weighting coefficients for Token Count Loss, Timestamp Regression Loss, and Blank Loss, respectively.

---

#### Token Count Loss

To enable the model to predict the number of tokens in an audio segment, I designed a Token Count Loss. Its formulation is:

$$
\ell_{\mathrm{count}} = \frac{1}{B} \sum_{i=1}^B \mathrm{Smooth}_{L_1}(C_i, c_i)
$$

where:

* \$B\$ is the batch size;
* \$C\_i = \sum\_{t=1}^T \alpha\_{i,t}\$ is the predicted total token count for the \$i\$-th sample;
* \$c\_i\$ is the ground-truth token count (true\_counts) for the \$i\$-th sample.

---

#### Timestamp Regression Loss

To train the model to predict the timestamps of each token within a segment, I designed a Timestamp Regression Loss:

$$
\ell_{\mathrm{time}} = \frac{1}{|\{i:c_i>0\}|} \sum_{i:c_i>0} \mathrm{Smooth}_{L_1}(\hat\tau_{i}, \tau_{i})
$$

where:

* \${i\:c\_i>0}\$ denotes the subset of samples with token counts greater than 0;
* \$\hat\tau\_{i}\$ is the predicted frame index list for the \$i\$-th sample, with each element \$\hat\tau\_{i,u}\$ computed as:

$$
\hat\tau_{i,u} = \sum_{t=1}^T w_{i,u,t} \cdot t
$$

* \$w\_{i,u,t}\$ is the attention weight, defined as:

$$
w_{i,u,t} = \frac{\exp(f_{i,u,t})}{\sum_{t'=1}^T \exp(f_{i,u,t'})}
$$

* \$f\_{i,u,t}\$ is the offset penalty, computed as:

$$
f_{i,u,t} = -\beta_1|A_{i,t} - u| - \beta_2 \max(A_{i,t} - u, 0)
$$

* \$A\_{i,t} = \sum\_{k=1}^t \alpha\_{i,k}\$ is the temporal integration of \$\alpha\$;
* \$u\$ ranges from \$1\$ to \$c\_i\$, where \$c\_i\$ is the ground-truth token count for the \$i\$-th sample;
* \$\tau\_{i} = (\tau\_{i,1}, \dots, \tau\_{i,C\_i})\$ is the ground-truth frame index list for the \$i\$-th sample.

---

#### Blank Loss

To prevent the model from producing excessive signal values in blank frames (e.g., silence or background noise), I designed a Blank Loss:

$$
\ell_{\mathrm{blank}} = \frac{1}{B} \sum_{i=1}^B \sum_{t=1}^T \alpha_{i,t} \cdot m_{i,t}
$$

where:

* \$m\_{i,t}\$ is the blank frame mask, defined as:

$$
m_{i,t} = \mathbf{1}\{l_{i,t} = 0\}
$$

* \$l\_{i,t}\$ is the label in the label sequence, with \$0\$ indicating a blank frame.

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

```
[69.49354553222656, 81.75992584228516, 92.08222961425781, 138.9051971435547, 
148.89248657226562, 160.82579040527344, 176.3285369873047, 214.15577697753906, 
225.21575927734375, 233.57644653320312, 243.09828186035156, 254.3844757080078, 
273.79620361328125] 
```
Predicted timestamps aligned closely with reference values, with only 2–3 frames of deviation, which is acceptable given timestamp variability.

### Robustness

* On blank segments (background noise >1s), the model output nearly zero alphas, showing robustness to silence.
    ```

    tensor([3.5944e-06, 2.8404e-07, 1.7056e-07, 1.7256e-07, 2.0879e-07, 2.6019e-07,
            3.2543e-07, 3.5060e-07, 3.5138e-07, 3.0055e-07, 2.8155e-07, 2.5605e-07,
            2.1286e-07, 2.0457e-07, 1.9383e-07, 1.8396e-07, 1.8302e-07, 1.7400e-07,
            1.6818e-07, 1.7000e-07, 1.6519e-07, 1.4737e-07, 1.6438e-07, 1.5200e-07,
            1.4527e-07, 1.3863e-07, 1.4256e-07, 1.4058e-07, 1.3799e-07, 1.2596e-07,
            1.3253e-07, 1.3737e-07, 1.5898e-07, 1.7688e-07, 1.9847e-07, 2.2142e-07,
            2.4273e-07, 3.0331e-07, 4.1480e-07, 5.2232e-07, 4.3415e-07, 5.9241e-07,
            5.9167e-07, 6.0761e-07, 4.8981e-07, 4.7571e-07, 4.4415e-07, 3.5544e-07,
            5.9094e-07, 8.9407e-06], device='cuda:0', grad_fn=<SelectBackward0>)
    ```
* On token-to-token gaps, alphas also remained near zero until a token appeared, further confirming robustness.
    ```
    tensor([1.4861e-06, 1.4406e-07, 8.3902e-08, 7.3880e-08, 7.4446e-08, 7.3631e-08,
            8.0129e-08, 8.8093e-08, 9.5247e-08, 1.0435e-07, 1.2186e-07, 1.2187e-07,
            1.3772e-07, 1.6681e-07, 1.7682e-07, 1.3715e-07, 1.3125e-07, 9.6394e-08,
            1.0572e-07, 8.6677e-08, 7.0555e-08, 5.6862e-08, 4.7397e-08, 4.9379e-08,
            5.1056e-07, 1.3614e-05, 2.3531e-05, 1.4020e-04, 2.1676e-02, 4.8366e-01,
            4.7016e-01, 4.5352e-03, 4.5237e-04, 1.7187e-04, 1.0282e-04, 5.6864e-05,
            2.5481e-05, 1.6732e-05, 2.0137e-05, 2.2847e-05, 3.2090e-05, 8.0079e-05,
            2.1820e-04, 9.8521e-04, 1.5554e-02, 1.3876e-01, 2.6434e-01, 3.0096e-01,
            2.4608e-01, 1.1959e-01], device='cuda:0', grad_fn=<SelectBackward0>)
    ```

