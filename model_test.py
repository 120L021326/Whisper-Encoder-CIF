import torch
import torch.nn as nn
from whisper import load_model
import whisper
import numpy as np
import io
import librosa
import soundfile
import wave
import whisperx

np.set_printoptions(threshold=np.inf)

Encoder_DIM = 512
Result_Offset = 0
Test_Cunk_Begin = int(Result_Offset * 16000 * 0.02)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
whisper_model = load_model(name='base', download_root='./').to(device)
whisper_model.eval()

with wave.open("E:\\学习历程\\打榜代码\\ASR学习\\asr demo\\test_zh1_16k.wav", 'rb') as wf:
# with wave.open("E:\dataset\\resource_aishell\\test\\data_input\\test\\00526.wav", 'rb') as wf:
    audio_chunk_file = soundfile.SoundFile(
        io.BytesIO(wf.readframes(wf.getnframes())),
        channels=1,
        endian="LITTLE",
        samplerate=16000,
        subtype="PCM_16",
        format="RAW"
    )
    waveform, _ = librosa.load(audio_chunk_file, sr=16000, mono=True, dtype="float32")

class LSTMTruncationDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_layers=2, bidirectional=True, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional, dropout=dropout)
        self.out_proj = nn.Linear(hidden_dim * (2 if bidirectional else 1), 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: [B, T, D]
        lstm_out, _ = self.lstm(x)          # [B, T, H]
        lstm_out = self.dropout(lstm_out)
        logits = self.out_proj(lstm_out)    # [B, T, 1]
        alphas = torch.sigmoid(logits).squeeze(-1)  # [B, T]
        return alphas

model = LSTMTruncationDetector(input_dim=Encoder_DIM).to(device)
model.load_state_dict(torch.load("model_v8.pth"))
model.eval()

# def build_truncation_labels(word_segments, num_frames, frame_stride=0.02):
#     labels = np.zeros(num_frames, dtype=np.float32)
#     end_time = 0
#     start_time = 0
#     for word in word_segments:
#         start_time = word['start']
#         end_time = word['end']
#         token_end_frame = int(round(end_time / frame_stride))
#         token_start_frame = int(round(start_time / frame_stride))
#         alpha = 1 / (token_end_frame - token_start_frame + 1)
#         labels[token_start_frame:token_end_frame + 1] = alpha

def build_truncation_labels(word_segments, num_frames, frame_stride=0.02):
    labels = np.zeros(num_frames, dtype=np.float32)
    end_time = 0
    start_time = 0
    for word in word_segments:
        start_time = word['start']
        end_time = word['end']
        token_end_frame = int(round(end_time / frame_stride))
        token_start_frame = int(round(start_time / frame_stride))
        alpha = 1 / (token_end_frame - token_start_frame)
        labels[token_start_frame:token_end_frame] = alpha

    return torch.tensor(labels, dtype=torch.float32)

def get_token_frame_intervals_single(alphas: torch.Tensor, threshold: float = 1.0):
    integrate = 0.0
    prev_fire = -1
    samps = alphas.tolist()

    intervals = []
    for t, a in enumerate(samps):
        integrate += a
        if integrate >= threshold:
            start = prev_fire + 1
            end = t
            intervals.append((start, end))
            prev_fire = t
            integrate -= threshold

    token_count = len(intervals)
    return token_count, intervals

import torch.nn.functional as F
def train_frame(alpha):
    U_i = int(alpha.sum().item())
    if U_i <= 0:
        return []
    A_i = torch.cumsum(alpha, dim=0)
    thresholds = torch.arange(1, U_i+1, device=device, dtype=A_i.dtype)

    diff = (A_i.unsqueeze(0) - thresholds.unsqueeze(1)).abs()
    penalty = (A_i.unsqueeze(0) - thresholds.unsqueeze(1)).clamp(min=0)
    f = diff
    weights = F.softmax(-10 * f - 10 * penalty, dim=1)

    t_idx = torch.arange(50, device=device, dtype=torch.float32)
    pred_frames = (weights * t_idx).sum(dim=1)
    return pred_frames

def find_small_runs_ends(x, threshold: float = 1e-5, min_run_len: int = 10):
    mask: list[bool] = (x < threshold).tolist()
    
    ends: list[int] = []
    count = 0
    
    for i, m in enumerate(mask):
        if m:
            count += 1
        else:
            if count >= min_run_len:
                ends.append(i - 1)
            count = 0
    
    if count >= min_run_len:
        ends.append(len(mask) - 1)
    
    return ends

# align_model, metadata = whisperx.load_align_model(language_code='zh', device=device)
# res = {
#         "segments": [{
#             "start": 0,
#             "end": len(waveform) / 16000,
#             "text": '大家好我叫小明很高兴认识你'
#             #"text": 'hello everyone my name is peter nice to meet you'
#         }]
#     }
# result = whisperx.align(res["segments"], align_model, metadata, waveform, device, return_char_alignments=False)
# word_segments = result["word_segments"]
# print(word_segments)
# label = build_truncation_labels(word_segments, 500)
# print(label.cpu().numpy())
# segments = get_token_frame_intervals_single(label, threshold=1.0)
# print(segments)

begin = Test_Cunk_Begin
offset = Result_Offset
flag = 0
res = []
while True:
    chunk_wf = waveform[begin:begin+16000]
    if len(chunk_wf) < 16000:
        chunk_wf = np.pad(chunk_wf, (0, 16000 - len(chunk_wf)), mode='constant')
        flag = 1
    mel = whisper.log_mel_spectrogram(chunk_wf).to(device)
    #print(mel.shape)  # [1, 80, 3000] (1, 80, T)
    mel = mel.unsqueeze(0) 
    with torch.no_grad():
        encoder_out = whisper_model.encoder(mel)
    alphas = model(encoder_out)
    train_frames = train_frame(alphas[0])
    if len(train_frames) != 0:
        train_frames = train_frames + offset
        res.extend( train_frames.tolist())
    print(f'权重方法: {train_frames}')
    print(alphas[0])
    sum = alphas[0].sum()
    ends = find_small_runs_ends(alphas[0])
    if len( train_frames) > 0:
        offset =  int(train_frames[-1].item()) + 1
        begin = int(offset * 16000 * 0.02)
    if len( train_frames) == 0 and flag == 0:
        print("存在空白音频段")
        offset = offset + ends[-1] + 1
        begin = int(offset * 16000 * 0.02)
    # print(sum.item())
    print(f'偏移量：{offset}')
    if flag == 1:
        break

print(res)
