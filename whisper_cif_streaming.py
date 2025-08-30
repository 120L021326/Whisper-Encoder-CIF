import torch
import torch.nn as nn
import whisper
import numpy as np
import cif_model
import torch.nn.functional as F

Encoder_DIM = 512
ids = ['.', '，', '。', '、', '；', '：', '？', '！', '“', '”', '（', '）', '《', '》', '【', '】',' ', '!', '?', ',']  # 需要过滤的标点符号和空格等

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

whisper_model = whisper.load_model(name='base.en', download_root='./').to(device)
whisper_model.eval()
whisper_model_infer = whisper.load_model(name='medium.en', download_root='./').to(device)
whisper_model_infer.eval()
cif_model = cif_model.LSTMTruncationDetector(Encoder_DIM).to(device)
cif_model.load_state_dict(torch.load("model_baseen_en.pth"))
cif_model.eval()

def train_frame(alpha, len):
    U_i = int(alpha.sum().item())
    if U_i <= 0:
        return []
    A_i = torch.cumsum(alpha, dim=0)
    thresholds = torch.arange(1, U_i+1, device=device, dtype=A_i.dtype)

    diff = (A_i.unsqueeze(0) - thresholds.unsqueeze(1)).abs()
    penalty = (A_i.unsqueeze(0) - thresholds.unsqueeze(1)).clamp(min=0)
    f = diff
    weights = F.softmax(-10 * f - 20 * penalty, dim=1)

    t_idx = torch.arange(len, device=device, dtype=torch.float32)
    pred_frames = (weights * t_idx).sum(dim=1)
    return pred_frames

def cif(chunk):
    mel = whisper.log_mel_spectrogram(chunk).to(device)
    mel = mel.unsqueeze(0) 
    with torch.no_grad():
        encoder_out = whisper_model.encoder(mel)
    with torch.no_grad():
        alphas = cif_model(encoder_out)
    return alphas

def cif_handler(alphas):
    pred_count = alphas[0].sum().item()
    pred_frames = train_frame(alphas[0], 50)
    return pred_count, pred_frames

def modify_logits(log):
    log[12] = -np.inf
    log[532] = -np.inf
    log[58] = -np.inf
    log[685] = -np.inf
    log[60] = -np.inf
    log[2361] = -np.inf
    log[986] = -np.inf
    log[2644] = -np.inf
    log[960] = -np.inf
    log[851] = -np.inf
    log[1906] = -np.inf
    log[784] = -np.inf
    log[7] = -np.inf
    log[357] = -np.inf
    log[8] = -np.inf
    log[1267] = -np.inf
    log[9] = -np.inf
    log[1635] = -np.inf
    log[438] = -np.inf
    log[1377] = -np.inf
    log[6329] = -np.inf
    log[11420] = -np.inf
    log[4211] = -np.inf
    log[9609] = -np.inf
    log[13] = -np.inf
    log[30] = -np.inf
    log[220] = -np.inf
    log[11] = -np.inf
    log[26] = -np.inf
    log[25] = -np.inf

def whisper_handler(audio, pred_num, prefix, flag=0):
    audio = whisper.pad_or_trim(audio) 
    mel = whisper.log_mel_spectrogram(audio).to(device)
    mel = mel.unsqueeze(0)
    with torch.no_grad():
        encoder_out = whisper_model_infer.encoder(mel)
    B = encoder_out.shape[0]
    tokenizer = whisper.tokenizer.get_tokenizer(whisper_model_infer.is_multilingual, num_languages=whisper_model_infer.num_languages, language="en")
    tokens_begin = torch.tensor(tokenizer.sot_sequence_including_notimestamps, dtype=torch.long, device=device)
    tokens_prev = torch.tensor([tokenizer.sot_prev], dtype=torch.long, device=device)
    prefix = tokenizer.encode(prefix)
    prefix = torch.tensor(prefix, dtype=torch.long, device=device)
    tokens = torch.cat([tokens_begin, prefix], dim=-1).unsqueeze(0)
    tokens = tokens.repeat(B, 1)
    text_tokens = []
    count = 0
    while True:
        logitss = whisper_model_infer.decoder(tokens, encoder_out)[:, -1, :]
        modify_logits(logitss[0])
        if count >= pred_num:
            penalty = 1 + 0.1*(count - pred_num)
            logitss[:, tokenizer.eot] *= penalty
        next_token = logitss.argmax(dim=-1, keepdim=True)
        token_text = tokenizer.decode([next_token[0].item()])
        if next_token[0].item() == tokenizer.eot:
            break
        text_tokens.append(next_token[0].item())
        tokens = torch.cat([tokens, next_token], dim=-1)
        count += 1
    text = tokenizer.decode(text_tokens)
    if flag != 1:
        text = ' '.join(text.split(' ')[:-1])
        text_tokens = tokenizer.encode(text)
    return text, text_tokens

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

import io
import wave
import soundfile
import librosa

def infer(wave_path):
    audio_chunk_file = soundfile.SoundFile(wave_path, mode='r')
    waveform, _ = librosa.load(audio_chunk_file, sr=16000, mono=True, dtype="float32")
    length = len(waveform)
    chunks = []
    scripts = []
    offset = 0
    flag = 0
    while True:
        chunk = waveform[int(offset*0.02*16000):int(offset*0.02*16000 + 16000)]
        if len(chunk) < 16000:
            chunk = np.pad(chunk, (0, 16000 - len(chunk)), mode='constant')
            flag = 1
        alphas = cif(chunk)
        pred_count, pred_frames = cif_handler(alphas)
        print(f"Predicted count: {pred_count}, Predicted frames: {pred_frames}")
        if pred_count < 0.5:
            print("No speech detected in this chunk, skipping...")
            ends = find_small_runs_ends(alphas[0])
            print(pred_count)
            print(alphas[0])
            print(ends)
            offset += ends[-1] + 1 
        else:
            if len(pred_frames) == 0:
                offset += 50
                modified_chunk = chunk
            else:
                offset += pred_frames[-1].item()+1
                modified_chunk = chunk[:int(pred_frames[-1].item() * 0.02 * 16000)]
            chunks.append(modified_chunk)
            audio = np.concatenate(chunks, axis=0)
            prefix = "".join(scripts)
            text, text_tokens = whisper_handler(audio, pred_count, prefix=prefix, flag=flag)
            scripts.append(text)
            print(f"Processed chunk {len(chunks)}: {text}")
        if flag == 1:
            break
    final_script = ''.join(scripts)
    print(f"Final script: {final_script}")

infer("E:\学习历程\打榜代码\ASR学习\lstm_cif\\test-clean\\LibriSpeech\\test-clean\\61\\70968\\61-70968-0015.flac")