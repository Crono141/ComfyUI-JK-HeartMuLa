import os
import torch
import torchaudio
import folder_paths
import comfy.model_management
import comfy.utils
import warnings
import logging
import math

warnings.filterwarnings("ignore", category=UserWarning, module="transformers.pipelines.base")
warnings.filterwarnings("ignore", message=".*chunk_length_s.*")
warnings.filterwarnings("ignore", message=".*generation_config.*")
warnings.filterwarnings("ignore", message=".*logits processor.*")
warnings.filterwarnings("ignore", message=".*Key value caches are already setup.*")
warnings.filterwarnings("ignore", message=".*Mismatch dtype between input and weight.*")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torchtune").setLevel(logging.ERROR)
logging.getLogger("torchtune.modules.transformer").setLevel(logging.ERROR)

def resolve_model_path(path):
    if os.path.isabs(path) and os.path.exists(path):
        return path
    
    h_path = os.path.join(folder_paths.models_dir, "HeartMuLa", path)
    if os.path.exists(h_path):
        return h_path
        
    m_path = os.path.join(folder_paths.models_dir, path)
    if os.path.exists(m_path):
        return m_path
        
    return path

class HeartMuLaLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_path": ("STRING", {"default": "HeartMuLa", "multiline": False}),
                "model_version": (
                    [
                        "HeartMuLa-RL-oss-3B-20260123",
                        "HeartMuLa-oss-3B",
                    ],
                    {"default": "HeartMuLa-RL-oss-3B-20260123"}
                ),
            }
        }

    RETURN_TYPES = ("HEARTMULA_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "HeartMuLa"

    def load_model(self, base_path, model_version):
        from .heartlib.pipelines.music_generation import HeartMuLaGenPipeline
        
        # Base path resolution (looks for 'HeartMuLa' in models dir)
        resolved_base_path = resolve_model_path(base_path)
        
        if not os.path.exists(resolved_base_path):
             raise FileNotFoundError(f"Base HeartMuLa folder not found at: {resolved_base_path}. Please create a 'HeartMuLa' folder in your ComfyUI models directory containing the model subfolders or select the correct base directory.")

        # Map selection to specific folders
        if model_version == "HeartMuLa-oss-3B":
            m_folder = "HeartMuLa-oss-3B"
            c_folder = "HeartCodec-oss"
        elif model_version == "HeartMuLa-RL-oss-3B-20260123":
            m_folder = "HeartMuLa-RL-oss-3B-20260123"
            c_folder = "HeartCodec-oss-20260123"
        else:
            raise ValueError(f"Unknown model version: {model_version}")

        m_path = os.path.join(resolved_base_path, m_folder)
        c_path = os.path.join(resolved_base_path, c_folder)

        if not os.path.exists(m_path):
             raise FileNotFoundError(f"Model folder not found at: {m_path}. Please download the model.")
             
        if not os.path.exists(c_path):
             raise FileNotFoundError(f"Codec folder not found at: {c_path}. Please download the codec.")

        device = torch.device("cpu")
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        
        print(f"Loading HeartMuLa model: {model_version}...")
        print(f"  - Base Path: {resolved_base_path}")
        print(f"  - Model: {m_folder}")
        print(f"  - Codec: {c_folder}")
        
        try:
            pipeline = HeartMuLaGenPipeline.from_pretrained(
                resolved_base_path,
                device=device,
                dtype=dtype,
                heartmula_path=m_path,
                heartcodec_path=c_path
            )
            return (pipeline,)
        except Exception as e:
            raise RuntimeError(f"Failed to load HeartMuLa model: {e}")

class HeartMuLaMusicGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("HEARTMULA_MODEL",),
                "lyrics": ("STRING", {"multiline": True, "default": ""}),
                "tags": ("STRING", {"multiline": True, "default": ""}),
                "duration_seconds": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 300.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0}),
                "top_k": ("INT", {"default": 50, "min": 1, "max": 1000}),
                "cfg_scale": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 5.0}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "HeartMuLa"

    def generate(self, model, lyrics, tags, duration_seconds, seed, temperature, top_k, cfg_scale):
        torch.manual_seed(seed)
        
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        
        device = comfy.model_management.get_torch_device()
        
        # Sanitize tags: remove spaces around commas to match model expectations
        cleaned_tags = ",".join([t.strip() for t in tags.split(",") if t.strip()])
        
        try:
            inputs = {
                "lyrics": lyrics,
                "tags": cleaned_tags,
            }
            
            processed_inputs = model.preprocess(inputs, cfg_scale)
            max_audio_length_ms = int(duration_seconds * 1000)

            # Step 1: Generate Tokens (HeartMuLa)
            print(f"Moving HeartMuLa (Generator) to {device}...")
            model.model.to(device)
            
            for k, v in processed_inputs.items():
                if isinstance(v, torch.Tensor):
                    processed_inputs[k] = v.to(device)
                    if v.is_floating_point():
                         processed_inputs[k] = processed_inputs[k].to(dtype=model.dtype)

            batch_size = processed_inputs["tokens"].shape[0]
            if not hasattr(model.model, "_caches_initialized") or getattr(model.model, "_cache_bs", 0) != batch_size:
                model.model.setup_caches(max_batch_size=batch_size)
                model.model._caches_initialized = True
                model.model._cache_bs = batch_size
            else:
                model.model.reset_caches()

            gen_steps = int(duration_seconds * 1000 // 80)
            decode_steps_visual = (gen_steps // 3) * 2
            
            pbar = comfy.utils.ProgressBar(gen_steps + decode_steps_visual)
            
            def gen_callback(step):
                pbar.update(1)

            with torch.no_grad():
                frames = model.generate_tokens(
                    processed_inputs,
                    max_audio_length_ms=max_audio_length_ms,
                    temperature=temperature,
                    topk=top_k,
                    cfg_scale=cfg_scale,
                    callback=gen_callback
                )
            
            print("Moving HeartMuLa to CPU...")
            model.model.to("cpu")
            torch.cuda.empty_cache()
            comfy.model_management.soft_empty_cache()

            # Step 2: Decode Audio (HeartCodec)
            print(f"Moving HeartCodec (Decoder) to {device}...")
            model.audio_codec.to(device)
            
            frames = frames.to(device)

            codes_len = frames.shape[-1]
            hop = 320
            ovlp_frames = 104
            
            if codes_len < 372:
                len_codes = 372
            elif (codes_len - ovlp_frames) % hop > 0:
                len_codes = math.ceil((codes_len - 52) / 320.0) * 320 + 52
            else:
                len_codes = codes_len
                
            actual_decode_chunks = (len_codes - 320) // 320 + 1
            actual_total_decode_steps = actual_decode_chunks * 10
            
            decode_scale = decode_steps_visual / max(1, actual_total_decode_steps)
            
            def decode_callback(step):
                pbar.update(decode_scale)

            with torch.no_grad():
                outputs = model.decode_tokens(frames, duration=29.76, callback=decode_callback)
            
            wav = outputs["wav"]
            
            print("Moving HeartCodec to CPU...")
            model.audio_codec.to("cpu")
            torch.cuda.empty_cache()
            comfy.model_management.soft_empty_cache()
            
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            
            wav = wav.unsqueeze(0)
            
            return ({"waveform": wav.cpu(), "sample_rate": 48000},)
        
        finally:
            if hasattr(model, "model"): model.model.to("cpu")
            if hasattr(model, "audio_codec"): model.audio_codec.to("cpu")
            comfy.model_management.soft_empty_cache()

class HeartMuLaTranscriptionLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_path": ("STRING", {"default": "HeartMuLa", "multiline": False}),
            }
        }

    RETURN_TYPES = ("HEARTMULA_TRANSCRIPTOR",)
    RETURN_NAMES = ("transcriptor",)
    FUNCTION = "load_model"
    CATEGORY = "HeartMuLa"

    def load_model(self, base_path):
        from .heartlib.pipelines.lyrics_transcription import HeartTranscriptorPipeline
        
        target_path = resolve_model_path(base_path)

        if not os.path.exists(target_path):
             raise FileNotFoundError(f"Model path does not exist: {target_path}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        print(f"Loading HeartMuLa Transcriptor from {target_path}...")
        
        try:
            pipeline = HeartTranscriptorPipeline.from_pretrained(
                target_path,
                device=device,
                dtype=dtype
            )
            return (pipeline,)
        except Exception as e:
            raise RuntimeError(f"Failed to load HeartMuLa Transcriptor: {e}")

class HeartMuLaLyricsTranscriber:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "transcriptor": ("HEARTMULA_TRANSCRIPTOR",),
                "audio": ("AUDIO",),
                "max_new_tokens": ("INT", {"default": 256, "min": 1, "max": 445, "step": 1}),
                "num_beams": ("INT", {"default": 2, "min": 1, "max": 5, "step": 1}),
                "condition_on_prev_tokens": ("BOOLEAN", {"default": False}),
                "logprob_threshold": ("FLOAT", {"default": -1.0, "min": -20.0, "max": 0.0, "step": 0.1}),
                "no_speech_threshold": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "transcribe"
    CATEGORY = "HeartMuLa"

    def transcribe(self, transcriptor, audio, max_new_tokens, num_beams, condition_on_prev_tokens, logprob_threshold, no_speech_threshold, temperature):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        wav = waveform.cpu()
        if wav.shape[1] > 1:
            wav = wav.mean(dim=1, keepdim=True)
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            wav = resampler(wav)
            sample_rate = 16000
        
        wav = wav.squeeze(0)
        wav_np = wav.numpy()
        
        if temperature == 0.0:
            temperature_arg = (0.0, 0.1, 0.2, 0.4)
        else:
            temperature_arg = temperature

        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "task": "transcribe",
            "condition_on_prev_tokens": condition_on_prev_tokens,
            "compression_ratio_threshold": 1.8,
            "temperature": temperature_arg,
            "logprob_threshold": logprob_threshold,
            "no_speech_threshold": no_speech_threshold,
        }
        
        print(f"Moving Whisper to {transcriptor.device}...")
        transcriptor.model.to(transcriptor.device)
        
        try:
            result = transcriptor(
                {"raw": wav_np, "sampling_rate": sample_rate},
                generate_kwargs=generate_kwargs
            )
            return (result["text"],)
        finally:
            print("Moving Whisper to CPU...")
            transcriptor.model.to("cpu")
            comfy.model_management.soft_empty_cache()

class HeartMuLaPostProcessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "normalize": ("BOOLEAN", {"default": True}),
                "stereo_width": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "high_pass": ("INT", {"default": 80, "min": 0, "max": 1000, "step": 1}),
                "low_pass": ("INT", {"default": 18000, "min": 0, "max": 24000, "step": 100}),
                "gain_db": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "process"
    CATEGORY = "HeartMuLa"

    def process(self, audio, normalize, stereo_width, high_pass, low_pass, gain_db):
        waveform = audio["waveform"].clone()
        sample_rate = audio["sample_rate"]
        
        device = waveform.device
        waveform = waveform.cpu()

        if high_pass > 0:
            waveform = torchaudio.functional.highpass_biquad(waveform, sample_rate, high_pass)

        if low_pass > 0 and low_pass < (sample_rate // 2):
            waveform = torchaudio.functional.lowpass_biquad(waveform, sample_rate, low_pass)

        if gain_db != 0:
            ratio = 10 ** (gain_db / 20)
            waveform = waveform * ratio

        if stereo_width != 1.0:
            if waveform.shape[1] == 1:
                waveform = waveform.repeat(1, 2, 1)
            
            mid = (waveform[:, 0:1, :] + waveform[:, 1:2, :]) / 2.0
            side = (waveform[:, 0:1, :] - waveform[:, 1:2, :]) / 2.0
            
            side = side * stereo_width
            
            new_l = mid + side
            new_r = mid - side
            waveform = torch.cat([new_l, new_r], dim=1)

        if normalize:
            max_val = torch.abs(waveform).max()
            if max_val > 1e-6:
                waveform = waveform / max_val

        return ({"waveform": waveform.to(device), "sample_rate": sample_rate},)

NODE_CLASS_MAPPINGS = {
    "HeartMuLaLoader": HeartMuLaLoader,
    "HeartMuLaMusicGenerator": HeartMuLaMusicGenerator,
    "HeartMuLaTranscriptionLoader": HeartMuLaTranscriptionLoader,
    "HeartMuLaLyricsTranscriber": HeartMuLaLyricsTranscriber,
    "HeartMuLaPostProcessor": HeartMuLaPostProcessor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HeartMuLaLoader": "HeartMuLa Loader",
    "HeartMuLaMusicGenerator": "HeartMuLa Music Generator",
    "HeartMuLaTranscriptionLoader": "HeartMuLa Transcription Loader",
    "HeartMuLaLyricsTranscriber": "HeartMuLa Lyrics Transcriber",
    "HeartMuLaPostProcessor": "Audio Post-Processor"
}
