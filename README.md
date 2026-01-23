# ComfyUI-HeartMuLa

A ComfyUI extension for music generation and lyrics transcription based on the [HeartMuLa model family](https://huggingface.co/HeartMuLa) and [heartlib source code](https://github.com/HeartMuLa/heartlib).

## Features
- **Text-to-Music**: Generate high-fidelity audio from lyrics and style tags.
- **Lyrics Transcription**: Automatic speech-to-text with support for long-form audio.
- **Audio Post-Processing**: Integrated tools for normalization, stereo widening, and frequency filtering.

## Installation

1. Navigate to your ComfyUI `custom_nodes` folder:

   ```bash
   cd ComfyUI/custom_nodes
   ```
2. Clone this repository:

   ```bash
   git clone https://github.com/BobRandomNumber/ComfyUI-HeartMuLa
   ```
3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Model Setup

The nodes require specific model weights and configuration files. Create a base folder named `HeartMuLa` inside your ComfyUI models directory (e.g., `ComfyUI/models/HeartMuLa/`) and organize the files as follows.

### 1. Base Configuration Files
Place these files directly in the root of the `HeartMuLa/` folder:
- [gen_config.json](https://huggingface.co/HeartMuLa/HeartMuLaGen/blob/main/gen_config.json)
- [tokenizer.json](https://huggingface.co/HeartMuLa/HeartMuLaGen/blob/main/tokenizer.json)

### 2. Model Directories
Download the repositories below as subfolders inside the `HeartMuLa/` directory. The folder names must match exactly for the loader to recognize them.

#### For Standard Generation (oss-3B)
- **Model**: [HeartMuLa-oss-3B](https://huggingface.co/HeartMuLa/HeartMuLa-oss-3B/tree/main)
- **Codec**: [HeartCodec-oss](https://huggingface.co/HeartMuLa/HeartCodec-oss/tree/main)

#### For RL-Tuned Generation (RL-oss-3B)
- **Model**: [HeartMuLa-RL-oss-3B-20260123](https://huggingface.co/HeartMuLa/HeartMuLa-RL-oss-3B-20260123/tree/main)
- **Codec**: [HeartCodec-oss-20260123](https://huggingface.co/HeartMuLa/HeartCodec-oss-20260123/tree/main)

#### For Transcription
- **Model**: [HeartTranscriptor-oss](https://huggingface.co/HeartMuLa/HeartTranscriptor-oss/tree/main)

### Final Directory Structure
```text
ComfyUI/models/HeartMuLa/
├── gen_config.json
├── tokenizer.json
├── HeartMuLa-oss-3B/
├── HeartCodec-oss/
├── HeartMuLa-RL-oss-3B-20260123/
├── HeartCodec-oss-20260123/
└── HeartTranscriptor-oss/
```

## Node Descriptions

### HeartMuLa Loader
Loads the generation and decoding pipelines.
- **base_path**: The root folder containing your HeartMuLa models. Use the "Select Base Path" button to set this.
- **model_version**: Select between the standard `oss-3B` or the reinforcement-learning tuned `RL-oss-3B` model. The loader automatically pairs the correct model with its specific codec.

### HeartMuLa Music Generator
Generates audio based on text inputs.
- **lyrics**: The text to be sung/spoken.
- **tags**: Style descriptions (e.g., "electronic, synthwave, 120bpm").
- **duration_seconds**: Desired length of the output (Default: 30s).
- **cfg_scale**: Guidance scale for tag adherence (Default: 1.5).

### Audio Post-Processor
A DSP utility for mastering the generated output.
- **normalize**: Peak normalization to 0dB.
- **stereo_width**: Adjusts stereo image width (Mid-Side processing).
- **high_pass / low_pass**: Removes unwanted frequencies.

### HeartMuLa Lyrics Transcriber
Converts audio back into text.
- **max_new_tokens**: Limit for the generated transcription length.
- **temperature**: Sampling temperature (0.0 enables robust multi-temperature decoding).

## Citation

```bibtex
@misc{yang2026heartmulafamilyopensourced,
      title={HeartMuLa: A Family of Open Sourced Music Foundation Models}, 
      author={Dongchao Yang and Yuxin Xie and Yuguo Yin and Zheyu Wang and Xiaoyu Yi and Gongxi Zhu and Xiaolong Weng and Zihan Xiong and Yingzhe Ma and Dading Cong and Jingliang Liu and Zihang Huang and Jinghan Ru and Rongjie Huang and Haoran Wan and Peixu Wang and Kuoxi Yu and Helin Wang and Liming Liang and Xianwei Zhuang and Yuanyuan Wang and Haohan Guo and Junjie Cao and Zeqian Ju and Songxiang Liu and Yuewen Cao and Heming Weng and Yuexian Zou},
      year={2026},
      eprint={2601.10547},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2601.10547}, 
}
```
