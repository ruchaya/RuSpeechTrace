#!/usr/bin/env python
from pathlib import Path
import shutil
import pandas as pd

CARD='https://huggingface.co/datasets/lab260/RuASD'
PAPER='https://doi.org/10.1109/ACCESS.2026.3706003'

def rec(name,family,checkpoint='unknown',task='TTS',vocoder='unknown',mode='unknown',transfer=None,evidence_extra='',notes=''):
    ev='; '.join(x for x in [CARD,PAPER,evidence_extra] if x)
    return dict(generator_name=name,base_model_family=family,checkpoint_or_voice=checkpoint,
                task_type=task,vocoder=vocoder,open_source_or_commercial=mode,
                family_label_for_transfer=transfer or family,evidence=ev,notes=notes)

M={
'11labs':rec('ElevenLabs TTS','ElevenLabs',mode='commercial',transfer='ElevenLabs'),
'TeraTTS':rec('TeraTTS-natasha','VITS','TeraTTS/natasha-g2p-vits',mode='open_source',transfer='VITS',evidence_extra='https://huggingface.co/TeraTTS/natasha-g2p-vits',notes='Matched to RuASD TeraTTS-natasha row by system list and local N/hours.'),
'bark_small_tts':rec('Bark-small','Bark','suno/bark-small',vocoder='EnCodec decoder',mode='open_source',evidence_extra='https://huggingface.co/suno/bark-small'),
'coquiTTS':rec('Coqui XTTS-v2','XTTS-v2','coqui/XTTS-v2',task='voice_cloning',vocoder='HiFi-GAN',mode='open_source',transfer='XTTS-v2',evidence_extra='https://huggingface.co/coqui/XTTS-v2'),
'edge_tts':rec('Microsoft Edge TTS','EdgeTTS','ru-RU-DmitryNeural; ru-RU-SvetlanaNeural',vocoder='neural vocoder (undisclosed)',mode='commercial',transfer='EdgeTTS',evidence_extra='https://github.com/rany2/edge-tts'),
'espeech_podcaster':rec('ESpeech Podcaster','F5-TTS','ESpeech/ESpeech-TTS-1_podcaster',mode='open_source',transfer='F5-TTS',evidence_extra='https://huggingface.co/ESpeech/ESpeech-TTS-1_podcaster'),
'espeech_rl-v1':rec('ESpeech RL-V1','F5-TTS','ESpeech/ESpeech-TTS-1_RL-V1',mode='open_source',transfer='F5-TTS',evidence_extra='https://huggingface.co/ESpeech/ESpeech-TTS-1_RL-V1',notes='RuASD paper: RL-V1 is based on SFT-95k.'),
'espeech_rl-v2':rec('ESpeech RL-V2','F5-TTS','ESpeech-TTS-1 RL-V2',mode='open_source',transfer='F5-TTS',notes='RuASD paper: RL-V2 is based on SFT-256k; dataset-card RL-V2 link currently points to RL-V1, so no exact repo URL asserted.'),
'espeech_sft-256k':rec('ESpeech SFT-256k','F5-TTS','ESpeech/ESpeech-TTS-1_SFT-256K',mode='open_source',transfer='F5-TTS',evidence_extra='https://huggingface.co/ESpeech/ESpeech-TTS-1_SFT-256K'),
'espeech_sft-95k':rec('ESpeech SFT-95k','F5-TTS','ESpeech/ESpeech-TTS-1_SFT-95K',mode='open_source',transfer='F5-TTS',evidence_extra='https://huggingface.co/ESpeech/ESpeech-TTS-1_SFT-95K'),
'fishsTTS':rec('Fish-Speech 1.5','Fish-Speech','fishaudio/fish-speech-1.5',vocoder='Firefly-GAN decoder',mode='open_source',transfer='Fish-Speech',evidence_extra='https://huggingface.co/fishaudio/fish-speech-1.5'),
'g2pTTS':rec('FastPitch G2P pipeline','FastPitch','Russian FastPitch + G2P front-end',vocoder='HiFi-GAN',mode='open_source',transfer='FastPitch',evidence_extra='https://huggingface.co/bene-ges/ru_g2p_ipa_bert_large; https://huggingface.co/bene-ges/tts_ru_hifigan_ruslan'),
'girl_nice':rec('TeraTTS-girl_nice','VITS','TeraTTS/girl_nice-g2p-vits',mode='open_source',transfer='VITS',evidence_extra='https://huggingface.co/TeraTTS/girl_nice-g2p-vits'),
'glados':rec('TeraTTS-glados','VITS','TeraTTS/glados-g2p-vits',mode='open_source',transfer='VITS',evidence_extra='https://huggingface.co/TeraTTS/glados-g2p-vits'),
'glados2':rec('TeraTTS-glados2','VITS','TeraTTS/glados2-g2p-vits',mode='open_source',transfer='VITS',evidence_extra='https://huggingface.co/TeraTTS/glados2-g2p-vits'),
'grad_tts':rec('Grad-TTS','Grad-TTS','10 reverse-diffusion step configurations aggregated',vocoder='separate neural vocoder (unspecified)',mode='open_source',transfer='Grad-TTS',evidence_extra='https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS',notes='RuASD paper explicitly aggregates 10 step-count variants into one system (50,108 files).'),
'mishaf5TTS':rec('Misha F5-TTS Russian','F5-TTS','Misha24-10/F5-TTS_RUSSIAN',mode='open_source',transfer='F5-TTS',evidence_extra='https://huggingface.co/Misha24-10/F5-TTS_RUSSIAN'),
'mms_tts_rus':rec('MMS TTS Russian','VITS','facebook/mms-tts-rus',mode='open_source',transfer='MMS-TTS',evidence_extra='https://huggingface.co/facebook/mms-tts-rus'),
'piperTTS':rec('PiperTTS','VITS','denis; dmitri; irina; ruslan',mode='open_source',transfer='Piper',evidence_extra='https://github.com/rhasspy/piper'),
'pyx3TTS':rec('pyttsx3','pyttsx3/system backend','OS-dependent backend/voice',vocoder='backend-specific',mode='open_source',transfer='pyttsx3',evidence_extra='https://github.com/nateshmbhat/pyttsx3'),
'rfsTTS':rec('RussianFastPitch','FastPitch','RussianFastPitch',vocoder='WaveGlow',mode='open_source',transfer='FastPitch',evidence_extra='https://github.com/safonovanastya/RussianFastPitch'),
'rhvoiceTTS':rec('RHVoice','RHVoice/statistical parametric','multiple Russian RHVoice voices',mode='open_source',transfer='RHVoice',evidence_extra='https://github.com/RHVoice/RHVoice'),
'ruxTTS':rec('Russian XTTS fine-tune (RuXTTS)','XTTS-v2','NeuroDonu/RU-XTTS-DonuModel',task='voice_cloning',vocoder='HiFi-GAN',mode='open_source',transfer='XTTS-v2',evidence_extra='https://huggingface.co/NeuroDonu/RU-XTTS-DonuModel'),
'sberTTS':rec('SaluteSpeech','SaluteSpeech','cloud API / Russian voices',mode='commercial',transfer='SaluteSpeech',evidence_extra='https://developers.sber.ru/portal/products/smartspeech'),
'sf5TTS':rec('Hotstone F5-TTS Russian','F5-TTS','hotstone228/F5-TTS-Russian',mode='open_source',transfer='F5-TTS',evidence_extra='https://huggingface.co/hotstone228/F5-TTS-Russian'),
'sileroTTS':rec('Silero TTS','Silero TTS','aidar; baya; kseniya; xenia; eugene',mode='open_source',transfer='Silero',evidence_extra='https://github.com/snakers4/silero-models'),
't5TTS':rec('SpeechT5 Russian fine-tune','SpeechT5','voxxer/speecht5_finetuned_commonvoice_ru_translit',vocoder='HiFi-GAN',mode='open_source',transfer='SpeechT5',evidence_extra='https://huggingface.co/voxxer/speecht5_finetuned_commonvoice_ru_translit'),
'tts_ru_free_hf_vits_high_multispeaker':rec('VITS high multispeaker','VITS','utrobinmv/tts_ru_free_hf_vits_high_multispeaker',mode='open_source',transfer='VITS',evidence_extra='https://huggingface.co/utrobinmv/tts_ru_free_hf_vits_high_multispeaker'),
'tts_ru_free_hf_vits_low_multispeaker':rec('VITS low multispeaker','VITS','utrobinmv/tts_ru_free_hf_vits_low_multispeaker',mode='open_source',transfer='VITS',evidence_extra='https://huggingface.co/utrobinmv/tts_ru_free_hf_vits_low_multispeaker'),
'tts_ru_hifigan_ruslan':rec('FastPitch IPA + HiFi-GAN pipeline','FastPitch','bene-ges FastPitch/RUSLAN pipeline',vocoder='HiFi-GAN',mode='open_source',transfer='FastPitch',evidence_extra='https://huggingface.co/bene-ges/tts_ru_ipa_fastpitch_ruslan; https://huggingface.co/bene-ges/tts_ru_hifigan_ruslan'),
'tts_transformer-ru-cv7_css10':rec('Fairseq Transformer TTS Russian','Fairseq Transformer TTS','facebook/tts_transformer-ru-cv7_css10',vocoder='neural vocoder (unspecified)',mode='open_source',transfer='Fairseq-TTS',evidence_extra='https://huggingface.co/facebook/tts_transformer-ru-cv7_css10'),
'tts_vits_ru_hf':rec('Russian VITS checkpoint','VITS','joefox/tts_vits_ru_hf',mode='open_source',transfer='VITS',evidence_extra='https://huggingface.co/joefox/tts_vits_ru_hf'),
'vits2TTS':rec('VITS2 Russian Natasha','VITS2','frappuccino/vits2_ru_natasha',mode='open_source',transfer='VITS2',evidence_extra='https://huggingface.co/frappuccino/vits2_ru_natasha'),
'vkTTS':rec('VK Cloud Voice','VK Cloud Voice','cloud API / six speakers',mode='commercial',transfer='VK-Cloud-TTS',evidence_extra='https://cloud.vk.com/'),
'vosk-gpt-sovits':rec('Vosk GPT-SoVITS checkpoint','GPT-SoVITS','alphacep/vosk-tts-ru-gpt-sovits',task='voice_cloning',vocoder='SoVITS decoder',mode='open_source',transfer='GPT-SoVITS',evidence_extra='https://huggingface.co/alphacep/vosk-tts-ru-gpt-sovits'),
'voskTTS':rec('Vosk-TTS','Vosk-TTS','five selectable speakers',vocoder='vocoder (unspecified)',mode='open_source',transfer='Vosk-TTS',evidence_extra='https://github.com/alphacep/vosk-tts'),
'xtts_ru_ipa':rec('XTTS Russian IPA fine-tune','XTTS-v2','omogr/xtts-ru-ipa',task='voice_cloning',vocoder='HiFi-GAN',mode='open_source',transfer='XTTS-v2',evidence_extra='https://huggingface.co/omogr/xtts-ru-ipa'),
}

path=Path('data_inventory/generator_ontology.csv')
if not path.exists():
    raise SystemExit(f'Missing {path}')
backup=path.with_name('generator_ontology_skeleton.csv')
if not backup.exists(): shutil.copy2(path,backup)
df=pd.read_csv(path,low_memory=False)
# Empty text columns are inferred by pandas as float64 because they contain only NaN.
# Cast all ontology text fields to object before assigning strings.
_text_cols=[
    'generator_name','base_model_family','checkpoint_or_voice','task_type','vocoder',
    'open_source_or_commercial','family_label_for_transfer','evidence','notes','ontology_status'
]
for _col in _text_cols:
    if _col not in df.columns:
        df[_col] = ''
    df[_col] = df[_col].astype('object')
labels=set(df.ruasd_label.astype(str))
if labels!=set(M):
    raise SystemExit(f'Mapping mismatch. missing={sorted(labels-set(M))}, extra={sorted(set(M)-labels)}')
for label,vals in M.items():
    mask=df.ruasd_label.astype(str).eq(label)
    for col,val in vals.items(): df.loc[mask,col]=val
    df.loc[mask,'ontology_status']='evidence_mapped_v1'
df.to_csv(path,index=False,encoding='utf-8-sig')
print('Updated:',path)
print('Backup: ',backup)
print('Rows:',len(df))
print('\nFamily counts:')
print(df.base_model_family.value_counts().to_string())
print('\nTask types:')
print(df.task_type.value_counts().to_string())
