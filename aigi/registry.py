"""Verified dataset/model identifiers. Counts are card descriptions, not download promises."""
DATASETS = {
    'kodak': dict(repo='danjacobellis/kodak', split='validation', image='image', prompt=None, kind='natural', loader='hf', count=24, evaluation_only=True),
    'div2k_train': dict(repo='yangtao9009/DIV2K', split='train', image='image', prompt=None, kind='natural', loader='hf', count=800),
    'lsdir': dict(repo='danjacobellis/LSDIR', split='train', image='image', prompt=None, kind='natural', loader='hf', count='about 85K'),
    'mlic100k': dict(repo='Whiteboat/MLIC-Train-100K', split='train', image='image', prompt=None, kind='natural', loader='7z', count='100K', patterns=['train512x512.7z.*']),
    'diffusiondb_1k': dict(repo='poloclub/diffusiondb', split='train', image='image', prompt='p', kind='aigi', loader='diffusiondb', count=1000, first=1, last=1, large=False),
    'diffusiondb_2m': dict(repo='poloclub/diffusiondb', split='train', image='image', prompt='p', kind='aigi', loader='diffusiondb', count=2000000, first=1, last=2000, large=False),
    'diffusiondb_14m': dict(repo='poloclub/diffusiondb', split='train', image='image', prompt='p', kind='aigi', loader='diffusiondb', count=14000000, first=1, last=14000, large=True),
    'diffusiondb_filtered': dict(repo='whosouravsharma/text-to-image-diffusiondb-2M', split='train', image='image', prompt='prompt', kind='aigi', loader='hf', count='18,219 across splits, not 2M'),
    'sdxl10k': dict(repo='ash12321/sdxl-generated-10k', split='train', image='image', prompt=None, kind='aigi', loader='hf', count=10000),
    'multi_generator': dict(repo='Shanmuk4622/ai-image-detection-dataset', split='train', image='image', prompt='prompt', kind='mixed', loader='hf', count='card: 10K real + 6 x 10K generated'),
}
MODELS = {
    'sana': 'Efficient-Large-Model/Sana_600M_1024px_diffusers',
    'ditic': 'JunqiShi/DiT-IC',
    'caption': 'Salesforce/blip-image-captioning-base',
}
