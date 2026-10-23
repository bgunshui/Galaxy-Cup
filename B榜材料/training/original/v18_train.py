# -*- coding: utf-8 -*-
"""V11 训练入口：解码器专用训练（v17_train 核心，no_lm 模式：冻结 LoRA，只训 dec/season）
    MIX_MANIFEST / OFF_CACHE / EXT_CACHE 环境变量含义同 v9_train.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v9_lib
import v17_train

MAN = os.environ.get('MIX_MANIFEST', 'area_qa_patch_v2/manifest.json')
OFF = os.environ.get('OFF_CACHE', 'code_cache_scl')
EXT = os.environ.get('EXT_CACHE', 'ext_cache')

print('=== V18 混合训练/留出评估（v17 核心）===')
print('  MIX_MANIFEST =', MAN)
print('  OFF_CACHE    =', OFF)
print('  EXT_CACHE    =', EXT, flush=True)
v17_train.CodeDS4 = lambda split: v9_lib.CodeDSMix(split, MAN, OFF, EXT)
v17_train.main()
