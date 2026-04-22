"""
文本描述模板模块 - 用于生成雷达干扰信号的 CLIP 文本描述
支持 3 种描述策略:
1. simple: 简单类别名，如 "DFTJ"
2. template: 模板描述，如 "DFTJ looks like dense false targets"
3. meta: 结合 metadata 的动态描述，如 "DFTJ with JNR=10dB, k=5"
"""

import yaml
from pathlib import Path

# ============================================================================
# 类别名称翻译字典（从 config.yaml 加载）
# ============================================================================
_LABEL_TRANSLATIONS = None

def load_label_translations(config_path: str = None) -> dict:
    """
    从配置文件加载类别名称翻译字典

    Args:
        config_path: 配置文件路径，默认为 multi/config.yaml

    Returns:
        翻译字典，如 {"DFTJ": "Dense False Target Jamming", ...}
    """
    global _LABEL_TRANSLATIONS

    if _LABEL_TRANSLATIONS is not None:
        return _LABEL_TRANSLATIONS

    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        _LABEL_TRANSLATIONS = config.get("label_translations", {})
    except Exception as e:
        print(f"Warning: Failed to load label translations: {e}")
        _LABEL_TRANSLATIONS = {}

    return _LABEL_TRANSLATIONS


def get_translated_name(jam_type, use_translation: bool = True) -> str:
    """
    获取干扰类型的翻译名称

    Args:
        jam_type: 干扰类型名称（如 "DFTJ"）
        use_translation: 是否使用翻译后的名称

    Returns:
        翻译后的名称（如 "Dense False Target Jamming"）或原始名称
    """
    name = get_jam_type_name(jam_type) if not isinstance(jam_type, str) else jam_type

    if use_translation:
        translations = load_label_translations()
        return translations.get(name, name)

    return name


# ============================================================================
# 干扰类型名称映射
# ============================================================================
JAM_TYPE_NAMES = {
    1: 'DFTJ',   # 离散频率调制干扰
    2: 'ISRJ',   # 间歇采样转发干扰
    3: 'RGPO',   # 距离门拖引干扰
    4: 'VGPO',   # 速度门拖引干扰
    5: 'AJ',     # 瞄准式压制干扰
    6: 'BJ',     # 阻塞式压制干扰
    7: 'SJ',     # 扫频式干扰
    8: 'NCJ',    # 噪声调频干扰
    9: 'NPJ',    # 噪声调相干扰
    10: 'SMSPJ', # 平滑伪 Wigner-Ville 分布干扰
    11: 'C&IJ',  # 切割与 interleaved 干扰
    12: 'NFMJ',  # 噪声调频干扰
    13: 'NPMJ',  # 噪声调相干扰
    14: 'NAMJ',  # 噪声调幅干扰
    15: 'CSJ',   # 组合谱干扰
    16: 'PJ',    # 相位编码干扰
}

# 反向映射：名称 -> ID
JAM_NAME_TO_ID = {v: k for k, v in JAM_TYPE_NAMES.items()}

# ============================================================================
# 视觉特征模板 (template 风格)
# ============================================================================
VISUAL_TEMPLATES = {
    1: {'base': 'dense false targets with diagonal lines', 'param': 'k'},
    2: {'base': 'discontinuous signal slices', 'param': 'M'},
    3: {'base': 'false targets in range domain', 'param': 'position'},
    4: {'base': 'doppler shift pattern', 'param': 'doppler'},
    5: {'base': 'concentrated frequency energy', 'param': None},
    6: {'base': 'wide frequency coverage', 'param': None},
    7: {'base': 'sweeping frequency pattern', 'param': None},
    8: {'base': 'noise-like frequency modulation', 'param': None},
    9: {'base': 'noise-like phase modulation', 'param': None},
    10: {'base': 'steep diagonal lines', 'param': 'M'},
    11: {'base': 'continuous signal segments', 'param': 'pattern'},
    12: {'base': 'noise frequency modulation', 'param': None},
    13: {'base': 'noise phase modulation', 'param': None},
    14: {'base': 'noise amplitude modulation', 'param': None},
    15: {'base': 'comb-like diagonal lines', 'param': 'M'},
    16: {'base': 'phase coded pattern', 'param': None},
}

# ============================================================================
# ImageNet 风格模板 (用于增强文本多样性)
# ============================================================================
IMAGENET_TEMPLATES = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]


def get_jam_type_name(jam_type) -> str:
    """
    获取干扰类型名称（支持整数或字符串输入）

    Args:
        jam_type: 干扰类型编号 (1-16) 或名称字符串 ("DFTJ")

    Returns:
        干扰类型名称
    """
    if isinstance(jam_type, str):
        # 已经是字符串名称，直接返回
        return jam_type
    elif isinstance(jam_type, int):
        # 整数编号，转换为名称
        return JAM_TYPE_NAMES.get(jam_type, f'Type{jam_type}')
    else:
        return f'Type{jam_type}'


def get_jam_type_id(jam_type) -> int:
    """
    获取干扰类型编号（支持整数或字符串输入）

    Args:
        jam_type: 干扰类型编号 (1-16) 或名称字符串 ("DFTJ")

    Returns:
        干扰类型编号，找不到返回 None
    """
    if isinstance(jam_type, int):
        return jam_type
    elif isinstance(jam_type, str):
        return JAM_NAME_TO_ID.get(jam_type)
    return None


def get_simple_description(jam_type) -> str:
    """
    生成简单类别名描述

    Args:
        jam_type: 干扰类型编号 (1-16) 或名称字符串 ("DFTJ")

    Returns:
        简单类别名，如 "DFTJ"
    """
    return get_jam_type_name(jam_type)


def get_template_description(jam_type) -> str:
    """
    生成模板描述

    Args:
        jam_type: 干扰类型编号 (1-16) 或名称字符串 ("DFTJ")

    Returns:
        模板描述，如 "DFTJ looks like dense false targets with diagonal lines"
    """
    name = get_jam_type_name(jam_type)
    jam_id = get_jam_type_id(jam_type)
    template = VISUAL_TEMPLATES.get(jam_id, {'base': 'unknown pattern'}) if jam_id else {'base': 'unknown pattern'}
    return f"{name} looks like {template['base']}"


def get_combined_template_description(jam_types: list) -> str:
    """
    生成组合干扰的模板描述

    Args:
        jam_types: 干扰类型列表，如 ["DFTJ", "ISRJ"]

    Returns:
        模板描述，如 "DFTJ and ISRJ combined: dense false targets with discontinuous signal slices"
    """
    if not jam_types:
        return "a radar signal with unknown jamming"

    if len(jam_types) == 1:
        return get_template_description(jam_types[0])

    # 组合情况：拼接各类型的名称和特征
    names = [get_jam_type_name(jt) for jt in jam_types]
    features = []
    for jt in jam_types:
        jam_id = get_jam_type_id(jt)
        template = VISUAL_TEMPLATES.get(jam_id, {'base': 'unknown pattern'}) if jam_id else {'base': 'unknown pattern'}
        features.append(template['base'])

    return f"{' and '.join(names)} combined: {', '.join(features)}"


def get_inference_description(jam_types: list, use_translation: bool = False) -> str:
    """
    生成推理用的描述（与训练格式一致，但不包含具体参数）

    Args:
        jam_types: 干扰类型列表，如 ["DFTJ"] 或 ["DFTJ", "ISRJ"]
        use_translation: 是否使用翻译后的名称（如 "Dense False Target Jamming"）

    Returns:
        描述字符串，如 "a radar signal with single jamming: CSJ" 或
        "a radar signal with combined jamming: CSJ, DFTJ" 或
        "a radar signal with single jamming: Dense False Target Jamming" (启用翻译时)
    """
    if not jam_types:
        return "a radar signal with unknown jamming"

    names = [get_translated_name(jt, use_translation) for jt in jam_types]

    if len(names) == 1:
        return f"a radar signal with single jamming: {names[0]}"
    else:
        return f"a radar signal with combined jamming: {', '.join(names)}"


def get_class_only_description(metadata: dict, use_translation: bool = False) -> str:
    """
    生成只包含干扰类型的描述（不包含 JNR 等参数）

    这个风格用于训练，确保同一干扰类型的所有样本共享相同的文本描述，
    避免因 JNR 不同而导致特征空间分裂。

    Args:
        metadata: metadata 字典，包含 jam_types 等
        use_translation: 是否使用翻译后的名称

    Returns:
        描述字符串，如 "a radar signal with single jamming: DFTJ"
    """
    jam_types = metadata.get('jam_types', [])

    # 确保 jam_types 是列表
    if isinstance(jam_types, str):
        jam_types = [jam_types] if jam_types else []
    elif not isinstance(jam_types, list):
        jam_types = []

    if not jam_types:
        return "a radar signal with unknown jamming"

    # 使用推理描述格式，保持一致性
    return get_inference_description(jam_types, use_translation=use_translation)


def get_meta_description(metadata: dict, use_translation: bool = False) -> str:
    """
    生成结合 metadata 的动态描述

    Args:
        metadata: metadata 字典，包含 jam_types, JNR, jam_params 等
                  jam_types 格式支持: "DFTJ", ["DFTJ"], ["DFTJ", "AJ"]
        use_translation: 是否使用翻译后的名称（如 "Dense False Target Jamming"）

    Returns:
        动态描述，如 "DFTJ with JNR=10dB, k=5 false targets" 或
        "Dense False Target Jamming with JNR=10dB, k=5 false targets" (启用翻译时)
    """
    jam_types = metadata.get('jam_types', [])
    jam_params = metadata.get('jam_params', {})
    jnr = metadata.get('JNR', None)

    # 确保 jam_types 是列表
    if isinstance(jam_types, str):
        jam_types = [jam_types] if jam_types else []
    elif not isinstance(jam_types, list):
        jam_types = []

    if not jam_types:
        return "a radar signal with unknown jamming"

    descriptions = []

    for jam_type in jam_types:
        name = get_translated_name(jam_type, use_translation)
        jam_id = get_jam_type_id(jam_type)
        template = VISUAL_TEMPLATES.get(jam_id, {'base': 'unknown pattern', 'param': None}) if jam_id else {'base': 'unknown pattern', 'param': None}

        # 构建描述
        parts = [name]

        # 添加 JNR 信息
        if jnr is not None:
            parts.append(f"JNR={jnr}dB")

        # 添加参数信息
        param_key = template.get('param')
        if param_key and param_key in jam_params:
            param_value = jam_params[param_key]
            if param_key == 'k':
                parts.append(f"{param_value} false targets")
            elif param_key == 'M':
                parts.append(f"M={param_value}")
            elif param_key == 'position':
                parts.append(f"position={param_key}")
            else:
                parts.append(f"{param_key}={param_value}")

        # 添加视觉特征
        parts.append(f"showing {template['base']}")

        descriptions.append(' with '.join(parts[:2]) if len(parts) > 1 else parts[0])

    if len(descriptions) == 1:
        return f"a radar signal with single jamming: {descriptions[0]}"
    else:
        return f"a radar signal with combined jamming: {', '.join(descriptions)}"


def get_imagenet_descriptions(jam_type, context: str = "radar signal") -> list:
    """
    用 ImageNet 模板生成多样化描述

    Args:
        jam_type: 干扰类型编号或名称
        context: 上下文描述，默认 "radar signal"

    Returns:
        描述列表，如 ["a photo of a radar signal with DFTJ.", ...]
    """
    if jam_type is None:
        subject = context
    else:
        name = get_jam_type_name(jam_type)
        subject = f"{context} with {name}"
    return [t.format(subject) for t in IMAGENET_TEMPLATES]


def _normalize_jam_types(jam_types) -> list:
    """
    规范化 jam_types 为列表格式

    Args:
        jam_types: "DFTJ", ["DFTJ"], [1], 或 None

    Returns:
        列表格式，如 ["DFTJ"] 或 [1]
    """
    if jam_types is None:
        return []
    if isinstance(jam_types, str):
        return [jam_types] if jam_types else []
    if isinstance(jam_types, list):
        return jam_types
    return []


def generate_text_descriptions(metadata: dict = None, style: str = 'all', use_translation: bool = False):
    """
    生成文本描述（支持多种策略）

    Args:
        metadata: metadata 字典（可选，用于 meta 风格）
        style: 描述风格
            - 'simple': 简单类别名
            - 'template': 模板描述
            - 'meta': 结合 metadata 的动态描述（包含 JNR 等参数）
            - 'class_only': 只包含干扰类型的描述（推荐用于训练，不含 JNR）
            - 'imagenet': ImageNet 风格多样化描述
            - 'all': 返回所有风格
        use_translation: 是否使用翻译后的名称（如 "Dense False Target Jamming"）

    Returns:
        文本描述（str 或 dict）
    """
    if style == 'simple':
        if metadata and 'jam_types' in metadata:
            jam_types = _normalize_jam_types(metadata.get('jam_types'))
            if jam_types:
                return get_translated_name(jam_types[0], use_translation)
        return "a radar signal"

    elif style == 'template':
        if metadata and 'jam_types' in metadata:
            jam_types = _normalize_jam_types(metadata.get('jam_types'))
            if jam_types:
                return get_combined_template_description(jam_types)
        return "a radar signal with jamming"

    elif style == 'meta':
        if metadata:
            return get_meta_description(metadata, use_translation=use_translation)
        return "a radar signal with jamming"

    elif style == 'class_only':
        # 推荐用于训练：只包含干扰类型，不包含 JNR
        if metadata:
            return get_class_only_description(metadata, use_translation=use_translation)
        return "a radar signal with unknown jamming"

    elif style == 'imagenet':
        if metadata and 'jam_types' in metadata:
            jam_types = _normalize_jam_types(metadata.get('jam_types'))
            if jam_types:
                return get_imagenet_descriptions(jam_types[0])
        return get_imagenet_descriptions(None, context="radar signal with unknown jamming")

    elif style == 'all':
        result = {
            'simple': 'a radar signal',
            'template': 'a radar signal with jamming',
            'meta': 'a radar signal with jamming',
            'class_only': 'a radar signal with unknown jamming',
        }
        if metadata and 'jam_types' in metadata:
            jam_types = _normalize_jam_types(metadata.get('jam_types'))
            if jam_types:
                result['simple'] = get_translated_name(jam_types[0], use_translation)
                result['template'] = get_template_description(jam_types[0])
        if metadata:
            result['meta'] = get_meta_description(metadata, use_translation=use_translation)
            result['class_only'] = get_class_only_description(metadata, use_translation=use_translation)
        return result

    return "a radar signal"


if __name__ == "__main__":
    # 测试文本生成

    # 测试 1: 无 metadata
    print("Test 1: No metadata")
    print(f"  simple: {generate_text_descriptions(style='simple')}")
    print(f"  template: {generate_text_descriptions(style='template')}")
    print(f"  meta: {generate_text_descriptions(style='meta')}")
    print()

    # 测试 2: 整数格式 jam_types（旧格式）
    print("Test 2: Integer format jam_types (old format)")
    meta1 = {
        'jam_types': [1],
        'JNR': 10,
        'jam_params': {'dftj_k': 5},
    }
    result = generate_text_descriptions(metadata=meta1, style='all')
    for style, text in result.items():
        print(f"  {style}: {text}")
    print()

    # 测试 3: 字符串格式 jam_types（新格式 - 列表）
    print("Test 3: String format jam_types (list)")
    meta2 = {
        'jam_types': ['DFTJ'],
        'JNR': 10,
        'jam_params': {'k': 5},
    }
    result = generate_text_descriptions(metadata=meta2, style='all')
    for style, text in result.items():
        print(f"  {style}: {text}")
    print()

    # 测试 4: 字符串格式 jam_types（新格式 - 单个字符串）
    print("Test 4: String format jam_types (single string)")
    meta_single = {
        'jam_types': 'AJ',  # 单个字符串，不是列表
        'JNR': 10,
    }
    result = generate_text_descriptions(metadata=meta_single, style='all')
    for style, text in result.items():
        print(f"  {style}: {text}")
    print()

    # 测试 5: 组合干扰（字符串格式）
    print("Test 5: Combined jamming (string format)")
    meta3 = {
        'jam_types': ['DFTJ', 'AJ'],
        'JNR': 15,
        'jam_params': {'k': 3},
    }
    result = generate_text_descriptions(metadata=meta3, style='all')
    for style, text in result.items():
        print(f"  {style}: {text}")
