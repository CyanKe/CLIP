"""
文本描述模板模块 - 用于生成雷达干扰信号的 CLIP 文本描述
支持 3 种描述策略:
1. simple: 简单类别名，如 "DFTJ"
2. template: 模板描述，如 "DFTJ looks like dense false targets"
3. meta: 结合 metadata 的动态描述，如 "DFTJ with JNR=10dB, k=5"
"""

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


def get_meta_description(metadata: dict) -> str:
    """
    生成结合 metadata 的动态描述

    Args:
        metadata: metadata 字典，包含 jam_types, JNR, jam_params 等
                  jam_types 格式支持: "DFTJ", ["DFTJ"], ["DFTJ", "AJ"]

    Returns:
        动态描述，如 "DFTJ with JNR=10dB, k=5 false targets"
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
        name = get_jam_type_name(jam_type)
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


def generate_text_descriptions(metadata: dict = None, style: str = 'all') -> dict:
    """
    生成文本描述（支持 3 种策略）

    Args:
        metadata: metadata 字典（可选，用于 meta 风格）
        style: 描述风格
            - 'simple': 简单类别名
            - 'template': 模板描述
            - 'meta': 结合 metadata 的动态描述
            - 'all': 返回所有风格

    Returns:
        文本描述（str 或 dict）
    """
    if style == 'simple':
        if metadata and 'jam_types' in metadata:
            jam_types = _normalize_jam_types(metadata.get('jam_types'))
            if jam_types:
                return get_simple_description(jam_types[0])
        return "a radar signal"

    elif style == 'template':
        if metadata and 'jam_types' in metadata:
            jam_types = _normalize_jam_types(metadata.get('jam_types'))
            if jam_types:
                return get_template_description(jam_types[0])
        return "a radar signal with jamming"

    elif style == 'meta':
        if metadata:
            return get_meta_description(metadata)
        return "a radar signal with jamming"

    elif style == 'all':
        result = {
            'simple': 'a radar signal',
            'template': 'a radar signal with jamming',
            'meta': 'a radar signal with jamming',
        }
        if metadata and 'jam_types' in metadata:
            jam_types = _normalize_jam_types(metadata.get('jam_types'))
            if jam_types:
                result['simple'] = get_simple_description(jam_types[0])
                result['template'] = get_template_description(jam_types[0])
        if metadata:
            result['meta'] = get_meta_description(metadata)
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
