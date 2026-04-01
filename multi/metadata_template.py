"""
Metadata模板设计 - 支持简短描述

核心思路：
1. 视觉特征描述：{Class Name} looks like {visual description}
2. 参数描述：a radar signal with {Class Name} at {strength} jamming power
3. 每句描述包含干扰类型名，便于识别
"""

import numpy as np

# =============================================================================
# 干扰类型名称映射
# =============================================================================

JAM_TYPE_NAMES = {
    1: 'DFTJ', 2: 'ISRJ', 3: 'RGPO', 4: 'VGPO',
    5: 'AJ', 6: 'BJ', 7: 'SJ', 8: 'NCJ',
    9: 'NPJ', 10: 'SMSPJ', 11: 'C&IJ', 12: 'NFMJ',
    13: 'NPMJ', 14: 'NAMJ', 15: 'CSJ', 16: 'PJ'
}

# =============================================================================
# 视觉特征描述模板 - {Class Name} looks like {visual description}
# =============================================================================

# 视觉特征模板：每种干扰类型的STFT图外观描述
VISUAL_TEMPLATES = {
    1: {  # DFTJ
        'base': 'short slanted lines in time-frequency domain',
        'param': {  # 根据假目标数量
            'few': 'with a few false targets',
            'several': 'with several false targets',
            'many': 'with many false targets'
        }
    },
    2: {  # ISRJ
        'base': 'discontinuous signal slices',
        'param': {
            'simple': 'with simple sampling pattern',
            'moderate': 'with moderate sampling pattern',
            'complex': 'with complex sampling pattern'
        }
    },
    3: {  # RGPO
        'base': 'false targets in range domain',
        'param': {
            'after': 'trailing behind real target',
            'before': 'leading ahead of real target',
            'overlap': 'overlapping with real target'
        }
    },
    4: {  # VGPO
        'base': 'doppler shift pattern',
        'param': {
            'up': 'with increasing frequency shift',
            'down': 'with decreasing frequency shift'
        }
    },
    5: {  # AJ - 瞄准干扰
        'base': 'a narrow horizontal noise band in the middle',
        'param': {}
    },
    6: {  # BJ - 阻塞干扰
        'base': 'a wide horizontal noise band in the middle',
        'param': {}
    },
    7: {  # SJ - 扫频干扰
        'base': 'multiple diagonal narrow interference bands',
        'param': {}
    },
    8: {  # NCJ - 噪声卷积干扰
        'base': 'a vertical narrow noise band with higher energy in the middle',
        'param': {}
    },
    9: {  # NPJ - 噪声乘积干扰
        'base': 'a vertical narrow noise band with uniform energy',
        'param': {}
    },
    10: {  # SMSPJ
        'base': 'steep diagonal lines',
        'param': {
            'few': 'with a few steep traces',
            'several': 'with several steep traces',
            'many': 'with many steep traces'
        }
    },
    11: {  # C&IJ
        'base': 'interleaved signal segments',
        'param': {
            'continuous': 'in continuous pattern',
            'discontinuous': 'in discontinuous pattern'
        }
    },
    12: {  # NFMJ - 噪声调频干扰
        'base': 'gradual energy decay from center without clear boundary',
        'param': {}
    },
    13: {  # NPMJ - 噪声调相干扰
        'base': 'a jittering horizontal high energy band in the middle',
        'param': {}
    },
    14: {  # NAMJ - 噪声调幅干扰
        'base': 'a flat horizontal high energy band with surrounding noise',
        'param': {}
    },
    15: {  # CSJ
        'base': 'comb-like diagonal lines',
        'param': {
            'few': 'with a few comb teeth',
            'several': 'with several comb teeth',
            'many': 'with many comb teeth'
        }
    },
    16: {  # PJ - 脉冲干扰
        'base': 'discontinuous horizontal lines in the middle',
        'param': {}
    },
}

# =============================================================================
# 参数映射函数
# =============================================================================

def get_jnr_level(jnr: float) -> str:
    """获取JNR强度等级"""
    if jnr < 5:
        return 'weak'
    elif jnr < 15:
        return 'moderate'
    elif jnr < 25:
        return 'strong'
    else:
        return 'very strong'


def get_count_level(count: int) -> str:
    """获取数量等级"""
    if count <= 4:
        return 'few'
    elif count <= 6:
        return 'several'
    else:
        return 'many'


def get_param_key(jam_type: int, jam_params: dict) -> str:
    """根据干扰类型和参数获取对应的param key"""
    if jam_type == 1:  # DFTJ
        k = jam_params.get('dftj_k', 5)
        return get_count_level(k)

    elif jam_type == 2:  # ISRJ
        M = jam_params.get('isrj_M', 3)
        if M <= 2:
            return 'simple'
        elif M <= 3:
            return 'moderate'
        else:
            return 'complex'

    elif jam_type == 3:  # RGPO
        pos_rel = jam_params.get('rgpo_position_relation', 'after')
        # 处理可能的bytes类型
        if isinstance(pos_rel, bytes):
            pos_rel = pos_rel.decode('utf-8')
        return pos_rel

    elif jam_type == 4:  # VGPO
        dop_dir = jam_params.get('vgpo_doppler_direction', 'up')
        if isinstance(dop_dir, bytes):
            dop_dir = dop_dir.decode('utf-8')
        return dop_dir

    elif jam_type == 10:  # SMSPJ
        M = jam_params.get('smspj_M', 5)
        return get_count_level(M)

    elif jam_type == 11:  # C&IJ
        is_cont = jam_params.get('cij_is_continuous', True)
        return 'continuous' if is_cont else 'discontinuous'

    elif jam_type == 15:  # CSJ
        M = jam_params.get('csj_M', 5)
        return get_count_level(M)

    return ''


# =============================================================================
# 描述生成函数
# =============================================================================

def generate_visual_description(jam_type: int, jam_params: dict) -> str:
    """
    生成视觉特征描述

    格式: {Class Name} looks like {visual description}

    Args:
        jam_type: 干扰类型编号
        jam_params: 干扰参数字典

    Returns:
        视觉特征描述字符串
    """
    class_name = JAM_TYPE_NAMES.get(jam_type, f'Type{jam_type}')
    template = VISUAL_TEMPLATES.get(jam_type, {'base': 'unknown pattern', 'param': {}})

    base_desc = template['base']
    param_templates = template.get('param', {})

    # 获取参数对应的描述
    param_key = get_param_key(jam_type, jam_params)
    param_desc = param_templates.get(param_key, '')

    # 组合描述
    if param_desc:
        return f"{class_name} looks like {base_desc} {param_desc}"
    else:
        return f"{class_name} looks like {base_desc}"


def generate_param_description(jam_type: int, jnr: float) -> str:
    """
    生成参数描述（强度）

    格式: a radar signal with {Class Name} at {strength} jamming power

    Args:
        jam_type: 干扰类型编号
        jnr: 干噪比

    Returns:
        参数描述字符串
    """
    class_name = JAM_TYPE_NAMES.get(jam_type, f'Type{jam_type}')
    jnr_level = get_jnr_level(jnr)

    return f"a radar signal with {class_name} at {jnr_level} jamming power"


def generate_short_description(metadata: dict, style: str = 'visual') -> str:
    """
    生成简短描述

    Args:
        metadata: 样本metadata字典
        style: 描述风格
            - 'visual': 视觉特征描述 (推荐) - {Class Name} looks like {description}
            - 'param': 参数描述 - a radar signal with {Class Name} at {strength}
            - 'visual_param': 视觉+参数组合

    Returns:
        简短描述字符串
    """
    jam_types = metadata.get('jam_types', [])
    JNR = metadata.get('JNR', 15)
    jam_params = metadata.get('jam_params', {})

    if not jam_types:
        return "a radar signal with no jamming"

    if style == 'random':
        style = np.random.choice(['visual', 'param', 'visual_param'])

    # 单干扰类型
    if len(jam_types) == 1:
        jam_type = jam_types[0]

        if style == 'visual':
            return generate_visual_description(jam_type, jam_params)

        elif style == 'param':
            return generate_param_description(jam_type, JNR)

        else:  # visual_param
            visual_desc = generate_visual_description(jam_type, jam_params)
            jnr_level = get_jnr_level(JNR)
            return f"{visual_desc}, at {jnr_level} jamming power"

    # 组合干扰
    else:
        class_names = [JAM_TYPE_NAMES.get(t, f'Type{t}') for t in jam_types]

        if style == 'visual':
            # 每个干扰类型的视觉描述
            visual_parts = []
            for jam_type in jam_types:
                visual_parts.append(generate_visual_description(jam_type, jam_params))
            return ', '.join(visual_parts)

        elif style == 'param':
            combined_names = ' and '.join(class_names)
            jnr_level = get_jnr_level(JNR)
            return f"a radar signal with combined {combined_names} at {jnr_level} jamming power"

        else:  # visual_param
            visual_parts = []
            for jam_type in jam_types:
                visual_parts.append(generate_visual_description(jam_type, jam_params))
            jnr_level = get_jnr_level(JNR)
            return f"{', '.join(visual_parts)}, at {jnr_level} jamming power"


# 主函数：保持向后兼容
def generate_abstract_description(metadata: dict) -> str:
    """
    根据metadata生成简短描述（适合CLIP理解）
    """
    return generate_short_description(metadata, style='visual')


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("新描述模板测试")
    print("=" * 70)

    test_cases = [
        # 欺骗干扰
        {'jam_types': [1], 'JNR': 5, 'jam_params': {'dftj_k': 4}},
        {'jam_types': [1], 'JNR': 20, 'jam_params': {'dftj_k': 7}},
        {'jam_types': [2], 'JNR': 15, 'jam_params': {'isrj_M': 3}},
        {'jam_types': [3], 'JNR': 15, 'jam_params': {'rgpo_position_relation': 'after'}},
        {'jam_types': [4], 'JNR': 25, 'jam_params': {'vgpo_doppler_direction': 'up'}},
        {'jam_types': [10], 'JNR': 15, 'jam_params': {'smspj_M': 6}},
        {'jam_types': [11], 'JNR': 15, 'jam_params': {'cij_is_continuous': True}},
        {'jam_types': [15], 'JNR': 15, 'jam_params': {'csj_M': 5}},
        # 压制干扰
        {'jam_types': [5], 'JNR': 15, 'jam_params': {}},  # AJ
        {'jam_types': [6], 'JNR': 15, 'jam_params': {}},  # BJ
        {'jam_types': [7], 'JNR': 15, 'jam_params': {}},  # SJ
        {'jam_types': [8], 'JNR': 15, 'jam_params': {}},  # NCJ
        {'jam_types': [9], 'JNR': 15, 'jam_params': {}},  # NPJ
        {'jam_types': [12], 'JNR': 15, 'jam_params': {}},  # NFMJ
        {'jam_types': [13], 'JNR': 15, 'jam_params': {}},  # NPMJ
        {'jam_types': [14], 'JNR': 15, 'jam_params': {}},  # NAMJ
        {'jam_types': [16], 'JNR': 15, 'jam_params': {}},  # PJ
        # 组合干扰
        {'jam_types': [1, 5], 'JNR': 20, 'jam_params': {'dftj_k': 5}},
        {'jam_types': [3, 5], 'JNR': 15, 'jam_params': {'rgpo_position_relation': 'before'}},
    ]

    for i, meta in enumerate(test_cases):
        jam_names = [JAM_TYPE_NAMES[t] for t in meta['jam_types']]
        print(f"\n{i+1}. {' + '.join(jam_names)}:")
        print(f"   visual:       {generate_short_description(meta, 'visual')}")
        print(f"   param:        {generate_short_description(meta, 'param')}")
        print(f"   visual_param: {generate_short_description(meta, 'visual_param')}")