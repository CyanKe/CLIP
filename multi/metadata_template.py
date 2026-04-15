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

# 反向映射：名称 -> ID
JAM_NAME_TO_ID = {v: k for k, v in JAM_TYPE_NAMES.items()}


# =============================================================================
# 格式规范化辅助函数
# =============================================================================

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
    if isinstance(jam_types, int):
        return [jam_types]
    if isinstance(jam_types, list):
        return jam_types
    return []


def _get_jam_type_name(jam_type) -> str:
    """
    获取干扰类型名称（支持整数或字符串输入）

    Args:
        jam_type: 干扰类型编号 (1-16) 或名称字符串 ("DFTJ")

    Returns:
        干扰类型名称
    """
    if isinstance(jam_type, str):
        return jam_type
    elif isinstance(jam_type, int):
        return JAM_TYPE_NAMES.get(jam_type, f'Type{jam_type}')
    else:
        return f'Type{jam_type}'


def _get_jam_type_id(jam_type) -> int:
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
    # === 扩展: AJ 带宽描述 ===
    5: {  # AJ - 瞄准干扰
        'base': 'a horizontal noise band in the middle',
        'param': {
            'narrow': 'with narrow bandwidth',      # < 30 MHz
            'moderate': 'with moderate bandwidth',  # 30-80 MHz
            'wide': 'with wide bandwidth'           # > 80 MHz
        }
    },
    # === 扩展: BJ 带宽描述 ===
    6: {  # BJ - 阻塞干扰
        'base': 'a wide horizontal noise band in the middle',
        'param': {
            'narrow': 'with narrow bandwidth',
            'moderate': 'with moderate bandwidth',
            'wide': 'with wide bandwidth'
        }
    },
    # === 扩展: SJ 带宽描述 ===
    7: {  # SJ - 扫频干扰
        'base': 'multiple diagonal interference bands',
        'param': {
            'narrow': 'with narrow bandwidth',
            'moderate': 'with moderate bandwidth',
            'wide': 'with wide bandwidth'
        }
    },
    8: {  # NCJ - 噪声卷积干扰
        'base': 'a vertical narrow noise band with higher energy in the middle',
        'param': {}
    },
    9: {  # NPJ - 噪声乘积干扰
        'base': 'a vertical narrow noise band with uniform energy',
        'param': {}
    },
    # === 扩展: SMSPJ 数量+斜率组合描述 ===
    10: {  # SMSPJ
        'base': 'steep diagonal lines',
        'param': {
            # 数量描述
            'few': 'with a few steep traces',
            'several': 'with several steep traces',
            'many': 'with many steep traces',
            # 数量+斜率组合 (优先匹配更具体的描述)
            'few_gentle': 'with a few gently sloping traces',
            'few_moderate': 'with a few moderately sloping traces',
            'few_steep': 'with a few steeply sloping traces',
            'several_gentle': 'with several gently sloping traces',
            'several_moderate': 'with several moderately sloping traces',
            'several_steep': 'with several steeply sloping traces',
            'many_gentle': 'with many gently sloping traces',
            'many_moderate': 'with many moderately sloping traces',
            'many_steep': 'with many steeply sloping traces'
        }
    },
    11: {  # C&IJ
        'base': 'interleaved signal segments',
        'param': {
            'continuous': 'in continuous pattern',
            'discontinuous': 'in discontinuous pattern'
        }
    },
    # === 扩展: NFMJ 带宽描述 ===
    12: {  # NFMJ - 噪声调频干扰
        'base': 'gradual energy decay from center without clear boundary',
        'param': {
            'narrow': 'with narrow bandwidth',
            'moderate': 'with moderate bandwidth',
            'wide': 'with wide bandwidth'
        }
    },
    # === 扩展: NPMJ 带宽描述 ===
    13: {  # NPMJ - 噪声调相干扰
        'base': 'a jittering horizontal high energy band in the middle',
        'param': {
            'narrow': 'with narrow bandwidth',
            'moderate': 'with moderate bandwidth',
            'wide': 'with wide bandwidth'
        }
    },
    # === 扩展: NAMJ 带宽描述 ===
    14: {  # NAMJ - 噪声调幅干扰
        'base': 'a flat horizontal high energy band with surrounding noise',
        'param': {
            'narrow': 'with narrow bandwidth',
            'moderate': 'with moderate bandwidth',
            'wide': 'with wide bandwidth'
        }
    },
    # === 扩展: CSJ 数量+梳齿组合描述 ===
    15: {  # CSJ
        'base': 'comb-like diagonal lines',
        'param': {
            # 数量描述 (当没有梳齿数时)
            'few': 'with a few comb teeth',
            'several': 'with several comb teeth',
            'many': 'with many comb teeth',
            # 数量+梳齿组合 (所有可能的组合)
            'few_fewteeth': 'with sparse comb structure',
            'few_severalteeth': 'with sparse-moderate comb structure',
            'few_manyteeth': 'with sparse-dense comb structure',
            'several_fewteeth': 'with moderate-sparse comb structure',
            'several_severalteeth': 'with moderate comb structure',
            'several_manyteeth': 'with moderate-dense comb structure',
            'many_fewteeth': 'with dense-sparse comb structure',
            'many_severalteeth': 'with dense-moderate comb structure',
            'many_manyteeth': 'with dense comb structure'
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


def get_bandwidth_level(bandwidth_hz: float) -> str:
    """
    获取带宽等级

    Args:
        bandwidth_hz: 带宽（Hz）

    Returns:
        带宽等级: 'narrow', 'moderate', 'wide'

    >>> 根据实际数据分布，可能需要调整阈值 <<<
    """
    bandwidth_mhz = bandwidth_hz / 1e6  # 转换为MHz
    if bandwidth_mhz < 25:
        return 'narrow'
    elif bandwidth_mhz < 45:
        return 'moderate'
    else:
        return 'wide'


def get_speed_level(speed: float) -> str:
    """
    获取拖引速度等级

    Args:
        speed: 速度参数值

    Returns:
        速度等级: 'slow', 'moderate', 'fast'

    >>> 阈值需要根据实际参数范围调整 <<<
    """
    if speed < 0.5:
        return 'slow'
    elif speed < 1.5:
        return 'moderate'
    else:
        return 'fast'


def get_slope_level(slope_factor: float) -> str:
    """
    获取斜率等级

    Args:
        slope_factor: 斜率因子

    Returns:
        斜率等级: 'gentle', 'moderate', 'steep'

    >>> 阈值需要根据实际参数范围调整 <<<
    """
    if slope_factor < 0.5:
        return 'gentle'
    elif slope_factor < 1.0:
        return 'moderate'
    else:
        return 'steep'


def get_param_key(jam_type, jam_params: dict) -> str:
    """
    根据干扰类型和参数获取对应的param key

    Args:
        jam_type: 干扰类型编号 (1-16) 或名称字符串 ("DFTJ")
        jam_params: 干扰参数字典

    Returns:
        参数描述 key
    """
    jam_id = _get_jam_type_id(jam_type)
    if jam_id is None:
        return ''

    if jam_id == 1:  # DFTJ
        k = jam_params.get('dftj_k') or jam_params.get('k')
        if k is not None:
            return get_count_level(k)
        return ''

    elif jam_id == 2:  # ISRJ
        M = jam_params.get('isrj_M') or jam_params.get('M')
        if M is not None:
            if M <= 2:
                return 'simple'
            elif M <= 3:
                return 'moderate'
            else:
                return 'complex'
        return ''

    elif jam_id == 3:  # RGPO
        pos_rel = jam_params.get('rgpo_position_relation') or jam_params.get('position_relation')
        if pos_rel:
            if isinstance(pos_rel, bytes):
                pos_rel = pos_rel.decode('utf-8')
            return pos_rel
        return ''

    elif jam_id == 4:  # VGPO
        dop_dir = jam_params.get('vgpo_doppler_direction') or jam_params.get('doppler_direction')
        if dop_dir:
            if isinstance(dop_dir, bytes):
                dop_dir = dop_dir.decode('utf-8')
            return dop_dir
        return ''

    # === 扩展: 带宽参数 ===
    elif jam_id == 5:  # AJ - 瞄准干扰
        BJ = jam_params.get('aj_BJ') or jam_params.get('BJ')
        if BJ is not None:
            return get_bandwidth_level(BJ)
        return ''

    elif jam_id == 6:  # BJ - 阻塞干扰
        BJ = jam_params.get('bj_BJ') or jam_params.get('BJ')
        if BJ is not None:
            return get_bandwidth_level(BJ)
        return ''

    elif jam_id == 7:  # SJ - 扫频干扰
        BJ = jam_params.get('sj_BJ') or jam_params.get('BJ')
        if BJ is not None:
            return get_bandwidth_level(BJ)
        return ''

    # === 扩展: SMSPJ 斜率参数 ===
    elif jam_id == 10:  # SMSPJ
        M = jam_params.get('smspj_M') or jam_params.get('M')
        slope = jam_params.get('smspj_slope_factor') or jam_params.get('slope_factor')
        if M is not None:
            count_level = get_count_level(M)
            if slope is not None:
                slope_level = get_slope_level(slope)
                return f"{count_level}_{slope_level}"
            return count_level
        return ''

    # === 扩展: C&IJ 切片参数 ===
    elif jam_id == 11:  # C&IJ
        is_cont = jam_params.get('cij_is_continuous') or jam_params.get('is_continuous')
        if is_cont is not None:
            return 'continuous' if is_cont else 'discontinuous'
        return ''

    # === 扩展: NFMJ/NPMJ/NAMJ 带宽参数 ===
    elif jam_id == 12:  # NFMJ
        BJ = jam_params.get('nfmj_BJ') or jam_params.get('BJ')
        if BJ is not None:
            return get_bandwidth_level(BJ)
        return ''

    elif jam_id == 13:  # NPMJ
        BJ = jam_params.get('npmj_BJ') or jam_params.get('BJ')
        if BJ is not None:
            return get_bandwidth_level(BJ)
        return ''

    elif jam_id == 14:  # NAMJ
        BJ = jam_params.get('namj_BJ') or jam_params.get('BJ')
        if BJ is not None:
            return get_bandwidth_level(BJ)
        return ''

    # === 扩展: CSJ 梳齿数 ===
    elif jam_id == 15:  # CSJ
        M = jam_params.get('csj_M') or jam_params.get('M')
        teeth = jam_params.get('csj_comb_teeth_count') or jam_params.get('comb_teeth_count')
        if M is not None:
            count_level = get_count_level(M)
            if teeth is not None:
                teeth_level = get_count_level(teeth)
                return f"{count_level}_{teeth_level}teeth"
            return count_level
        return ''

    return ''


# =============================================================================
# 描述生成函数
# =============================================================================

def generate_visual_description(jam_type, jam_params: dict) -> str:
    """
    生成视觉特征描述

    格式: {Class Name} looks like {visual description}

    Args:
        jam_type: 干扰类型编号 (1-16) 或名称字符串 ("DFTJ")
        jam_params: 干扰参数字典

    Returns:
        视觉特征描述字符串
    """
    class_name = _get_jam_type_name(jam_type)
    jam_id = _get_jam_type_id(jam_type)

    template = VISUAL_TEMPLATES.get(jam_id, {'base': 'unknown pattern', 'param': {}}) if jam_id else {'base': 'unknown pattern', 'param': {}}

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


def generate_param_description(jam_type, jnr: float) -> str:
    """
    生成参数描述（强度）

    格式: a radar signal with {Class Name} at {strength} jamming power

    Args:
        jam_type: 干扰类型编号 (1-16) 或名称字符串 ("DFTJ")
        jnr: 干噪比

    Returns:
        参数描述字符串
    """
    class_name = _get_jam_type_name(jam_type)
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

    # 规范化 jam_types 为列表
    jam_types = _normalize_jam_types(jam_types)

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
        class_names = [_get_jam_type_name(t) for t in jam_types]

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
        # 整数格式（旧格式）
        {'jam_types': [1], 'JNR': 5, 'jam_params': {'dftj_k': 4}},
        {'jam_types': [5], 'JNR': 15, 'jam_params': {}},  # AJ
        {'jam_types': [1, 5], 'JNR': 20, 'jam_params': {'dftj_k': 5}},  # 组合
        # 字符串格式（新格式 - 列表）
        {'jam_types': ['DFTJ'], 'JNR': 20, 'jam_params': {'k': 7}},
        {'jam_types': ['AJ'], 'JNR': 15, 'jam_params': {}},
        {'jam_types': ['DFTJ', 'AJ'], 'JNR': 20, 'jam_params': {'k': 5}},  # 组合
        # 字符串格式（新格式 - 单个字符串）
        {'jam_types': 'AJ', 'JNR': 15, 'jam_params': {}},
    ]

    for i, meta in enumerate(test_cases):
        jam_types = _normalize_jam_types(meta['jam_types'])
        jam_names = [_get_jam_type_name(t) for t in jam_types]
        print(f"\n{i+1}. {' + '.join(jam_names)} (input: {meta['jam_types']}):")
        print(f"   visual:       {generate_short_description(meta, 'visual')}")
        print(f"   param:        {generate_short_description(meta, 'param')}")
        print(f"   visual_param: {generate_short_description(meta, 'visual_param')}")