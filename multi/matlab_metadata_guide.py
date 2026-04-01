"""
MATLAB侧修改建议 - 在数据生成时记录metadata

============================================================
修改位置: main_generation.m
============================================================

在 Line 131-166 的循环中，需要记录每个样本的生成参数：

%% 修改1: 在循环前初始化metadata数组
% Line 131 之后添加:
all_metadata = struct();
all_positions = zeros(SAMPLE_NUM, 1);
all_jam_params = struct();

%% 修改2: 在 multi_generation 调用后记录metadata
% 需要修改 multi_generation.m 函数，使其返回额外的参数信息
% 或者在 main_generation.m 中根据 generation_plan 推断

for i = 1:len
    jam_type = generation_plan{i, 1};
    label = generation_plan{i, 2};
    num_to_generate = generation_plan{i, 3};

    % 调用生成函数
    [new_times, new_label, new_metadata] = multi_generation_with_metadata(label, params, current_jnr, num_to_generate);

    % 记录到总metadata
    point_l = ...; point_r = ...;
    all_metadata(point_l:point_r) = new_metadata;
end

%% 修改3: 保存metadata
% 在 Line 188 之后添加:
path_metadata = fullfile(snr_output_dir, 'test_echo_metadata.mat');
save(path_metadata, 'all_metadata', '-v7.3');

============================================================
修改位置: multi_generation.m
============================================================

添加返回metadata的功能:

function [samples, labels, metadata] = multi_generation_with_metadata(label, params, current_jnr, data_num)
    % ... 原有代码 ...

    % 初始化metadata
    metadata = struct('sample_idx', {}, 'jam_types', {}, 'JNR', {}, 'pos', {}, 'jam_params', {});

    for m = 1:data_num
        % 记录当前位置
        current_pos = params.pos + randi([0 4000]);

        metadata(m).sample_idx = m;
        metadata(m).jam_types = label;
        metadata(m).JNR = current_jnr;
        metadata(m).pos = current_pos;

        % 根据干扰类型记录特定参数
        if any(label == 1)  % DFTJ
            metadata(m).dftj_k = k;  % 假目标数量
        end
        if any(label == 2)  % ISRJ
            metadata(m).isrj_M = M;  % 转发次数
            metadata(m).isrj_N = N;  % 采样次数
        end
        if any(label == 5) || any(label == 6)  % AJ/BJ
            metadata(m).jam_BJ = jam_params.BJ;  % 带宽
        end
        % ... 其他干扰类型 ...
    end
end

============================================================
简化方案: 不修改multi_generation.m
============================================================

如果不想修改 multi_generation.m，可以在 main_generation.m 中推断metadata:

for i = 1:len
    jam_type = generation_plan{i, 1};
    label = generation_plan{i, 2};
    num_to_generate = generation_plan{i, 3};

    [new_times, new_label] = multi_generation(label, params, current_jnr, num_to_generate);

    % 在这里推断并创建metadata
    for m = point_l:point_r
        all_metadata(m).sample_idx = m;
        all_metadata(m).jam_types = label;
        all_metadata(m).JNR = current_jnr;
        all_metadata(m).pos = 'unknown';  % 或使用固定值
        % 干扰特定参数可设置为典型值
        if any(label == 1)
            all_metadata(m).dftj_k = 'unknown';
        end
    end
end

============================================================
Python侧读取metadata
============================================================

使用 scipy.io.loadmat 或 h5py 读取.mat文件:

import scipy.io as sio

metadata = sio.loadmat('test_echo_metadata.mat')
all_metadata = metadata['all_metadata']

# MATLAB struct会被转换为numpy structured array
# 每个元素是一个struct，可通过字段名访问
"""

# =============================================================================
# 测试用MATLAB代码片段
# =============================================================================

MATLAB_TEST_CODE = """
% 测试代码 - 创建小规模metadata并保存
test_metadata = struct();
for m = 1:10
    test_metadata(m).sample_idx = m;
    test_metadata(m).jam_types = [1, 5];  % DFTJ + AJ
    test_metadata(m).JNR = 15;
    test_metadata(m).pos = 2500 + m*100;
    test_metadata(m).dftj_k = randi([4, 8]);
    test_metadata(m).jam_BJ = (18.5 + 5*rand)*1e6;
end
save('test_metadata.mat', 'test_metadata', '-v7.3');
"""

PYTHON_TEST_CODE = """
# Python读取测试
import scipy.io as sio
import numpy as np

# 加载MATLAB文件
data = sio.loadmat('test_metadata.mat')
metadata_struct = data['test_metadata']

print(f"Metadata shape: {metadata_struct.shape}")
print(f"Number of samples: {len(metadata_struct[0])}")

# 访问第一个样本的metadata
sample0 = metadata_struct[0, 0]
print(f"Sample 0:")
print(f"  sample_idx: {sample0['sample_idx'][0,0]}")
print(f"  jam_types: {sample0['jam_types'][0]}")
print(f"  JNR: {sample0['JNR'][0,0]}")
print(f"  pos: {sample0['pos'][0,0]}")
"""

if __name__ == "__main__":
    print("=" * 60)
    print("MATLAB metadata 修改建议")
    print("=" * 60)
    print("\n关键修改位置:")
    print("  1. main_generation.m Line 131: 初始化metadata数组")
    print("  2. main_generation.m Line 186-188: 保存metadata文件")
    print("  3. multi_generation.m: 返回metadata信息")
    print("\n推荐格式: .mat文件，与stft/label文件放在同一目录")