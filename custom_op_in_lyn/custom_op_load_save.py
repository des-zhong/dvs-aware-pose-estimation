# Copyright (c) OpenBII
# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.

import lyngor as lyn


def load(inputs, input_types):
    '''
    Parameters
    inputs      : Input data
    input_types : Types of input data
    '''
    # check
    assert isinstance(inputs[1], str)
    data = inputs[0]
    flag = inputs[1]
    is_init = inputs[2]
    shape = inputs[3]
    dtype = inputs[4]
    offset = inputs[5]
    step = inputs[6]
    uselookup = inputs[7]
    mode = inputs[8]
    group_id = inputs[9]
    out = lyn.apu.load(data=data,
                       flag=flag,
                       is_initial=is_init,
                       shape=shape,
                       dtype=dtype,
                       offset=offset,
                       step=step,
                       uselookup=uselookup,
                       mode=mode,
                       group_id=group_id)
    return out


# op save definition
def save(inputs, input_types):
    '''
    Parameters
    inputs      : Input data
    input_types : Types of input data
    '''
    # check
    assert isinstance(inputs[1], str)
    data = inputs[0]
    flag = inputs[1]

    if len(inputs) > 2:
        offset = inputs[2]
        step = inputs[3]
        isOutput = inputs[4]
        uselookup = inputs[5]
        mode = inputs[6]
        group_id = inputs[7]
        out = lyn.apu.save(data, flag, offset, step, isOutput, uselookup, mode, group_id)
    else:
        out = lyn.apu.save(data, flag)

    return out


# register operators load save
lyn.pytorch_convert_map.update({'custom::load': load})
lyn.pytorch_convert_map.update({'custom::save': save})
