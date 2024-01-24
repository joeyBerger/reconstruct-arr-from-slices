
import random

symbol_ceiling = 12
total_reel_length = 10
total_input_slices = 150
slice_length = 4
total_samples = 30000

def generate_uniform_inputs_with_output():
    output = []
    input_slices = []

    for i in range(total_reel_length):
        output.append(random.randint(0, symbol_ceiling))

    doubled_output = output + output

    for i in range(total_input_slices):
        rand_index = random.randint(0, total_reel_length - 1)
        input_slices.append(
            doubled_output[rand_index: rand_index + slice_length])

    return input_slices, output

def expand_output_arr(output):
    expanded_output = []
    doubled_output = output + output
    output_length = len(output)
    for i in range(len(output)):
        expanded_output.append(doubled_output[i: i + output_length])
    return expanded_output

def recursive_divide(arr, divisor):
    if not arr:
        return []
    result = [arr[0] / divisor] + recursive_divide(arr[1:], divisor)
    return result

def get_final_inputs_and_outputs():
    input_slices, output = generate_uniform_inputs_with_output()
    expanded_output = expand_output_arr(output)
    return input_slices, output

def divide_input_slices(input_slices):
    divided_slices = []
    for slice in input_slices:
        divided_slices.append(recursive_divide(slice, 12))
    return divided_slices

def wrap_around_slice(arr, start, end):
    if len(arr) == 0:
        return []

    start %= len(arr)
    end %= len(arr)

    if start <= end:
        return_arr = arr[start:end+1]
    else:
            return_arr = arr[start:] + arr[:end+1]

    return return_arr

def start_arr_at_lowest_value(arr):
    min_index = arr.index(min(arr))
    wrap_index = min_index - 1
    if wrap_index < 0:
        wrap_index = len(arr) - 1
    return wrap_around_slice(arr, min_index, wrap_index)

def get_data():
    inputs = []
    outputs = []
    for _ in range(total_samples):
        input_slices, expanded_output = get_final_inputs_and_outputs()
        inputs.append(divide_input_slices(input_slices))
        outputs.append(start_arr_at_lowest_value(
            recursive_divide(expanded_output, symbol_ceiling)))
        
    return inputs, outputs