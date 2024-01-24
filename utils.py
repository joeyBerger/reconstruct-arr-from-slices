import numpy as np
import torch

def find_best_fit(outputs, y, device):
    best_tensor_rotations = torch.empty(0).to(device)
    for i in range(len(outputs)):
        lowest_idx = lowest_delta = 100000
        for j in range(len(outputs[i])):
            new_tensor = return_wrapped_tensor(outputs[i], j)
            diff = torch.sum(abs(new_tensor - y))
            if diff < lowest_delta:
                lowest_delta = diff
                lowest_idx = j

        updated_tensor = return_wrapped_tensor(outputs[i], lowest_idx)
        best_tensor_rotations = torch.cat((best_tensor_rotations, updated_tensor.unsqueeze(0)), dim=0)

    return best_tensor_rotations

def create_sequences(inputs, outputs):
    y = []
    for i in range(len(outputs)):
        y.append(outputs[i])

    X = inputs

    return np.array(inputs), np.array(y)


def convert_data_to_tensors(X_train, y_train, X_valid, y_valid, X_test, y_test):
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_valid = torch.from_numpy(X_valid).float()
    y_valid = torch.from_numpy(y_valid).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def return_wrapped_tensor(tensor, offset, device):
    tensor_size = tensor.shape[0]
    return_tensor = torch.empty(tensor.shape).to(device)
    for i in range(tensor_size):
        return_tensor[i] = tensor[(i + offset) % tensor_size]
    return return_tensor