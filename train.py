import torch
import torch.nn as nn
from dataHandler import get_data
from utils import create_sequences, convert_data_to_tensors
from model import LSTM

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.is_available())

total_samples = 30000

inputs, outputs = get_data()
    
# Split the data into training and testing sets
train_size = int(total_samples * 0.8)
test_valid_size = int((total_samples - train_size) / 2)
train_data = inputs[:train_size]
valid_data = inputs[train_size:train_size + test_valid_size]
test_data = inputs[train_size + test_valid_size:]


# Create sequences for training, validation and testing data
X_train, y_train = create_sequences(train_data, outputs[:train_size])
X_valid, y_valid = create_sequences(valid_data, outputs[train_size: train_size + test_valid_size])
X_test, y_test = create_sequences(test_data, outputs[train_size + test_valid_size:])
    
# Instantiate the model
input_size = X_train.shape[2]
hidden_size = 32
output_size = 10
model = LSTM(input_size, hidden_size, output_size)
model.to(device)


# Define the loss function and optimizer
criterion = nn.MSELoss()
momentum = 0.9

optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001, betas=(momentum, 0.9))


# Convert numpy arrays to Pytorch tensors
X_train, y_train, X_valid, y_valid, X_test, y_test = convert_data_to_tensors(X_train, y_train, X_valid, y_valid, X_test, y_test)

# Define the batch size and number of epochs
batch_size = 32
num_epochs = 250
lowest_train_loss = lowest_valid_loss= 1000
validation_step = 5

# Train the model
for epoch in range(num_epochs):

    model.train()

    # Shuffle the training data
    perm = torch.randperm(X_train.shape[0])
    X_train = X_train[perm]
    y_train = y_train[perm]

    # Loop over batches
    for i in range(0, X_train.shape[0], batch_size):
        # Get batch
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]

        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    loss_amount = loss.item()

    # Print loss for this epoch
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch +
          1, num_epochs, loss_amount))
    
    lowest_train_loss = min(lowest_train_loss, loss_amount)

    if epoch % validation_step == 0:
        model.eval()
        with torch.no_grad():
            y_pred = model(X_valid.to(device))
            test_loss = criterion(y_pred, y_valid.to(device))
            loss_amount = test_loss.item()
            lowest_valid_loss = min(lowest_valid_loss, loss_amount)
            print('Validation Loss: {:.4f}'.format(loss_amount))

    

print('Lowest Training Loss: {:.4f}'.format(lowest_train_loss))
print('Lowest Validation Loss: {:.4f}'.format(lowest_train_loss))

# Evaluate the model on the test data
model.eval()
with torch.no_grad():
    y_pred = model(X_test.to(device))

print(y_pred[1])
print(y_test[1])

for i in range(100):
    print(y_pred[i])
    print(y_test[i])
    print('\n\n')

# Calculate the test loss
test_loss = criterion(y_pred, y_test.to(device))
print('Test Loss: {:.4f}'.format(test_loss.item()))

