import json
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from elois_net import EloisNet
# from sklearn.model_selection import train_test_split
import torch.optim as optim
from itertools import islice

# Constants
BATCH_SIZE = 256
LEARNING_RATE = 4e-7
NUM_EPOCHS = 10000
N_HEAD = 128
N_LAYER = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(42)

# Fields to predict
output_vector_fields = ['netincomeloss']
OUTPUT_SIZE = len(output_vector_fields)

def load_data(filepath='financial_statements.json'):
    # Load financial statements
    with open(filepath, 'r+') as file:
        financial_statements = json.load(file)

    # Count field occurrences
    all_fields_counter = {}
    # financial_statements = dict(islice(financial_statements.items(), 5))
    for financial_statement in financial_statements.values():
        for train_yearly_report in financial_statement:
            for key, value in train_yearly_report.items():
                if not isinstance(value, (int, float)):
                    continue
                all_fields_counter[key] = all_fields_counter.get(key, 0) + 1

    # Get top 128 most common fields
    sorted_all_fields_counter = dict(
        sorted(all_fields_counter.items(), 
              key=lambda item: item[1], 
              reverse=True)[:256]
    )

    test_input_data = []
    test_ground_truth = []

    
    train_input_data = []
    train_ground_truth = []
    
    for financial_statement in financial_statements.values():
        if len(financial_statement) < 5:
            continue
            
        train_input_vector = []
        train_masking_vector = []

        train_output_vector = []

        test_input_vector = []
        test_masking_vector = []

        test_output_vector = []
        
        # Get input data from first 3 years
        for index, train_yearly_report in enumerate(financial_statement):
            if (index > 3):
                break
            for key in sorted_all_fields_counter:
                value = train_yearly_report.get(key, 0)
                try:
                    value = float(value)/1e6
                except:
                    value = 0
                masking_value = 1 if key in train_yearly_report else 0
                if index > 0:
                    test_input_vector.append(value)
                    test_masking_vector.append(masking_value)
                if index < 3:
                    train_input_vector.append(value)
                    train_masking_vector.append(masking_value)
        

        train_yearly_report = financial_statement[3]
        test_yearly_report = financial_statement[4]

        for key in output_vector_fields:
            train_value = train_yearly_report.get(key, None)
            test_value = test_yearly_report.get(key, None)
            if train_value is not None:
                train_value /= 1e6
            if test_value is not None:
                test_value /= 1e6
            train_output_vector.append(train_value)
            test_output_vector.append(test_value)
        
        train_input_data.append((train_input_vector, train_masking_vector))
        train_ground_truth.append(train_output_vector)

        test_input_data.append((test_input_vector, test_masking_vector))
        test_ground_truth.append(test_output_vector)

    return train_input_data, train_ground_truth, test_input_data, test_ground_truth, len(train_input_data[0][0])

def prepare_sub_data(input_data, ground_truth):
    """
    Helper function to filter out samples with None values in ground truth
    and convert the data to tensors.
    """
    # Filter out any samples with None values in ground truth
    valid_samples = []
    valid_targets = []
    
    for idx, (sample, target) in enumerate(zip(input_data, ground_truth)):
        if None in target:
            continue
        valid_samples.append(sample)
        valid_targets.append(target)
    
    # Convert to tensors
    values = torch.FloatTensor([x[0] for x in valid_samples])
    mask = torch.FloatTensor([x[1] for x in valid_samples])
    targets = torch.FloatTensor(valid_targets)
    
    # Create proper input format (B, 2, n_embd)
    data = torch.stack([values, mask], dim=1)
    
    return data, targets

def prepare_data(train_input_data, train_ground_truth, test_input_data, test_ground_truth):
    """
    Main function to prepare training and evaluation data.
    Evaluation data combines both validation and test data.
    """
    # Prepare training data
    X_train, y_train = prepare_sub_data(train_input_data, train_ground_truth)
    
    # Prepare evaluation data (combine validation and test data)
    X_eval, y_eval = prepare_sub_data(test_input_data, test_ground_truth)
    
    return X_train, y_train, X_eval, y_eval

def create_dataloaders(X_train, y_train, X_val, y_val, batch_size):
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    length = 0
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        logits, loss = model(data, targets)
        
        if loss is None:
            continue
            
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        length += 1
    
    return total_loss / length

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    length = 0
    
    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            logits, loss = model(data, targets)
            
            if loss is None:
                continue
                
            total_loss += loss.item()
            length += 1
    
    return total_loss / length

def main():
    # Load and prepare data
    train_input_data, train_ground_truth, test_input_data, test_ground_truth, n_embd = load_data()
    X_train, y_train, X_val, y_val = prepare_data(train_input_data, train_ground_truth, test_input_data, test_ground_truth,)
    train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val, BATCH_SIZE)
    
    # Initialize model
    model = EloisNet(
        n_embd=n_embd,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        output_size=OUTPUT_SIZE
    ).to(DEVICE)
    
    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        val_loss = validate(model, val_loader, DEVICE)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'test_model.pt')
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

if __name__ == '__main__':
    main()


#best_val_loss: ~1330 for the Eloi's net, ~400,000 for a standard MLP