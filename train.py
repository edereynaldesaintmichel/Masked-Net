import json
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import torch.optim as optim
from itertools import islice
import torch.nn as nn
from torch.nn import functional as F

# Hyperparams
BATCH_SIZE = 512
LEARNING_RATE = 0.5e-3
NUM_EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_DROPOUT_PROB = 0.0
NUM_INPUT_FIELDS = 32
BAD_EXAMPLE_CUTOFF = 20
WEIGHT_DECAY = 0.0

# Fields to predict:
# OUTPUT_VECTOR_FIELDS = ["interestIncome", "interestExpense", "ebitda", "operatingIncome", "incomeBeforeTax", "netIncome", "eps", "epsdiluted",] # These output fields are for net_income_and_stuff_model.pt
INPUT_FIELDS = ["revenue", "costOfRevenue", "grossProfit", "grossProfitRatio", "researchAndDevelopmentExpenses", "generalAndAdministrativeExpenses", "sellingAndMarketingExpenses", "sellingGeneralAndAdministrativeExpenses", "otherExpenses", "operatingExpenses", "costAndExpenses", "interestIncome", "interestExpense", "depreciationAndAmortization", "ebitda", "ebitdaratio", "operatingIncome", "operatingIncomeRatio", "totalOtherIncomeExpensesNet", "incomeBeforeTax", "incomeBeforeTaxRatio", "incomeTaxExpense", "netIncome", "netIncomeRatio", "eps", "epsdiluted", "weightedAverageShsOut", "weightedAverageShsOutDil", "cashAndCashEquivalents", "shortTermInvestments", "cashAndShortTermInvestments", "netReceivables", "inventory", "otherCurrentAssets", "totalCurrentAssets", "propertyPlantEquipmentNet", "goodwill", "intangibleAssets", "goodwillAndIntangibleAssets", "longTermInvestments", "taxAssets", "otherNonCurrentAssets", "totalNonCurrentAssets", "otherAssets", "totalAssets", "accountPayables", "shortTermDebt", "taxPayables", "deferredRevenue", "otherCurrentLiabilities", "totalCurrentLiabilities", "longTermDebt", "deferredRevenueNonCurrent", "deferredTaxLiabilitiesNonCurrent", "otherNonCurrentLiabilities", "totalNonCurrentLiabilities", "otherLiabilities", "capitalLeaseObligations", "totalLiabilities", "preferredStock", "commonStock", "retainedEarnings", "accumulatedOtherComprehensiveIncomeLoss", "othertotalStockholdersEquity", "totalStockholdersEquity", "totalEquity", "totalLiabilitiesAndStockholdersEquity", "minorityInterest", "totalLiabilitiesAndTotalEquity", "totalInvestments", "totalDebt", "netDebt", "deferredIncomeTax", "stockBasedCompensation", "changeInWorkingCapital", "accountsReceivables", "accountsPayables", "otherWorkingCapital", "otherNonCashItems", "netCashProvidedByOperatingActivities", "investmentsInPropertyPlantAndEquipment", "acquisitionsNet", "purchasesOfInvestments", "salesMaturitiesOfInvestments", "otherInvestingActivites", "netCashUsedForInvestingActivites", "debtRepayment", "commonStockIssued", "commonStockRepurchased", "dividendsPaid", "otherFinancingActivites", "netCashUsedProvidedByFinancingActivities", "effectOfForexChangesOnCash", "netChangeInCash", "cashAtEndOfPeriod", "cashAtBeginningOfPeriod", "operatingCashFlow", "capitalExpenditure", "freeCashFlow", "calendarYear", "reportedCurrency"]

OUTPUT_VECTOR_FIELDS = ["revenue", "netIncome", "eps", "epsdiluted", "freeCashFlow", "totalStockholdersEquity", "operatingCashFlow", "dividendsPaid"]
# OUTPUT_VECTOR_FIELDS = ["totalStockholdersEquity"]


torch.manual_seed(42)

std, mean = 1, 1


class MaskedLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.main_proj = nn.Linear(in_features, out_features, bias=False)
        # LoRA projections
        self.result_proj_1 = nn.Linear(3*out_features, out_features, bias=False)
        self.result_proj_2 = nn.Linear(out_features, 3*out_features, bias=False)
        # general bias, saving the bias params of the previous layers
        self.bias = nn.Parameter(torch.zeros(out_features*3))
        # non-linearity
        self.leaky_relu = nn.LeakyReLU()
        # stored for output splitting
        self.out_features = out_features

    def forward(self, values, mask_zero, mask_invalid):     
        # Weight sharing, very similar to a 1D convo with out_features filters. Full rank *local* operation with limited number of parameters.
        general_vector = torch.cat((self.main_proj(values), self.main_proj(mask_zero), self.main_proj(mask_invalid)), dim=1)
        
        # Low (relatively speaking) rank operation with lots of interaction between all the input components. Exactly the opposite of the previous step.
        out = self.result_proj_2(self.result_proj_1(general_vector))
        out = out + self.bias
        out = self.leaky_relu(out)
        
        # Split the output into three equal parts
        values_out = out[:, :self.out_features]
        mask_zero_out = out[:, self.out_features:2*self.out_features]
        mask_invalid_out = out[:, 2*self.out_features:]

        return values_out, mask_zero_out, mask_invalid_out
    
class MaskedSequential(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x, mask_zero, mask_invalid):
        for layer in self.layers:
            if isinstance(layer, MaskedLayer):
                x, mask_zero, mask_invalid = layer(x, mask_zero, mask_invalid)
            else:
                x = layer(x)
        return x

class MaskedNet(nn.Module):
    def __init__(self, input_size, output_size, number_of_currencies, dropout_prob=0.1):
        super().__init__()
        self.currency_embedding = nn.Embedding(num_embeddings=number_of_currencies, embedding_dim=2)
        self.dropout = nn.Dropout(MAX_DROPOUT_PROB)
        self.lm_head = MaskedSequential(
            MaskedLayer(input_size + 1, input_size//8),
            MaskedLayer(input_size//8, input_size//12),
            nn.Linear(input_size//12, output_size)
        )

    def forward(self, input, targets=None, clean_dataset=False):
        main_features = input[:, :-1]
        mask_invalid = (main_features == -.01).float()
        mask_zero = (main_features == 0).float()
        currency_idx = input[:, -1].long()
        currency_embedding = self.currency_embedding(currency_idx)
        combined_input = torch.cat([main_features, currency_embedding], dim=1)
        mask_invalid = torch.cat([mask_invalid, torch.zeros(mask_invalid.shape[0], 2)], dim=1)
        mask_zero = torch.cat([mask_zero, torch.zeros(mask_zero.shape[0], 2)], dim=1)

        output = self.lm_head(combined_input, mask_zero, mask_invalid)

        if targets is None:
            return output, None
        if clean_dataset:
            return output, torch.mean(output-targets, dim=1)

        loss = F.l1_loss(output, targets)
        return output, loss


# class MaskedNet(nn.Module):
#     def __init__(self, input_size, output_size, number_of_currencies, dropout_prob=0.1):
#         super().__init__()
#         self.currency_embedding = nn.Embedding(num_embeddings=number_of_currencies, embedding_dim=2)
#         self.lm_head = nn.Sequential(
#             nn.Dropout(MAX_DROPOUT_PROB),
#             nn.Linear(input_size + 1, input_size//3),
#             nn.LeakyReLU(),
#             nn.Linear(input_size//3, input_size//3),
#             nn.LeakyReLU(),
#             nn.Linear(input_size//3, input_size//4),
#             nn.LeakyReLU(),
#             nn.Linear(input_size//4, output_size)
#         )

#     def forward(self, input, targets=None, dropout_factor=1):
#         main_features = input[:, :-1]
#         currency_idx = input[:, -1].long()
#         currency_embedding = self.currency_embedding(currency_idx)  # Shape: (batch_size, 2)
#         combined_input = torch.cat([main_features, currency_embedding], dim=1)

#         output = self.lm_head(combined_input)  # (B, output_size)

#         if targets is None:
#             return output, None

#         loss = F.l1_loss(output, targets)
#         return output, loss





def get_val_dataloader(full_dataset, output_field_indices, batch_size):
    input_data = []
    ground_truth = []
    excluded_counter = 0
    total_counter = 0
    for company_statements in full_dataset:
        if (company_statements.shape[0] < 4):
            continue
        total_counter += 1
        to_append = torch.flatten(company_statements[-4:-1])
        mask = to_append == 0
        if to_append[mask].shape[0] > to_append.shape[0]*0.5:
            excluded_counter += 1
            continue
        input_data.append(to_append)
        sub_ground_truth = torch.index_select(company_statements[-1], dim=-1, index=output_field_indices)
        ground_truth.append(sub_ground_truth)

    ground_truth = torch.stack(ground_truth)  # Changed from torch.tensor to torch.stack
    input_data = torch.stack(input_data)      # Changed from torch.tensor to torch.stack
    val_dataset = TensorDataset(input_data, ground_truth)
    print(f'Share of bad val examples: {excluded_counter/total_counter:.4f}')
    return DataLoader(val_dataset, batch_size=batch_size)


def get_train_dataloader(full_dataset, output_field_indices, batch_size):
    input_data = []
    ground_truth = []
    excluded_counter = 0
    total_counter = 0
    for i in range(0, 50):
        for company_statements in full_dataset:
            if (company_statements.shape[0] < i + 5):
                continue
            total_counter += 1
            to_append = torch.flatten(company_statements[-i-5:-i-2])
            mask = to_append == 0
            # if to_append[mask].shape[0] > to_append.shape[0]*0.5:
            #     excluded_counter += 1
            #     continue
            input_data.append(to_append)
            sub_ground_truth = torch.index_select(company_statements[-i-2], dim=-1, index=output_field_indices)
            ground_truth.append(sub_ground_truth)

    ground_truth = torch.stack(ground_truth)  # Changed from torch.tensor to torch.stack
    input_data = torch.stack(input_data)      # Changed from torch.tensor to torch.stack
    training_dataset = TensorDataset(input_data, ground_truth)

    print(f'Share of bad training examples: {excluded_counter/total_counter:.4f}')

    return DataLoader(training_dataset, batch_size=batch_size, shuffle=True)


def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    length = 0

    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)

        optimizer.zero_grad()
        output, loss = model(data, targets)

        if loss is None:
            continue

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        length += 1

        if batch_idx%1000 == 0:
            print(f'Training loss: {loss}')

    return total_loss / length


def get_target_loss(val_loader, output_data_indices=None):
    total_target_loss = 0
    total_length = 0

    for data, targets in val_loader:
        carry_over_data = torch.index_select(data, dim=-1, index=output_data_indices)

        length = data.shape[0]
        loss = F.l1_loss(carry_over_data, targets).item() * length
        total_target_loss += loss
        total_length += length

    return total_target_loss / total_length


def validate(model, val_loader, device, output_data_indices=None):
    model.eval()
    total_loss = 0
    total_length = 0
    total_target_loss = 0

    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            output, loss = model(data, targets)

            if loss is None:
                continue
            length = len(data)
            total_length += length
            total_loss += loss.item() * length

    return total_loss / total_length


def cleanDataset(model, train_loader, device, train_loss):
    model.eval()
    good_data = []
    good_targets = []
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            _, losses = model(data, targets, clean_dataset=True)
            good_indices = losses <= BAD_EXAMPLE_CUTOFF*train_loss
            good_data.append(data[good_indices].cpu())
            good_targets.append(targets[good_indices].cpu())
    
    clean_data = torch.cat(good_data, dim=0)
    clean_targets = torch.cat(good_targets, dim=0)
    
    clean_dataset = torch.utils.data.TensorDataset(clean_data, clean_targets)
    
    clean_loader = torch.utils.data.DataLoader(
        clean_dataset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=train_loader.num_workers if hasattr(train_loader, 'num_workers') else 0
    )
    
    original_size = len(train_loader.dataset)
    clean_size = len(clean_dataset)
    removed_percentage = (original_size - clean_size) / original_size * 100
    
    print(f"Original dataset size: {original_size}")
    print(f"Clean dataset size: {clean_size}")
    print(f"Removed {removed_percentage:.2f}% of examples")
    
    return clean_loader


def get_output_gradients(model, val_loader, device, output_field:str, input_fields:list, output_fields:list):
    model.eval()
    all_gradients = []

    output_index = output_fields.index(output_field)

    input_dict = {f'{input_fields[i]}': i for i in range(len(INPUT_FIELDS))}

    for data, targets in val_loader:
        data = data.to(device)
        data.requires_grad = True

        output, _ = model(data, targets.to(device))
        output[:, output_index].sum().backward()
        
        gradients = data.grad.clone().sum(dim=0)
        # immediate_gradients_dict = {key: (float(gradients[value]), float(gradients[value+len(INPUT_FIELDS)]), float(gradients[value+len(INPUT_FIELDS)*2])) for key, value in input_dict.items()}
        all_gradients.append(gradients)  # Store gradients for this batch

    all_gradients = torch.stack(all_gradients, dim=0)
    all_gradients_sum = all_gradients.sum(dim=0)
    all_gradients_sum /= all_gradients_sum.max()
    full_gradients_dict = {key: (float(all_gradients_sum[value]), float(all_gradients_sum[value+len(INPUT_FIELDS)]), float(all_gradients_sum[value+len(INPUT_FIELDS)*2])) for key, value in input_dict.items()}
    csv_string = '"Metric", "n-3", "n-2", "n-1"\n'
    for metric, values in full_gradients_dict.items():
        csv_string += f'"{metric}", {values[0]}, {values[1]}, {values[2]}\n'
    print(csv_string)
    return full_gradients_dict


def get_predictibility(model, val_loader, device, output_fields:list[str], input_fields:list[str]):
    model.eval()
    total_loss = 0
    total_length = 0
    total_target_loss = 0
    output_data_indices = [input_fields.index(field) for field in output_fields]
    target_losses = torch.zeros(len(output_fields))
    attained_losses = torch.zeros(len(output_fields))
    composite_losses = torch.zeros(len(output_fields))
    predictibility = {}

    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            output, _ = model(data, targets)
            length = data.shape[0]
            for output_index, input_index in enumerate(output_data_indices):
                carry_over_data = data[:, len(input_fields) * 2 + input_index]
                target_losses[output_index] += F.l1_loss(carry_over_data, targets[:,output_index]).item() * length
                attained_losses[output_index] += F.l1_loss(output[:,output_index], targets[:,output_index]).item() * length
                composite_losses[output_index] += F.l1_loss((output[:,output_index] + carry_over_data*2.5)/3.5, targets[:,output_index]).item() * length

            total_length += length
            
        target_losses /= total_length
        attained_losses /= total_length
        composite_losses /= total_length

        predictibility = {field: (float(target_losses[index]), float(attained_losses[index]), float(composite_losses[index])) for index, field in enumerate(output_fields)}

    return predictibility


def train():
    OUTPUT_SIZE = len(OUTPUT_VECTOR_FIELDS)
    # Load and prepare data
    output_field_indices = torch.tensor([INPUT_FIELDS.index(field) for field in OUTPUT_VECTOR_FIELDS])
    full_dataset = torch.load('full_data.pt')
    val_data_loader = get_val_dataloader(full_dataset=full_dataset, output_field_indices=output_field_indices, batch_size=BATCH_SIZE)
    train_data_loader = get_train_dataloader(full_dataset=full_dataset, output_field_indices=output_field_indices, batch_size=BATCH_SIZE)

    # Initialize model
    model = MaskedNet(
        input_size=3*len(INPUT_FIELDS),
        output_size=OUTPUT_SIZE,
        number_of_currencies=47,
        dropout_prob=MAX_DROPOUT_PROB,
    ).to(DEVICE)
    # model.load_state_dict(torch.load('test_model.pt', map_location=torch.device('cpu')))

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Training loop
    best_val_loss = float('inf')

    target_loss = get_target_loss(val_loader=val_data_loader, output_data_indices=output_field_indices+len(INPUT_FIELDS)*2)

    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_data_loader, optimizer, DEVICE)
        val_loss = validate(model, val_data_loader, DEVICE)
        # train_data_loader = cleanDataset(model, train_data_loader, DEVICE, val_loss)
        # train_loss = train_epoch(model, fine_tune_dataloader, optimizer, DEVICE)
        # val_loss = validate(model, val_data_loader, DEVICE)
        # print(f'Epoch {epoch}-ft: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Best VL: {best_val_loss:.4f}, Target: {target_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'test_model.pt')

        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Best VL: {best_val_loss:.4f}, Target: {target_loss:.4f}')

def test():
        # Fields to predict
    OUTPUT_SIZE = len(OUTPUT_VECTOR_FIELDS)
    # Load and prepare data
    output_field_indices = torch.tensor([INPUT_FIELDS.index(field) for field in OUTPUT_VECTOR_FIELDS])
    full_dataset = torch.load('full_data.pt')
    val_data_loader = get_val_dataloader(full_dataset=full_dataset, output_field_indices=output_field_indices, batch_size=BATCH_SIZE)

     # Initialize model
    model = MaskedNet(
        input_size=3*len(INPUT_FIELDS),
        output_size=OUTPUT_SIZE,
        number_of_currencies=47,
        dropout_prob=MAX_DROPOUT_PROB,
    ).to(DEVICE)


    model.load_state_dict(torch.load('test_model.pt', map_location=torch.device('cpu')))
    # std = torch.load('std.pt')
    # mean = torch.load('mean.pt')
    # currency_indices = torch.load('currency_indices.pt')
    # currency_exchange_rates = torch.load('currency_exchange_rates.pt')


    pred = get_predictibility(model, val_data_loader, DEVICE, input_fields=INPUT_FIELDS, output_fields=OUTPUT_VECTOR_FIELDS)
    gradients = get_output_gradients(model, val_data_loader, DEVICE, output_field='freeCashFlow', input_fields=INPUT_FIELDS, output_fields=OUTPUT_VECTOR_FIELDS)

    return gradients


def get_layers_condition_number():
    """
    Calculate condition numbers for each layer in the neural network model.
    
    Returns:
        dict: Dictionary containing condition numbers for each layer's weight matrix
    """
    # Load the model
    model = MaskedNet(
        input_size=3*len(INPUT_FIELDS),
        output_size=len(OUTPUT_VECTOR_FIELDS),
        number_of_currencies=47,
        dropout_prob=MAX_DROPOUT_PROB,
    )
    
    # Load the state dict
    model.load_state_dict(torch.load('test_model.pt', map_location=torch.device('cpu')))
    
    condition_numbers = {}
    for name, parameter in model.named_parameters():
        if 'weight' in name:
            weight_matrix = parameter.data
            
            # Calculate SVD
            try:
                U, S, V = torch.svd(weight_matrix)
                condition_number = S[0] / S[-1]
                
                condition_numbers[name] = condition_number.item()
            except:
                condition_numbers[name] = float('inf')
    
    return condition_numbers


if __name__ == '__main__':

    train()
    # test()
    # get_layers_condition_number()

