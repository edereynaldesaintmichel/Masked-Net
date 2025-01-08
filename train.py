import json
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import torch.optim as optim
from itertools import islice
import torch.nn as nn
from torch.nn import functional as F

# Hyperparams
BATCH_SIZE = 128
LEARNING_RATE = 2e-3
NUM_EPOCHS = 50
N_HEAD = 32
N_LAYER = 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_GRAD_NORM = 1e7
MAX_DROPOUT_PROB = 0.0
NUM_INPUT_FIELDS = 32


# Fields to predict:
OUTPUT_VECTOR_FIELDS = ["interestIncome", "interestExpense", "ebitda", "operatingIncome", "incomeBeforeTax", "netIncome", "eps", "epsdiluted",] # These output fields are for net_income_and_stuff_model.pt
# OUTPUT_VECTOR_FIELDS = ["revenue", "operatingIncome"] # These output fields are for revenue_model.pt


torch.manual_seed(42)

std, mean = 1, 1


class EloisHead(nn.Module):
    def __init__(self, n_embd, head_size, dropout = 0.1):
        super().__init__()
        self.head_size = head_size
        self.values_proj = nn.Linear(n_embd, head_size, bias=False)
        self.cov = nn.Parameter(torch.randn(head_size, head_size))
        self.loadings = nn.Parameter(torch.randn(head_size, head_size))

        # nn.init.xavier_normal_(self.loadings)
        # nn.init.xavier_normal_(self.cov)

    def forward(self, x):
        values = self.values_proj(x[:,0,:])
        mask = self.values_proj(x[:,1,:]) # (B, hs)
        one_minus_mask = self.values_proj(1-x[:,1,:])
        weighted_avg_coefficients = torch.eye(self.head_size).to(DEVICE) + torch.diag_embed(mask) @ self.cov @ torch.diag_embed(one_minus_mask)# (B, hs, hs)

        weighted_avg_coefficients = weighted_avg_coefficients / weighted_avg_coefficients.sum(dim = -1, keepdim=True)
        weighted_avg_coefficients = F.softmax(weighted_avg_coefficients, dim=-1)
        loadings = self.loadings * weighted_avg_coefficients # check if this is a real pointwise operation
        # weighted_avg_coefficients = self.dropout(weighted_avg_coefficients)

        # Add dimension to values using unsqueeze
        values = values.unsqueeze(1)  # (B, 1, hs)
        out = values @ loadings.transpose(-2,-1) # (B, 1, hs)
        out = out.squeeze(1)  # (B, hs) - Remove the middle dimension
        # out = self.batch_norm(out)  # Now BatchNorm1d will work

        return out


class MultiHeadLayer(nn.Module):
    """Multiple EloisHead in parallel."""

    def __init__(self, n_embd, n_head, head_size, dropout = 0.1):
        super().__init__()
        self.heads = nn.ModuleList([EloisHead(n_embd=n_embd, head_size=head_size, dropout=dropout) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.layerNorm = nn.LayerNorm(n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        # out = self.layerNorm(out)
        # out = torch.cat([out, x[:,1,:]], dim=1)

        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd, dropout = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            # nn.Linear(n_embd, 2*n_embd),
            nn.GELU(),
            nn.Linear(1*n_embd, n_embd),
            # nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.net(x)
        # out = torch.cat([out, x[:,1,:]], dim=1)

        return out

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.mh = MultiHeadLayer(n_embd=n_embd, n_head=n_head, head_size=head_size)
        self.ffwd = FeedFoward(n_embd)

    def forward(self, x):
        out = self.mh(x)
        # out = out + x[:,0,:]
        out = self.ffwd(out)
        out = torch.cat([out.unsqueeze(1), x[:,1,:].unsqueeze(1)], dim=1)

        return out


class FinancialDropout(nn.Module):
    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x, targets, dropout_factor):
        if not self.training:
            return x, targets

        # x shape is (batch_size, 2, n_embd)
        values = x[:, 0, :]  # (batch_size, n_embd)
        masks = x[:, 1, :]   # (batch_size, n_embd)

        # Create random dropout mask
        random_mask = (torch.rand_like(values) > self.drop_prob * dropout_factor).float()

        # Generate random scaling factors between 0.5 and 2 for each sample in the batch
        scale_factors = torch.randn(values.shape[0], 1, device=values.device) * 0.5 + 1

        # Apply dropout and scaling to values, only dropout to masks
        new_values = values * random_mask * scale_factors
        new_targets = targets * scale_factors
        new_masks = masks * random_mask

        # Recombine into original format
        return torch.stack([new_values, new_masks], dim=1), new_targets

class EloisNet(nn.Module):
    def __init__(self, input_size, output_size, number_of_currencies, dropout_prob=0.1):
        super().__init__()
        self.currency_embedding = nn.Embedding(num_embeddings=number_of_currencies, embedding_dim=2)
        self.lm_head = nn.Sequential(
            nn.Linear(input_size + 1, input_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(input_size, input_size//2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(input_size//2, input_size//4),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(input_size//4, input_size//4),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(input_size//4, input_size//4),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(input_size//4, input_size//4),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(input_size//4, output_size)
        )

    #     self.apply(self._init_weights)

    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         torch.nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, input, targets=None, dropout_factor=1):
        main_features = input[:, :-1]
        currency_idx = input[:, -1].long()
        currency_embedding = self.currency_embedding(currency_idx)  # Shape: (batch_size, 2)
        combined_input = torch.cat([main_features, currency_embedding], dim=1)

        output = self.lm_head(combined_input)  # (B, output_size)

        if targets is None:
            return output, None

        loss = F.l1_loss(output, targets)
        return output, loss





def get_val_dataloader(full_dataset, output_field_indices, batch_size):
    input_data = []
    ground_truth = []
    for company_statements in full_dataset:
        if (company_statements.shape[0] < 4):
            continue
        input_data.append(torch.flatten(company_statements[-4:-1]))
        sub_ground_truth = torch.index_select(company_statements[-1], dim=-1, index=output_field_indices)
        ground_truth.append(sub_ground_truth)

    ground_truth = torch.stack(ground_truth)  # Changed from torch.tensor to torch.stack
    input_data = torch.stack(input_data)      # Changed from torch.tensor to torch.stack
    val_dataset = TensorDataset(input_data, ground_truth)

    return DataLoader(val_dataset, batch_size=batch_size)


def get_train_dataloader(full_dataset, output_field_indices, batch_size):
    input_data = []
    ground_truth = []
    for i in range(0, 50):
        for company_statements in full_dataset:
            if (company_statements.shape[0] < i + 5):
                continue
            input_data.append(torch.flatten(company_statements[-i-5:-i-2]))
            sub_ground_truth = torch.index_select(company_statements[-i-2], dim=-1, index=output_field_indices)
            ground_truth.append(sub_ground_truth)

    ground_truth = torch.stack(ground_truth)  # Changed from torch.tensor to torch.stack
    input_data = torch.stack(input_data)      # Changed from torch.tensor to torch.stack
    train_dataset = TensorDataset(input_data, ground_truth)

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


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


def get_output_gradients(model, val_loader, device, output_field:str, input_fields:list, output_fields:list):
    model.eval()
    all_gradients = []

    output_index = output_fields.index(output_field)

    input_dict = {f'{input_fields[i]}': i for i in range(74)}

    for data, targets in val_loader:
        data = data.to(device)
        data.requires_grad = True

        output, _ = model(data, targets.to(device))
        output[:, output_index].sum().backward()
        
        gradients = data.grad.clone().sum(dim=0)
        immediate_gradients_dict = {key: (float(gradients[value]), float(gradients[value+74]), float(gradients[value+148])) for key, value in input_dict.items()}
        all_gradients.append(gradients)  # Store gradients for this batch

    all_gradients = torch.stack(all_gradients, dim=0)
    all_gradients_sum = all_gradients.sum(dim=0)
    all_gradients_sum /= all_gradients_sum.max()
    full_gradients_dict = {key: (float(all_gradients_sum[value]), float(all_gradients_sum[value+74]), float(all_gradients_sum[value+148])) for key, value in input_dict.items()}

    return full_gradients_dict


def train():
    OUTPUT_SIZE = len(OUTPUT_VECTOR_FIELDS)
    # Load and prepare data
    fields = ["revenue", "costOfRevenue", "grossProfit", "grossProfitRatio", "researchAndDevelopmentExpenses", "generalAndAdministrativeExpenses", "sellingAndMarketingExpenses", "sellingGeneralAndAdministrativeExpenses", "otherExpenses", "operatingExpenses", "costAndExpenses", "interestIncome", "interestExpense", "depreciationAndAmortization", "ebitda", "ebitdaratio", "operatingIncome", "operatingIncomeRatio", "totalOtherIncomeExpensesNet", "incomeBeforeTax", "incomeBeforeTaxRatio", "incomeTaxExpense", "netIncome", "netIncomeRatio", "eps", "epsdiluted", "weightedAverageShsOut", "weightedAverageShsOutDil", "cashAndCashEquivalents", "shortTermInvestments", "cashAndShortTermInvestments", "netReceivables", "inventory", "otherCurrentAssets", "totalCurrentAssets", "propertyPlantEquipmentNet", "goodwill", "intangibleAssets", "goodwillAndIntangibleAssets", "longTermInvestments", "taxAssets", "otherNonCurrentAssets", "totalNonCurrentAssets", "otherAssets", "totalAssets", "accountPayables", "shortTermDebt", "taxPayables", "deferredRevenue", "otherCurrentLiabilities", "totalCurrentLiabilities", "longTermDebt", "deferredRevenueNonCurrent", "deferredTaxLiabilitiesNonCurrent", "otherNonCurrentLiabilities", "totalNonCurrentLiabilities", "otherLiabilities", "capitalLeaseObligations", "totalLiabilities", "preferredStock", "commonStock", "retainedEarnings", "accumulatedOtherComprehensiveIncomeLoss", "othertotalStockholdersEquity", "totalStockholdersEquity", "totalEquity", "totalLiabilitiesAndStockholdersEquity", "minorityInterest", "totalLiabilitiesAndTotalEquity", "totalInvestments", "totalDebt", "netDebt", "calendarYear", "reportedCurrency"]
    output_field_indices = torch.tensor([fields.index(field) for field in OUTPUT_VECTOR_FIELDS])
    full_dataset = torch.load('full_data.pt')
    val_data_loader = get_val_dataloader(full_dataset=full_dataset, output_field_indices=output_field_indices, batch_size=BATCH_SIZE)
    train_data_loader = get_train_dataloader(full_dataset=full_dataset, output_field_indices=output_field_indices, batch_size=BATCH_SIZE)

    # Initialize model
    model = EloisNet(
        input_size=222,
        output_size=OUTPUT_SIZE,
        number_of_currencies=46,
        dropout_prob=MAX_DROPOUT_PROB,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    best_val_loss = float('inf')

    target_loss = get_target_loss(val_loader=val_data_loader, output_data_indices=output_field_indices+148)

    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_data_loader, optimizer, DEVICE)
        val_loss = validate(model, val_data_loader, DEVICE)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'test_model.pt')

        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Best VL: {best_val_loss:.4f}, Target: {target_loss:.4f}')

def test():
        # Fields to predict
    OUTPUT_SIZE = len(OUTPUT_VECTOR_FIELDS)
    # Load and prepare data
    fields = ["revenue", "costOfRevenue", "grossProfit", "grossProfitRatio", "researchAndDevelopmentExpenses", "generalAndAdministrativeExpenses", "sellingAndMarketingExpenses", "sellingGeneralAndAdministrativeExpenses", "otherExpenses", "operatingExpenses", "costAndExpenses", "interestIncome", "interestExpense", "depreciationAndAmortization", "ebitda", "ebitdaratio", "operatingIncome", "operatingIncomeRatio", "totalOtherIncomeExpensesNet", "incomeBeforeTax", "incomeBeforeTaxRatio", "incomeTaxExpense", "netIncome", "netIncomeRatio", "eps", "epsdiluted", "weightedAverageShsOut", "weightedAverageShsOutDil", "cashAndCashEquivalents", "shortTermInvestments", "cashAndShortTermInvestments", "netReceivables", "inventory", "otherCurrentAssets", "totalCurrentAssets", "propertyPlantEquipmentNet", "goodwill", "intangibleAssets", "goodwillAndIntangibleAssets", "longTermInvestments", "taxAssets", "otherNonCurrentAssets", "totalNonCurrentAssets", "otherAssets", "totalAssets", "accountPayables", "shortTermDebt", "taxPayables", "deferredRevenue", "otherCurrentLiabilities", "totalCurrentLiabilities", "longTermDebt", "deferredRevenueNonCurrent", "deferredTaxLiabilitiesNonCurrent", "otherNonCurrentLiabilities", "totalNonCurrentLiabilities", "otherLiabilities", "capitalLeaseObligations", "totalLiabilities", "preferredStock", "commonStock", "retainedEarnings", "accumulatedOtherComprehensiveIncomeLoss", "othertotalStockholdersEquity", "totalStockholdersEquity", "totalEquity", "totalLiabilitiesAndStockholdersEquity", "minorityInterest", "totalLiabilitiesAndTotalEquity", "totalInvestments", "totalDebt", "netDebt", "calendarYear", "reportedCurrency"]
    output_field_indices = torch.tensor([fields.index(field) for field in OUTPUT_VECTOR_FIELDS])
    full_dataset = torch.load('full_data.pt')
    val_data_loader = get_val_dataloader(full_dataset=full_dataset, output_field_indices=output_field_indices, batch_size=BATCH_SIZE)

     # Initialize model
    model = EloisNet(
        input_size=222,
        output_size=OUTPUT_SIZE,
        number_of_currencies=46,
        dropout_prob=MAX_DROPOUT_PROB,
    ).to(DEVICE)


    model.load_state_dict(torch.load('net_income_and_stuff_model.pt', map_location=torch.device('cpu')))
    std = torch.load('std.pt')
    mean = torch.load('mean.pt')
    currency_indices = torch.load('currency_indices.pt')
    currency_exchange_rates = torch.load('currency_exchange_rates.pt')



    gradients = get_output_gradients(model, val_data_loader, DEVICE, output_field='epsdiluted', input_fields=fields, output_fields=OUTPUT_VECTOR_FIELDS)

    return gradients

if __name__ == '__main__':

    test()
    # train()

