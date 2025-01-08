import json
import torch

FIELDS_AND_LIMITS = {
    "revenue": (0, 1e12),  # Revenue can't be negative
    "costOfRevenue": (0, 8e11),  # Costs are typically positive
    "grossProfit": (-4e11, 4e11),  # Can be negative in extreme cases
    "grossProfitRatio": (-5.0, 1.0),  # Updated: can be well below -1
    "researchAndDevelopmentExpenses": (0, 5e10),  # Expenses are typically positive
    "generalAndAdministrativeExpenses": (0, 5e10),
    "sellingAndMarketingExpenses": (0, 5e10),
    "sellingGeneralAndAdministrativeExpenses": (0, 1e11),
    "otherExpenses": (-5e10, 5e10),  # Can be negative (if it's actually income)
    "operatingExpenses": (0, 2e11),
    "costAndExpenses": (0, 9e11),
    "interestIncome": (0, 5e10),
    "interestExpense": (0, 5e10),
    "depreciationAndAmortization": (0, 5e10),
    "ebitda": (-4e11, 4e11),
    "ebitdaratio": (-5.0, 1.0),  # Updated: can be well below -1
    "operatingIncome": (-3e11, 3e11),
    "operatingIncomeRatio": (-5.0, 1.0),  # Updated: can be well below -1
    "totalOtherIncomeExpensesNet": (-5e10, 5e10),
    "incomeBeforeTax": (-3e11, 3e11),
    "incomeBeforeTaxRatio": (-5.0, 1.0),  # Updated: can be well below -1
    "incomeTaxExpense": (-1e11, 1e11),  # Can be negative (tax benefit)
    "netIncome": (-2e11, 2e11),
    "netIncomeRatio": (-5.0, 1.0),  # Updated: can be well below -1
    "eps": (-1000, 1000),
    "epsdiluted": (-1000, 1000),
    "weightedAverageShsOut": (0, 2e10),  # Shares can't be negative
    "weightedAverageShsOutDil": (0, 2e10),
    "cashAndCashEquivalents": (0, 3e11),
    "shortTermInvestments": (-3e11, 3e11),
    "cashAndShortTermInvestments": (-4e11, 4e11),
    "netReceivables": (-2e11, 2e11),
    "inventory": (0, 2e11),
    "otherCurrentAssets": (-2e11, 2e11),
    "totalCurrentAssets": (0, 5e11),
    "propertyPlantEquipmentNet": (0, 5e11),
    "goodwill": (0, 4e11),
    "intangibleAssets": (0, 4e11),
    "goodwillAndIntangibleAssets": (0, 5e11),
    "longTermInvestments": (-5e11, 5e11),
    "taxAssets": (0, 1e11),
    "otherNonCurrentAssets": (-3e11, 3e11),
    "totalNonCurrentAssets": (0, 2e12),
    "otherAssets": (-3e11, 3e11),
    "totalAssets": (0, 3e12),
    "accountPayables": (0, 2e11),
    "shortTermDebt": (0, 3e11),
    "taxPayables": (0, 1e11),
    "deferredRevenue": (0, 1e11),
    "otherCurrentLiabilities": (0, 2e11),
    "totalCurrentLiabilities": (0, 5e11),
    "longTermDebt": (0, 5e11),
    "deferredRevenueNonCurrent": (0, 1e11),
    "deferredTaxLiabilitiesNonCurrent": (0, 1e11),
    "otherNonCurrentLiabilities": (0, 2e11),
    "totalNonCurrentLiabilities": (0, 1e12),
    "otherLiabilities": (0, 2e11),
    "capitalLeaseObligations": (0, 2e11),
    "totalLiabilities": (0, 2e12),
    "preferredStock": (0, 1e11),
    "commonStock": (0, 1e11),
    "retainedEarnings": (-5e11, 5e11),  # Can be negative for companies with accumulated losses
    "accumulatedOtherComprehensiveIncomeLoss": (-1e11, 1e11),
    "othertotalStockholdersEquity": (-2e11, 2e11),
    "totalStockholdersEquity": (-1e12, 1e12),  # Can be negative if accumulated losses exceed capital
    "totalEquity": (-1e12, 1e12),
    "totalLiabilitiesAndStockholdersEquity": (0, 3e12),
    "minorityInterest": (-2e11, 2e11),
    "totalLiabilitiesAndTotalEquity": (0, 3e12),
    "totalInvestments": (-1e12, 1e12),
    "totalDebt": (0, 1e12),
    "netDebt": (-1e12, 1e12),  # Can be negative if cash > debt
    'deferredIncomeTax': (-1e9, 1e9),
    'stockBasedCompensation': (0, 1e10),
    'changeInWorkingCapital': (-1e10, 1e10),
    'accountsReceivables': (-5e9, 5e9),
    'accountsPayables': (-5e9, 5e9),
    'otherWorkingCapital': (-5e9, 5e9),
    'otherNonCashItems': (-1e10, 1e10),
    'netCashProvidedByOperatingActivities': (-2e10, 2e10),
    'investmentsInPropertyPlantAndEquipment': (-5e10, 0),
    'acquisitionsNet': (-1e11, 0),
    'purchasesOfInvestments': (-1e11, 0),
    'salesMaturitiesOfInvestments': (0, 1e11),
    'otherInvestingActivites': (-1e10, 1e10),
    'netCashUsedForInvestingActivites': (-1e11, 1e11),
    'debtRepayment': (-5e10, 0),
    'commonStockIssued': (0, 5e10),
    'commonStockRepurchased': (-5e10, 0),
    'dividendsPaid': (-5e10, 0),
    'otherFinancingActivites': (-1e10, 1e10),
    'netCashUsedProvidedByFinancingActivities': (-5e10, 5e10),
    'effectOfForexChangesOnCash': (-5e9, 5e9),
    'netChangeInCash': (-1e11, 1e11),
    'cashAtEndOfPeriod': (0, 2e11),
    'cashAtBeginningOfPeriod': (0, 2e11),
    'operatingCashFlow': (-2e10, 2e10),
    'capitalExpenditure': (-5e10, 0),
    'freeCashFlow': (-1e11, 1e11),
    "calendarYear": None,
    "reportedCurrency": None,  # This should be a string
}

def nanstd(x): 
    return torch.sqrt(torch.mean(torch.pow(x-torch.nanmean(x,dim=1).unsqueeze(-1),2)))

def niceify_data():
    financial_statements = {}
    for i in range(26):
        with open(f'data/full_reports_{i}.json', 'r+') as file:
            to_add = json.load(file)
            financial_statements = {**financial_statements, **to_add}

    # Count field occurrences
    all_fields_counter = {}
    # financial_statements = dict(islice(financial_statements.items(), 5))
    # fields = [key for key, value in next(iter(financial_statements.values()))[0].items() if isinstance(value, (int, float))] + ["calendarYear", "reportedCurrency"]

    data = []
    currency_indices = {"USD": 0, "EUR": 1, "CAD": 2, "CNY": 3, "IDR": 4, "AUD": 5, "ILS": 6, "GBP": 7, "DKK": 8, "BRL": 9, "NOK": 10, "PHP": 11, "SEK": 12, "TWD": 13, "CHF": 14, "TRY": 15, "NZD": 16, "SGD": 17, "JPY": 18, "HKD": 19, "NGN": 20, "ZAR": 21, "PEN": 22, "MYR": 23, "THB": 24, "CLP": 26, "PLN": 27, "MXN": 28, "NIS": 29, "SAR": 30, "PGK": 31, "COP": 32, "INR": 33, "ARS": 34, "GEL": 35, "GHS": 36, "CZK": 37, "EGP": 38, "RON": 39, "HUF": 40, "RUB": 41, "KRW": 42, "KZT": 43, "NAD": 44, "VND": 45}
    currency_exchange_rates = {"USD": 1.0, "EUR": 0.93, "CAD": 1.37, "CNY": 7.24, "IDR": 15865.0, "AUD": 1.52, "ILS": 3.74, "GBP": 0.8, "DKK": 6.97, "BRL": 4.95, "NOK": 10.85, "PHP": 57.68, "SEK": 10.57, "TWD": 32.45, "CHF": 0.91, "TRY": 32.24, "NZD": 1.65, "SGD": 1.35, "JPY": 151.64, "HKD": 7.82, "NGN": 1487.96, "ZAR": 18.96, "PEN": 3.72, "MYR": 4.77, "THB": 36.26, "CLP": 971.46, "PLN": 4.02, "MXN": 17.06, "NIS": 3.74, "SAR": 3.75, "PGK": 3.8, "COP": 3918.96, "INR": 83.5, "ARS": 879.65, "GEL": 2.68, "GHS": 13.89, "CZK": 23.47, "EGP": 47.6, "RON": 4.64, "HUF": 366.1, "RUB": 91.62, "KRW": 1334.42, "KZT": 450.82, "NAD": 18.96, "VND": 24535.0}
    all_financial_statements = []
    ratio_fields = ['grossProfitRatio', 'ebitdaratio', 'operatingIncomeRatio', 'incomeBeforeTaxRatio', 'netIncomeRatio']
    to_not_scale_with_exchange_rate = set(['calendarYear', 'reportedCurrency'] + ratio_fields)
    invalid_counter = 0
    general_counter = 0
    for company_statements in financial_statements.values():
        general_counter += 1
        list_statements = []
        is_valid = True
        for statement in company_statements:
            if float(statement['calendarYear']) < 2000:
                break
            currency = statement['reportedCurrency']
            if currency not in currency_indices:
                continue
            statement['reportedCurrency'] = currency_indices[currency] + 1
            vector = []
            for field, limit in FIELDS_AND_LIMITS.items():
                value = statement[field]
                try:
                    value = float(value)
                except:
                    value = 0.0
                if field not in to_not_scale_with_exchange_rate:
                    value = value / currency_exchange_rates[currency]
                
                if limit is not None and (value < limit[0] or value > limit[1]):
                    # is_valid = False
                    invalid_counter += 1
                    value = 0
                    # break

                vector.append(value)
            if not is_valid:
                break
            list_statements.append(vector)
        if not is_valid:
            continue
        all_financial_statements += list_statements
        list_statements = torch.tensor(list_statements)
        data.append(list_statements)

    all_financial_statements = torch.tensor(all_financial_statements)
    all_financial_statements = torch.nan_to_num(all_financial_statements, nan=0.0, posinf=0.0, neginf=0.0)
    mask = all_financial_statements[:,:-1] != 0
    # all_financial_statements[:,:-1][~mask] = float('nan')
    # mean = torch.nanmean(all_financial_statements[:,:-1], dim=0)
    # std = 1
    std, mean = torch.std_mean(all_financial_statements[:,:-1], dim=0)
    std = torch.cat((std, torch.tensor(1).unsqueeze(0)), -1)
    mean = torch.cat((mean, torch.tensor(0).unsqueeze(0)), -1)

    normalized_data = []
    for reports in data:
        if reports.shape[0] == 0:
            continue
        mask = reports != 0
        reports[~mask] = float('nan')
        reports = (reports - mean) / std
        reports = torch.nan_to_num(reports, nan=0.0, posinf=0.0, neginf=0.0)
        normalized_data.append(reports)

    print(f'Share of invalid data: {invalid_counter/general_counter}')
    torch.save(std, 'std.pt')
    torch.save(mean, 'mean.pt')
    torch.save(normalized_data, 'full_data.pt')
    torch.save(currency_indices, 'currency_indices.pt')
    torch.save(currency_exchange_rates, 'currency_exchange_rates.pt')

    return normalized_data


niceify_data()