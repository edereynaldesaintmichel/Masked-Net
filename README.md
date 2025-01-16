# Masked Net for financial statements prediction.

This repo features a whole financial statements prediciton model. Its architecture is novel (as far as I know). I called it the masked Net because it helps handling sparse data (with a lot of zeros that don't don't bear the value of a normal zero) using a mask system. It helps achieve the same convergence speed as a normal MLP with ~half of the parameters. Nothing too fancy here, but it definitely helps.

## Acknowledgments
The data (which is heavier than the 100MB limit of github repo) comes from [Financial Modeling Prep](https://site.financialmodelingprep.com/)
