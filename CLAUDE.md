# Dual-Patch Transformer

This model is designed for time series data with both numeric and categorical features,
handling missing values efficiently by encoding them in a dual-patch format.


## Development

Use `uv` to install the required packages:
use `ruff to format and check the code:

```bash
uvx ruff format .
uvx ruff check . ---fix
```


DO NOT run the model yourself. The model outputs tqdm progress bars which will blow
up your context window. Ask me to run it for you if you need to see the output.


## Architecture 

Review @instructions.md for a detailed architecture overview.