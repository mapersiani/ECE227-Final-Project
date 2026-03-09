# Data Layout

- `nodes.json`: persona dataset (personas mode). Node count is dynamic.
- `senate_nodes.json`: US senator dataset (senate mode). Node count is dynamic.

Node format: each object has `name`, `prompt`, `style`, `initial`.
- `name`: must follow `party_firstname_lastname` (e.g., `democrat_Joe_Biden`, `republican_Mitch_McConnell`).

Select via `--persona-set personas` (default) or `--persona-set senate`.
