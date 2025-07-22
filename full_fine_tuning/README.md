## Full Fine-Tuning

We test several temporal adaptation strategies using full fine-tuning.  
To illustrate the input transformations, we use a sample text (`CLMET3_1_1_67`, year: 1776, opening: *"Preface Of The Author."*).

| Model | Description | Temporal Embedding | Example Input |
|-------|-------------|---------------------|----------------|
| **B** | OLMo 1B (no temporal context) | None | `Preface Of The Author...` |
| **C** | OLMo 1B + year range tokens | Special token | `[1710–1780] Preface Of The Author...` |
| **D** | OLMo 1B + year range prompt | Natural language | `This is English text written between 1710 and 1780.\n\nPreface Of The Author...` |
| **E** | OLMo 1B + exact year prompt | Natural language | `This text was written in the year 1776.\n\nPreface Of The Author...` |
| **F** | OLMo 1B + exact year as word | Natural language | `1776.\n\nPreface Of The Author...` |

Each model (B–F) has corresponding training and evaluation scripts located in the `full_fine_tuning/` folder.  
Fake-temporal baselines (with wrong year prompts) are also included for models C–F.

Key implementation details:

- Models **D–F** only differ in the natural language prompts generated in `prepare_chunks`
- Model **C** requires modifying the tokenizer to add special tokens
- Model **B** has no explicit temporal input modifications
