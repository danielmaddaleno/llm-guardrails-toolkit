![Tests](https://github.com/danielmaddaleno/llm-guardrails-toolkit/actions/workflows/tests.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

# LLM Guardrails Toolkit

Lightweight Python framework for adding input/output guardrails to LLM applications. Designed for enterprise GenAI deployments that need content filtering, PII detection, prompt injection prevention, and response validation.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![AWS](https://img.shields.io/badge/AWS-Bedrock-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

When deploying LLMs in enterprise environments, you need guardrails to ensure safe, compliant, and useful responses. This toolkit provides a modular pipeline of validators that run before and after every LLM call.

## Features

- 🛡️ **Prompt Injection Detection** — Detects and blocks common injection patterns
- 🔒 **PII Redaction** — Automatically masks emails, phone numbers, SSNs, and credit cards
- 📏 **Token Budget Control** — Enforces input/output token limits
- ✅ **Response Validation** — Schema validation, toxicity checks, and hallucination flags
- 🔌 **Pluggable Architecture** — Easy to add custom validators
- ☁️ **AWS Bedrock Integration** — Works out of the box with Bedrock models

## Quick Start

```python
from guardrails import GuardrailsPipeline, PIIRedactor, PromptInjectionDetector, TokenBudget

pipeline = GuardrailsPipeline(
    input_guards=[
        PromptInjectionDetector(),
        PIIRedactor(),
        TokenBudget(max_input_tokens=2000),
    ],
    output_guards=[
        PIIRedactor(),
        TokenBudget(max_output_tokens=1000),
    ],
)

# Validate input before sending to LLM
safe_prompt = pipeline.validate_input("Summarize this customer record: John Doe, john@email.com, SSN 123-45-6789")
# -> "Summarize this customer record: [NAME], [EMAIL], SSN [SSN]"

# Validate output after receiving from LLM
safe_response = pipeline.validate_output(llm_response)
```

## Project Structure

```
├── guardrails/
│   ├── __init__.py
│   ├── pipeline.py            # Core pipeline orchestrator
│   ├── validators/
│   │   ├── __init__.py
│   │   ├── pii_redactor.py    # PII detection & masking
│   │   ├── injection.py       # Prompt injection detection
│   │   ├── token_budget.py    # Token limit enforcement
│   │   └── toxicity.py        # Toxicity scoring
│   └── integrations/
│       ├── __init__.py
│       └── bedrock.py         # AWS Bedrock wrapper
├── tests/
│   ├── test_pii.py
│   ├── test_injection.py
│   └── test_pipeline.py
├── examples/
│   └── bedrock_example.py
├── requirements.txt
└── README.md
```

## License

MIT


## Installation

```bash
git clone https://github.com/danielmaddaleno/llm-guardrails-toolkit.git
cd llm-guardrails-toolkit
pip install -r requirements.txt
```

## Usage

See `docs/` for detailed examples.

## Configuration

Configuration files live in `configs/`. Copy the sample and edit.

## Development

```bash
make install  # Install deps
make test     # Run tests
make lint     # Linters
```

## Architecture

See [docs/architecture.md](docs/architecture.md).

## Roadmap

- [ ] Improve test coverage
- [ ] Add benchmarks
- [ ] Docker support

## Acknowledgements

Built with Python and open-source tools.
