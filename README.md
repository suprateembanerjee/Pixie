# Pixie

## Philosophy

Local LLMs have been made readily available by open source models served by Ollama and repositories like Huggingface. Still, full fledged applications like ChatGPT remain much more useful for most tasks, and not only because the model itself is more proficient. For instance, the same query would result in vastly different experiences using an OpenAI API Key, vs using a built out application like ChatGPT. Such applications do not exist for local models, and the closest we have is the great initiative by H2O AI, which is LMStudio. This is my attempt at creating an application that can leverage hot pluggable LLMs offered by Ollama and build an experience that can be used to perform day to day tasks.

## Usage
From the home directory of the application, run `streamlit run src/app.py`.

## Roadmap
- Pipelines: To build several common use cases that take specific workflows. Examples include Text2SQL pipelines, which would consist of Database profiling, schema linking, with intermitten reasoning and validation steps.
- Guardrailing: To build guardrails at varying stages of the chat interface.
- Code Execution Sandbox: To build a docker sandbox that can be used to dynamically generate connectors to MCP tools, or API connectors to databases.
