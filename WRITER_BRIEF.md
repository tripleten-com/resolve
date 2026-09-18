# Writer brief for Project 4 lessons (read with SPEC.md)

You are rewriting or writing theory lessons in the Notion database "Advanced AI/ML Lesson Database" (data source `collection://6d4d34ae-1b4a-487a-856e-9c089fb23167`) for Project 4, Resolve. SPEC.md is the contract (company, node names, state fields, tools, stack). This file is how to write and how to ship.

## What a lesson is

- Theory, no tasks. No exercises, quizzes, "try it yourself", or "your turn". Worked code is fine and expected. One "Working with AI" callout at most, as advice, never as an assignment.
- A gentle, complete on-ramp for a beginner who has done Projects 1 to 3 (Python, pandas, PyTorch basics, FastAPI, Docker Compose, the OpenRouter/Ollama setup from chapter 7.3) but has never built an agent. Define every term the first time it appears (tool, schema, node, edge, checkpointer, trace, chunk, embedding, and so on). Explain what a line of code does before or after showing it, not instead of.
- The scenario is always Quillstack and Resolve (SPEC section 1). Use the fixed node names, state fields, ticket types, and tool names from SPEC sections 2 to 4. Use facts from `data/facts.yaml` and articles from `data/kb/` when you need a concrete example; never invent a different company.
- Cross-reference other chapters by chapter name only, in bold, for example **Retrieval and RAG**. Never link to or name an individual lesson number in another chapter. Never reference sprint numbers ("Sprint 5"). Never reference AWS, Bedrock, OpenSearch Serverless, IAM, CloudWatch, SageMaker, Jira Cloud, or Confluence as required parts of the course. Atlassian may appear once, in 11.3, as an optional aside.
- Length: 1,200 to 2,200 words of prose plus code. Worked-example lessons may run longer. Every lesson ends with one or two sentences that say what the next lesson (by title, same chapter) does.

## Voice and prose rules (these are checked)

- Second person, present tense, dry and direct, the register of the Project 1 and 2 pages. Short paragraphs. Fragments are allowed as voice.
- No em dashes or en dashes anywhere. Use commas, colons, or a new sentence.
- No "not X, but Y" constructions. No forced triads. No one-line rhetorical closers. No "Let's dive in", "In this lesson we will", "It's important to note", "In today's world".
- No bold lead-ins on paragraphs or list items ("**Step one.** ..."). Bold is for a defined term on first use and for chapter names. Use colons after code-formatted names in lists instead of dashes.
- Numbers stay soft in prose unless they are facts from facts.yaml or the exact value a code block uses.
- Headings: `##` for sections, `###` sparingly. No heading is a question.

## Code rules

- Python 3.12. Every code block must run against the venv at `resolve/.venv` (Windows: `.venv/Scripts/python`) with `PYTHONPATH=src`, OpenSearch at `http://localhost:9200`, the helpdesk at the port in `.env` (`HELPDESK_URL`), and the key in `resolve/.env`. Run every block you write (a scratch script under `resolve/scratch/<chapter>/` is fine; delete it when done, keep nothing under `src/` unless SPEC section 7 names it). If a block cannot be run (needs a LangSmith key you don't have, or a GitHub push), say so in your report and keep the block minimal and checked by reading the installed library's signatures.
- Installed versions, use their current APIs and no deprecated ones: openai 3.15, langchain-core 1.6, langchain-openai 1.6, langgraph 1.2 (use `interrupt` and `Command`, `ToolNode`, `tools_condition`, `MemorySaver`, `SqliteSaver`), langgraph-checkpoint-sqlite 3.1, langchain-mcp-adapters 0.3 (`MultiServerMCPClient`), mcp 1.30 (`FastMCP`), langsmith 0.12, sentence-transformers 6.0, opensearch-py 3.2, pydantic 2.13, fastapi 0.141 (the helpdesk container pins 0.115 separately). Exact pins are in `requirements.txt`. Check a signature with `python -c "import inspect, x; print(inspect.signature(x.f))"` before you rely on it.
- The model client: raw-loop lessons (10.1, 11.1, 12.1) use `from resolve.llm import chat, ask, MODEL, MODEL_REVIEW` (file `src/resolve/llm.py`; read it). LangChain lessons use `ChatOpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"], model=os.environ["LLM_MODEL"], temperature=0)`. Never hardcode a provider or a key. Say once per lesson, where the client first appears, that the endpoint and model come from `.env` and any OpenAI-compatible provider works.
- Embeddings: `SentenceTransformer("all-MiniLM-L6-v2")`, 384 dims. Index name `resolve-kb`. Tool schemas: OpenAI function format in 11.1 and 12.x; LangChain `@tool` from 11.2 on.
- The first code block in a lesson that introduces a library shows its `pip install` line with the pinned version from `requirements.txt` (read it).
- Keep code blocks short: 10 to 40 lines each, one idea per block, explained around it. Cumulative "here is the whole file again" blocks are not allowed; show the delta.

## Notion markup

Pages use Notion-flavored markdown. Use only what the existing pages use:

```
## Heading
Plain paragraphs. **defined term**. `code`. [link text](https://...)
1. numbered   - bulleted   - [ ] checkbox (task pages only)
<callout icon="📖" color="gray_bg">
	Tab-indented body. Used for theory pointers, notes, and "Working with AI" (icon 🤖, color blue_bg).
</callout>
<details>
<summary>Toggle title</summary>
	Tab-indented body.
</details>
<table header-row="true">
<tr>
<td>col</td>
<td>col</td>
</tr>
</table>
```python
code
```
```
Callout bodies and toggle bodies are indented with a tab. Do not use HTML, images you cannot upload, or "!Placeholder" notes; if a diagram would help, write one line in italics starting with "Diagram idea:" and move on.

## Shipping a lesson

1. Fetch the existing page (notion-fetch by URL) and read it fully before rewriting; keep what is good, fix what SPEC and this brief require, cut all AWS.
2. Write the new body to `resolve/lessons/<chapter>/<nn>-<slug>.md` first, then apply with `notion-update-page` command `replace_content` (and `update_properties` for `Name` and `Learning objective` when they change). Never change `Chapter`, `Project`, `Sprint`, or `Status`.
3. New lessons: `notion-create-pages` with parent `{"type":"data_source_id","data_source_id":"6d4d34ae-1b4a-487a-856e-9c089fb23167"}` and properties `Name`, `Chapter` (must be one of the existing option strings given in your task), `Project` = `P4 - Resolve (Agentic Support System)`, `Sprint`, `Status` = `Ready for review`, `Source` = `New`, `Learning objective`.
4. After applying, notion-fetch the page once and confirm the body rendered (no stray `<` or broken callouts).
5. Report per lesson: URL, word count, code blocks run (yes/no and how), anything you could not verify, and any SPEC inconsistency you found.
