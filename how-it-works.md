AI agents today can be lightweight or distributed, simple or complex, but their core is always the same: they reason and plan, they orchestrate tool calls, and most importantly, they can learn over time. Azure Managed Redis -Microsoft’s first-party Redis Enterprise, makes memory management both seamless and production-ready in the Azure ecosystem. 

Understanding agent memory architecture 

An agent’s usefulness depends on how and what it remembers. A typical agentic memory architecture contains two key components: 

Working short-term memory 

A play-by-play of the current interaction, so the agent can track ongoing conversation history and state without asking you to repeat yourself. Microsoft Agent Framework can help manage this short-term or working memory tied to a user’s session. In the Agent Framework SDK, it is implemented as Thread with an associated ChatMessageStore that can be offloaded to Redis.  

Durable long-term Memory 

Preferences, behavioral patterns, durable facts and interests pieced together from many interactions so the agent can learn and adapt over time. Long-term memory is typically shared across sessions. Within Microsoft Agent Framework SDK, long-term memory is provided through ContextProviders.  

Azure Managed Redis can be integrated directly as the context provider to maximize performance, control and advanced features. One of the popular context providers is powered by Mem0. Mem0 handles intelligent memory extraction, deduplication, and contextual grounding while recording these memories in Azure Managed Redis. 

In this agent, here’s a snapshot of how the memory flow works with Agent Framework and Redis while processing a user request: 

- Prompt ingestion: The user sends a prompt through the UI. The Agent Framework collects that prompt and fetches relevant memories from Azure Managed Redis using semantic search. 
- Context assembly: The Agent Framework composes the system prompt, user input prompt, chat history, and retrieved memory.  
- LLM decision and tool orchestration: The LLM (such as Azure OpenAI) decides whether to call tools like flight or hotel search, answer directly, or loop through multiple steps. Contextual Memory itself is one such tool — the model can choose to store useful facts for later recall and filter out irrelevant ones, keeping the information flow tight and focused 
- Response creation: Once enough information is gathered, the LLM returns a final response to the UI. 
- Asynchronous memory update: The user’s message and relevant assistant responses are pushed into memory. Durable facts are extracted and stored in Azure Managed Redis as vectors, JSON, or metadata. 
- Context for the next interaction: The next time the user interacts, the agent uses what’s stored in Azure Managed Redis to recall, personalize, and adapt. The interaction is richer because context is preserved. 
