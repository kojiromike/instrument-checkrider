from anthropic.types import TextBlock, ToolUseBlock

from .util import ANTHROPIC, INDEX, VOYAGE


def query_with_context(
    user_query: list[str], top_k: int = 3
) -> list[TextBlock | ToolUseBlock]:
    # Generate embedding for the query using VoyageAI
    query_embedding = VOYAGE.embed(user_query).embeddings[0]

    # Query Pinecone
    results = INDEX.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    # Format context from retrieved documents
    context = "\n\n".join([match.metadata.get("text", "") for match in results.matches])

    # Create prompt with context
    prompt = f"""Here is some relevant context: {context}

Based on this context, please answer the following question: {user_query}"""

    # Query Claude
    message = ANTHROPIC.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content
