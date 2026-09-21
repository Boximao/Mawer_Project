Citation hit: gold `document_id` appears in `top_k_context` after hybrid retrieve (and rerank if a key is set).
Numeric exact: digit string in the claim appears in the cited chunk.
Abstain: `out_of_corpus` items must not produce a RELEASED answer without HITL; `answer` stays null.
Claim support: each kept claim has `quote` as a substring of an allowlisted chunk.
