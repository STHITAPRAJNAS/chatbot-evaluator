# LLM Critique and Non-LLM Evaluation Methods Research

## LLM-as-a-Judge Approaches

### Overview
LLM-as-a-Judge is an evaluation method that uses large language models to assess the quality of text outputs from LLM-powered products, including chatbots, Q&A systems, and RAG applications. This approach leverages the language understanding capabilities of LLMs to evaluate outputs based on specific criteria.

### Types of LLM Judges

1. **Pairwise Comparison**
   - Gives the LLM two responses and asks it to choose the better one
   - Useful for comparing models, prompts, or configurations
   - Typically done offline during development and testing phases
   - Helps identify which system performs best in specific scenarios

2. **Evaluation by Criteria (Reference-Free)**
   - Asks the LLM to assess a response based on specific qualities
   - Evaluates dimensions like tone, clarity, correctness, or other criteria
   - Can be used for continuous monitoring in production
   - No ground truth or reference answer required

3. **Evaluation by Criteria (Reference-Based)**
   - Provides extra context like a source document or reference
   - Asks the LLM to score the response against this reference
   - Useful for evaluating factual accuracy and faithfulness
   - Similar to RAGAS's reference-based evaluation metrics

### Best Practices for LLM Judges

1. **Define Clear Evaluation Criteria**
   - Use binary or low-precision scoring (yes/no, 1-3 scale)
   - Clearly explain the meaning of each score
   - Break down complex criteria into simpler components

2. **Create a Labeled Dataset**
   - Prepare a small dataset to test the LLM judge
   - Manually label this dataset as ground truth
   - Use diverse examples that challenge evaluation criteria

3. **Craft Effective Evaluation Prompts**
   - Write detailed instructions for the LLM
   - Ask for reasoning to improve quality and debugging
   - Test prompts against labeled data to ensure alignment

4. **Iterate and Refine**
   - Evaluate the LLM judge against ground truth
   - Adjust prompts based on performance
   - Consider domain expert input for setting standards

## Non-LLM Evaluation Methods

### Traditional Retrieval Metrics

1. **Ranking Metrics**
   - Hit Rate: Measures if at least one retrieved chunk is relevant
   - NDCG (Normalized Discounted Cumulative Gain): Evaluates ranking quality
   - MRR (Mean Reciprocal Rank): Measures where the first relevant result appears

2. **Relevance Scoring**
   - Embedding-based similarity: Measures semantic similarity between query and retrieved chunks
   - Keyword matching: Evaluates lexical overlap
   - Entity recall: Checks if important entities are present in retrieved contexts

### Automated Evaluation Without Ground Truth

1. **Context Relevance Evaluation**
   - Score each retrieved chunk individually for relevance to the query
   - Aggregate scores across chunks to evaluate overall retrieval quality
   - Can be implemented using embedding similarity or other algorithmic approaches

2. **Production Monitoring Metrics**
   - Track average relevance scores over time
   - Identify query groups with lower performance
   - Monitor retrieval patterns and distribution changes

### Evidently Tool Capabilities

Evidently is an open-source tool that provides comprehensive RAG evaluation:

1. **Retrieval Evaluation**
   - Score context relevance at the chunk level
   - Run ranking metrics like Hit Rate
   - Evaluate with or without ground truth

2. **Generation Evaluation**
   - Assess generation quality with or without reference answers
   - Evaluate factual correctness and coherence
   - Support for different LLMs as evaluators

3. **Integration Capabilities**
   - Test Suites for automated evaluation
   - Dashboard for visualization and monitoring
   - Python library for integration into existing workflows

## Applications to Text-to-SQL Evaluation

For text-to-SQL applications, specialized evaluation methods include:

1. **SQL Equivalence Checking**
   - LLMSQLEquivalence: Uses LLMs to determine if two SQL queries are functionally equivalent
   - Execution-based comparison: Runs queries and compares results
   - Syntax-based comparison: Analyzes query structure and logic

2. **SQL Result Validation**
   - Compares query results against expected outputs
   - Evaluates if the SQL query retrieves the correct data
   - Can be automated using test datasets with known answers

## Integration with pytest-bdd

To integrate these evaluation methods with pytest-bdd for automated quality checks:

1. **Define BDD Scenarios**
   - Create feature files describing evaluation criteria
   - Define scenarios for different types of queries and expected behaviors
   - Include both retrieval and generation quality checks

2. **Implement Step Definitions**
   - Connect BDD steps to evaluation functions
   - Set up test fixtures for evaluation datasets
   - Configure thresholds for pass/fail criteria

3. **Automate Testing in CI/CD**
   - Run evaluations during build process
   - Generate reports on evaluation metrics
   - Set quality gates based on performance thresholds
