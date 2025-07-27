# README Prioritization Fixes

This document outlines the fixes implemented to address the issues with the chatbot not properly referencing README.md files and hallucinating complex answers when simpler ones are available.

## Issues Identified

### 1. README Files Not Being Prioritized
- **Problem**: The chatbot was referencing docs/ without looking at README.md first
- **Root Cause**: The original retriever only prioritized 60% of results from README files
- **Impact**: Users got complex answers from documentation when simple answers existed in README

### 2. Hallucination and Complex Answers
- **Problem**: The chatbot created complex answers even when simple ones were available
- **Root Cause**: System prompt wasn't explicit enough about preferring simple answers
- **Impact**: Confusing and overly complex responses

### 3. Inadequate Retrieval Strategy
- **Problem**: Fixed retrieval parameters didn't optimize for README content
- **Root Cause**: Simple similarity search without README-specific logic
- **Impact**: Poor document selection for user queries

## Fixes Implemented

### 1. Enhanced System Prompt (`app/prompts/system.txt`)

**Changes Made:**
- Added explicit "ALWAYS CHECK README FIRST" directive
- Emphasized "PREFER SIMPLE, DIRECT ANSWERS"
- Added "AVOID HALLUCINATION" guideline
- Created clear answer priority order: README.md → docs/ → other files
- Added instruction to avoid overcomplicating when simple answers exist

**Before:**
```
1. **Prioritize README files**: When multiple sources contain relevant information, prefer information from README.md files...
```

**After:**
```
1. **ALWAYS CHECK README FIRST**: Before providing any answer, prioritize information from README.md files. README files contain the most authoritative and up-to-date information for users.

2. **PREFER SIMPLE, DIRECT ANSWERS**: When multiple sources contain information, choose the simplest, most direct answer. Avoid complex explanations when a straightforward answer exists in the README.
```

### 2. New Hybrid Retriever (`app/rag/chain.py`)

**New Class: `HybridREADMERetriever`**

**Key Features:**
- **README-First Strategy**: Attempts to get README documents first
- **Fallback Mechanism**: If insufficient README content, combines with other docs
- **Smart Filtering**: Better detection of README files (case-insensitive, multiple patterns)
- **Query Analysis**: Determines if a query should prioritize README based on keywords

**Implementation:**
```python
def _get_relevant_documents(self, query: str) -> List[Document]:
    # First, try to get README-specific results
    readme_docs = self._get_readme_documents(query)

    # If we have enough README docs, return them
    if len(readme_docs) >= self.k:
        return readme_docs[:self.k]

    # Otherwise, get general results and prioritize README
    general_docs = self.base_retriever.get_relevant_documents(query)
    # ... prioritize README from general results
```

### 3. Improved README Detection

**Enhanced Logic:**
```python
def _is_readme_document(self, doc: Document) -> bool:
    source = doc.metadata.get("source", "")
    file_name = doc.metadata.get("file", "")

    return (
        "README" in source.upper() or
        "README" in file_name.upper() or
        source.endswith("README.md") or
        file_name.endswith("README.md")
    )
```

### 4. Query Analysis for README Prioritization

**Smart Detection:**
```python
def _should_check_readme(self, query: str) -> bool:
    query_lower = query.lower()
    readme_keywords = [
        "how to", "how do i", "getting started", "install", "setup",
        "usage", "quick start", "basic", "simple", "command", "run",
        "what is", "overview", "introduction", "guide"
    ]
    return any(keyword in query_lower for keyword in readme_keywords)
```

### 5. Increased Answer Length Limit

**Change:** Increased from 120 to 200 words in `app/rag/citations.py`

**Reason:** Allows for more complete answers while still maintaining conciseness

### 6. Enhanced User Prompt Template

**New Template (`app/prompts/user.txt`):**
```
Context: {context}

Question: {question}

Remember: If a simple answer exists in the README, use it. Don't overcomplicate or elaborate unnecessarily. Be direct and concise.
```

## Testing and Validation

### Test Script Created: `test_readme_prioritization.py`

**Features:**
- Tests README prioritization with common questions
- Validates that README files are cited appropriately
- Checks for simple vs complex answer patterns
- Provides feedback on improvement areas

**Example Test Questions:**
- "How do I install this project?"
- "What is the quick start guide?"
- "How do I run the tests?"
- "What are the basic usage instructions?"

## Expected Improvements

### 1. Better README Prioritization
- **Before**: 60% README content in results
- **After**: Up to 100% README content when available, with smart fallback

### 2. Simpler, More Direct Answers
- **Before**: Complex explanations from documentation
- **After**: Simple, direct answers from README when available

### 3. Reduced Hallucination
- **Before**: LLM might elaborate beyond source content
- **After**: Explicit instructions to avoid speculation and stick to sources

### 4. Better User Experience
- **Before**: Confusing, overly complex responses
- **After**: Clear, actionable answers from authoritative sources

## Usage

The fixes are automatically applied when using the QA system:

```bash
# The improvements are built into the system
readme-mentor qa https://github.com/user/repo
```

## Monitoring and Validation

To validate the improvements:

1. **Run the test script:**
   ```bash
   python test_readme_prioritization.py
   ```

2. **Check citation patterns:**
   - Look for README.md citations in responses
   - Verify that simple questions get simple answers

3. **Monitor user feedback:**
   - Track if users report simpler, more helpful responses
   - Check if README content is being properly referenced

## Future Enhancements

1. **A/B Testing**: Compare old vs new retrieval strategies
2. **User Feedback Integration**: Collect feedback on answer quality
3. **Dynamic Prompting**: Adjust prompts based on query type
4. **Citation Quality Metrics**: Track citation accuracy and relevance

## Conclusion

These fixes address the core issues by:
- **Prioritizing README files** through enhanced retrieval logic
- **Preventing hallucination** with explicit system instructions
- **Encouraging simple answers** when they exist in README
- **Improving user experience** with more relevant, actionable responses

The system now properly prioritizes README.md files and provides simpler, more accurate answers based on the most authoritative source in any repository.
