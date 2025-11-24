"""
LangExtract extraction schemas for Paul Graham essays.

This module defines extraction schemas tailored for Paul Graham's writing style,
which typically includes: personal narratives, startup/business advice, life lessons,
technology insights, and philosophical observations.
"""

import textwrap
import langextract as lx


def get_paul_graham_essay_schema():
    """
    Get extraction schema for Paul Graham essays.
    
    This schema extracts:
    - Key concepts and ideas (e.g., "startups", "programming", "YC")
    - Advice and lessons (actionable insights)
    - Personal experiences (autobiographical elements)
    - Time references (years, periods, events)
    - People and organizations mentioned
    
    Returns:
    dict: Contains 'prompt' and 'examples' for LangExtract
    """
    
    prompt = textwrap.dedent(
        """\
        Extract structured information from Paul Graham's essay text.
        Focus on identifying key concepts, advice, personal experiences, and entities.
        Use exact text for extractions. Do not paraphrase or create overlapping entities.
        
        For each extraction, provide attributes from these predefined sets:
        
        Concept attributes:
        - category: ["technology", "business", "startups", "programming", "art", "education", "life", "philosophy", "writing"]
        - importance: ["high", "medium", "low"]
        
        Advice attributes:
        - type: ["actionable", "cautionary", "philosophical", "tactical"]
        - domain: ["career", "startups", "learning", "creativity", "relationships", "decision_making"]
        
        Experience attributes:
        - period: ["childhood", "college", "grad_school", "viaweb", "yc", "post_yc", "general"]
        - sentiment: ["positive", "negative", "neutral", "mixed"]
        
        Entity attributes (people/organizations):
        - role: ["founder", "colleague", "investor", "friend", "company", "institution"]
        - significance: ["major", "minor"]
        
        Time attributes:
        - decade: ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s", "unspecified"]
        - specificity: ["exact_year", "approximate", "era"]
        
        Focus on extracting information that would be useful for:
        - Finding specific advice or lessons
        - Identifying personal narratives and experiences
        - Connecting concepts and ideas
        - Timeline construction
        - Thematic analysis
        """
    )
    
    # Provide few-shot examples to guide extraction
    examples = [
        lx.data.ExampleData(
            text="I learned to program in BASIC when I was about 13 years old, on a TRS-80 computer. It was the most exciting thing I'd ever done.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="experience",
                    extraction_text="learned to program in BASIC when I was about 13 years old",
                    attributes={"period": "childhood", "sentiment": "positive"}
                ),
                lx.data.Extraction(
                    extraction_class="concept",
                    extraction_text="program in BASIC",
                    attributes={"category": "programming", "importance": "high"}
                ),
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="TRS-80",
                    attributes={"role": "company", "significance": "minor"}
                ),
                lx.data.Extraction(
                    extraction_class="time",
                    extraction_text="about 13 years old",
                    attributes={"decade": "1970s", "specificity": "approximate"}
                ),
            ],
        ),
        lx.data.ExampleData(
            text="The most important thing is to work on something you're genuinely interested in. Don't force yourself to work on something because you think you should.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="advice",
                    extraction_text="work on something you're genuinely interested in",
                    attributes={"type": "actionable", "domain": "career"}
                ),
                lx.data.Extraction(
                    extraction_class="advice",
                    extraction_text="Don't force yourself to work on something because you think you should",
                    attributes={"type": "cautionary", "domain": "decision_making"}
                ),
                lx.data.Extraction(
                    extraction_class="concept",
                    extraction_text="genuinely interested",
                    attributes={"category": "life", "importance": "high"}
                ),
            ],
        ),
        lx.data.ExampleData(
            text="When we started Y Combinator in 2005, we had no idea it would grow into what it is today. We just wanted to help startups the way we wished someone had helped us.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="Y Combinator",
                    attributes={"role": "company", "significance": "major"}
                ),
                lx.data.Extraction(
                    extraction_class="time",
                    extraction_text="2005",
                    attributes={"decade": "2000s", "specificity": "exact_year"}
                ),
                lx.data.Extraction(
                    extraction_class="experience",
                    extraction_text="started Y Combinator in 2005",
                    attributes={"period": "yc", "sentiment": "positive"}
                ),
                lx.data.Extraction(
                    extraction_class="concept",
                    extraction_text="help startups",
                    attributes={"category": "startups", "importance": "high"}
                ),
            ],
        ),
        lx.data.ExampleData(
            text="The best way to get startup ideas is to become the kind of person who has them. Live in the future and build what seems interesting.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="advice",
                    extraction_text="become the kind of person who has them",
                    attributes={"type": "philosophical", "domain": "startups"}
                ),
                lx.data.Extraction(
                    extraction_class="advice",
                    extraction_text="Live in the future and build what seems interesting",
                    attributes={"type": "actionable", "domain": "creativity"}
                ),
                lx.data.Extraction(
                    extraction_class="concept",
                    extraction_text="startup ideas",
                    attributes={"category": "startups", "importance": "high"}
                ),
            ],
        ),
    ]
    
    return {
        "prompt": prompt,
        "examples": examples,
    }


def get_simple_paul_graham_schema():
    """
    Get a simpler extraction schema focusing on main themes and advice.
    
    This is a lighter-weight version that extracts only:
    - Main topics/themes
    - Key advice
    - Important entities
    
    Returns:
    dict: Contains 'prompt' and 'examples' for LangExtract
    """
    
    prompt = textwrap.dedent(
        """\
        Extract main themes, advice, and key entities from Paul Graham's essay.
        Use exact text. Do not paraphrase.
        
        Topic attributes:
        - theme: ["startups", "programming", "education", "creativity", "career", "life_lessons"]
        
        Advice attributes:
        - actionable: ["yes", "no"]
        
        Entity attributes:
        - type: ["person", "company", "technology", "concept"]
        """
    )
    
    examples = [
        lx.data.ExampleData(
            text="If you want to start a startup, the most important thing is to build something people want.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="topic",
                    extraction_text="start a startup",
                    attributes={"theme": "startups"}
                ),
                lx.data.Extraction(
                    extraction_class="advice",
                    extraction_text="build something people want",
                    attributes={"actionable": "yes"}
                ),
            ],
        ),
        lx.data.ExampleData(
            text="I met Robert Morris at Harvard, and we became good friends.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="Robert Morris",
                    attributes={"type": "person"}
                ),
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="Harvard",
                    attributes={"type": "company"}
                ),
            ],
        ),
    ]
    
    return {
        "prompt": prompt,
        "examples": examples,
    }


# Export schemas
SCHEMAS = {
    "paul_graham_detailed": get_paul_graham_essay_schema,
    "paul_graham_simple": get_simple_paul_graham_schema,
}


def get_schema(schema_name="paul_graham_detailed"):
    """
    Get extraction schema by name.
    
    Args:
        schema_name (str): Name of the schema to retrieve
        
    Returns:
        dict: Schema configuration with prompt and examples
    """
    if schema_name not in SCHEMAS:
        raise ValueError(f"Unknown schema: {schema_name}. Available: {list(SCHEMAS.keys())}")
    
    return SCHEMAS[schema_name]()
