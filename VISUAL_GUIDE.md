# Visual Guide: Metadata Extraction Options

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│                      METADATA EXTRACTION DECISION TREE                          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

                              Do you need metadata?
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                   YES                                 NO
                    │                                   │
                    ▼                                   ▼
         Do you have budget for        ┌───────────────────────────┐
            API calls (GPT-4)?          │   Option: None (Basic)    │
                    │                   │   Speed: ⚡⚡⚡           │
            ┌───────┴────────┐          │   Cost: FREE              │
            │                │          │   Metadata: Page numbers  │
           YES              NO          └───────────────────────────┘
            │                │
            │                ▼
            │    ┌───────────────────────────┐
            │    │   Option: EntityExtractor │
            │    │   Speed: ⚡⚡              │
            │    │   Cost: FREE              │
            │    │   Metadata: Named entities│
            │    └───────────────────────────┘
            │
            ▼
   Need semantic metadata
     (concepts, advice)?
            │
      ┌─────┴─────┐
      │           │
     YES         NO
      │           │
      ▼           ▼
  Need both   ┌────────────────────────────┐
  entities &  │  Option: LangExtract       │
  semantic?   │  Speed: ⚡                 │
      │       │  Cost: ~$2/30pg            │
  ┌───┴───┐   │  Metadata: Rich semantic   │
  │       │   └────────────────────────────┘
 YES     NO
  │       │
  ▼       │
┌──────────────────────────┐
│   Option: Both           │
│   Speed: ⚡ (slowest)    │
│   Cost: ~$2/30pg         │
│   Metadata: Maximum      │
└──────────────────────────┘




┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│                         METADATA FIELDS COMPARISON                              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

NONE (Basic)
├── source (page number)
├── file_path
├── file_name
├── file_type
├── file_size
└── page_label

ENTITYEXTRACTOR
├── source (page number)
├── file_path
├── file_name
├── file_type
├── file_size
├── page_label
├── PER (persons) ◄── NEW
├── ORG (organizations) ◄── NEW
├── LOC (locations) ◄── NEW
└── [other entity types] ◄── NEW

LANGEXTRACT
├── source (page number)
├── file_path
├── file_name
├── file_type
├── file_size
├── page_label
├── langextract_concepts ◄── NEW
├── concept_categories ◄── NEW
├── concept_importance ◄── NEW
├── langextract_advice ◄── NEW
├── advice_types ◄── NEW
├── advice_domains ◄── NEW
├── langextract_entities ◄── NEW
├── entity_roles ◄── NEW
├── entity_names ◄── NEW
├── langextract_experiences ◄── NEW
├── experience_periods ◄── NEW
├── experience_sentiments ◄── NEW
├── time_references ◄── NEW
└── time_decades ◄── NEW

BOTH (EntityExtractor + LangExtract)
├── source (page number)
├── file_path
├── file_name
├── file_type
├── file_size
├── page_label
├── PER (persons) ◄── EntityExtractor
├── ORG (organizations) ◄── EntityExtractor
├── LOC (locations) ◄── EntityExtractor
├── [other entity types] ◄── EntityExtractor
├── langextract_concepts ◄── LangExtract
├── concept_categories ◄── LangExtract
├── concept_importance ◄── LangExtract
├── langextract_advice ◄── LangExtract
├── advice_types ◄── LangExtract
├── advice_domains ◄── LangExtract
├── langextract_entities ◄── LangExtract
├── entity_roles ◄── LangExtract
├── entity_names ◄── LangExtract
├── langextract_experiences ◄── LangExtract
├── experience_periods ◄── LangExtract
├── experience_sentiments ◄── LangExtract
├── time_references ◄── LangExtract
└── time_decades ◄── LangExtract




┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│                            PROCESSING PIPELINE                                  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

OPTION 1: None (Basic)
┌──────────┐    ┌──────────────┐    ┌──────────────┐
│ Load PDF │───▶│ Split Chunks │───▶│ Basic        │
│          │    │ (Sentence    │    │ Metadata     │
│          │    │  Splitter)   │    │ Only         │
└──────────┘    └──────────────┘    └──────────────┘
                                           │
                                           ▼
                                    ┌──────────────┐
                                    │ Store to DB  │
                                    └──────────────┘


OPTION 2: EntityExtractor
┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Load PDF │───▶│ Split Chunks │───▶│ Entity       │───▶│ Entity       │
│          │    │ (Sentence    │    │ Extractor    │    │ Metadata     │
│          │    │  Splitter)   │    │ (Local HF)   │    │ Added        │
└──────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                │
                                                                ▼
                                                         ┌──────────────┐
                                                         │ Store to DB  │
                                                         └──────────────┘


OPTION 3: LangExtract
┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Load PDF │───▶│ Split Chunks │───▶│ LangExtract  │───▶│ Semantic     │
│          │    │ (Sentence    │    │ (GPT-4 API)  │    │ Metadata     │
│          │    │  Splitter)   │    │              │    │ Added        │
└──────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                │
                                                                ▼
                                                         ┌──────────────┐
                                                         │ Store to DB  │
                                                         └──────────────┘


OPTION 4: Both
┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Load PDF │───▶│ Split Chunks │───▶│ Entity       │───▶│ LangExtract  │───▶│ Combined     │
│          │    │ (Sentence    │    │ Extractor    │    │ (GPT-4 API)  │    │ Metadata     │
│          │    │  Splitter)   │    │ (Local HF)   │    │              │    │              │
└──────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                                     │
                                                                                     ▼
                                                                              ┌──────────────┐
                                                                              │ Store to DB  │
                                                                              └──────────────┘




┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│                         COST & TIME COMPARISON                                  │
│                           (30-page document)                                    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

None (Basic)
    Time: ████ 10 seconds
    Cost: FREE
    ────────────────────────────────────────────────────────────────────────▶

EntityExtractor
    Time: ████████████ 30 seconds
    Cost: FREE
    ────────────────────────────────────────────────────────────────────────▶

LangExtract
    Time: ████████████████████████████████████████████████████████ 15 minutes
    Cost: ~$2.00 (GPT-4o API)
    ────────────────────────────────────────────────────────────────────────▶

Both
    Time: ████████████████████████████████████████████████████████████ 16 minutes
    Cost: ~$2.00 (GPT-4o API)
    ────────────────────────────────────────────────────────────────────────▶




┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│                            QUERY CAPABILITY MAP                                 │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

Query Type                     │ None │ Entity │ Lang │ Both │
───────────────────────────────┼──────┼────────┼──────┼──────┤
Basic page retrieval           │  ✓   │   ✓    │  ✓   │  ✓   │
Document summarization         │  ✓   │   ✓    │  ✓   │  ✓   │
                               │      │        │      │      │
Entity-based queries:          │      │        │      │      │
  - List people                │  ✗   │   ✓    │  ✓   │  ✓✓  │
  - Find organizations         │  ✗   │   ✓    │  ✓   │  ✓✓  │
  - Locate places              │  ✗   │   ✓    │  ✓   │  ✓✓  │
                               │      │        │      │      │
Semantic queries:              │      │        │      │      │
  - Conceptual search          │  ✗   │   ✗    │  ✓   │  ✓   │
  - Advice extraction          │  ✗   │   ✗    │  ✓   │  ✓   │
  - Experience analysis        │  ✗   │   ✗    │  ✓   │  ✓   │
  - Temporal analysis          │  ✗   │   ✗    │  ✓   │  ✓   │
                               │      │        │      │      │
Complex cross-type queries:    │      │        │      │      │
  - Entity + Concept           │  ✗   │   ✗    │  ✗   │  ✓   │
  - Entity + Advice            │  ✗   │   ✗    │  ✗   │  ✓   │
  - Entity + Experience        │  ✗   │   ✗    │  ✗   │  ✓   │

Legend: ✓ = Good, ✓✓ = Excellent, ✗ = Not supported




┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│                         RECOMMENDED USAGE PATTERNS                              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

Development Phase
    │
    ├─ Prototype & Testing ────────▶ Use: None or EntityExtractor
    │                                Why: Fast iteration, no costs
    │
    ├─ Feature Development ────────▶ Use: EntityExtractor
    │                                Why: Realistic metadata, still free
    │
    └─ Pre-Production Testing ─────▶ Use: LangExtract (small sample)
                                     Why: Verify quality before scaling


Production Phase
    │
    ├─ Budget-Constrained ─────────▶ Use: EntityExtractor
    │                                Why: Free, good entity recognition
    │
    ├─ Quality-Focused ────────────▶ Use: LangExtract
    │                                Why: Rich metadata, worth the cost
    │
    └─ Maximum Capability ─────────▶ Use: Both
                                     Why: Best of both worlds


Research/Analysis
    │
    └─ Academic/Deep Analysis ─────▶ Use: LangExtract or Both
                                     Why: Need rich semantic understanding




┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│                              QUICK START GUIDE                                  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

Step 1: Choose Your Option
    ┌──────────────────────────────────────────────────────┐
    │ In langextract_simple.py, find these lines:         │
    │                                                      │
    │   metadata = "entity"  # ◄── Change this line      │
    │   schema_name = "paul_graham_detailed"             │
    └──────────────────────────────────────────────────────┘

Step 2: Set Your Choice
    ┌──────────────────────────────────────────────────────┐
    │ For None:         metadata = None                    │
    │ For Entity:       metadata = "entity"                │
    │ For LangExtract:  metadata = "langextract"           │
    │ For Both:         metadata = "both"                  │
    └──────────────────────────────────────────────────────┘

Step 3: Set API Keys (if needed)
    ┌──────────────────────────────────────────────────────┐
    │ All options need:                                    │
    │   export ANTHROPIC_API_KEY=your_key                  │
    │                                                      │
    │ LangExtract/Both also need:                          │
    │   export OPENAI_API_KEY=your_key                     │
    └──────────────────────────────────────────────────────┘

Step 4: Run the Script
    ┌──────────────────────────────────────────────────────┐
    │ python langextract_simple.py                         │
    └──────────────────────────────────────────────────────┘

Step 5: View Results
    ┌──────────────────────────────────────────────────────┐
    │ The script will:                                     │
    │ 1. Show configuration info                           │
    │ 2. Process the document                              │
    │ 3. Print sample metadata                             │
    │ 4. Save to database                                  │
    └──────────────────────────────────────────────────────┘
```
