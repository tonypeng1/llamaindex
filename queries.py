"""
Query configurations for different articles.
This file centralizes the active test query for each article.
Simply uncomment the line you want to use for each article.
"""

from typing import Dict

# =============================================================================
# 1. PAUL GRAHAM ESSAY
# =============================================================================
# PG_ACTIVE = "What was mentioned about Jessica from pages 17 to 22?"
# PG_ACTIVE = "What was mentioned about Jessica from pages 17 to 22? Please cite page numbers in your answer."
# PG_ACTIVE = "What did Paul Graham do in 1980, in 1996 and in 2019?"
# PG_ACTIVE = "What did the author do after handing off Y Combinator to Sam Altman?"
# PG_ACTIVE = "What did Paul Graham say about Lisp?"
# PG_ACTIVE = "What events are associated with England?"
# PG_ACTIVE = "What events are associated with locations in USA?"
PG_ACTIVE = "Find all locations that are mentioned in the essay and describe the events associated with them?"
# PG_ACTIVE = "What advice is given to become a successful developer?"
# PG_ACTIVE = "What programming languages are discussed in the document?"
# PG_ACTIVE = "What strategic advice is given about startups?"
# PG_ACTIVE = "Has the author been to Europe?"
# PG_ACTIVE = "What was mentioned about Jessica from pages 17 to 19?"
# PG_ACTIVE = "List all people mentioned in the document."
# PG_ACTIVE = "What experiences from the 1990s are described?"
# PG_ACTIVE = "What programming concepts are given in the document?"
# PG_ACTIVE = "Who are mentioned as colleagues in the document?"
# PG_ACTIVE = "Does the author have any advice on relationships?"
# PG_ACTIVE = "Create table of contents for this article."
# PG_ACTIVE = "How did rejecting prestigious conventional paths lead to the most influential creative projects?"


# =============================================================================
# 2. HOW TO DO GREAT WORK
# =============================================================================
# GREAT_WORK_ACTIVE = "What did the author advice on choosing what to work on?"
# GREAT_WORK_ACTIVE = "Why morale needs to be nurtured and protected?" 
# GREAT_WORK_ACTIVE = "What are the contents from pages 26 to 29?"
# GREAT_WORK_ACTIVE = "What are the contents from pages 20 to 24 (one page at a time)?"
# GREAT_WORK_ACTIVE = "What are the concise contents from pages 20 to 24 (one page at a time) in the voice of the author?"
# GREAT_WORK_ACTIVE = "Summarize the content from pages 1 to 5 (one page at a time) in the voice of the author by NOT retrieving the text verbatim."
# GREAT_WORK_ACTIVE = "Summarize the key takeaways from pages 1 to 5 (one page at a time) in a sequential order and in the voice of the author by NOT retrieving the text verbatim."
GREAT_WORK_ACTIVE = "Summarize the key contents from pages 1 to 5 (one page at a time) in the voice of the author by NOT retrieving the text verbatim."


# =============================================================================
# 3. UBER 10Q MARCH 2022
# =============================================================================
# UBER_ACTIVE = "what is UBER Short-term insurance reserves reported in 2022?"
UBER_ACTIVE = "what is UBER's common stock subject to repurchase in 2021?"
# UBER_ACTIVE = "What is UBER Long-term insurance reserves reported in 2021?"
# UBER_ACTIVE = "What is the number of monthly active platform consumers in Q2 2021?"
# UBER_ACTIVE = "What is the number of monthly active platform consumers in 2022?"
# UBER_ACTIVE = "What is the number of trips in 2021?"
# UBER_ACTIVE = "What is the free cash flow in 2021?"
# UBER_ACTIVE = "What is the gross bookings of delivery in Q3 2021?"
# UBER_ACTIVE = "What is the gross bookings in 2022?"
# UBER_ACTIVE = "What is the value of mobility adjusted EBITDA in 2022?"
# UBER_ACTIVE = "What is the status of the classification of drivers?"
# UBER_ACTIVE = "What is the comprehensive income (loss) attributable to Uber reported in 2021?"
# UBER_ACTIVE = "What is the comprehensive income (loss) attributable to Uber Technologies reported in 2022?"
# UBER_ACTIVE = "What are the data shown in the bar graph titled 'Monthly Active Platform Consumers'?"
# UBER_ACTIVE = "Can you tell me the page number on which the bar graph titled 'Monthly Active Platform Consumers' is located?"
# UBER_ACTIVE = "What are the data shown in the bar graph titled 'Monthly Active Platform Consumers' on page 43?"
# UBER_ACTIVE = "What is the Q2 2020 value shown in the bar graph titled 'Monthly Active Platform Consumers' on page 43?"
# UBER_ACTIVE = "What are the main risk factors for Uber?"
# UBER_ACTIVE = "What are the data shown in the bar graph titled 'Trips'?"
# UBER_ACTIVE = "What are the data shown in the bar graph titled 'Gross Bookings'?"


# =============================================================================
# 4. ATTENTION IS ALL YOU NEED
# =============================================================================
ATTENTION_ACTIVE = "What is the benefit of multi-head attention instead of single-head attention?"
# ATTENTION_ACTIVE = "Describe the content of section 3.1"  # not working
# ATTENTION_ACTIVE = "Describe the content of section 3.1 with the title 'Encoder and Decoder Stacks'."  # WORK 
# ATTENTION_ACTIVE = "What is the caption of Figure 2?"
# ATTENTION_ACTIVE = "What is in equation (1)."
# ATTENTION_ACTIVE = "What is in equation (2)."  # not working
# ATTENTION_ACTIVE = "What is in equation (3)."  # not working
# ATTENTION_ACTIVE = "Is there any equation in section 5.3?"  # not working
# ATTENTION_ACTIVE = "Is there any equation in section 5.3 titled 'Optimizer'?"
# ATTENTION_ACTIVE = "How many equations are there in the full context of this document?"  # WORK!
# ATTENTION_ACTIVE = "How many equations are there in this document?"  # WORK!
# ATTENTION_ACTIVE = "What is on page 6?"  # WORK!
# ATTENTION_ACTIVE = "How many tables are there in this document?"
# ATTENTION_ACTIVE = "What is table 1 about?"
# ATTENTION_ACTIVE = "What do the results in table 1 show?"
# ATTENTION_ACTIVE = "List all sections and subsections in this document. Keep the original section/subsection numbers."  # not working
# ATTENTION_ACTIVE = "List all sections and subsections in the full context of this document. Use the original section/subsection numbers."  # WORK!
# ATTENTION_ACTIVE = "Find out how many sections and subsections does this document have and use the results to describe the content of subsection 3.1."
# ATTENTION_ACTIVE = "List all sections with the section number and section title?"  # not working
# ATTENTION_ACTIVE = "Create a table of content."  # not working
# ATTENTION_ACTIVE = "What does Figure 1 show?" 
# ATTENTION_ACTIVE = "Describe Figure 1 in detail." 
# ATTENTION_ACTIVE = "What does table 1 show?"
# ATTENTION_ACTIVE = "What are the results in table 1?"
# ATTENTION_ACTIVE = "Describe Figure 2 in detail."
# ATTENTION_ACTIVE = "What is the title of table 2?"
# ATTENTION_ACTIVE = "In table 2 what do 'EN-DE' and 'EN-FR' mean?"
# ATTENTION_ACTIVE = "What is the BLEU score of the model 'MoE' in EN-FR in Table 2?" 
# ATTENTION_ACTIVE = "How do a query and a set of key value pairs work together in an attention function?" 
# ATTENTION_ACTIVE = "What is the formula for Scaled Dot-Product Attention?"
# ATTENTION_ACTIVE = "Describe the Transformer architecture shown in Figure 1."
# ATTENTION_ACTIVE = "Describe Figure 2 in detail. What visual elements does it contain?"


# =============================================================================
# 5. METAGPT
# =============================================================================
METAGPT_ACTIVE = "What is MetaGPT?"
# METAGPT_ACTIVE = "How does the multi-agent system work?"


# =============================================================================
# 6. CAREER IN AI
# =============================================================================
CAREER_ACTIVE = "What skills are needed for an AI career?"
# CAREER_ACTIVE = "How should one start learning AI?"


# =============================================================================
# 7. RAG ANYTHING
# =============================================================================
# RAG_ANYTHING_ACTIVE = "Describe Figure 1 in detail. What visual elements and workflow does it show?"
# RAG_ANYTHING_ACTIVE = "Describe the content in Figure 4 in detail."
# RAG_ANYTHING_ACTIVE = "In the Accuracy (%) on DocBench Dataset table (table 2), what methods are being compared and what is the worst performing method?"
RAG_ANYTHING_ACTIVE = "In Table 2, what methods are being compared and what is the worst performing method?"
# RAG_ANYTHING_ACTIVE = "In the Accuracy (%) on MMLongBench Dataset table (table 3), what methods are being compared, and what is the best performing method?"
# RAG_ANYTHING_ACTIVE = "Please summarize the content in the Introduction section."
# RAG_ANYTHING_ACTIVE = "Please summarize the content from pages 1 to 2."
# RAG_ANYTHING_ACTIVE = "Please summarize the content from pages 15 to 16."
# RAG_ANYTHING_ACTIVE = "Please summarize the content in the section in the Appendix: ADDITIONAL CASE STUDIES."
# RAG_ANYTHING_ACTIVE = "Please summarize the content in the Appendix section:CHALLENGES AND FUTURE DIRECTIONS FOR MULTI-MODAL RAG."
# RAG_ANYTHING_ACTIVE = "Please summarize the content in the Appendix section A.5: CHALLENGES AND FUTURE DIRECTIONS FOR MULTI-MODAL RAG."
# RAG_ANYTHING_ACTIVE = "Please summarize the content in Section A.2 ADDITIONAL CASE STUDIES"
# RAG_ANYTHING_ACTIVE = "Please summarize the content in Section 4: RELATED WORK"
# RAG_ANYTHING_ACTIVE = "Describe the content of Section 2.3 CROSS-MODAL HYBRID RETRIEVAL"
# RAG_ANYTHING_ACTIVE = "What is the content of the Evaluation section?"
# RAG_ANYTHING_ACTIVE = "Summarize the content of the Evaluation section."
# RAG_ANYTHING_ACTIVE = "Summarize the Conclusion section."
# RAG_ANYTHING_ACTIVE = "Summarize the section 3.4, CASE STUDIES."
# RAG_ANYTHING_ACTIVE = "Summarize the CASE STUDIES section."
# RAG_ANYTHING_ACTIVE = "Summarize the CROSS-MODAL HYBRID RETRIEVAL section."
# RAG_ANYTHING_ACTIVE = "What is in equation (1)?"
# RAG_ANYTHING_ACTIVE = "What are in equation (3) and (4)?"
# RAG_ANYTHING_ACTIVE = "What are in equation (4)?"
# RAG_ANYTHING_ACTIVE = "What are in the equations (1), (2), (3), and (4)? What are they trying to represent collectively?"
# RAG_ANYTHING_ACTIVE = "What are in the equations (2) and (3)?"
# RAG_ANYTHING_ACTIVE = "How graphs are used in RAG-Anything's retrieval process as described in the paper?"
# RAG_ANYTHING_ACTIVE = "Did the paper mention about any tool used to parse mathematical equations from the PDF? If so, what is the name of the tool?"
# RAG_ANYTHING_ACTIVE = "What are the specific challenges mentioned regarding text-centric retrieval bias?"
# RAG_ANYTHING_ACTIVE = "Does the paper mention an example of a merged graph for cross-modal hybrid retrieval? If so, describe the example in detail."


# =============================================================================
# 7. Laser_coprop_RA
# =============================================================================

# LASER_COPROP_ACTIVE = "Summarize the introduction section."
# LASER_COPROP_ACTIVE = "What are the most effective dithering frequencies for stimulated Brillouin scattering (SBS) suppression as reported in the paper?"
# LASER_COPROP_ACTIVE = "What is the signal wavelength and pump wavelength used in the experiments described in the paper?"
# LASER_COPROP_ACTIVE = "Describe the content in Figure 4."
LASER_COPROP_ACTIVE = "What is the signal wavelength and pump wavelength used in the experiments described in the paper? Infer from contents in figures if necessary."
# LASER_COPROP_ACTIVE = "Using the formula in Equation (2), calculate the fiber length (L) when the effective length (Leff) is 20 km and the attenuation coefficient (α) is 0.21 dB/km."


# =============================================================================
# 8. Noise_in_DRA
# =============================================================================

# NOISE_IN_DRA_ACTIVE = "Does this paper distinguish Rayleigh-reflected signal and Rayleigh-reflected pump? If so, how are they different?"
# NOISE_IN_DRA_ACTIVE = "Please summarize the content of Section 4."
# NOISE_IN_DRA_ACTIVE = "Please explain the contents in Fig 5.1 in detail. There are three sub-figures in it: (a) the DRB crosstalk coefficient, (b) the ASE enhancement factor and (c) the Q penalty. Explain what each parameter means and how they are related."
# NOISE_IN_DRA_ACTIVE = "Please explain the contents in Fig 6.1 in detail. Explain what each transfer function (i.e. SOP transfer and RIN transfer) means and how they are related."
NOISE_IN_DRA_ACTIVE = "Please explain the contents in Fig 6.2 (a) and (b) in detail. Explain what each axis represents and what is shown in the figures means."
# NOISE_IN_DRA_ACTIVE = "Please describe the contents of fig. 4.1 and fig. 4.3 in detail and examine how they differ in trends."


# =============================================================================
# 9. ASE_noise_pump_depletion
# =============================================================================

# ASE_NOISE_PUMP_DEPL_ACTIVE = "Summarize the introduction section."
ASE_NOISE_PUMP_DEPL_ACTIVE = "What is the relationship between signal power and ASE power as described in the paper?"


# =============================================================================
# 10. Pump_depletion_FRA
# =============================================================================

# PUMP_DEPLETION_FRA_ACTIVE = "How does pump depletion dependent on fiber length, input pump power, and input signal power?"
# PUMP_DEPLETION_FRA_ACTIVE = "What is the definition of unity gain length (Lu(fw)) as given in the paper?"
# PUMP_DEPLETION_FRA_ACTIVE = "Please describe Figure 6 (a) and (b) in detail. How do the two sub-figures contrast with each other?"
# PUMP_DEPLETION_FRA_ACTIVE = "Please describe Figure 5 in detail. How do the threshold depletion pump power Pp(dep) vary with input signal power?"
PUMP_DEPLETION_FRA_ACTIVE = "Please describe Figure 8 in detail. What is the definition of safe fiber length (y-axis) as given in the paper? Why is safe fiber length only define in backward pumping case?"


# =============================================================================
# 11. NF_Analysis_DFRA
# =============================================================================

# NF_Analysis_DFRA_ACTIVE = "Does anywhere in this paper mention the signal power used in the analysis/experiments?"
# NF_Analysis_DFRA_ACTIVE = "How does NF dependent on fiber length and pump power?"
NF_Analysis_DFRA_ACTIVE = "What is the definition of NF (noise figure)? How does NF dependent on fiber length and pump power?"


# =============================================================================
# MAPPINGS (Used by config.py)
# =============================================================================

ACTIVE_QUERIES: Dict[str, str] = {
    "paul_graham_essay": PG_ACTIVE,
    "How_to_do_great_work": GREAT_WORK_ACTIVE,
    "attention_all": ATTENTION_ACTIVE,
    "metagpt": METAGPT_ACTIVE,
    "uber_10q_march_2022": UBER_ACTIVE,
    "eBook-How-to-Build-a-Career-in-AI": CAREER_ACTIVE,
    "RAG_Anything": RAG_ANYTHING_ACTIVE,
    "Laser_coprop_RA": LASER_COPROP_ACTIVE,
    "Noise_in_DRA": NOISE_IN_DRA_ACTIVE,
    "ASE_noise_pump_depletion": ASE_NOISE_PUMP_DEPL_ACTIVE,
    "Pump_depletion_FRA": PUMP_DEPLETION_FRA_ACTIVE,
    "NF_Analysis_DFRA": NF_Analysis_DFRA_ACTIVE,
}

def get_query_for_article(article_key: str) -> str:
    """
    Retrieve the active query for a given article.
    """
    return ACTIVE_QUERIES.get(article_key, "What is this document about?")
