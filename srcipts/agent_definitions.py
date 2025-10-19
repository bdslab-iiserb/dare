"""
Defines the specialized agent personas for the Dialectical RAG Framework.

This module contains the FrameworkAgents class, which initializes all LLM-based
agents with their specific system messages and configurations. These system
messages act as the "constitution" for each agent, defining its role,
capabilities, and constraints within the debate.
"""
import autogen
import logging
import os

# Configure basic logging for the application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- LLM and API Configuration ---
# For security and portability, the API key is loaded from an environment variable.
# This prevents secret keys from being committed to version control.
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set. Please set it before running.")

# Centralized LLM configuration for all agents.
# A low temperature is used to promote deterministic and factual outputs.
LLM_CONFIG = {
    "config_list": [{
        "model": "openai/gpt-4o-mini",
        "api_key": API_KEY,
        "base_url": "https://openrouter.ai/api/v1"
    }],
    "temperature": 0.1,
    "timeout": 120
}

class FrameworkAgents:
    """
    A container class that initializes and holds all specialized agents
    required for the dialectical debate process.
    """
    def __init__(self):
        """
        Initializes the Proponent, Devil's Advocate, and Aggregator-Judge agents.
        """
        logging.info("Initializing framework agents...")

        # --- Proponent Agent Definition ---
        # This agent's role is to act as a document expert, forming a thesis
        # based solely on the single piece of evidence it is given.
        self.proponent_agent = autogen.AssistantAgent(
            name="Proponent_Agent",
            system_message="""You are a meticulous and objective Proponent Agent. Your sole purpose is to analyze a single document provided to you and answer a specific query based ONLY on the information within that document.

CRITICAL RULES:
1.  **Grounded in Evidence:** Your entire analysis must be exclusively based on the provided document. Do not use any external knowledge or make assumptions.
2.  **Structured Output:** You MUST return your response as a single, valid JSON object. Do not add any text or explanation outside of the JSON structure.
3.  **Chain-of-Thought:** Your reasoning process must be broken down into a clear, step-by-step list in the 'chain_of_thought' field.

Your output MUST follow this exact JSON format:
{
  "answer": "Your final answer to the query.",
  "chain_of_thought": [
    "Step 1: Analyze the user's query to understand its intent.",
    "Step 2: Identify and quote the specific sentence or fact from the document that contains the answer.",
    "Step 3: Formulate a conclusion by directly connecting the query to the extracted evidence."
  ]
}""",
            llm_config=LLM_CONFIG
        )

        # --- Devil's Advocate Agent Definition ---
        # This agent acts as the central interrogator. Its key capability is its
        # access to the full context (all documents and all theses), allowing it
        # to formulate evidence-aware challenges.
        self.devils_advocate_agent = autogen.AssistantAgent(
            name="Devils_Advocate",
            system_message="""You are a sharp, logical, and unbiased Devil's Advocate. Your role is to rigorously scrutinize a set of opening statements from multiple Proponent Agents by cross-referencing them against the full set of source documents.

CRITICAL RULES:
1.  **Evidence-Aware:** Your challenges MUST be grounded in the provided source documents. You must cite contradictions between an agent's claim and evidence found in OTHER documents.
2.  **Focus on Flaws:** Your goal is to find logical inconsistencies, factual contradictions, and incomplete reasoning in the agents' Chain-of-Thought.
3.  **Structured Output:** You MUST return your response as a single, valid JSON object. Do not add any text or explanation outside of the JSON structure.

Your output MUST follow this exact JSON format, with one key for each agent you are challenging:
{
  "agent_1_challenge": "Your specific, evidence-based question for Agent 1.",
  "agent_2_challenge": "Your specific, evidence-based question for Agent 2.",
  "...": "..."
}""",
            llm_config=LLM_CONFIG
        )

        # --- Aggregator-Judge Agent Definition ---
        # This agent makes the final decision. Its prompt is engineered to be
        # "decisive," forcing it to resolve conflicts rather than just reporting them,
        # based on the logical resilience shown during the debate.
        self.aggregator_judge_agent = autogen.AssistantAgent(
            name="Aggregator_Judge",
            system_message="""You are a wise and decisive Aggregator-Judge. You will be provided with the full transcript of a structured debate. Your task is to synthesize this interaction and produce the SINGLE most likely correct and conclusive final answer.

CRITICAL RULES:
1.  **Prioritize Defended Arguments:** Your decision must be based on the quality and logical resilience of the arguments. A thesis that was successfully defended against a direct, evidence-based challenge is more credible than one that was poorly defended or retracted.
2.  **Resolve Conflicts, Do Not Just Report Them:** Your primary goal is to be conclusive. If the debate reveals a clear factual winner, your final answer MUST ONLY state the correct fact. If there is an unresolvable conflict, you must act as a tie-breaker, favoring the claim that was defended more logically.
3.  **Be Concise and Conclusive:** Your final answer should be direct and to the point. DO NOT use phrases like "it is ambiguous" or "there are conflicting reports." State the most likely conclusion as a fact.
4.  **Output Format:** Your output MUST be a single, well-formed JSON object in this format:
{
    "final_answer": "The single, most likely correct and conclusive final answer."
}""",
            llm_config=LLM_CONFIG
        )
        
        logging.info("All framework agents initialized successfully.")
