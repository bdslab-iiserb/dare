import autogen
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
API_KEY = " Replace with your actual OpenRouter key" 
LLM_CONFIG = {
    "config_list": [{
        "model": "openai/gpt-4o-mini", # A capable and fast model
        "api_key": API_KEY,
        "base_url": "https://openrouter.ai/api/v1"
    }],
    "temperature": 0.1, # Lower temperature for more deterministic, factual outputs
    "timeout": 120
}

class FrameworkAgents:
    """
    A class to initialize and hold all the specialized agents for our framework.
    """
    def __init__(self):
        """
        Initializes the Proponent, Devil's Advocate, and Aggregator-Judge agents
        with their specific system messages and configurations.
        """
        logging.info("Initializing framework agents...")

        # --- 1. The Proponent Agent ---
        # Role: To analyze a single document and form an initial, reasoned thesis.
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
    "Step 1: A brief summary of the query's intent.",
    "Step 2: The specific sentence or fact from the document that contains the answer.",
    "Step 3: Your conclusion based on connecting the query to the evidence."
  ]
}""",
            llm_config=LLM_CONFIG
        )

        # --- 2. The Devil's Advocate Agent ---
        # Role: To act as a centralized, evidence-aware interrogator.
        self.devils_advocate_agent = autogen.AssistantAgent(
            name="Devils_Advocate",
            system_message="""You are a sharp, logical, and unbiased Devil's Advocate. Your role is to rigorously scrutinize a set of opening statements from multiple Proponent Agents by cross-referencing them against the full set of source documents.

CRITICAL RULES:
1.  **Evidence-Aware:** Your challenges MUST be grounded in the provided source documents. You can and should cite contradictions between an agent's claim and evidence found in OTHER documents.
2.  **Focus on Flaws:** Your goal is to find logical inconsistencies, factual contradictions, and incomplete reasoning.
3.  **Structured Output:** You MUST return your response as a single, valid JSON object. Do not add any text or explanation outside of the JSON structure.

Your output MUST follow this exact JSON format, with one key for each agent you are challenging:
{
  "agent_1_challenge": "Your specific, evidence-based question for Agent 1.",
  "agent_2_challenge": "Your specific, evidence-based question for Agent 2.",
  "...": "..."
}""",
            llm_config=LLM_CONFIG
        )

        # --- 3. The Aggregator-Judge Agent ---
        # Role: To synthesize the entire debate and render a final, justified verdict.
        self.aggregator_judge_agent = autogen.AssistantAgent(
            name="Aggregator_Judge",
            system_message="""You are a wise and impartial Aggregator-Judge. You will be provided with the full transcript of a structured debate, including opening statements, adversarial challenges, and rebuttals. Your task is to synthesize this entire interaction and produce the most logically sound and complete final answer.

CRITICAL RULES:
1.  **Synthesize, Don't Vote:** Your decision must be based on the quality and logical resilience of the arguments, not on the number of agents who proposed an answer.
2.  **Prioritize Defended Arguments:** A thesis that was successfully defended against a direct, evidence-based challenge is more credible than one that was not challenged or was poorly defended.
3.  **Protect Valid Minority Opinions:** If the debate reveals a valid ambiguity (e.g., two different people with the same name), you must include all valid answers in your final output. Do not default to the majority.
4.  **Be Coherent and Final:** Your output should be a clean, final answer written for the end-user. Do not mention the debate, the agents, or the internal process.

Your output should be a single, well-formed JSON object in this format:
{
    "final_answer": "The comprehensive, synthesized final answer to the original query."
}""",
            llm_config=LLM_CONFIG
        )
        
        logging.info("All framework agents initialized successfully.")

# --- Built-in Test Block ---
# This test verifies that the agents can be initialized without configuration errors.
if __name__ == "__main__":
    logging.info("--- Running Built-in Test for Agent Definitions ---")
    try:
        agents = FrameworkAgents()
        
        # Verification Checks
        assert isinstance(agents.proponent_agent, autogen.AssistantAgent), "Proponent Agent is not an AssistantAgent."
        assert isinstance(agents.devils_advocate_agent, autogen.AssistantAgent), "Devil's Advocate is not an AssistantAgent."
        assert isinstance(agents.aggregator_judge_agent, autogen.AssistantAgent), "Aggregator-Judge is not an AssistantAgent."
        
        logging.info("Check 1/1: All agents initialized as autogen.AssistantAgent instances. PASSED.")
        
        # You can also print the system messages to manually verify them
        # print("\n--- Proponent System Message ---")
        # print(agents.proponent_agent.system_message)
        
        logging.info("--- AGENT DEFINITION TESTS PASSED SUCCESSFULLY ---")

    except Exception as e:
        logging.error(f"TEST FAILED: An error occurred during agent initialization: {e}", exc_info=True)
