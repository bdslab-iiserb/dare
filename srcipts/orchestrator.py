"""
Orchestrates the multi-agent dialectical debate process.

This module contains the DebateOrchestrator class, which programmatically
controls the three-phase debate (Thesis, Antithesis, Synthesis) for a given
query and set of documents. It manages the flow of information between agents,
ensuring a structured and deterministic interaction.
"""
import autogen
import json
import logging
from typing import Dict, List, Any

# Import the classes from our other modules
from agent_definitions import FrameworkAgents
from data_loader import QueryContext

# Configure basic logging for the application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DebateOrchestrator:
    """
    Manages the entire three-phase dialectical debate for a single query.
    """
    def __init__(self):
        """Initializes the orchestrator with the necessary agents."""
        self.agents = FrameworkAgents()
        # The UserProxyAgent acts as the programmatic entry point to initiate chats.
        self.chat_initiator = autogen.UserProxyAgent(
            name="Chat_Initiator",
            human_input_mode="NEVER",
            code_execution_config=False,
        )
        logging.info("Debate Orchestrator initialized.")

    def _execute_single_turn_chat(self, recipient_agent: autogen.AssistantAgent, message: str) -> str:
        """
        A helper function to run a single, atomic request-response chat turn.
        This ensures the orchestrator maintains full control over the debate flow.
        """
        self.chat_initiator.initiate_chat(
            recipient_agent,
            message=message,
            max_turns=1,
            silent=True  # Suppress default AutoGen logging for cleaner output
        )
        # The agent's response is the last message in the chat history.
        return self.chat_initiator.last_message(recipient_agent)["content"]

    def run_debate(self, query_context: QueryContext) -> Dict[str, Any]:
        """
        Executes the full, three-phase dialectical debate for a given QueryContext.

        Args:
            query_context: The standardized data object containing the query,
                           documents, and ground truth.

        Returns:
            A dictionary containing the full debate transcript.
        """
        debate_transcript = {}
        
        # --- PHASE 1: THESIS GENERATION ---
        # Each Proponent Agent generates an initial thesis based on its single document.
        logging.info(f"--- Starting Phase 1: Thesis Generation for Query ID: {query_context.query_id} ---")
        opening_statements = {}
        for i, doc in enumerate(query_context.documents):
            agent_name = f"agent_{i+1}"
            prompt = f"""Your task is to generate a structured thesis based on the query and the single document provided.

Query: "{query_context.question}"

Document (doc_id: {doc['doc_id']}):
---
{doc['text']}
---
"""
            logging.info(f"  > Getting opening statement from {agent_name}...")
            response_str = self._execute_single_turn_chat(self.agents.proponent_agent, prompt)
            try:
                opening_statements[agent_name] = json.loads(response_str)
            except json.JSONDecodeError:
                logging.warning(f"  > FAILED to parse JSON from {agent_name}. Storing raw response.")
                opening_statements[agent_name] = {"answer": "JSON_PARSE_ERROR", "chain_of_thought": [response_str]}
        
        debate_transcript["phase_1_opening_statements"] = opening_statements
        logging.info("--- Phase 1 Complete ---")

        # --- PHASE 2: ANTITHESIS GENERATION & REBUTTAL ---
        # The Devil's Advocate challenges each thesis, and the Proponents must respond.
        logging.info("--- Starting Phase 2: Antithesis Generation & Rebuttal ---")
        
        # 2a: The Challenge
        # The Devil's Advocate receives the full context: the query, all theses, and all documents.
        challenge_prompt = f"""You are the Devil's Advocate. Your task is to generate targeted, evidence-based challenges based on the original query and the provided evidence.

Original Query: "{query_context.question}"

--- OPENING STATEMENTS ---
{json.dumps(opening_statements, indent=2)}
---

--- ALL SOURCE DOCUMENTS ---
{json.dumps(query_context.documents, indent=2)}
---
"""
        logging.info("  > Getting challenges from Devil's Advocate...")
        challenges_str = self._execute_single_turn_chat(self.agents.devils_advocate_agent, challenge_prompt)
        try:
            challenges = json.loads(challenges_str)
        except json.JSONDecodeError:
            logging.error("  > FAILED to parse JSON from Devil's Advocate. Aborting debate.")
            return {"error": "Failed to parse Devil's Advocate challenges.", "transcript": debate_transcript}
        
        debate_transcript["phase_2a_challenges"] = challenges

        # 2b: The Rebuttal
        # Each Proponent Agent must now defend its thesis against the specific challenge.
        rebuttals = {}
        for i, doc in enumerate(query_context.documents):
            agent_name = f"agent_{i+1}"
            challenge_key = f"{agent_name}_challenge"
            
            if challenge_key not in challenges:
                logging.warning(f"  > No challenge found for {agent_name}. Skipping rebuttal.")
                continue

            # The rebuttal prompt explicitly includes the Structured Rebuttal Protocol.
            rebuttal_prompt = f"""You are {agent_name}. You must formulate a rebuttal to a direct challenge.

**Your Task:** Follow the Structured Rebuttal Protocol below precisely.

**Structured Rebuttal Protocol:**
1.  **Acknowledge Challenge:** Start by paraphrasing the core of the challenge you received.
2.  **Review Contradictory Evidence:** Explicitly state the piece of evidence from the other document(s) that your challenger presented.
3.  **Argument & Justification:** Choose ONE of the following paths and justify it:
    *   **DEFEND:** Argue why your original evidence/reasoning is superior.
    *   **CONCEDE:** Acknowledge that the contradictory evidence is superior and that your initial answer was flawed.
    *   **RECONCILE:** Explain how both pieces of evidence can be true simultaneously (e.g., in cases of ambiguity).
4.  **Final Revised Statement:** Provide your final, updated answer and Chain-of-Thought in the required JSON format.

**Context for your Rebuttal:**
- Original Query: "{query_context.question}"
- Your Assigned Document (doc_id: {doc['doc_id']}): {doc['text']}
- Your Opening Statement: {json.dumps(opening_statements.get(agent_name, {}), indent=2)}
- THE CHALLENGE YOU MUST ADDRESS: "{challenges[challenge_key]}"

Now, provide your rebuttal. Your entire response must be a single JSON object.
"""
            logging.info(f"  > Getting rebuttal from {agent_name}...")
            rebuttal_str = self._execute_single_turn_chat(self.agents.proponent_agent, rebuttal_prompt)
            try:
                rebuttals[agent_name] = json.loads(rebuttal_str)
            except json.JSONDecodeError:
                logging.warning(f"  > FAILED to parse JSON rebuttal from {agent_name}. Storing raw response.")
                rebuttals[agent_name] = {"answer": "JSON_PARSE_ERROR", "chain_of_thought": [rebuttal_str]}

        debate_transcript["phase_2b_rebuttals"] = rebuttals
        logging.info("--- Phase 2 Complete ---")

        # --- PHASE 3: SYNTHESIS (Final Judgment) ---
        # The Aggregator-Judge reviews the entire debate to produce a final, conclusive answer.
        logging.info("--- Starting Phase 3: Synthesis ---")
        final_prompt = f"""You are the Aggregator-Judge. Below is the full transcript of a debate. Your task is to synthesize this interaction and produce the SINGLE most likely correct and conclusive final answer, as per your core instructions.

Original Query: "{query_context.question}"

--- DEBATE TRANSCRIPT ---
{json.dumps(debate_transcript, indent=2)}
---

Based on your analysis of the entire debate, provide the final, synthesized answer in the required JSON format. Remember to be decisive and resolve conflicts.
"""
        logging.info("  > Getting final answer from Aggregator-Judge...")
        final_answer_str = self._execute_single_turn_chat(self.agents.aggregator_judge_agent, final_prompt)
        try:
            final_answer = json.loads(final_answer_str)
        except json.JSONDecodeError:
            logging.error("  > FAILED to parse JSON from Aggregator-Judge. Storing raw response.")
            final_answer = {"final_answer": final_answer_str}

        debate_transcript["phase_3_final_answer"] = final_answer
        logging.info("--- Phase 3 Complete ---")
        
        return debate_transcript```
