import autogen
import json
import logging
from typing import Dict, List, Any

# Import the classes we've already built
from agent_definitions import FrameworkAgents, LLM_CONFIG
from data_loader import QueryContext

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DebateOrchestrator:
    """
    Orchestrates the entire three-phase dialectical debate for a single query.
    """
    def __init__(self):
        """
        Initializes the orchestrator with the necessary agents.
        """
        self.agents = FrameworkAgents()
        # The UserProxyAgent acts as the primary interface to initiate chats.
        self.chat_initiator = autogen.UserProxyAgent(
            name="Chat_Initiator",
            human_input_mode="NEVER",
            code_execution_config=False,
        )
        logging.info("Debate Orchestrator initialized.")

    def _execute_single_turn_chat(self, recipient_agent: autogen.AssistantAgent, message: str) -> str:
        """
        A helper function to run a single, atomic request-response chat turn.
        This is the core communication mechanism.
        """
        self.chat_initiator.initiate_chat(
            recipient_agent,
            message=message,
            max_turns=1,
            silent=True # We will control logging ourselves.
        )
        # The response is the last message in the chat history.
        return self.chat_initiator.last_message(recipient_agent)["content"]

    def run_debate(self, query_context: QueryContext) -> Dict[str, Any]:
        """
        Executes the full, three-phase dialectical debate for a given QueryContext.

        Args:
            query_context (QueryContext): The standardized data object containing the
                                          query, documents, and ground truth.

        Returns:
            A dictionary containing the full debate transcript and the final answer.
        """
        debate_transcript = {}
        
        # --- PHASE 1: THESIS GENERATION (Opening Statements) ---
        logging.info(f"--- Starting Phase 1: Thesis Generation for Query ID: {query_context.query_id} ---")
        opening_statements = {}
        for i, doc in enumerate(query_context.documents):
            agent_name = f"agent_{i+1}"
            prompt = f"""Here is the query and a single document. Your task is to generate a structured thesis.

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
                opening_statements[agent_name] = {"answer": "JSON PARSE ERROR", "chain_of_thought": [response_str]}
        
        debate_transcript["phase_1_opening_statements"] = opening_statements
        logging.info("--- Phase 1 Complete ---")

        # --- PHASE 2: ANTITHESIS GENERATION (Cross-Examination) ---
        logging.info("--- Starting Phase 2: Antithesis Generation & Rebuttal ---")
        # 2a: The Challenge
        challenge_prompt = f"""Here are the opening statements from multiple Proponent Agents and the full set of source documents. Your task is to act as the Devil's Advocate and generate targeted, evidence-based challenges.

Query: "{query_context.question}"

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
        rebuttals = {}
        for i, doc in enumerate(query_context.documents):
            agent_name = f"agent_{i+1}"
            challenge_key = f"{agent_name}_challenge"
            
            if challenge_key not in challenges:
                logging.warning(f"  > No challenge found for {agent_name}. Skipping rebuttal.")
                continue

            rebuttal_prompt = f"""You are {agent_name}. You must formulate a rebuttal to a direct challenge from the Devil's Advocate. Follow the Structured Rebuttal Protocol.

Original Query: "{query_context.question}"
Your Assigned Document (doc_id: {doc['doc_id']}):
---
{doc['text']}
---
Your Opening Statement:
---
{json.dumps(opening_statements.get(agent_name, {}), indent=2)}
---
THE CHALLENGE YOU MUST ADDRESS:
---
"{challenges[challenge_key]}"
---

Now, provide your rebuttal in the required structured JSON format (answer, chain_of_thought).
"""
            logging.info(f"  > Getting rebuttal from {agent_name}...")
            rebuttal_str = self._execute_single_turn_chat(self.agents.proponent_agent, rebuttal_prompt)
            try:
                rebuttals[agent_name] = json.loads(rebuttal_str)
            except json.JSONDecodeError:
                logging.warning(f"  > FAILED to parse JSON rebuttal from {agent_name}. Storing raw response.")
                rebuttals[agent_name] = {"answer": "JSON PARSE ERROR", "chain_of_thought": [rebuttal_str]}

        debate_transcript["phase_2b_rebuttals"] = rebuttals
        logging.info("--- Phase 2 Complete ---")

        # --- PHASE 3: SYNTHESIS (Final Judgment) ---
        logging.info("--- Starting Phase 3: Synthesis ---")
        final_prompt = f"""You are the Aggregator-Judge. Below is the full transcript of a debate. Your task is to synthesize this interaction and produce the most logically sound and complete final answer.

Original Query: "{query_context.question}"

--- DEBATE TRANSCRIPT ---
{json.dumps(debate_transcript, indent=2)}
---

Based on your analysis of the entire debate, provide the final, synthesized answer in the required JSON format.
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
        
        return debate_transcript


# --- Built-in End-to-End Test Block ---
if __name__ == "__main__":
    logging.info("--- Running Built-in End-to-End Test for Orchestrator ---")

    # We will use our classic "Michael Jordan" example as a hard-coded test case.
    # This simulates loading one QueryContext object.
    test_query = QueryContext(
        query_id="test_michael_jordan",
        question="In which year was Michael Jordan born?",
        documents=[
            {"doc_id": "doc_1", "text": "Michael Jeffrey Jordan (born February 17, 1963) is an American businessman and former professional basketball player."},
            {"doc_id": "doc_2", "text": "Michael I. Jordan (born February 25, 1956) is an American scientist, professor at the University of California, Berkeley."},
            {"doc_id": "doc_3", "text": "A popular blog post claims that basketball star Michael Jordan was born in 1998, a common misconception."}
        ],
        gold_answers=["1963", "1956"],
        wrong_answers=["1998"]
    )

    try:
        # 1. Initialize the orchestrator
        orchestrator = DebateOrchestrator()
        
        # 2. Run the full debate on our single test case
        final_transcript = orchestrator.run_debate(test_query)
        
        # 3. Print the results for manual inspection
        print("\n" + "="*80)
        print("--- DEBATE COMPLETE. FINAL TRANSCRIPT: ---")
        print(json.dumps(final_transcript, indent=2))
        print("="*80)
        
        final_answer = final_transcript.get("phase_3_final_answer", {}).get("final_answer", "ERROR: No final answer found.")
        
        logging.info(f"FINAL SYNTHESIZED ANSWER: {final_answer}")
        
        # Verification Check
        if "ERROR" not in final_answer and final_answer:
            logging.info("--- ORCHESTRATOR TEST PASSED SUCCESSFULLY ---")
        else:
            logging.error("--- ORCHESTRATOR TEST FAILED: Final answer was not generated correctly. ---")

    except Exception as e:
        logging.error(f"TEST FAILED: An unexpected error occurred during the orchestration test: {e}", exc_info=True)
