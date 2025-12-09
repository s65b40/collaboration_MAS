from api import chatgpt_response
import re
from calculate_tokens import calculate_tokens
import json

# Extract via regex
def re_extract(pattern, input):
    result_match = re.search(pattern, input)
    if result_match:
        return result_match.group(1).strip()
    else:
        return ""
    
class Agent:
    def __init__(self, name, claim, evidence, label):
        self.name = name
        self.claim = claim
        self.evidence = evidence 
        self.label = label
        self.history = []
        self.judgement = ''
        self.explanation = ''

    def chat(self, message):
        medical_dict = {self.reference_medical_data: self.medical_data}
        prompt = f"# Medical data:\n{medical_dict}\n\n\n # Historical analysis:\n{self.history}\n\n\n# Instruction:\n{message}"
        response = chatgpt_response(prompt)

        # Calculate tokens
        input_tokens = calculate_tokens(prompt)
        output_tokens = calculate_tokens(response)

        # Extract result and explanation via regex
        result_pattern = r'Result:\s*(.*)'
        explanation_pattern = r'Explanation:\s*(.*)'
        result = re_extract(result_pattern, response) or response
        explanation = re_extract(explanation_pattern, response)

        prediction = {"predict_result": result, "explanation": explanation}
        print("############")
        print(self.name)
        print(self.reference_medical_data)
        print(prompt)
        print(prediction)

        return prediction, input_tokens, output_tokens

    def add_history(self, name, message):
        self.history.append({"name":name, "message":message})

    def add_history_last_round(self, name, message, current_round):
        """Keep the current round and previous two rounds of history; drop older ones."""
        # Add the current round's entry to the history list
        self.history.append({"name": name,"message": message, "round": current_round})
        
        # Clean history: keep only current and previous round
        self.history = [entry for entry in self.history if entry['round'] >= current_round - 1]

    def update_history(self, name, task, message, current_round):
        self.history = []
        self.history.append({"name":name, "task":task, "message":message, "round": current_round})

    def generate_history(self, judgements):
        prompt = {}
        prompt["instruction"] = f"You are {self.name}. Based on the previous conversation and the judgments from the current round, please generate a comprehensive summary that highlights the key evidence, important information, and judgment outcomes relevant to claim verification, focusing on the historical context that contributes to the assessment of the claim's truthfulness."    
        prompt["claim"] = self.claim
        prompt["previous conversation"] = self.history
        prompt["current round judgments"] = judgements
        prompt = json.dumps(prompt, indent=2)

        response = chatgpt_response(prompt)

        # Calculate tokens
        input_tokens = calculate_tokens(prompt)
        output_tokens = calculate_tokens(response)

        print("############")
        print(self.name)
        print(prompt)
        print(response)

        return response, input_tokens, output_tokens


class Instructor(Agent):
    def __init__(self, name, claim):
        super().__init__(name, claim, None, None)
    def generate_historical_information(self, judgements):
        prompt = {}
        prompt["instruction"] = "You are an instructor overseeing multiple agents and tasked with evaluating the truth of a given claim. Based on the previous conversation and the judgments from the current round, please generate a comprehensive summary that highlights the key evidence, important information, and judgment outcomes relevant to claim verification, focusing on the historical context that contributes to the assessment of the claim's truthfulness."
        prompt["claim"] = self.claim
        prompt["previous conversation"] = self.history
        prompt["current round judgments"] = judgements
        prompt = json.dumps(prompt, indent=2)

        response = chatgpt_response(prompt)

        # Calculate tokens
        input_tokens = calculate_tokens(prompt)
        output_tokens = calculate_tokens(response)

        print("############")
        print(self.name)
        print(prompt)
        print(response)

        return response, input_tokens, output_tokens
    
    def decide_next_agent(self, agents):
        prompt = {}
        prompt["instruction"] = "You are an instructor overseeing multiple assistants and tasked with evaluating the truth of a given claim. Based on their judgments and explanations, as well as the previous conversation provided, please decide which agents should speak in the next round and the order in which they will speak."
        prompt["claim"] = self.claim
        prompt["previous conversation"] = self.history
        prompt["assistants information"] = {}
        for agent in agents:
            prompt["assistants information"][agent.name] = {"judgement": agent.judgement, "explanation": agent.explanation}
        prompt["output requirements"] = "Please provide the names of the agents in the order they should speak.\nPlease strictly follow this format:\nAgentNames: [AgentX, AgentY, ...] (where X, Y, ... are the agent numbers)"
        prompt = json.dumps(prompt, indent=2)

        response = chatgpt_response(prompt)

        # Calculate tokens
        input_tokens = calculate_tokens(prompt)
        output_tokens = calculate_tokens(response)

        selected_agent_names = re.findall(r'Agent(\d+)', response)
        selected_agents = sorted((agent for agent in agents if agent.name in [f"Agent{num}" for num in selected_agent_names]), key=lambda agent: selected_agent_names.index(agent.name[5:]))
        print("############")
        print(self.name)
        print(prompt)
        print(response)
        print("Selected agents:", [agent.name for agent in selected_agents])

        return selected_agents, input_tokens, output_tokens
    
    def decide_final_decision(self, judgements, agents):
        instruction = "You are an instructor overseeing multiple agents and your task is to judge the claim's truthfulness.Please review the rationale for each judgement, as well as relevant previous conversation. Assess whether whether the claim's truthfulness can be judged or if further discussion or information is needed."
        prompt = {}
        prompt["instruction"] = instruction
        prompt["claim"] = self.claim
        prompt["previous conversation"] = self.history
        prompt["current round agent judgements"] = judgements
        prompt["previous agent judgements"] = {}
        for agent in agents:
            if agent.name not in judgements:
                # Retrieve this agent's past prediction
                prompt["previous agent judgements"][agent.name] = {"judgement": agent.judgement, "explanation": agent.explanation}
        prompt["output requirements"] = "Your response should indicate one of the following:\nFinal Decision: The judgement (Please choose from these three answers: supporting, neutral, refuting)\nNeed Further Discussion: More information or clarification is needed before making a final decision.\nPlease provide a brief explanation for your choice."
        prompt = json.dumps(prompt, indent=2)

        response = chatgpt_response(prompt)

        # Calculate tokens
        input_tokens = calculate_tokens(prompt)
        output_tokens = calculate_tokens(response)

        print("############")
        print(self.name)
        print(prompt)
        print(response)

        if "Need Further Discussion" in response:
            return '', input_tokens, output_tokens
        else:
            final_decision_pattern = r'Final Decision:\s*(.*)'
            final_decision = re_extract(final_decision_pattern, response)
            return final_decision, input_tokens, output_tokens
