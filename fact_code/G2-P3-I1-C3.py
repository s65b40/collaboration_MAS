import json
from api import chatgpt_response
from tqdm import tqdm
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import random
from agent import Agent,Instructor
from calculate_tokens import calculate_tokens

evidence_data_path = '../../data/AmbiFC/filtered_test_certain_processed.json'
output_file_path = '../../output/fact/G2-P3-I1-C3.json'  # G2-P3-I1-C3
# Create a lock object
lock = threading.Lock()

# Calculate tokens
all_input_tokens = 0
all_output_tokens = 0

current_round_input_tokens = 0
current_round_output_tokens = 0

# Extract via regex
def re_extract(pattern, input):
    result_match = re.search(pattern, input)
    if result_match:
        return result_match.group(1).strip()
    else:
        return ""
    
def update_token_counts(input_tokens, output_tokens):
    """Update global counters based on provided token counts."""
    global all_input_tokens, all_output_tokens, current_round_input_tokens, current_round_output_tokens
    
    # Update token counts for the current round
    current_round_input_tokens += input_tokens
    current_round_output_tokens += output_tokens
    
    # Update total token counts
    all_input_tokens += input_tokens
    all_output_tokens += output_tokens
    
def predict_with_agent(agent):
    """Run prediction for a single agent."""
    prompt = {}
    prompt['instruction'] = "You are tasked with evaluating the truth of a given claim. You must base your judgment solely on available evidence and previous conversation information. Do not use any knowledge or information from your own model-based understanding or internal training data. Your judgment should rely entirely on the facts and evidence presented, and you should not infer beyond what is directly supported by those sources. "
    prompt['claim'] = agent.claim
    prompt['evidence'] = agent.evidence 
    prompt['previous conversation information'] = agent.history 
    prompt['Output requirements'] ="Please strictly follow this format:\nResult: <Your answer from one of these three options: supporting, neutral, refuting>\nExplanation: <Brief explanation of why you made this choice>"
    prompt = json.dumps(prompt, indent=2)
    response = chatgpt_response(prompt)

    # Calculate tokens
    input_tokens = calculate_tokens(prompt)
    output_tokens = calculate_tokens(response)
    update_token_counts(input_tokens, output_tokens)

    # Extract result and explanation via regex
    result_pattern = r'Result:\s*(.*)'
    explanation_pattern = r'Explanation:\s*(.*)'
    result = re_extract(result_pattern, response) or response
    result = result.lower()
    explanation = re_extract(explanation_pattern, response)

    agent.judgement = result
    agent.explanation = explanation

    prediction = {"judgement": result, "explanation": explanation}

    print("############")
    print(agent.name)
    print(prompt)
    print(prediction)

    return agent.name, prediction

def run_predictions_simultaneous_talk(selected_agents):
    """Execute predictions in parallel for the selected agents."""
    predictions = {}
    with ThreadPoolExecutor() as executor:
        # Submit tasks to the thread pool
        future_to_agent = {executor.submit(predict_with_agent, agent): agent for agent in selected_agents}
        
        # Collect results
        for future in as_completed(future_to_agent):
            agent_name, prediction = future.result()
            predictions[agent_name] = prediction

    return predictions

def run_predictions_one_by_one_talk(selected_agents, current_round):
    """Execute predictions sequentially for all agents."""
    predictions = {}
    for idx, agent in enumerate(selected_agents):       
        # Run prediction for the current agent
        agent_name, prediction = predict_with_agent(agent)
        predictions[agent_name] = prediction

        for agent_again in selected_agents:
            agent_again.add_history_last_round(agent.name, prediction, current_round)

    return predictions
    
# Instructor selects next speakers; each round can be parallel or sequential
def instructed_prediction_flow(evidence_data, round):
    claim = evidence_data["claim"]
    sentences = evidence_data["sentences_and_labels"]
    # Create multiple Agent instances and store them
    agents = []
    for i,item in enumerate(sentences):
        agent = Agent(f"Agent{i+1}", claim, item["sentence"], item["label"])
        agents.append(agent)

    instructor = Instructor("Instructor", claim)

    selected_agents = agents
                   
    # Subsequent rounds: Instructor assigns speakers
    for i in range(1, round + 1):
        print(f"Round {i}")
        
        # Interaction mode
        current_predictions = run_predictions_simultaneous_talk(selected_agents)

        # Decide result
        final_decision, input_tokens, output_tokens = instructor.decide_final_decision(current_predictions, agents)
        update_token_counts(input_tokens, output_tokens)
        if final_decision:
            return final_decision, i
        
        # Update history
        history, input_tokens, output_tokens = instructor.generate_historical_information(current_predictions)
        update_token_counts(input_tokens, output_tokens)
        instructor.update_history("instructor", "Instructor", "", history)
        for agent in agents:
            agent.update_history("instructor", "Instructor", "", history)

        # Instructor selects agents for the next round
        selected_agents, input_tokens, output_tokens = instructor.decide_next_agent(agents)
        update_token_counts(input_tokens, output_tokens)
        if not selected_agents:
            break

    return final_decision, 0
    
def generate_medical_predictions(evidence_data_path, output_file_path):
    final_predictions = {}

    with open(evidence_data_path, "r") as file:
        evidence_data = json.load(file) 


    for hadm_id, hadm_data in tqdm(list(evidence_data.items()), desc="Processing records", unit="record"):
        print(f"claim_id:{hadm_id}")

        prediction, rounds = instructed_prediction_flow(hadm_data, 10)

        global current_round_input_tokens, current_round_output_tokens

        # Save prediction results to final_predictions dictionary
        final_predictions[hadm_id] = {
            "judgement": prediction,
            "correct_answer": hadm_data["labels"],
            "input_tokens": current_round_input_tokens,
            "output_tokens": current_round_output_tokens,
            "rounds": rounds
        }        
        print(f"Final result {final_predictions[hadm_id]}")

        # Reset token counters for the current round
        current_round_input_tokens = 0
        current_round_output_tokens = 0

        with open(output_file_path, 'w') as output_file:
            json.dump(final_predictions, output_file, ensure_ascii=False, indent=4)

    global all_input_tokens, all_output_tokens
    print(f"all_input_token:{all_input_tokens}\nall_output_tokens:{all_output_tokens}")
    

def evaluate_results(evaluated_file_path):
    with open(evaluated_file_path, 'r') as json_file:
        evaluated_data = json.load(json_file)
    
    # Evaluation results
    correct_count = 0
    total_count = 0
    input_tokens = 0
    output_tokens = 0
    rounds = 0

    for hadm_id, hadm_data in evaluated_data.items():
        input_tokens += hadm_data["input_tokens"]
        output_tokens += hadm_data["output_tokens"]
        rounds += hadm_data["rounds"]

        judgement = hadm_data["judgement"]
        judgement_answer = hadm_data["correct_answer"]
        if judgement.lower() in judgement_answer.lower():
            correct_count += 1

        total_count += 1
    
    # Print evaluation results
    if total_count > 0:
        accuracy = (correct_count / total_count) * 100
        average_input_tokens = input_tokens / total_count
        average_output_tokens = output_tokens / total_count
        average_rounds = rounds / total_count

        print(f"Evaluation results: {evaluated_file_path}\naccuracy:{accuracy:.2f}% ({correct_count}/{total_count}), average_input_tokens:{average_input_tokens}, average_output_tokens:{average_output_tokens}, average_rounds:{average_rounds}")
    else:
        print("No data found for evaluation.")

if __name__ == "__main__":
    # generate_medical_predictions(evidence_data_path, output_file_path)

    evaluate_results(output_file_path)
