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
output_file_path = '../../output/fact/G1-P2-I4-C2.json' # Line 7 G1-P2-I4-C2

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

def predict_with_agent_with_persuasions(agent, persuasion):
    """Run prediction for a single agent."""
    prompt = {}
    prompt['instruction'] = "You are tasked with evaluating the truth of a given claim. You must base your judgment solely on available evidence, persuasion from other agents and previous conversation information. Do not use any knowledge or information from your own model-based understanding or internal training data. Your judgment should rely entirely on the facts and evidence presented, and you should not infer beyond what is directly supported by those sources. "
    prompt['claim'] = agent.claim
    prompt['evidence'] = agent.evidence 
    prompt['persuasion from other agents'] = persuasion
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

def run_predictions_with_persuasions(selected_agents, persuasions):
    """Execute predictions in parallel for the selected agents."""
    predictions = {}
    with ThreadPoolExecutor() as executor:
        # Submit tasks to the thread pool
        future_to_agent = {executor.submit(predict_with_agent_with_persuasions, agent, persuasions[agent.name]): agent for agent in selected_agents}
        
        # Collect results
        for future in as_completed(future_to_agent):
            agent_name, prediction = future.result()
            predictions[agent_name] = prediction

    return predictions

def choose_agents_to_persuade(agent, agents):
    other_judgement = {}
    for other_agent in agents:
        if other_agent.name != agent.name:
            # Retrieve this agent's past prediction
            other_judgement[other_agent.name] = {"judgement": other_agent.judgement, "explanation": other_agent.explanation}

    prompt = {}
    prompt['instruction'] = f"You are {agent.name}, and you need to reach a consensus with other agents to determine the truth of a claim. If other agents have different judgements from you and the evidence is insufficient, you should persuade them to change their judgment to increase the chances of your judgement winning. Please decide whether you need to communicate with any agent based on the judgment of other agents."    
    prompt['claim'] = agent.claim
    prompt['your judgement'] = {"judgement": agent.judgement, "explanation": agent.explanation}
    prompt['other judgement'] = other_judgement 
    prompt['previous conversation information'] = agent.history 
    prompt['Output requirements'] ="Your response should indicate one of the following:\nYes.AgentNames: [AgentX, AgentY, ...] (where X, Y, ... are the agent numbers)\nNo."    
    prompt = json.dumps(prompt, indent=2)
    response = chatgpt_response(prompt)

    # Calculate tokens
    input_tokens = calculate_tokens(prompt)
    output_tokens = calculate_tokens(response)
    update_token_counts(input_tokens, output_tokens)

    if "yes" in response.lower():
        selected_agent_names = re.findall(r'Agent(\d+)', response)
        selected_agents = [agent for agent in agents if agent.name in [f"Agent{num}" for num in selected_agent_names]]
    else:
        selected_agents = []

    print("############")
    print(agent.name)
    print(prompt)
    print(response)
    print("Selected agents:", [agent.name for agent in selected_agents])

    return selected_agents


def generate_persuasion(agent, agent_to_persuade):
    prompt = {}
    prompt['instruction'] = f"You are {agent.name}, please leave your judgement to {agent_to_persuade.name}. You should deliver your judgement once you are confident enough and in a way to convince other agent with a short reason."    
    prompt['claim'] = agent.claim
    prompt['your judgement'] = {"judgement": agent.judgement, "explanation": agent.explanation}
    prompt['his judgement'] = {"judgement": agent_to_persuade.judgement, "explanation": agent_to_persuade.explanation} 
    prompt['previous conversation information'] = agent.history 
    prompt = json.dumps(prompt, indent=2)
    response = chatgpt_response(prompt)

    # Calculate tokens
    input_tokens = calculate_tokens(prompt)
    output_tokens = calculate_tokens(response)
    update_token_counts(input_tokens, output_tokens)

    print("############")
    print(agent.name)
    print(prompt)
    print(response)

    return agent_to_persuade.name, response


def run_generate_persuasion(agent, agents):
    persuasions = {}
    chosen_agents = choose_agents_to_persuade(agent, agents)
    with ThreadPoolExecutor() as executor:
        # Submit tasks to the thread pool
        future_to_agent = {executor.submit(generate_persuasion, agent, chosen_agent): chosen_agent for chosen_agent in chosen_agents}
        
        # Collect results
        for future in as_completed(future_to_agent):
            agent_name, persuasion = future.result()
            persuasions[agent_name] = persuasion

    return agent.name, persuasions

def merge_predictions(agents):
    all_predictions = {}

    for agent in agents:
        # Retrieve this agent's past prediction
        all_predictions[agent.name] = {"judgement": agent.judgement, "explanation": agent.explanation}
    
    return all_predictions

# Agent voting
def vote(predictions, type):
    """
    Vote across multiple predictions and return the most frequent result (if its frequency exceeds 2/3); otherwise return False
    :param predictions: A list of prediction dicts, each containing "result" and "explanation".
    :param type: If type=1, return the option exceeding 2/3; if type=2, return the most frequent option.
    :return: The most frequent result if it exceeds 2/3; otherwise False.
    """
    # Extract all 'result' values
    result_values = [prediction["judgement"] for prediction in predictions.values()]

    # Count how often each result appears
    result_counter = Counter(result_values)

    # Find the most frequent result and its count
    most_common_result, most_common_count = result_counter.most_common(1)[0]
    print(result_counter)
    print(f"Vote result: {most_common_result},{most_common_count},{most_common_count/len(predictions)}")

    if type == 1:
        if most_common_count == len(predictions): 
            return most_common_result
        else:
            return False
    else:
        return most_common_result
    
def generate_agents_history(agents, judgements, current_round):
    # Use a thread pool to parallelize each agent's history generation and update
    def process_agent(agent):
        history, input_tokens, output_tokens = agent.generate_history(judgements)
        update_token_counts(input_tokens, output_tokens)
        agent.update_history(agent.name, "", history, current_round)

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        executor.map(process_agent, agents)


# Each round: parallel speaking
def parallel_prediction_flow(evidence_data, round):
    claim = evidence_data["claim"]
    sentences = evidence_data["sentences_and_labels"]
    # Create multiple Agent instances and store them
    agents = []  # Used to store multiple Agent instances
    for i,item in enumerate(sentences):  # Create Agent instance
        agent = Agent(f"Agent{i+1}", claim, item["sentence"], item["label"])
        agents.append(agent)
        
    print(f"Round 1")
    predictions = run_predictions_simultaneous_talk(agents)

    vote_result = vote(predictions, 1)
    if vote_result:
        return vote_result, 1
    
    generate_agents_history(agents, predictions, 1)
    
    for i in range(2, round+1):
        print(f"Round {i}")
        persuasions = {}

        with ThreadPoolExecutor() as executor:
            # Submit tasks to the thread pool
            future_to_agent = {executor.submit(run_generate_persuasion, agent, agents): agent for agent in agents}
            
            # Collect results
            for future in as_completed(future_to_agent):
                agent_name, current_persuasion = future.result()
                for agent_to_persuade in current_persuasion:
                    if agent_to_persuade in persuasions:
                        persuasions[agent_to_persuade][agent_name] = current_persuasion[agent_to_persuade]
                    else:
                        persuasions[agent_to_persuade] = {agent_name: current_persuasion[agent_to_persuade]}

        print(json.dumps(persuasions, indent=2))

        selected_agents = [agent for agent in agents if agent.name in persuasions]

        current_predictions = run_predictions_with_persuasions(selected_agents, persuasions)

        predictions = merge_predictions(agents)

        vote_result = vote(predictions, 1)
        if vote_result:
            return vote_result, i
        
        generate_agents_history(agents, current_predictions, i)

    # If rounds exceed the limit and no consensus, return the most frequent result
    vote_result = vote(predictions, 2)
    return vote_result, round


def generate_medical_predictions(evidence_data_path, output_file_path):
    final_predictions = {}

    with open(evidence_data_path, "r") as file:
        evidence_data = json.load(file) 


    for hadm_id, hadm_data in tqdm(list(evidence_data.items()), desc="Processing records", unit="record"):
        print(f"claim_id:{hadm_id}")

        prediction, rounds = parallel_prediction_flow(hadm_data, 10)

        global current_round_input_tokens, current_round_output_tokens

        # Save prediction results to final_predictions dictionary
        final_predictions[hadm_id] = {
            "judgement": prediction,
            "correct_answer": hadm_data["labels"],
            "input_tokens": current_round_input_tokens,
            "output_tokens": current_round_output_tokens,
            "rounds": rounds
        }        
        print(f"Final result{final_predictions[hadm_id]}")

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
