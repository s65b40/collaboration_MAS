import json

def filter_claim_data(data):
    """
    Filter data, keep sentences whose annotations agree, and return a new structure with claim, labels, and sentences_and_labels.
    If all sentences for an item share the same label, drop that item.

    Args:
        data (list): Input data containing multiple claim objects, each with sentences and their annotations.

    Returns:
        dict: Filtered data keyed by claim_id with claim, labels, and sentences_and_labels.
    """
    filtered_data = {}

    # Iterate over dataset
    for item in data:
        # Check whether sentences have annotations
        if "sentence_annotations" not in item:
            continue
        
        # Store sentence/label pairs that meet criteria
        valid_sentences_and_labels = []
        
        # Collect labels for all sentences
        all_labels = set()

        # Iterate sentences and annotations
        for sentence_index, annotations in item["sentence_annotations"].items():
            # Get all annotations for current sentence
            annotations_set = set([annotation["annotation"] for annotation in annotations])
            
            # If annotations agree, keep the sentence and label
            if len(annotations_set) == 1:  # annotations_set has one element, so they agree
                sentence = item["sentences"][sentence_index]
                label = annotations[0]["annotation"]  # take any annotation (they're identical)
                valid_sentences_and_labels.append({"sentence": sentence, "label": label})
                all_labels.add(label)  # add this sentence's label to the set
        
        # Skip item if all sentence labels are identical
        if len(all_labels) == 1:
            continue
        
        # Keep item if filtered sentence list is non-empty
        if valid_sentences_and_labels:
            # Use passage value as the final label
            final_label = item["labels"]["passage"]
            
            # Build final structure, dropping claim_id since it's the dict key
            filtered_item = {
                "claim": item["claim"],
                "labels": final_label,  # use passage directly as label
                "sentences_and_labels": valid_sentences_and_labels  # contains valid sentences and labels
            }
            
            # Use claim_id as the key
            filtered_data[item["claim_id"]] = filtered_item

    return filtered_data


def load_jsonl(file_path):
    """
    Read JSON objects from a JSONL file into a list.
    
    Args:
        file_path (str): File path
    
    Returns:
        list: List of JSON objects
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # Parse each line's JSON object
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    return data


def save_json(data, file_path):
    """
    Save data as JSON, keyed by claim_id.
    
    Args:
        data (dict): Data to save; must be a dict keyed by claim_id
        file_path (str): Output file path
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def process_filtered_data(input_file_path, output_file_path):
    """
    Further filter processed data:
    1. If the total sentence count is 6 or fewer, keep all sentences regardless of label.
    2. If there are more than 6 sentences, keep up to 6, prioritizing sentences whose label matches labels["passage"].

    Args:
        input_file_path (str): Input file path (processed filtered_test_certain.json)
        output_file_path (str): Output file path for filtered results
    """
    # Load existing processed results
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    filtered_data = {}

    for claim_id, item in data.items():
        # Store sentences that meet criteria
        passage_sentences = []
        other_sentences = []

        # Iterate sentences and labels
        for sentence_info in item["sentences_and_labels"]:
            sentence = sentence_info["sentence"]
            label = sentence_info["label"]
            if sentence:
                if label == item["labels"]:
                    passage_sentences.append({"sentence": sentence, "label": label})
                else:
                    other_sentences.append({"sentence": sentence, "label": label})

        # If total sentences <= 6, keep all
        if len(passage_sentences) + len(other_sentences) <= 6:
            valid_sentences_and_labels = passage_sentences + other_sentences
        else:
            # If more than 6, first keep sentences whose label matches passage
            valid_sentences_and_labels = passage_sentences[:6]

            # If fewer than 6 matching sentences, fill from others
            remaining_count = 6 - len(valid_sentences_and_labels)
            valid_sentences_and_labels += other_sentences[:remaining_count]

        # If filtered list is not empty, keep the item
        if valid_sentences_and_labels:
            # Create new data structure
            filtered_item = {
                "claim": item["claim"],
                "labels": item["labels"],  # keep original labels
                "sentences_and_labels": valid_sentences_and_labels
            }
            filtered_data[claim_id] = filtered_item

    # Save final result to a new file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=4)


# Example usage
if __name__ == "__main__":
    # # Example: data stored in a JSONL file
    # input_file_path = '../../data/AmbiFC/test.certain.jsonl'
    # output_file_path = '../../data/AmbiFC/filtered_test_certain.json'

    # # Read JSONL file line by line
    # data = load_jsonl(input_file_path)

    # # Call the filter function
    # filtered_result = filter_claim_data(data)

    # # Save filtered results to a new file
    # save_json(filtered_result, output_file_path)

    input_file_path = '../../data/AmbiFC/filtered_test_certain.json'
    output_file_path = '../../data/AmbiFC/filtered_test_certain_processed_new.json'
    process_filtered_data(input_file_path, output_file_path)
