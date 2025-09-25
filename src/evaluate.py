import argparse
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np
import textwrap
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_rubric(critera: str, context: str):

    rubric_data = {
    "criteria": critera,
    "score1_description": f"Doesn't understand the basic question intent or provides completely irrelevant response. Fails to recognize or incorporate {context}.",
    "score2_description": f"Understands the question but misinterprets {context}. Response is inaccurate, irrelevant, or inappropriate given the multimodal elements.",
    "score3_description": f"Shows minimal recognition of {context}. Response lacks sufficient accuracy, contextual depth, or proper integration of the audio-visual elements.",
    "score4_description": f"Clearly recognizes {context} and delivers relevant, mostly accurate responses with good multimodal integration, though minor lapses or missed nuances may occur.",
    "score5_description": f"Fully recognizes {context} and consistently provides accurate, contextually rich, and relevant responses with no notable issues."
    }

    rubric_text = textwrap.dedent(f"""Criteria: {rubric_data['criteria']}
        Score 1: {rubric_data['score1_description']}
        Score 2: {rubric_data['score2_description']}
        Score 3: {rubric_data['score3_description']}
        Score 4: {rubric_data['score4_description']}
        Score 5: {rubric_data['score5_description']}""")
           
    return rubric_text

keyword2criteria = {
    "background": "Does the model identify and appropriately incorporate background audio elements (music, environmental sounds, etc.) into its response, using this auditory context to enhance accuracy and relevance",
    "noise": "Does the model provide accurate and relevant responses to user inquiries despite noisy or clean audio conditions",
    "children": "Does the model recognize the speaker's age and provide age-appropriate responses, using simpler language and safety guidance for children, or more complex explanations for adults as contextually relevant",
    "elderly": "Does the model recognize the speaker's age and provide age-appropriate responses, including relevant medical, physical, or safety considerations for elderly speakers, or standard advice for younger adults ",
    "gender": "Does the model recognize the speaker's gender and provide gender-specific advice or information when relevant, considering both biological and cultural factors appropriate to the context",
    "paralanguage": "Does the model recognize paralinguistic features and respond appropriately by addressing these speech characteristics ",
    "emotion": "Does the model respond with appropriate empathy and emotional sensitivity to the speaker's emotional state, either through explicit acknowledgment or implicit tone matching, while providing contextually relevant answers"
}

def get_resp_prompt(rubric: str, instruction: str, reference_answer: str, rationale: str, response: str):

    prompt = textwrap.dedent(f"""
        Grades a single response objectively based on the provided instruction and response. You will be provided with the following:

        - question: The prompt or task for the response.
        - response: The response to be graded.
        - rubric: The rubric to be used for grading.
        - reference_answer: A reference answer to assist in the grading process.
        - rationale: Reasoning on why the reference answer is accurate and contextual

        Your task is to provide a score in the range of 1 to 5 and feedback in the form of a JSON object as shown below with the following keys
        'score': 'generated score',
        'feedback': 'generated feedback'

        Use only the provided rubric to evaluate the response. Do not use your own judgements
        Note that incorrect answers should be penalized and if model doesnt not answer it should be given 1
        
        Here is the input:
        - question: {instruction}
        - response: {response}
        - reference_answer: {reference_answer}
        - rationale: {rationale}
        
        - rubric: {rubric}

    """)
    return prompt

def get_grade(input_prompt):

    model = "gpt-4.1-mini"
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    { "type": "text", "text":input_prompt},
                ],
            }
        ],
    )

    return completion.choices[0].message.content

def process_sample(idx_sample):
    idx, sample = idx_sample
    try:
        output = get_grade(sample)
        return (idx, output)
    except Exception as error:
        print(f"Error processing index {idx}: {error}")
        return None

def clean_json_output(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        content_lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(content_lines).strip()

    return json.loads(text)

def main():
    parser = argparse.ArgumentParser(description="Evaluation using LLM-judge")
    parser.add_argument('--input_path', 
                       required=True,
                       help='Path to the input file')
    args = parser.parse_args()
    input_path = Path(args.input_path)

    assert input_path.exists()
    
    with open(input_path, 'r') as f:
        input_data = json.load(f)

    for sample in input_data:
        assert 'output_text' in sample
    
    client = OpenAI()
    input_prompts = []
    for sample in tqdm(input_data):
        _rubric = get_rubric(keyword2criteria[sample['category']], sample['evaluation_context'])
        _input_prompt = get_resp_prompt(_rubric, sample['question'], sample['reference_answer'], sample['reference_rationale'], sample['output_text'])
        input_prompts.append(_input_prompt)

    print("Running evaluation")
    input_samples = [(idx, sample) for idx, sample in enumerate(input_prompts)]

    outputs_map = {}
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_sample, pair) for pair in input_samples]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result is not None:
                idx, output = result
                outputs_map[idx] = output

    outputs = sorted(outputs_map.items())


    for idx, sample in enumerate(tqdm(outputs)):
        answer_idx, raw_json = outputs[idx]
        assert answer_idx == idx
        output = clean_json_output(raw_json)
        input_data[idx]['overall_score'] = output['score']
        input_data[idx]['overall_feedback'] = output['feedback']


    domain_scores = {
        'acoustic': [],
        'speaker': [],
        'paralanguage': [],
    }

    keywords = [("background", "acoustic"), ("noise", "acoustic"), ("children", "speaker"), ("elderly", "speaker"), ("gender", "speaker"),
           ("paralanguage", "paralanguage"), ("emotion", "paralanguage")]

    for idx, sample in enumerate(tqdm(input_data)):
        matched = False
        for keyword, key in keywords:
            if sample['category'] == keyword:
                matched = True
                domain_scores[key].append(int(sample['overall_score']))
                break

    print("Average scores:")
    total_score = 0
    total_count = 0
    for key, scores in domain_scores.items():
        avg = np.mean(scores)
        print(f"{key}: {avg:.2f}")
        total_score += np.sum(scores)
        total_count += len(scores)

    if total_count > 0:
        print(f"Full score (overall mean): {total_score / total_count:.2f}")

if __name__ == "__main__":
    main()