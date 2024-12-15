from datasets import load_dataset
import json
import pandas as pd
import os
import re
import sys
class Dataset():
    '''
    Base class for dataset
    '''

    def __init__(self, ds_config):
        self.questions = []
        self.answers = []    

    def generate_prompts(self, prompts):
        '''
        Generate prompt for the dataset
        '''
        new_prompts = []
        for q, prompt in zip(self.questions, prompts):
            if isinstance(prompt, list):
                new_prompts.append([p.replace('<QUESTION>', q) for p in prompt])
            else:
                new_prompts.append(prompt.replace('<QUESTION>', q))
        return new_prompts

    def __len__(self):
        return len(self.questions)

class GSM8kDataset(Dataset):
    '''
    GSM8k dataset
    '''

    def __init__(self):
        '''
        Load the dataset
        '''
        self.name = 'gsm8k'
        split = 'test'
        if True:
            dataset = load_dataset('NLP_assignment3/NLPDL-task2/data')
        else:
            dataset = load_dataset(self.name)
        dataloader = dataset[split]
        self.questions = [batch['question'] for batch in dataloader]
        self.answers = [batch['answer'] for batch in dataloader]
        self.questions = self.questions[:10]
        self.answers = self.answers[:10]


    def generate_prompts(self, prompts):
        '''
        Generate prompt for the dataset
        '''
        new_prompts = []

        for q, prompt in zip(self.questions, prompts):
            if isinstance(prompt, list):
                new_prompts.append([p.replace('<QUESTION>', q) for p in prompt])
            else:
                new_prompts.append(prompt.replace('<QUESTION>', q))
        #print(new_prompts[0])
        return new_prompts

    def evaluate(self, raw_output, file_path, cnt_round=0):
        '''
        Evaluate the generated answers by the debate system
        '''
        cnt_match, cnt_sum = evaluate_gsm8k(raw_output, self.questions, self.answers, file_path, cnt_round)
        return cnt_match, cnt_sum
    
    def __len__(self):
        return len(self.questions)

def evaluate_gsm8k(raw_output, questions, correct_answers, file_path, cnt_cound=0):
    '''
    Evaluate the generated answers by the debate system
    raw_output: list[dict{'agent_n':response}], the generated answers by the debate system
    questions: list[str], the questions in the dataset
    correct_answers: list[dict{'question':str, 'answer':str}], the correct answers in the dataset
    '''
        # Create a DataFrame for evaluation
    # Create a DataFrame for evaluation
    df = pd.DataFrame({
        'question': questions,
        'response': raw_output,
        'correct_answer': [ans.split('####')[-1].strip() for ans in correct_answers],
        'qid': [id for id in range(len(questions))]
    })
    
    # Process responses
    def process_row(row):
        correct_answer = row['correct_answer']
        responses = row['response']
        
        if isinstance(responses, list):  # Multiple responses
            generated_answers = [get_generated_answer((r)) for r in responses]
            match_count = sum(1 for ans in generated_answers if ans == correct_answer)
            is_correct = match_count > len(responses) // 2
            match_agents = [i for i, ans in enumerate(generated_answers) if ans == correct_answer]
            return is_correct, match_count, match_agents
        else:  # Single response
            generated_answer = get_generated_answer((responses))
            is_correct = generated_answer == correct_answer
            match_count = 1 if is_correct else 0
            match_agents = [0] if is_correct else []
            return is_correct, match_count, match_agents
    
    # Apply processing to all rows
    df[['is_correct', 'match_count', 'match_agents']] = df.apply(process_row, axis=1, result_type='expand')
    
    # Format results in the desired structure
    eval_results = []
    for _, row in df.iterrows():
        eval_results.append({
            "question": row['question'],
            "response": row['response'],
            "correct_answer": row['correct_answer'],
            "is_correct": row['is_correct'],
            "match_count": row['match_count'],
            "match_agents": row['match_agents'],
            "cnt_round": cnt_cound
        })
    
    # Save results to the file in the requested format
    # if not os.path.exists(file_path):
    #     with open(file_path, "w", encoding="utf-8") as file:
    #         json.dump(eval_results, file, ensure_ascii=False, indent=4)
    # else:
    #     try:
    #         with open(file_path, "r", encoding="utf-8") as file:
    #             original_data = json.load(file)
    #     except FileNotFoundError:
    #         original_data = []

    #     # Combine with existing data if it exists
    #     if isinstance(original_data, list):
    #         updated_data = original_data + eval_results
    #     else:
    #         raise ValueError("Unsupported JSON structure in the existing file!")

    #     with open(file_path, "w", encoding="utf-8") as file:
    #         json.dump(updated_data, file, ensure_ascii=False, indent=4)
    
    # Summary statistics
    cnt_match = df['is_correct'].sum()
    cnt_sum = len(df)
    
    return cnt_match, cnt_sum

def main():
    '''
    Main function
    '''
    # Load the dataset
    dataset = GSM8kDataset()
    # Load the prompts
    if len(sys.argv) > 1:
        if sys.argv[1] == 'cot':
            prompts = ["What is the answer to the following math word problem? <QUESTION> Please provide the final answer after ####. example: #### 123. Let's think step by step."]
        elif sys.argv[1] == 'few-shot':
            prompts = ["Here are some examples. Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? ** Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nHow many clips did Natalia sell altogether in April and May? ** Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72. What is the answer to the following math word problem? Janet, a third grade teacher, is picking up the sack lunch order from a local deli for the field trip she is taking her class on. There are 35 children in her class, 5 volunteer chaperones, and herself. She she also ordered three additional sack lunches, just in case there was a problem. Each sack lunch costs $7. How much do all the lunches cost in total? Janet needs 35 lunches for the kids + 5 for the chaperones + 1 for herself + 3 extras = <<35+5+1+3=44>>44 lunches.\nEach lunch is $7, so lunch for the field trip costs $7 per lunch * 44 lunches = $<<7*44=308>>308 total\n#### 308 Please answer the following math question <QUESTION>. Please provide the final answer after ####. example: #### 123"]
        else:
            prompts = ['What is the answer to the following math word problem? <QUESTION> Please provide the final answer after ####. example: #### 123']
    else:
        prompts = ['What is the answer to the following math word problem? <QUESTION> Please provide the final answer after ####. example: #### 123']
    prompts = prompts * len(dataset.questions)
    prompts = dataset.generate_prompts(prompts)
    # Evaluate the dataset
    from openai import OpenAI

    client = OpenAI(api_key="sk-fe165b786cdd444cb582f16a261a389b", base_url="https://api.deepseek.com")

    cnt_match = 0
    for prompt, answer in zip(prompts, dataset.answers):
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        if get_generated_answer(response.choices[0].message.content) == answer.split('####')[-1].strip():
            cnt_match += 1
    print(f'Accuracy: {cnt_match}/{len(dataset.questions)}')

def get_last_number(data):
    data = data.replace(',', '')
    numbers = re.findall(r'\d+', data)
    if numbers:
        return numbers[-1]  # 返回最后一个完整数字串
    return ''

def get_generated_answer(data):
    answer = ''
    if '####' in data:
        part = data.split('####')[-1].strip()
        extracted = get_last_number(part)
        if extracted:  # 确保提取到了内容
            answer = extracted
    if not answer:
        extracted = get_last_number(data)
        if extracted:
            answer = extracted
    return answer

if __name__ == '__main__':
    main()    