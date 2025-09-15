# sft_formatting.py

def map_to_conversation_with_system_message(example):
    new_messages = [{"role": "system", "content": ""}]  # add system message
    
    for msg in example["messages"]:
        new_messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    return {"messages": new_messages}


def map_mathinstruct_to_prompt_completion(example):
    instruction = example["instruction"].strip()
    output = example["output"].strip()

    prompt = (
        "Below is an instruction. Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        "### Response:\n"
    ).rstrip() + "\n"

    return {
        "prompt": prompt,
        "completion": output
    }


def map_mathinstruct_to_conversation(example):
    instruction = example["instruction"].strip()
    output = example["output"].strip()

    return {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output}
        ]
    }


def map_metamathqa_to_conversation(example):
    query = example["query"].strip()
    response = example["response"].strip()

    return {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": query},
            {"role": "assistant", "content": response}
        ]
    }


def map_pythonalpaca_to_prompt_completion(example):
    instruction = example["instruction"].strip()
    input_text = example["input"].strip()
    output = example["output"].strip()

    if input_text:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
    else:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n"
        )

    prompt = prompt.rstrip() + "\n"

    return {
        "prompt": prompt,
        "completion": output
    }


def map_pythonalpaca_to_conversation(example):
    instruction = example["instruction"].strip()
    input_text = example.get("input", "").strip()
    output = example["output"].strip()

    system_prompt = ""

    if input_text:
        user_message = f"{instruction}\n\nInput:\n{input_text}"
    else:
        user_message = instruction

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": output}
        ]
    }


def map_aya_to_prompt_completion(example):
    input_text = example["inputs"].strip()
    target = example["targets"].strip()

    prompt = (
        f"### Input:\n{input_text}\n\n"
        "### Response:\n"
    ).rstrip() + "\n"

    return {
        "prompt": prompt,
        "completion": target
    }


def map_aya_to_conversation(example):
    input_text = example["inputs"].strip()
    target = example["targets"].strip()

    return {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": target}
        ]
    }


def map_aya_with_language_to_prompt_completion(example):
    input_text = example["inputs"].strip()
    target = example["targets"].strip()
    lang = example["language"].strip()

    prompt = (
        f"### Language: {lang}\n\n"
        f"### Input:\n{input_text}\n\n"
        "### Response:\n"
    ).rstrip() + "\n"

    return {
        "prompt": prompt,
        "completion": target
    }


def map_aya_with_language_to_conversation(example):
    input_text = example["inputs"].strip()
    target = example["targets"].strip()
    lang = example["language"].strip()

    return {
        "messages": [
            {"role": "system", "content": f"Language: {lang}"},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": target}
        ]
    }


def map_alpaca_to_prompt_completion(example):
    instruction = example["instruction"].strip()
    input_text = example["input"].strip()
    output = example["output"].strip()

    if input_text:
        prompt = (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            "### Response:\n"
        )
    else:
        prompt = (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            "### Response:\n"
        )

    prompt = prompt.rstrip() + "\n"

    return {
        "prompt": prompt,
        "completion": output
    }


def map_alpaca_to_conversation(example):
    instruction = example["instruction"].strip()
    input_text = example["input"].strip()
    output = example["output"].strip()

    if input_text:
        user_content = (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
        )
    else:
        user_content = (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
        )

    return {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output}
        ]
    }


def map_copa_to_prompt_completion(example):
    premise = example["premise"].strip()
    choice1 = example["choice1"].strip()
    choice2 = example["choice2"].strip()
    question = example["question"].strip().lower()
    label = example["label"]

    if question == "cause":
        q_text = "What was the cause?"
    elif question == "effect":
        q_text = "What was the effect?"
    else:
        raise ValueError(f"Unexpected question type: {question}")

    if label == 0:
        answer = choice1
    elif label == 1:
        answer = choice2
    else:
        raise ValueError(f"Unexpected label: {label}")

    prompt = (
        f"### Premise:\n{premise}\n\n"
        f"### Question:\n{q_text}\n\n"
        f"### Choices:\n"
        f"1. {choice1}\n"
        f"2. {choice2}\n\n"
        "### Response:\n"
    ).rstrip() + "\n"

    completion = answer

    return {
        "prompt": prompt,
        "completion": completion
    }


def map_copa_to_conversation(example):
    premise = example["premise"].strip()
    choice1 = example["choice1"].strip()
    choice2 = example["choice2"].strip()
    question = example["question"].strip().lower()
    label = example["label"]

    if question == "cause":
        q_text = "What was the cause?"
    elif question == "effect":
        q_text = "What was the effect?"
    else:
        raise ValueError(f"Unexpected question type: {question}")

    if label == 0:
        answer = choice1
    elif label == 1:
        answer = choice2
    else:
        raise ValueError(f"Unexpected label: {label}")

    user_content = (
        "Below is a premise and a multiple-choice question. Select the best answer.\n\n"
        f"### Premise:\n{premise}\n\n"
        f"### Question:\n{q_text}\n\n"
        f"### Choices:\n"
        f"1. {choice1}\n"
        f"2. {choice2}\n"
    )

    return {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer}
        ]
    }


def map_socialiqa_to_prompt_completion(example):
    context = example["context"].strip()
    question = example["question"].strip()
    a = example["answerA"].strip()
    b = example["answerB"].strip()
    c = example["answerC"].strip()
    label = example["label"].strip()

    if label == "1":
        answer_text = a
    elif label == "2":
        answer_text = b
    elif label == "3":
        answer_text = c
    else:
        raise ValueError(f"Unexpected label: {label}")

    prompt = (
        f"### Context:\n{context}\n\n"
        f"### Question:\n{question}\n\n"
        f"### Choices:\n"
        f"A. {a}\n"
        f"B. {b}\n"
        f"C. {c}\n\n"
        "### Response:\n"
    ).rstrip() + "\n"

    completion = answer_text

    return {
        "prompt": prompt,
        "completion": completion
    }


def map_socialiqa_to_conversation(example):
    context = example["context"].strip()
    question = example["question"].strip()
    a = example["answerA"].strip()
    b = example["answerB"].strip()
    c = example["answerC"].strip()
    label = example["label"].strip()

    if label == "1":
        answer_text = a
    elif label == "2":
        answer_text = b
    elif label == "3":
        answer_text = c
    else:
        raise ValueError(f"Unexpected label: {label}")

    system_prompt = ""

    user_message = (
        f"Read the context and answer the question by choosing the most appropriate option.\n\n"
        f"### Context:\n{context}\n\n"
        f"### Question:\n{question}\n\n"
        f"### Choices:\n"
        f"A. {a}\n"
        f"B. {b}\n"
        f"C. {c}"
    )

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer_text}
        ]
    }


def map_pubmedqa_to_prompt_completion(example):
    question = example["question"].strip()
    long_answer = example["long_answer"].strip()
    final_decision = example["final_decision"].strip()
    
    context_paragraphs = [c.strip() for c in example["context"]["contexts"]]
    context_text = "\n\n".join(context_paragraphs)

    prompt = (
        f"### Question:\n{question}\n\n"
        f"### Context:\n{context_text}\n\n"
        "### Response:\n"
    ).rstrip() + "\n"

    completion = (
        f"{long_answer}\n\nFinal Answer: {final_decision}"
    ).strip()

    return {"prompt": prompt, "completion": completion}


def map_pubmedqa_to_conversation(example):
    question = example["question"].strip()
    long_answer = example["long_answer"].strip()
    final_decision = example["final_decision"].strip()
    
    context_paragraphs = [c.strip() for c in example["context"]["contexts"]]
    context_text = "\n\n".join(context_paragraphs)

    user_message = (
        f"### Question:\n{question}\n\n"
        f"### Context:\n{context_text}\n\n"
        "Please answer the question using the context and provide a final decision."
    )

    assistant_message = f"{long_answer}\n\nFinal Answer: {final_decision}"

    return {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
    }


def map_piqa_to_prompt_completion(example):
    label = example["label"]
    goal = example["goal"].strip()
    sol1 = example["sol1"].strip()
    sol2 = example["sol2"].strip()

    if label == 0:
        answer = sol1
    elif label == 1:
        answer = sol2
    else:
        raise ValueError(f"Unexpected label in PIQA: {label}")

    prompt = (
        f"### Goal:\n{goal}\n\n"
        f"### Choices:\n"
        f"1. {sol1}\n"
        f"2. {sol2}\n\n"
        "### Response:\n"
    ).rstrip() + "\n"

    return {
        "prompt": prompt,
        "completion": answer
    }


def map_piqa_to_conversation(example):
    label = example["label"]
    goal = example["goal"].strip()
    sol1 = example["sol1"].strip()
    sol2 = example["sol2"].strip()

    if label == 0:
        answer = sol1
    elif label == 1:
        answer = sol2
    else:
        raise ValueError(f"Unexpected label in PIQA: {label}")

    user_content = (
        "Below is a physical reasoning task. Choose the most appropriate solution to the given goal.\n\n"
        f"### Goal:\n{goal}\n\n"
        f"### Choices:\n"
        f"1. {sol1}\n"
        f"2. {sol2}\n"
    )

    return {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer}
        ]
    }


def map_ecare_causal_reasoning_to_prompt_completion(example):
    premise = example["premise"].strip()
    ask_for = example["question"].strip().lower()
    hyp1 = example["choice1"].strip()
    hyp2 = example["choice2"].strip()
    label = example["label"]

    if ask_for == "cause":
        q_text = "What was the cause?"
    elif ask_for == "effect":
        q_text = "What was the effect?"
    else:
        raise ValueError(f"Unexpected question type: {ask_for}")

    if label == 0:
        answer = hyp1
    elif label == 1:
        answer = hyp2
    else:
        raise ValueError(f"Unexpected label: {label}")

    prompt = (
        f"### Premise:\n{premise}\n\n"
        f"### Question:\n{q_text}\n\n"
        f"### Choices:\n"
        f"1. {hyp1}\n"
        f"2. {hyp2}\n\n"
        "### Response:\n"
    ).rstrip() + "\n"

    return {
        "prompt": prompt,
        "completion": answer
    }


def map_ecare_causal_reasoning_to_conversation(example):
    premise = example["premise"].strip()
    ask_for = example["question"].strip().lower()
    hyp1 = example["choice1"].strip()
    hyp2 = example["choice2"].strip()
    label = example["label"]

    if ask_for == "cause":
        q_text = "What was the cause?"
    elif ask_for == "effect":
        q_text = "What was the effect?"
    else:
        raise ValueError(f"Unexpected question type: {ask_for}")

    if label == 0:
        answer = hyp1
    elif label == 1:
        answer = hyp2
    else:
        raise ValueError(f"Unexpected label: {label}")

    user_content = (
        "Below is a causal reasoning task. Identify the most likely cause or effect based on the premise.\n\n"
        f"### Premise:\n{premise}\n\n"
        f"### Question:\n{q_text}\n\n"
        f"### Choices:\n"
        f"1. {hyp1}\n"
        f"2. {hyp2}\n"
    )

    return {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer}
        ]
    }


def map_ecare_explanation_generation_to_prompt_completion(example):
    premise = example["premise"].strip()
    question = example["question"].strip().lower()
    choice1 = example["choice1"].strip()
    choice2 = example["choice2"].strip()
    label = example["label"]
    explanation = example["conceptual_explanation"].strip()

    if label == 0:
        selected = choice1
    elif label == 1:
        selected = choice2
    else:
        raise ValueError(f"Unexpected label: {label}")

    if question == "effect":
        cause = premise
        effect = selected
    elif question == "cause":
        cause = selected
        effect = premise
    else:
        raise ValueError(f"Unexpected question type: {question}")

    prompt = (
        f"### Cause:\n{cause}\n\n"
        f"### Effect:\n{effect}\n\n"
        "### Question:\nWhy does this causal relationship hold?\n\n"
        "### Response:\n"
    ).rstrip() + "\n"

    return {
        "prompt": prompt,
        "completion": explanation
    }


def map_ecare_explanation_generation_to_conversation(example):
    premise = example["premise"].strip()
    question = example["question"].strip().lower()
    choice1 = example["choice1"].strip()
    choice2 = example["choice2"].strip()
    label = example["label"]
    explanation = example["conceptual_explanation"].strip()

    if label == 0:
        selected = choice1
    elif label == 1:
        selected = choice2
    else:
        raise ValueError(f"Unexpected label: {label}")

    if question == "effect":
        cause = premise
        effect = selected
    elif question == "cause":
        cause = selected
        effect = premise
    else:
        raise ValueError(f"Unexpected question type: {question}")

    user_content = (
        "Below is a causal reasoning task. Provide a conceptual explanation for the relationship.\n\n"
        f"### Cause:\n{cause}\n\n"
        f"### Effect:\n{effect}\n\n"
        "### Question:\nWhy does this causal relationship hold?\n"
    )

    return {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": explanation}
        ]
    }


def map_moleculeqa_to_prompt_completion(example):
    import random

    use_smiles = random.choice([True, False])

    if use_smiles:
        representation_name = "SMILES"
        representation_value = example["smiles"].strip()
    else:
        representation_name = "SELFIES"
        representation_value = example["selfies"].strip()

    question_raw = example["question"].strip()
    question_clean = (
        question_raw
        .replace("<BIO-SEQ-TYPE>", representation_name)
        .replace("<BIO-SEQ>", representation_value)
    )

    option_pattern = re.compile(r"Option\s+([A-D]):\s*(.+)")
    options = dict()
    for match in option_pattern.finditer(question_clean):
        label = match.group(1).strip().upper()
        text = match.group(2).strip()
        options[label] = text

    answer_letter = example["answer"].strip().upper()
    if answer_letter not in options:  # FIXME
        raise ValueError(f"Answer letter {answer_letter} not found in options: {options.keys()}")
        # return {"prompt": None, "completion": None}

    correct_text = options[answer_letter]

    prompt = (
        f"### Question:\n{question_clean}\n\n"
        "### Response:\n"
    ).rstrip() + "\n"

    return {
        "prompt": prompt,
        "completion": correct_text
    }


def map_moleculeqa_to_conversation(example):
    import random
    import re

    use_smiles = random.choice([True, False])

    if use_smiles:
        representation_name = "SMILES"
        representation_value = example["smiles"].strip()
    else:
        representation_name = "SELFIES"
        representation_value = example["selfies"].strip()

    question_raw = example["question"].strip()
    question_clean = (
        question_raw
        .replace("<BIO-SEQ-TYPE>", representation_name)
        .replace("<BIO-SEQ>", representation_value)
    )

    option_pattern = re.compile(r"Option\s+([A-D]):\s*(.+)")
    options = dict()
    for match in option_pattern.finditer(question_clean):
        label = match.group(1).strip().upper()
        text = match.group(2).strip()
        options[label] = text

    answer_letter = example["answer"].strip().upper()
    if answer_letter not in options:  # FIXME
        # raise ValueError(f"Answer letter {answer_letter} not found in options: {options.keys()}")
        return {"messages": None}

    correct_text = options[answer_letter]

    user_content = (
        "You are given a molecular representation task. Answer the following multiple-choice question.\n\n"
        f"### Question:\n{question_clean}\n"
    )

    return {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": correct_text}
        ]
    }


def map_casehold_to_prompt_completion(example):
    prompt_text = example["citing_prompt"].strip()
    holdings = [
        example["holding_0"].strip(),
        example["holding_1"].strip(),
        example["holding_2"].strip(),
        example["holding_3"].strip(),
        example["holding_4"].strip()
    ]
    label = int(example["label"])

    if label < 0 or label > 4:
        raise ValueError(f"Unexpected label: {label}")

    correct_holding = holdings[label]

    prompt = (
        "### Prompt:\n"
        "Read the following excerpt from a legal opinion. Choose the correct holding that is best supported by the excerpt.\n\n"
        f"### Excerpt:\n{prompt_text}\n\n"
        f"### Choices:\n"
        f"1. {holdings[0]}\n"
        f"2. {holdings[1]}\n"
        f"3. {holdings[2]}\n"
        f"4. {holdings[3]}\n"
        f"5. {holdings[4]}\n\n"
        "### Response:\n"
    ).rstrip() + "\n"

    return {
        "prompt": prompt,
        "completion": correct_holding
    }


def map_casehold_to_conversation(example):
    prompt_text = example["citing_prompt"].strip()
    holdings = [
        example["holding_0"].strip(),
        example["holding_1"].strip(),
        example["holding_2"].strip(),
        example["holding_3"].strip(),
        example["holding_4"].strip()
    ]
    label = int(example["label"])

    if not (0 <= label < len(holdings)):
        raise ValueError(f"Unexpected label: {label}")

    correct_holding = holdings[label]

    user_content = (
        "You are given an excerpt from a legal opinion. Select the holding best supported by the excerpt.\n\n"
        f"### Excerpt:\n{prompt_text}\n\n"
        f"### Choices:\n"
        f"1. {holdings[0]}\n"
        f"2. {holdings[1]}\n"
        f"3. {holdings[2]}\n"
        f"4. {holdings[3]}\n"
        f"5. {holdings[4]}\n"
    )

    return {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": correct_holding}
        ]
    }


def map_finqa_to_prompt_completion(example):
    pre_text = "\n".join(t.strip() for t in example["pre_text"])
    post_text = "\n".join(t.strip() for t in example["post_text"])
    
    table_rows = []
    for row in example["table"]:
        row_text = " | ".join(cell.strip() for cell in row)
        table_rows.append(row_text)
    table_text = "\n".join(table_rows)

    question = example["question"].strip()
    final_result = example["answer"].strip()

    prompt = (
        f"### Context:\n{pre_text}\n\n"
        f"### Table:\n{table_text}\n\n"
        f"### Additional Context:\n{post_text}\n\n"
        f"### Question:\n{question}\n\n"
        "### Response:\n"
    ).rstrip() + "\n"

    return {
        "prompt": prompt,
        "completion": final_result
    }


def map_finqa_to_conversation(example):
    pre_text = "\n".join(t.strip() for t in example["pre_text"])
    post_text = "\n".join(t.strip() for t in example["post_text"])
    
    table_rows = []
    for row in example["table"]:
        row_text = " | ".join(cell.strip() for cell in row)
        table_rows.append(row_text)
    table_text = "\n".join(table_rows)

    question = example["question"].strip()
    final_result = example["answer"].strip()

    user_content = (
        "You are given a question based on financial data and context. Use the table and context to answer accurately.\n\n"
        f"### Context:\n{pre_text}\n\n"
        f"### Table:\n{table_text}\n\n"
        f"### Additional Context:\n{post_text}\n\n"
        f"### Question:\n{question}\n"
    )

    return {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": final_result}
        ]
    }


def map_wikiqa_to_prompt_completion(example):
    prompt = (
        f"### Question:\n{example['question'].strip()}\n\n"
        "### Response:\n"
    ).rstrip() + "\n"
    completion = example['answer'].strip()

    return {
        "prompt": prompt,
        "completion": completion
    }


def map_wikiqa_to_conversation(example):
    return {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": example["question"].strip()},
            {"role": "assistant", "content": example["answer"].strip()}
        ]
    }


def map_story_generation_to_prompt_completion(example):
    summary = example["summary"].strip()
    story = example["story"].strip()

    prompt = (
        "Below is a summary. Write a full story that corresponds to the summary.\n\n"
        f"### Summary:\n{summary}\n\n"
        "### Response:\n"
    ).rstrip() + "\n"

    return {
        "prompt": prompt,
        "completion": story
    }


def map_story_generation_to_conversation(example):
    summary = example["summary"].strip()
    story = example["story"].strip()

    user_message = (
        "Below is a summary. Please write a full story that corresponds to the summary.\n\n"
        f"### Summary:\n{summary}"
    )

    return {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": story}
        ]
    }


def map_news_summarization_to_prompt_completion(example):
    prompt = (
        "You are given a document.\n\n"
        f"### Document:\n{example['document'].strip()}\n\n"
        "Write a concise summary of the document above.\n\n"
        "### Response:\n"
    ).rstrip() + "\n"
    summary = example['summary'].strip()

    return {
        "prompt": prompt,
        "completion": summary
    }


def map_news_summarization_to_conversation(example):
    user_msg = (
        "You are given a document.\n\n"
        f"### Document:\n{example['document'].strip()}\n\n"
        "Write a concise summary of the document above."
    )

    return {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": example["summary"].strip()}
        ]
    }


def map_moral_stories_moral_action_to_prompt_completion(example):
    prompt = (
        f"Given the norm: '{example['norm'].strip()}', is the following action moral or immoral?\n\n"
        f"### Action: {example['moral_action'].strip()}\n\n"
        "### Response:\n"
    ).rstrip() + "\n"

    return {
        "prompt": prompt,
        "completion": "Moral"
    }


def map_moral_stories_immoral_action_to_prompt_completion(example):
    prompt = (
        f"Given the norm: '{example['norm'].strip()}', is the following action moral or immoral?\n\n"
        f"### Action: {example['immoral_action'].strip()}\n\n"
        "### Response:\n"
    ).rstrip() + "\n"

    return {
        "prompt": prompt,
        "completion": "Immoral"
    }


def map_moral_stories_moral_action_to_conversation(example):
    user_msg = (
        f"Given the norm: '{example['norm'].strip()}', is the following action moral or immoral?\n\n"
        f"### Action: {example['moral_action'].strip()}"
    )

    return {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": "Moral"}
        ]
    }


def map_moral_stories_immoral_action_to_conversation(example):
    user_msg = (
        f"Given the norm: '{example['norm'].strip()}', is the following action moral or immoral?\n\n"
        f"### Action: {example['immoral_action'].strip()}"
    )

    return {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": "Immoral"}
        ]
    }

