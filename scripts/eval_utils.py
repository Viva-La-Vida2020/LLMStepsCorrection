import re
import numpy as np


def extract_steps(step_string):
    """
    Extract and parse steps from a given string.

    Parameters:
    - step_string (str): A string containing steps, e.g., 'Steps: m_1 = a + b ; m_2 = m_1 - c. '

    Returns:
    - (list of lists): A parsed list of steps, e.g., [['m_1', '=', 'a', '+', 'b'], ['m_2', '=', 'm_1', '-', 'c']]
    """
    # Remove 'Steps: ' prefix and split by ';'
    step_string = step_string.replace('. ', '')
    step_string = step_string.replace('Steps:', '').strip()
    steps = step_string.split(';')

    parsed_steps = []
    for step in steps:
        # Split each step by spaces and remove extra spaces
        parsed_step = [s.strip() for s in re.split(r'[\s=]+', step) if s.strip()]
        if parsed_step:  # Only add non-empty steps
            parsed_steps.append(parsed_step)

    return parsed_steps


def compute_steps_acc(pred_steps, true_steps):
    """
    Compute the steps accuracy by comparing predicted steps with true steps.

    Parameters:
    - pred_steps (list of lists): Predicted steps extracted from model output.
    - true_steps (list of lists): True steps extracted from example data.

    Returns:
    - (float): Steps accuracy (1.0 if all steps match exactly, otherwise 0.0).
    """
    return 1.0 if pred_steps == true_steps else 0.0


def compute_answer(pred_steps, values):
    """
    Compute the final answer based on predicted steps and given values.

    Parameters:
    - pred_steps (list of lists): Predicted steps extracted from model output.
    - values (dict): A dictionary of variable values, e.g., {'a': 6.0, 'b': 9.0, 'c': 11.0}.

    Returns:
    - (float): The final computed answer.
    """
    final_answer = None

    try:
        for step in pred_steps:
            var = step[0]  # Variable to store the result of the current step

            # Get operands and operator from the step
            if step[1] in values.keys():
                operand1 = values.get(step[1], step[1])
            else:
                operand1 = step[1]
            operator = step[2]
            if step[3] in values.keys():
                operand2 = values.get(step[3], step[3])
            else:
                operand2 = step[3]

            # Create an expression string to evaluate
            expression = f"{operand1} {operator} {operand2}"

            # Evaluate the expression and store the result in values
            values[var] = eval(expression)
            final_answer = values[var]
    except:
        final_answer = None

    return final_answer


def evaluate(pred, true_steps, values, answer):
    """
    Evaluate the model output against the expected steps and answer.

    Parameters:
    - pred (str): The model's output, e.g., 'Steps: m_1 = a + b ; m_2 = m_1 - c. '
    - example (dict): The example data containing the true steps, answer, and values.

    Returns:
    - (dict): A dictionary containing steps accuracy and answer accuracy.
    """
    # Extract steps from model output and true example
    pred_steps = extract_steps(pred)
    true_steps = extract_steps(true_steps)

    # Compute steps accuracy
    steps_acc = compute_steps_acc(pred_steps, true_steps)

    # Compute final answer from predicted steps
    computed_answer = compute_answer(pred_steps, values)

    # Compute answer accuracy
    if computed_answer is not None:
        answer_acc = 1.0 if np.isclose(computed_answer, answer.cpu()) else 0.0
    else:
        answer_acc = 0.0

    return {
        "steps_accuracy": steps_acc,
        "answer_accuracy": answer_acc
    }