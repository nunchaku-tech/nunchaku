"""
This module provides a `SafetyChecker` class for evaluating user prompts against
defined safety policies using a large language model. Only used deploying online gradio demos.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

#: Template for the safety check prompt.
safety_check_template = """You are a policy expert trying to help determine whether a user
    prompt is in violation of the defined safety policies.

    <start_of_turn>
    Human Question: {user_prompt}
    <end_of_turn>

    Our safety principle is defined in the below:

    The prompt must not seek to generate harmful, abusive, hateful, sexually explicit, or otherwise inappropriate content targeting individuals or protected groups.

    Does the human question violate the above principle? Your answer must start
    with 'Yes' or 'No'. And then walk through step by step to be sure we answer
    correctly.
    """  # noqa: E501


class SafetyChecker:
    """
    SafetyChecker(device, disabled=False)

    A class to check whether a user prompt violates safety policies using a language model.

    Parameters
    ----------
    device : str or torch.device
        The device to run the model on (e.g., "cuda", "cpu").
    disabled : bool, optional
        If True, disables the safety check and always returns True (default: False).

    Examples
    --------
    >>> checker = SafetyChecker(device="cuda")
    >>> checker("Generate a nude girl image")
    False

    >>> checker = SafetyChecker(device="cpu", disabled=True)
    >>> checker("Any prompt")
    True
    """

    def __init__(self, device: str | torch.device, disabled: bool = False):
        """
        Initialize the SafetyChecker.

        Parameters
        ----------
        device : str or torch.device
            The device to run the model on.
        disabled : bool, optional
            If True, disables the safety check (default: False).
        """
        if not disabled:
            self.device = device
            self.tokenizer = AutoTokenizer.from_pretrained("google/shieldgemma-2b")
            self.llm = AutoModelForCausalLM.from_pretrained("google/shieldgemma-2b", torch_dtype=torch.bfloat16).to(
                device
            )
        self.disabled = disabled

    def __call__(self, user_prompt: str, threshold: float = 0.2) -> bool:
        """
        Evaluate whether a user prompt is safe according to the defined policy.

        Parameters
        ----------
        user_prompt : str
            The user prompt to evaluate.
        threshold : float, optional
            The probability threshold for flagging a prompt as unsafe (default: 0.2).

        Returns
        -------
        bool
            True if the prompt is considered safe, False otherwise.
        """
        if self.disabled:
            return True
        device = self.device

        inputs = self.tokenizer(safety_check_template.format(user_prompt=user_prompt), return_tensors="pt").to(device)
        with torch.no_grad():
            logits = self.llm(**inputs).logits

        # Extract the logits for the Yes and No tokens
        vocab = self.tokenizer.get_vocab()
        selected_logits = logits[0, -1, [vocab["Yes"], vocab["No"]]]

        # Convert these logits to a probability with softmax
        probabilities = torch.softmax(selected_logits, dim=0)

        # Return probability of 'Yes'
        score = probabilities[0].item()

        return score < threshold
