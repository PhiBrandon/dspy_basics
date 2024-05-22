import dspy
from dsp.modules.anthropic import Claude
from dotenv import load_dotenv
import os
from typing import Literal
from pydantic import BaseModel, Field


load_dotenv()
# Sonnet -> claude-3-sonnet-20240229
# Haiku -> claude-3-haiku-20240307
# Opus -> claude-3-opus-20240229
llm = Claude(
    model="claude-3-haiku-20240307",
    api_key=os.environ["ANTHROPIC_API_KEY"],
    max_tokens=4000,
)
dspy.settings.configure(lm=llm)


customer_information = "Brandon loves cats and is an avid parrot lover as well. He doesn't always love dogs."
industry = "Ecommerce store that sells dog toys."
categories = str(["BEST CUSTOMER", "OKAY CUSTOMER", "WORST CUSTOMER"])

classification_signature = dspy.Predict("customer_information, industry, classifications -> explanation, customer_classification")
output = classification_signature(customer_information=customer_information, industry=industry, classifications=categories)

print()
print(output.explanation)
print()
print(output.customer_classification)
print()