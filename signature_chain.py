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


class CustomerClassification(dspy.Signature):
    customer_information: str = dspy.InputField(desc="information about the customer")
    industry: str = dspy.InputField(desc="store information")
    classifications: str = dspy.InputField(desc="potential classifications of customer")
    explanation: str = dspy.OutputField(desc="Explanation of the customer classification")
    customer_classification: str = dspy.OutputField(desc="classifcation of customer based on categories list")

class_sign = dspy.Predict(CustomerClassification)
output = class_sign(customer_information=customer_information, industry=industry, classifications=categories)
print()
print(output.explanation)
print()
print(output.customer_classification)

class ListClassification(dspy.Signature):
    customer_classification: str = dspy.InputField()
    industry: str = dspy.InputField()
    email_lists: str = dspy.InputField()
    list_classification: str = dspy.OutputField(desc="email lists customer should be added to based on classification and industry")

email_lists = str(["CAT LIST", "PARROT LIST", "DOG LIST"])
customer_classification = f"{output.explanation}\n{output.customer_classification}"
list_class = dspy.Predict(ListClassification)
output_2 = list_class(customer_classification=output.customer_classification, industry=industry, email_lists=email_lists)

print("\n\n")
print(output_2.list_classification)




