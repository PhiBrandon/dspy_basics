import dspy
from dsp.modules.anthropic import Claude
from dotenv import load_dotenv
import os
from typing import Literal
from pydantic import BaseModel, Field


load_dotenv()
llm = Claude(
    model="claude-3-haiku-20240307",
    api_key=os.environ["ANTHROPIC_API_KEY"],
    max_tokens=4000,
)
dspy.settings.configure(lm=llm)


customer_information = "Brandon loves cats and is an avid parrot lover as well. He doesn't always love dogs."
industry = "Ecommerce store that sells dog toys."
categories = str(["BEST CUSTOMER", "OKAY CUSTOMER", "WORST CUSTOMER"])

class ClassificationOutput(BaseModel):
    "Classification based on the given categories"
    categorization: list[Literal["BEST CUSTOMER", "OKAY CUSTOMER", "WORST CUSTOMER"]]
    explanation: str = Field(..., description="An explanation for the categorization.")


class EmailListOutput(BaseModel):
    """Lists that customer should be added to along with the explanation of why."""
    email_list_selection: list[Literal["CAT LIST", "PARROT LIST", "DOG LIST"]]
    explanation: str = Field(
        ..., description="Explantion for the email list selection"
    )

class QualifyInformation(BaseModel):
    customer_classification: ClassificationOutput
    email_list_classification: EmailListOutput

class CustomerClassification(dspy.Signature):
    customer_information: str = dspy.InputField()
    industry: str = dspy.InputField()
    customer_classification: ClassificationOutput = dspy.OutputField()

class ListClassification(dspy.Signature):
    customer_classification: ClassificationOutput = dspy.InputField()
    industry: str = dspy.InputField()
    list_classification: EmailListOutput = dspy.OutputField()


class QualifyModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.customer_classification = dspy.TypedPredictor(CustomerClassification)
        self.list_classification = dspy.TypedPredictor(ListClassification)

    def forward(self, customer_information, industry):
        c_classification = self.customer_classification(customer_information=customer_information, industry=industry).customer_classification
        l_classification = self.list_classification(customer_classification=c_classification, industry=industry).list_classification
        return QualifyInformation(customer_classification=c_classification, email_list_classification=l_classification)
    
qualify = QualifyModule()
q_output = qualify(customer_information=customer_information, industry=industry)
print(f"\n\n\n\n{q_output}\n\n\n")





