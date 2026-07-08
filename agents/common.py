from typing import TypedDict, Annotated
from botocore.exceptions import ClientError, EndpointConnectionError
from langgraph.graph.message import add_messages
from langchain_aws import ChatBedrock
from langchain_openai import ChatOpenAI

from settings import get_llm_settings

MAX_TOTAL_ATTEMPTS = 5


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    structured_intent: dict
    config: str
    validation_result: str
    retry_count: int
    failure_context: str
    repair_attempts: int
    deployment_result: str      # written by DeploymentAgent
    network_memory: dict
    network_memory_summary: str
    current_question_index: int
    total_attempts: int


llm_settings = get_llm_settings()

if llm_settings.provider == "bedrock":
    model = ChatBedrock(
        model=llm_settings.model_id,
        region_name=llm_settings.aws_region,
    )
else:
    model = ChatOpenAI(
        model=llm_settings.model_id,
        api_key=llm_settings.lm_studio_api_key,
        base_url=f"{llm_settings.lm_studio_endpoint}/v1",
        temperature=0,
    )


def invoke_model(messages):
    try:
        return model.invoke(messages)
    except EndpointConnectionError as exc:
        raise RuntimeError(
            "Unable to reach the LLM endpoint. Check your internet/DNS connection "
            "or Bedrock endpoint access, then retry."
        ) from exc
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "Unknown")
        if error_code in {"UnrecognizedClientException", "InvalidClientTokenId", "ExpiredTokenException"}:
            raise RuntimeError(
                "Bedrock authentication failed. Check AWS_ACCESS_KEY_ID, "
                "AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN (if using temporary credentials), "
                "and AWS_DEFAULT_REGION in .env."
            ) from exc
        raise
    except Exception as exc:
        error_text = str(exc)
        if "NameResolutionError" in error_text or "Failed to resolve" in error_text:
            raise RuntimeError(
                "Unable to reach the LLM endpoint. Check your internet/DNS connection "
                "or Bedrock endpoint access, then retry."
            ) from exc
        raise
