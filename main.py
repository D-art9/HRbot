from core.llm_provider import llm
from core.logger import logger


def main():
    logger.info("HR Intelligence Agent Started")

    response = llm.invoke("Say system initialized")

    print(response.content)


if __name__ == "__main__":
    main()
