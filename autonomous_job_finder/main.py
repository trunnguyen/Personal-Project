import asyncio
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT= str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.agent.graph import workflow
from src.utils.logger import logger

RUN_INTERVAL_SECONDS = 259200

async def run_auto():
    app = workflow.compile()

    while True:
        logger.info("Automation trigger")
        try:
            initial_state = {
                "found_jobs":[],
                "unfound_jobs":[],
                "highly_relevant_jobs":[],
            }
            await app.ainvoke(initial_state)
            logger.info("Cycle Complete")
        except Exception as e:
            logger.critical(f"Daemon execution failed: {e}")

        logger.info("Sleep for 3 days")
        await asyncio.sleep(RUN_INTERVAL_SECONDS)

if __name__ == "__main__":
    asyncio.run(run_auto())