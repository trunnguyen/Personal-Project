import os
import sys
import asyncio
from typing import List, Dict, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from datetime import datetime
from pathlib import Path

PROJECT_ROOT= str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.analytics.training_pipeline import run_model_training
from src.utils.db_manager import JobDB
from src.utils.logger import logger
from src.analytics.job_recommender import Recommender
from src.scraper import crawler as job_crawler
from src.agent.notifier import Notifier

class AgentState(TypedDict):
    found_jobs:List[Dict[str,Any]]
    unfound_jobs:List[Dict[str,Any]]
    highly_relevant_jobs:List[Dict[str,Any]]

base_data_path=os.path.join(PROJECT_ROOT, "data")
db= JobDB(os.path.join(base_data_path, "jobs.db"),os.path.join(base_data_path, "jobs.csv"))
recommender = Recommender()
notifier = Notifier()


async def scrape_node(state:AgentState) -> Dict[str, Any]:

    logger.info("LangGraph Node [Scrape]: Triggering Crawler")
    target_db = os.path.join(base_data_path, "jobs.db")
    target_csv = os.path.join(base_data_path, "jobs.csv")


    await job_crawler.main(db_path=target_db,csv_path=target_csv)

    all_jobs= db.get_all_jobs()
    return {"found_jobs":all_jobs}

def score_node(state:AgentState) -> Dict[str, Any]:
    logger.info("Ranking and Filtering Jobs")
    unscored=db.get_unscored_jobs()
    if not unscored:
        logger.info("No unscored jobs found")
        return{"highly_relevant_jobs":[]}

    scored_jobs=recommender.update_ai_score(unscored)

    for job in scored_jobs:
        db.update_job_score(job['id'],job['ai_score'])

    high_matches = []
    for job in scored_jobs:
        threshold=0.7 if recommender.is_trained else 0.3
        if job.get('ai_score',0.0) >= threshold:
            high_matches.append(job)
    return {"highly_relevant_jobs": high_matches}

def alert_node(state:AgentState) -> Dict[str, Any]:

    logger.info("LangGraph Node [Alert]: Triggering Alerter")
    high_matches = state.get('highly_relevant_jobs',[])

    notifier.send_report(high_matches)
    notifier.send_notification(high_matches)

    return {}


def route_decision_edge(state:AgentState) -> str:
    high_matches = state.get('highly_relevant_jobs',[])

    high_matches.sort(key=lambda x: x.get('ai_score',0.0),reverse=True)
    if len(high_matches)>0:
        logger.info(f"Routing Rule: Found {len(high_matches)} highly relevant jobs")
        return "trigger_alert"
    else:
        logger.info(f"Routing Rule: No highly relevant jobs found")
        return "finish"

def retrain_node(state:AgentState) -> Dict[str, Any]:

    if datetime.now().weekday()==6:
        logger.info("Retraining Model")
        try:
            run_model_training()
            logger.info("Training Complete")
        except Exception as e:
            logger.error(f"Training failed: {e}")
    else:
        logger.info("Skip Weekly Retrain")

    return state


workflow = StateGraph(AgentState)


workflow.add_node("scrape_linkedin", scrape_node)
workflow.add_node("retrain_node", retrain_node)
workflow.add_node("ranking_job",score_node)
workflow.add_node("dispatch_alert",alert_node)

workflow.set_entry_point("scrape_linkedin")
workflow.add_edge("scrape_linkedin","retrain_node")
workflow.add_edge("retrain_node","ranking_job")


workflow.add_conditional_edges(
    "ranking_job",
    route_decision_edge,
    {
        "trigger_alert":"dispatch_alert",
        "finish":END
    }
)

workflow.add_edge("dispatch_alert",END)

app=workflow.compile()

async def main():
    initial_state:AgentState = {
        "found_jobs":[],
        "unfound_jobs":[],
        "highly_relevant_jobs":[]
    }
    await app.ainvoke(initial_state)

if __name__=="__main__":
    asyncio.run(main())

