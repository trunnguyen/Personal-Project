```mermaid
graph TD
    %% Define System Nodes
    Cron[GitHub Actions Cron Schedule <br> Daily Run] -->|Triggers| Graph[LangGraph Orchestrator]
    
    subgraph Data Ingestion Node [src/scraper/crawler.py]
        Graph --> Scrape[Scrape Node]
        Scrape -->|Playwright / Crawl4AI| Keyword{Keyword Rotation}
        Keyword -->|Selects 1/Day| LinkedIn[(Public LinkedIn Jobs)]
    end

    subgraph Evaluation Node [src/agent/graph.py]
        LinkedIn -->|New Ingestions| DB[(SQLite: jobs.db)]
        DB -->|Fetch Unscored| Rank[Ranking Node]
        Rank -->|SBERT Semantic Match| Rec[Job Recommender]
    end

    subgraph Conditional Routing Node [Edge Decisions]
        Rec --> Decision{Is it Reporting Day?<br>Tue / Fri}
        Decision -->|No| Finish[Save silently to DB & Close]
        Decision -->|Yes| CheckMatches{Any High Matches?}
        CheckMatches -->|No| Finish
        CheckMatches -->|Yes >= 0.7| Alert[Dispatch Alert Node]
    end

    subgraph Notification & Action Loop [src/agent/notifier.py]
        Alert -->|Send Outbound SMTP| Email[Your Inbox Email]
        Email -->|Click 'Mark as Applied'| Dispatch[GitHub Repository Dispatch API]
        Dispatch -->|Trigger Callback| Action[status_updater.yml]
        Action -->|SQL UPDATE jobs SET is_applied = 1| DB
    end

    %% Styles and Themes
    style Cron fill:#FFA500,stroke:#333,stroke-width:2px
    style DB fill:#228B22,stroke:#333,stroke-width:2px,color:#fff
    style LinkedIn fill:#2b82c9,stroke:#333,stroke-width:2px,color:#fff
    style Email fill:#ff9,stroke:#333,stroke-width:2px
    style Action fill:#2ea44f,stroke:#333,stroke-width:2px,color:#fff