```mermaid
graph TD
    Start --> Supervisor
    Supervisor -->|greeting| greeting
    Supervisor -->|document_drafting| document_drafting
    Supervisor -->|legal_research| legal_research
    Supervisor -->|consultation_request| consultation_request
    Supervisor -->|general_legal| general_legal
    Supervisor -->|finish| END
    greeting --> END
    document_drafting --> END
    legal_research --> END
    consultation_request --> END
    general_legal --> END
