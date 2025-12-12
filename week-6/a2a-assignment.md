# Week 6 Assignment: A2A Protocol - Multi-Department Workflow Design

## Assignment Overview

## Scenario

You are an **AI Solutions Architect** at **TechCorp Industries**, a mid-sized technology company with 2,000 employees. The CEO has tasked you with automating the **Quarterly Business Review (QBR)** process using A2A protocol.

### Current State (Manual Process)

The quarterly business review currently takes **3 weeks** to complete and involves:

1. **Finance Department** - Prepares financial statements, budget analysis, and forecasts
2. **Sales Department** - Compiles sales metrics, pipeline data, and customer acquisition numbers
3. **Operations Department** - Gathers operational KPIs, efficiency metrics, and project status
4. **HR Department** - Provides workforce analytics, hiring metrics, and employee satisfaction data
5. **Executive Team** - Reviews all data, makes strategic decisions, and approves final report

### Pain Points

- Data collection takes 10+ days due to manual coordination
- Inconsistent report formats across departments
- No real-time visibility into review progress
- Delays in getting executive approval
- Historical data not easily accessible for comparison

### Your Task

Design an A2A-based multi-agent system that automates this process, reducing the timeline from 3 weeks to **24-48 hours**.

---

## Part 1: Agent Design (40 Points)

### Requirements

Design **Agent Cards** for at least **5 agents** that will participate in this workflow.

For each agent, specify:

1. **Agent ID and Name** (5 points)
2. **Description** - What the agent does (5 points)
3. **Capabilities** - At least 3 capabilities per agent with schemas (15 points)
4. **Authentication & Authorization** (5 points)
5. **SLA Specifications** (5 points)
6. **Rate Limits** (5 points)

### Template

Use this JSON template for each agent:

```json
{
  "agent_card": {
    "id": "your-agent-id",
    "name": "Your Agent Name",
    "description": "What this agent does",
    "version": "1.0.0",
    "vendor": "TechCorp Industries",
    "endpoints": {
      "a2a": "https://agents.techcorp.com/[agent-name]/a2a",
      "health": "https://agents.techcorp.com/[agent-name]/health"
    },
    "capabilities": {
      "skills": [
        {
          "name": "capability_name",
          "description": "What this capability does",
          "input_schema": {
            "type": "object",
            "properties": {
              "param1": {"type": "string", "description": "..."},
              "param2": {"type": "integer", "description": "..."}
            },
            "required": ["param1"]
          },
          "output_schema": {
            "type": "object",
            "properties": {
              "result": {"type": "string"}
            }
          }
        }
      ],
      "delegation_support": true,
      "streaming_support": false
    },
    "authentication": {
      "methods": ["oauth2"],
      "required_scopes": ["read:data", "write:reports"]
    },
    "rate_limits": {
      "requests_per_minute": 100,
      "concurrent_tasks": 5
    },
    "sla": {
      "availability": "99.9%",
      "max_response_time_ms": 5000
    }
  }
}
```

### Suggested Agents (You may modify or add more)

1. **Finance Agent** - Handles all financial data and analysis
2. **Sales Agent** - Manages sales metrics and forecasts
3. **Operations Agent** - Tracks operational KPIs
4. **HR Agent** - Provides workforce analytics
5. **Orchestrator Agent** - Coordinates the entire workflow
6. **Report Generator Agent** - Creates unified reports
7. **Executive Agent** - Handles approvals and decisions

### Deliverable

📄 **File:** `agents.json` - Contains all agent card definitions

---

## Part 2: Workflow Definition (30 Points)

### Requirements

Design a complete workflow that:

1. **Triggers automatically** at the end of each quarter (5 points)
2. **Collects data in parallel** from all departments (5 points)
3. **Validates and normalizes** data from different sources (5 points)
4. **Aggregates results** using an appropriate strategy (5 points)
5. **Generates a unified report** (5 points)
6. **Routes for executive approval** with appropriate escalation (5 points)

### Template

Use this YAML template:

```yaml
workflow:
  id: "quarterly-business-review-v1"
  name: "Quarterly Business Review Automation"
  description: "Automated QBR data collection, analysis, and reporting"
  version: "1.0.0"
  
  trigger:
    type: "scheduled"  # or "event", "manual"
    schedule: "0 0 1 1,4,7,10 *"  # First day of each quarter
    # OR
    event: "quarter_end"
    source: "calendar_system"
    
  input_schema:
    type: "object"
    properties:
      quarter:
        type: "string"
        description: "Quarter identifier (e.g., Q1-2024)"
      year:
        type: "integer"
    required: ["quarter", "year"]
    
  steps:
    - step_id: "step_1_initialize"
      name: "Initialize Review"
      agent: "orchestrator_agent"
      action: "initialize_qbr"
      input:
        quarter: "${workflow.input.quarter}"
        year: "${workflow.input.year}"
      output: "initialization_result"
      timeout: "5m"
      
    - step_id: "step_2_collect_data"
      name: "Parallel Data Collection"
      parallel: true
      depends_on: ["step_1_initialize"]
      sub_steps:
        - agent: "finance_agent"
          action: "get_financial_data"
          output: "finance_data"
        - agent: "sales_agent"
          action: "get_sales_metrics"
          output: "sales_data"
        # Add more parallel steps...
        
    # Continue adding steps...
    
  error_handling:
    retry_policy:
      max_retries: 3
      backoff: "exponential"
      initial_delay: "1s"
    fallback:
      action: "notify_admin"
      
  aggregation:
    strategy: "merge_all"
    conflict_resolution: "latest_wins"
    
  notifications:
    on_complete:
      - channel: "email"
        recipients: ["executive-team@techcorp.com"]
    on_failure:
      - channel: "slack"
        webhook: "${secrets.SLACK_WEBHOOK}"
```

### Workflow Diagram

Include a visual diagram of your workflow (ASCII art or describe it clearly):

```
[Your workflow diagram here]

Example:
                    ┌──────────────┐
                    │   Trigger    │
                    │ (Quarter End)│
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ Orchestrator │
                    │    Agent     │
                    └──────┬───────┘
                           │
         ┌────────┬────────┼────────┬────────┐
         ▼        ▼        ▼        ▼        ▼
      [Finance] [Sales] [Ops]    [HR]    [...]
         │        │        │        │        │
         └────────┴────────┼────────┴────────┘
                           ▼
                    [Aggregation]
                           │
                    [Report Gen]
                           │
                    [Executive Approval]
                           │
                    [Complete]
```

### Deliverable

📄 **File:** `workflow.yaml` - Complete workflow definition

---

## Part 3: Governance Model (20 Points)

### Requirements

Define governance structures including:

1. **Permission Matrix** - Who can do what (8 points)
2. **Escalation Paths** - At least 3 scenarios (6 points)
3. **Conflict Resolution** - Strategy for handling conflicts (3 points)
4. **Audit Requirements** - What to log and retain (3 points)

### Template

```yaml
governance:
  # Permission Matrix
  permissions:
    agents:
      orchestrator_agent:
        trust_level: "high"
        can_delegate_to: ["*"]  # Can delegate to all agents
        can_access_data:
          - "all_departments"
          - "historical_qbr"
        restricted_actions: []
        
      finance_agent:
        trust_level: "medium"
        can_delegate_to: 
          - "reporting_agent"
        can_access_data:
          - "financial_data"
          - "budget_data"
        restricted_actions:
          - "modify_approved_reports"
          
      # Define for all agents...
      
  # Escalation Paths
  escalations:
    - scenario: "data_collection_timeout"
      description: "When a department agent fails to respond within SLA"
      trigger:
        condition: "response_time > sla_threshold"
        threshold: "30m"
      path:
        - level: 1
          action: "retry_with_fallback"
          target: "same_agent"
          timeout: "15m"
        - level: 2
          action: "notify_department_lead"
          target: "human_escalation"
          timeout: "1h"
        - level: 3
          action: "skip_and_note"
          target: "orchestrator_agent"
          note: "Mark data as unavailable"
          
    # Add 2 more escalation scenarios...
    
  # Conflict Resolution
  conflict_resolution:
    strategy: "hierarchy_based"
    rules:
      - conflict_type: "data_discrepancy"
        resolution: "use_source_of_truth"
        source_priority:
          - "erp_system"
          - "crm_system"
          - "manual_input"
      - conflict_type: "approval_conflict"
        resolution: "escalate_to_higher"
        
  # Audit Trail
  audit:
    enabled: true
    log_level: "detailed"
    events_to_log:
      - "all_agent_communications"
      - "data_access"
      - "delegations"
      - "escalations"
      - "approvals"
      - "errors"
    retention:
      duration_days: 365
      archive_after_days: 90
    compliance:
      - "SOX"
      - "GDPR"
      - "internal_policy"
```

### Deliverable

📄 **File:** `governance.yaml` - Complete governance documentation

---

## Part 4: Implementation Plan (10 Points)

### Requirements

Provide a brief implementation plan covering:

1. **Phased Rollout Approach** (4 points)
2. **Success Metrics** (3 points)
3. **Risk Mitigation** (3 points)

### Template

```markdown
# Implementation Plan: QBR Automation

## Phase 1: Foundation (Weeks 1-2)
- [ ] Task 1: ...
- [ ] Task 2: ...
- Success Criteria: ...

## Phase 2: Development (Weeks 3-6)
- [ ] Task 1: ...
- [ ] Task 2: ...
- Success Criteria: ...

## Phase 3: Testing (Weeks 7-8)
- [ ] Task 1: ...
- [ ] Task 2: ...
- Success Criteria: ...

## Phase 4: Pilot (Weeks 9-10)
- [ ] Task 1: ...
- [ ] Task 2: ...
- Success Criteria: ...

## Phase 5: Full Rollout (Weeks 11-12)
- [ ] Task 1: ...
- [ ] Task 2: ...
- Success Criteria: ...

---

## Success Metrics

| Metric | Current | Target | How to Measure |
|--------|---------|--------|----------------|
| QBR Completion Time | 3 weeks | 48 hours | ... |
| ... | ... | ... | ... |

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation Strategy |
|------|-------------|--------|---------------------|
| ... | High/Med/Low | High/Med/Low | ... |

---

## Dependencies
- ...
- ...

## Resource Requirements
- ...
- ...
```

### Deliverable

📄 **File:** `implementation-plan.md` - Implementation brief

---

## Submission Checklist

Before submitting, ensure your folder contains:

- [ ] `agents.json` - Agent card definitions for all agents
- [ ] `workflow.yaml` - Complete workflow definition with diagram
- [ ] `governance.yaml` - Governance model documentation
- [ ] `implementation-plan.md` - Implementation brief
- [ ] `README.md` - Overview of your design decisions (optional but recommended)

### Folder Structure

```
A2A_Assignment_[YourName]_[Date]/
├── README.md                 # (Optional) Design decisions overview
├── agents.json               # Part 1: Agent Cards
├── workflow.yaml             # Part 2: Workflow Definition
├── governance.yaml           # Part 3: Governance Model
└── implementation-plan.md    # Part 4: Implementation Plan
```

### Naming Convention

`A2A_Assignment_[YourName]_[Date].zip`

Example: `A2A_Assignment_JohnDoe_2024-01-15.zip`

---

## Evaluation Rubric

### Part 1: Agent Design (40 Points)

| Criteria | Points | Description |
|----------|--------|-------------|
| Completeness | 15 | All required agents with complete specifications |
| Schema Quality | 10 | Well-defined input/output schemas |
| Capability Design | 10 | Logical and comprehensive capabilities |
| Security Considerations | 5 | Appropriate auth and rate limiting |

### Part 2: Workflow Definition (30 Points)

| Criteria | Points | Description |
|----------|--------|-------------|
| Logic & Flow | 10 | Correct sequencing and parallel execution |
| Error Handling | 8 | Comprehensive retry and fallback logic |
| Integration | 7 | Proper agent coordination |
| Completeness | 5 | All workflow components present |

### Part 3: Governance Model (20 Points)

| Criteria | Points | Description |
|----------|--------|-------------|
| Permission Matrix | 8 | Clear and appropriate access controls |
| Escalation Paths | 6 | Realistic and well-defined scenarios |
| Conflict Resolution | 3 | Practical resolution strategies |
| Audit Trail | 3 | Comprehensive logging requirements |

### Part 4: Implementation Plan (10 Points)

| Criteria | Points | Description |
|----------|--------|-------------|
| Phased Approach | 4 | Realistic and logical phases |
| Success Metrics | 3 | Measurable and relevant metrics |
| Risk Mitigation | 3 | Identified risks with mitigation strategies |

---

## Tips for Success

1. **Be Specific** - Generic answers score lower than detailed, specific ones
2. **Think Practically** - Design for real-world implementation, not theory
3. **Consider Edge Cases** - What happens when things go wrong?
4. **Document Your Decisions** - Explain why you made certain choices
5. **Review the Session Notes** - Reference the patterns and examples discussed


*Good luck with your assignment!*

